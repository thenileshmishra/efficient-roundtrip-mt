import torch
import copy
from pytorch_lightning import LightningModule


class TranslationGRPOTask(LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        lr,
        num_return_sequences=8,
        max_new_tokens=128,
        gen_temperature=0.9,
        beta=0.04,
        clip_param=0.2,
        tgt_lang_id=None,
        reference_model_name: str = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tokenizer"]) 

        self.model = model
        self.tokenizer = tokenizer
        self.end_of_sentence_token_id = tokenizer.eos_token_id
        self.tgt_lang_id = tgt_lang_id

        # Reference policy (frozen)
        if reference_model_name is not None:
            from transformers import AutoModelForSeq2SeqLM
            self.ref_model = AutoModelForSeq2SeqLM.from_pretrained(
                reference_model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=getattr(self.model, "dtype", None),
            ).to(self.model.device if self.model.device is not None else self.device)
        else:
            # Default to a frozen copy of current model's initial weights (loaded freshly)
            self.ref_model = None

        # Lazily freeze ref model in setup() after devices are set
        self._ref_initialized = False

    def setup(self, stage: str):
        if not self._ref_initialized:
            if self.ref_model is None:
                # Safely clone the current policy as the frozen reference
                self.ref_model = copy.deepcopy(self.model)
            # Ensure proper device placement and freeze
            self.ref_model.to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
            self._ref_initialized = True

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)

    @staticmethod
    def _gather_log_probs_from_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        log_probs = logits.log_softmax(dim=-1)
        gathered = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        return gathered

    def _compute_decoder_per_token_logps(self, model, encoder_inputs, decoder_input_ids, target_ids) -> torch.Tensor:
        # Teacher-forced forward pass over decoder tokens to get logits at each step
        outputs = model(
            input_ids=encoder_inputs["input_ids"],
            attention_mask=encoder_inputs.get("attention_mask", None),
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )
        logits = outputs.logits  # (B, L, V)
        per_token_logps = self._gather_log_probs_from_logits(logits, target_ids)  # (B, L)
        return per_token_logps

    @torch.no_grad()
    def _generate_sequences(self, encoder_inputs):
        generation_kwargs = dict(
            max_new_tokens=self.hparams.max_new_tokens,
            do_sample=True,
            temperature=self.hparams.gen_temperature,
            num_return_sequences=self.hparams.num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.end_of_sentence_token_id,
        )
        gen = self.model.generate(
            input_ids=encoder_inputs["input_ids"],
            attention_mask=encoder_inputs.get("attention_mask", None),
            forced_bos_token_id=self.tgt_lang_id,
            **generation_kwargs,
        )
        return gen

    def training_step(self, prompt, batch_idx):
        # prompt is a tuple: (source_encoded, target_string)
        src_prompt = prompt[0]
        ground_truth = prompt[1]

        # Move encoder inputs to device
        encoder_inputs = {
            k: v.to(self.device) for k, v in src_prompt.items()
        }

        # Generate K sequences (on-policy)
        generated_sequences = self._generate_sequences(encoder_inputs)  # (K, Ldec)

        # Build decoder inputs/targets for per-token log-prob computation
        # Shift: input is tokens up to last-1, target is tokens from position 1..
        decoder_input_ids = generated_sequences[:, :-1]
        target_ids = generated_sequences[:, 1:]

        # Compute per-token logps under current policy
        per_token_logps = self._compute_decoder_per_token_logps(
            self.model, encoder_inputs, decoder_input_ids, target_ids
        )  # (B, L-1)

        # Compute per-token logps under reference policy
        with torch.no_grad():
            ref_per_token_logps = self._compute_decoder_per_token_logps(
                self.ref_model, encoder_inputs, decoder_input_ids, target_ids
            )

        # Build completion mask: valid tokens until first eos (exclude pads)
        is_pad = target_ids == self.tokenizer.pad_token_id
        is_eos = target_ids == self.end_of_sentence_token_id
        # Mask positions after first eos per sequence
        eos_cumsum = is_eos.cumsum(dim=-1)
        after_eos = eos_cumsum >= 1
        completion_mask = (~is_pad) & (~after_eos)

        # Compute scalar rewards per sequence using BLEU+chrF against ground truth
        # Decode generated sequences without special tokens
        generated_texts = self.tokenizer.batch_decode(
            torch.where(generated_sequences == self.tokenizer.pad_token_id, self.end_of_sentence_token_id, generated_sequences),
            skip_special_tokens=True,
        )
        # Ground truth may be a single string replicated across K
        if isinstance(ground_truth, str):
            references = [ground_truth] * len(generated_texts)
        else:
            references = [ground_truth for _ in range(len(generated_texts))]

        import sacrebleu
        rewards = []
        for hyp, ref in zip(generated_texts, references):
            bleu = sacrebleu.sentence_bleu(hyp, [ref], smooth_method='exp').score / 100.0
            chrf = sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score / 100.0
            combined = 0.5 * bleu + 0.5 * chrf
            rewards.append(torch.log(torch.tensor(combined + 1e-8, device=self.device)))
        rewards = torch.stack(rewards, dim=0)  # (B,)

        # Normalize rewards -> advantages
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        advantages = advantages.unsqueeze(1).expand_as(per_token_logps)

        # PPO-style ratio without stored behavior logps (on-policy baseline trick)
        ratio = torch.exp(per_token_logps - per_token_logps.detach())
        clipped_ratio = torch.clamp(ratio, 1 - self.hparams.clip_param, 1 + self.hparams.clip_param)
        per_token_adv_loss = torch.min(ratio * advantages, clipped_ratio * advantages)

        # KL penalty versus reference policy
        ref_minus_pi = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(ref_minus_pi) - ref_minus_pi - 1.0

        per_token_obj = per_token_adv_loss - self.hparams.beta * per_token_kl
        per_token_obj = per_token_obj * completion_mask

        # Mean over valid tokens per sequence, then batch mean
        token_counts = completion_mask.sum(dim=-1).clamp_min(1)
        seq_losses = -per_token_obj.sum(dim=-1) / token_counts
        loss = seq_losses.mean()

        # Logging
        with torch.no_grad():
            kl_mean = (per_token_kl * completion_mask).sum() / token_counts.sum()
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/kl", kl_mean, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/reward", rewards.mean(), on_step=False, on_epoch=True, sync_dist=True)

        return loss
