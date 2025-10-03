import torch
import copy
import os
from pytorch_lightning import LightningModule
import sacrebleu


class TranslationGRPOTask(LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        lr,
        num_return_sequences=8,
        max_new_tokens=256,
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
                reference_model_name, attn_implementation="flash_attention_2", dtype=self.model.dtype
            ).to(self.model.device if self.model.device is not None else self.device)
        else:
            # Default to a frozen copy of current model's initial weights (loaded freshly)
            self.ref_model = None

    def on_train_epoch_start(self):
        # Refresh reference policy at epoch boundaries: π_ref ← π_θ
        self.ref_model = copy.deepcopy(self.model).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.config = self.model.config

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)

    @staticmethod
    def _gather_log_probs_from_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        log_probs = logits.log_softmax(dim=-1)
        gathered = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        return gathered

    def _compute_decoder_per_token_logps(self, model, encoder_inputs, decoder_input_ids, target_ids) -> torch.Tensor:
        # Teacher-forced forward pass over decoder tokens to get logits at each step
        repeated_input_ids = encoder_inputs["input_ids"].repeat(self.hparams.num_return_sequences, 1)
        attention_mask = encoder_inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(self.hparams.num_return_sequences, 1)

        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id).long()

        outputs = model(
            input_ids=repeated_input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
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

        encoder_inputs = {
            k: v.to(self.device) for k, v in src_prompt.items()
        }

        generated_sequences = self._generate_sequences(encoder_inputs)  # (K, Ldec)

        decoder_input_ids = generated_sequences[:, :-1]
        target_ids = generated_sequences[:, 1:]
        # Compute per-token logps under current policy
        per_token_logps = self._compute_decoder_per_token_logps(
            self.model, encoder_inputs, decoder_input_ids, target_ids
        )  # (B, L-1)

        with torch.no_grad():
            ref_per_token_logps = self._compute_decoder_per_token_logps(
                self.ref_model, encoder_inputs, decoder_input_ids, target_ids
            )

        # Build completion mask
        is_pad = target_ids == self.tokenizer.pad_token_id
        is_eos = target_ids == self.end_of_sentence_token_id
        eos_cumsum = is_eos.cumsum(dim=-1)
        after_eos = eos_cumsum >= 1
        completion_mask = (~is_pad) & (~after_eos)

        # Decode generated sequences without special tokens for reward computation
        generated_texts = self.tokenizer.batch_decode(
            torch.where(generated_sequences == self.tokenizer.pad_token_id, self.end_of_sentence_token_id, generated_sequences),
            skip_special_tokens=True,
        )
        
        # Ground truth may be a single string replicated across K
        if isinstance(ground_truth, str):
            references = [ground_truth] * len(generated_texts)
        else:
            references = [ground_truth for _ in range(len(generated_texts))]

        rewards = []
        # bleu_scores = []
        chrf_scores = []
        for hyp, ref in zip(generated_texts, references):
            # bleu = sacrebleu.sentence_bleu(hyp, [ref], smooth_method='exp').score / 100.0
            chrf = sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score / 100.0
            # bleu_scores.append(bleu)
            chrf_scores.append(chrf)
            # combined = 0.5 * bleu + 0.5 * chrf
            combined = chrf
            rewards.append(torch.log(torch.tensor(combined + 1e-8, device=self.device)))
        # bleu_mean = torch.tensor(bleu_scores, device=self.device).mean()
        chrf_mean = torch.tensor(chrf_scores, device=self.device).mean()
        rewards = torch.stack(rewards, dim=0)  # (B,)

        ### GRPO OBJECTIVE ###
        # Normalize rewards -> advantages (group-normalized over G samples for the prompt)
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
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/kl", kl_mean, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/reward", rewards.mean(), on_step=True, on_epoch=True, sync_dist=True)
        # log bleu and chrf
        # self.log("train/bleu", bleu_mean, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/chrf", chrf_mean, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src_prompt, ground_truth = batch
        encoder_inputs = {k: v.to(self.device) for k, v in src_prompt.items()}

        with torch.no_grad():
            gen = self.model.generate(
                input_ids=encoder_inputs["input_ids"],
                attention_mask=encoder_inputs.get("attention_mask", None),
                forced_bos_token_id=self.tgt_lang_id,
                max_new_tokens=self.hparams.max_new_tokens,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.end_of_sentence_token_id,
            )

        decoded = self.tokenizer.batch_decode(
            torch.where(gen == self.tokenizer.pad_token_id, self.end_of_sentence_token_id, gen),
            skip_special_tokens=True,
        )

        hyp = decoded[0]
        ref = ground_truth if isinstance(ground_truth, str) else str(ground_truth)
        chrf = sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score / 100.0
        self.log("val/chrf", chrf, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_train_end(self):
        # Save HF model and tokenizer at the end of training in the model directory
        model_dir = os.path.join(getattr(self.trainer, "default_root_dir", "."), "model")
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
