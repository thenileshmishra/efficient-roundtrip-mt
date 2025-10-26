import os
import copy
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import wandb
import evaluate
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from lora_utils import apply_lora
from utils import (
    grpo_generate_sequences,
    grpo_compute_loss_and_logs,
)
from dl import TranslationDataModule


def _run_evaluation(
    model: torch.nn.Module,
    tokenizer,
    data_module,
    eval_cfg,
    *,
    tgt_lang_id: int,
    device: torch.device,
    max_new_tokens: int,
):
    if eval_cfg is None:
        return {}

    requested_metrics = getattr(eval_cfg, "translation_metric", [])
    if isinstance(requested_metrics, str):
        requested_metrics = [requested_metrics]
    requested_metrics = [metric.lower() for metric in requested_metrics]

    was_training = model.training
    model.eval()

    val_loader = data_module.val_dataloader()
    predictions = []
    references = []
    sample_records = []

    # Progress bar setup
    try:
        total_batches = len(val_loader)
    except TypeError:
        # Fallback if len(val_loader) is not available
        total_batches = None

    num_beams = int(getattr(eval_cfg, "num_beams", 1))
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        forced_bos_token_id=tgt_lang_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if num_beams > 1:
        generation_kwargs["early_stopping"] = True

    with torch.no_grad():
        with tqdm(total=total_batches, desc="Evaluating", leave=False) as pbar:
            for idx, batch in enumerate(val_loader):
                if len(batch) == 4:
                    encoder_inputs, ground_truths, _, sample_ids = batch
                else:
                    encoder_inputs, ground_truths, sample_ids = batch
                if isinstance(ground_truths, str):
                    ground_truths = [ground_truths]
                else:
                    ground_truths = list(ground_truths)
                sample_ids = [int(sid) for sid in (sample_ids if isinstance(sample_ids, (list, tuple)) else [sample_ids])]
                encoder_inputs = {k: v.to(device, non_blocking=True) for k, v in encoder_inputs.items()}
                generated = model.generate(
                    input_ids=encoder_inputs["input_ids"],
                    attention_mask=encoder_inputs.get("attention_mask"),
                    **generation_kwargs,
                )
                if generated.dim() == 1:
                    generated = generated.unsqueeze(0)
                for sample_idx, reference in enumerate(ground_truths):
                    hypothesis = tokenizer.decode(generated[sample_idx], skip_special_tokens=True)
                    ref_text = reference if isinstance(reference, str) else str(reference)
                    predictions.append(hypothesis)
                    references.append(ref_text)
                    sample_records.append((ref_text, hypothesis, sample_ids[sample_idx]))
                if pbar.total is not None:
                    pbar.update(1)

    results = {}
    if predictions and "bleu" in requested_metrics:
        bleu_metric = evaluate.load("sacrebleu")
        bleu_res = bleu_metric.compute(
            predictions=predictions,
            references=[[r] for r in references],
        )
        results["eval/bleu"] = float(bleu_res.get("score", 0.0))
    if predictions and "chrf++" in requested_metrics:
        chrf_metric = evaluate.load("chrf")
        chrf_res = chrf_metric.compute(
            predictions=predictions,
            references=[[r] for r in references],
            char_order=6,
            word_order=2,
        )
        results["eval/chrf++"] = float(chrf_res.get("score", 0.0))

    if results:
        print("Evaluation results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")

    if wandb.run is not None and (results or sample_records):
        log_payload = results.copy()
        # if sample_records:
        #     eval_table = wandb.Table(columns=["reference", "generated", "sample_id"], log_mode="MUTABLE")
        #     for reference, hypothesis, sample_id in sample_records:
        #         eval_table.add_data(reference, hypothesis, sample_id)
        #     log_payload["eval/Translations"] = eval_table
        wandb.log(log_payload)

    if was_training:
        model.train()

    return results


@hydra.main(version_base=None, config_path="./configs/", config_name="train")
def train(config: DictConfig):
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build model and tokenizer
    model, tokenizer = get_model(config)
    model.to(device)

    source_lang_code = config.task.data.source_lang
    target_lang_code = config.task.data.target_lang
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = source_lang_code
    if hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = target_lang_code

    def _tokenize_with_lang(texts, src_lang):
        if isinstance(texts, str):
            texts = [texts]
        previous_src_lang = getattr(tokenizer, "src_lang", None)
        if previous_src_lang is not None:
            tokenizer.src_lang = src_lang
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if previous_src_lang is not None:
            tokenizer.src_lang = previous_src_lang
        return encoded

    def _compute_grad_norm():
        grads = [
            p.grad.detach().float().norm(2)
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        if not grads:
            return 0.0
        return torch.norm(torch.stack(grads), 2).item()

    eval_cfg = getattr(config.task, "eval", None)
    eval_only = bool(getattr(eval_cfg, "only", False)) if eval_cfg is not None else False
    run_eval = bool(getattr(eval_cfg, "run", False)) if eval_cfg is not None else False
    if eval_only:
        run_eval = True
    run_training = not eval_only

    # Initialize Weights & Biases
    wandb.init(
        project="grpo-translation-nllb-multi-domain",
        name="25-gradient-steps-600m-outcome-reward-batch-6",
        config=OmegaConf.to_container(config, resolve=True),
        dir="/root/wandb",
    )
    train_table = None
    if run_training:
        train_table = wandb.Table(columns=["reference", "generated", "sample_id"], log_mode="MUTABLE")
    # Reference model (frozen copy of the policy)
    reference_name = getattr(config.task.model, "reference_name", None)
    ref_model = None
    if run_training:
        if reference_name is not None:
            ref_model = AutoModelForSeq2SeqLM.from_pretrained(
                reference_name,
                attn_implementation="flash_attention_2",
                dtype=model.dtype,
            ).to(device)
        else:
            ref_model = copy.deepcopy(model).to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    # Data module
    illegal_token_mask = None
    data = TranslationDataModule(
        tokenizer=tokenizer,
        illegal_token_mask=illegal_token_mask,
        data_path=config.task.data.path,
        dataset_config_name=config.task.data.dataset_config_name,
        source_lang=config.task.data.source_lang,
        target_lang=config.task.data.target_lang,
        sort_by_length=False,
        train_batch_size=int(getattr(config.task.training, "batch_size", 1)),
    )
    data.setup("fit")

    # Training hyperparameters
    max_epochs = int(config.task.training.epochs)
    updates_per_batch = int(getattr(config.task.training, "updates_per_batch", 50))
    num_return_sequences = int(getattr(config.task.training, "num_return_sequences", 4))
    max_new_tokens = int(getattr(config.task.constraints, "max_sentence_len", 128))
    gen_temperature = float(getattr(config.task.training, "gen_temperature", 1.3))
    beta = float(getattr(config.task.training, "beta", 0.04))
    clip_param = float(getattr(config.task.training, "clip_param", 0.2))
    tgt_lang_id = tokenizer.convert_tokens_to_ids(target_lang_code)
    src_lang_id = tokenizer.convert_tokens_to_ids(source_lang_code)

    # Optimizer setup
    if run_training:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found for the optimizer.")
        optimizer = torch.optim.AdamW(
            trainable_params, lr=float(getattr(config.task.training, "lr", 7e-6))
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)
        print(
            f"Trainable parameters: {trainable_params_count:,} "
            f"({trainable_params_count / max(total_params, 1):.2%}) out of {total_params:,}"
        )

    if run_training:
        for epoch in range(max_epochs):
            train_loader = data.train_dataloader()
            step_idx = 0

            for batch in train_loader:
                src_prompt, ground_truths, source_texts, sample_ids = batch
                encoder_inputs = {k: v.to(device, non_blocking=True) for k, v in src_prompt.items()}
                batch_size = encoder_inputs["input_ids"].size(0)

                for _ in range(updates_per_batch):
                    # Generate candidate sequences with current policy
                    generated_local = grpo_generate_sequences(
                        model,
                        tokenizer,
                        encoder_inputs,
                        tgt_lang_id,
                        max_new_tokens=max_new_tokens,
                        gen_temperature=gen_temperature,
                        num_return_sequences=num_return_sequences,
                        top_k=int(getattr(config.task.training, "top_k", 100)),
                        top_p=float(getattr(config.task.training, "top_p", 0.95)),
                        end_of_sentence_token_id=tokenizer.eos_token_id,
                    )

                    seq_len = generated_local.size(1)
                    try:
                        generated_all = generated_local.reshape(batch_size, num_return_sequences, seq_len)
                    except RuntimeError as exc:
                        raise RuntimeError(
                            "Unable to reshape generated sequences into (batch_size, num_return_sequences, seq_len). "
                            f"Batch size={batch_size}, num_return_sequences={num_return_sequences}, seq_len={seq_len}."
                        ) from exc

                    # Compute GRPO loss and update model
                    loss_forward, logs_forward = grpo_compute_loss_and_logs(
                        model,
                        ref_model,
                        tokenizer,
                        encoder_inputs,
                        generated_all,
                        ground_truths,
                        end_of_sentence_token_id=tokenizer.eos_token_id,
                        beta=beta,
                        clip_param=clip_param,
                        tgt_lang_id=tgt_lang_id,
                    )

                    optimizer.zero_grad()
                    loss_forward.backward()
                    forward_grad_norm = _compute_grad_norm()
                    optimizer.step()

                    # Prepare backward direction prompts from generated target sentences
                    forward_generated_flat = generated_all.reshape(
                        batch_size * num_return_sequences, seq_len
                    )
                    forward_prompt_texts = tokenizer.batch_decode(
                        forward_generated_flat, skip_special_tokens=True
                    )
                    backward_inputs_encoded = _tokenize_with_lang(
                        forward_prompt_texts, target_lang_code
                    )
                    backward_inputs = {
                        k: v.to(device, non_blocking=True)
                        for k, v in backward_inputs_encoded.items()
                    }
                    backward_batch_size = backward_inputs["input_ids"].size(0)
                    generated_backward = grpo_generate_sequences(
                        model,
                        tokenizer,
                        backward_inputs,
                        src_lang_id,
                        max_new_tokens=max_new_tokens,
                        gen_temperature=gen_temperature,
                        num_return_sequences=num_return_sequences,
                        top_k=int(getattr(config.task.training, "top_k", 100)),
                        top_p=float(getattr(config.task.training, "top_p", 0.95)),
                        end_of_sentence_token_id=tokenizer.eos_token_id,
                    )

                    back_seq_len = generated_backward.size(1)
                    try:
                        generated_backward_all = generated_backward.reshape(
                            backward_batch_size, num_return_sequences, back_seq_len
                        )
                    except RuntimeError as exc:
                        raise RuntimeError(
                            "Unable to reshape backward generated sequences into "
                            "(batch_size, num_return_sequences, seq_len). "
                            f"Batch size={backward_batch_size}, num_return_sequences={num_return_sequences}, "
                            f"seq_len={back_seq_len}."
                        ) from exc

                    expanded_source_texts = [
                        source_texts[idx // num_return_sequences]
                        for idx in range(backward_batch_size)
                    ]

                    loss_backward, logs_backward = grpo_compute_loss_and_logs(
                        model,
                        ref_model,
                        tokenizer,
                        backward_inputs,
                        generated_backward_all,
                        expanded_source_texts,
                        end_of_sentence_token_id=tokenizer.eos_token_id,
                        beta=beta,
                        clip_param=clip_param,
                        tgt_lang_id=src_lang_id,
                    )

                    optimizer.zero_grad()
                    loss_backward.backward()
                    backward_grad_norm = _compute_grad_norm()
                    optimizer.step()

                    if step_idx % int(getattr(config.task.training, "log_every_n_steps", 10)) == 0:
                        print(
                            f"[epoch {epoch}] step {step_idx} | "
                            f"f_loss={logs_forward['loss'].item():.4f} "
                            f"f_kl={logs_forward['kl'].item():.4f} f_reward={logs_forward['reward'].item():.4f} "
                            f"f_chrf={logs_forward['chrf'].item():.4f} f_grad={forward_grad_norm:.4f} | "
                            f"b_loss={logs_backward['loss'].item():.4f} "
                            f"b_kl={logs_backward['kl'].item():.4f} b_reward={logs_backward['reward'].item():.4f} "
                            f"b_chrf={logs_backward['chrf'].item():.4f} b_grad={backward_grad_norm:.4f}"
                        )
                        # Print the reference and one generated sequence for inspection
                        best_candidates = []
                        for idx, reference in enumerate(ground_truths):
                            decoded = tokenizer.decode(
                                generated_all[idx, 0], skip_special_tokens=True
                            )
                            best_candidates.append((reference, decoded, source_texts[idx], sample_ids[idx]))
                        ref_text, gen_text, src_text, ref_sample_id = best_candidates[0]
                        print(f"Source[{ref_sample_id}]: {src_text}")
                        print(f"Reference[{ref_sample_id}]: {ref_text}")
                        print(f"Generated[{ref_sample_id}]: {gen_text}")
                        back_best = tokenizer.decode(
                            generated_backward_all[0, 0], skip_special_tokens=True
                        )
                        print(f"Back-Generated[{ref_sample_id}]: {back_best}")

                        # Log to Weights & Biases
                        wandb.log(
                            {
                                "train/forward_loss": float(logs_forward["loss"].item()),
                                "train/forward_kl": float(logs_forward["kl"].item()),
                                "train/forward_chrf": float(logs_forward["chrf"].item()),
                                "train/forward_reward": float(logs_forward["reward"].item()),
                                "train/forward_gradient_norm": float(forward_grad_norm),
                                "train/backward_loss": float(logs_backward["loss"].item()),
                                "train/backward_kl": float(logs_backward["kl"].item()),
                                "train/backward_chrf": float(logs_backward["chrf"].item()),
                                "train/backward_reward": float(logs_backward["reward"].item()),
                                "train/backward_gradient_norm": float(backward_grad_norm),
                            }
                        )
                        # if train_table is not None:
                        #     for reference, decoded, ref_sample_id in best_candidates:
                        #         train_table.add_data(
                        #             reference,
                        #             decoded,
                        #             ref_sample_id,
                        #         )
                        #     wandb.log({"train/Translations": train_table}, step=step_idx)

                    # Optional periodic evaluation every N updates
                    if run_eval:
                        eval_every_n_batches = int(getattr(eval_cfg, "every_n_batches", 0))
                        if eval_every_n_batches > 0 and step_idx > 0 and (step_idx % eval_every_n_batches == 0):
                            _run_evaluation(
                                model,
                                tokenizer,
                                data,
                                eval_cfg,
                                tgt_lang_id=tgt_lang_id,
                                device=device,
                                max_new_tokens=max_new_tokens,
                            )

                    step_idx += 1

            # Refresh reference model at epoch boundaries
            ref_model = copy.deepcopy(model).to(device)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)

    if run_eval:
        _run_evaluation(
            model,
            tokenizer,
            data,
            eval_cfg,
            tgt_lang_id=tgt_lang_id,
            device=device,
            max_new_tokens=max_new_tokens,
        )

    # Save final model when training occurred
    if run_training:
        save_dir = os.path.join(os.getcwd(), "model")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    if wandb.run is not None:
        wandb.finish()




def get_model(config: DictConfig):
    model_cfg = config.task.model
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_cfg.name, attn_implementation="flash_attention_2", dtype=torch.bfloat16
    )

    use_lora = bool(getattr(model_cfg, "use_lora", False))
    if use_lora:
        for param in model.parameters():
            param.requires_grad_(False)

        lora_cfg = getattr(model_cfg, "lora", {})
        target_modules = tuple(getattr(lora_cfg, "target_modules", ("q_proj", "v_proj")))
        if not target_modules:
            raise ValueError("LoRA target modules must be a non-empty sequence of module suffixes.")

        replaced_modules = apply_lora(
            model=model,
            target_modules=target_modules,
            r=int(getattr(lora_cfg, "r", 8)),
            lora_alpha=float(getattr(lora_cfg, "alpha", 32)),
            lora_dropout=float(getattr(lora_cfg, "dropout", 0.05)),
        )

        print(f"LoRA enabled: adapted {len(replaced_modules)} modules with rank={int(getattr(lora_cfg, 'r', 8))}.")

    return model, tokenizer


if __name__ == "__main__":
    train()
