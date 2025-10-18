import os
import copy
import hydra
import torch
import torch.distributed as dist
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
import time

def _is_dist():
    return dist.is_available() and dist.is_initialized()

def _get_dist_info():
    if _is_dist():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def _broadcast_model_(model, src: int = 0):
    if not _is_dist():
        return
    for param in model.parameters():
        dist.broadcast(param.data, src=src)
    for buf in model.buffers():
        dist.broadcast(buf.data, src=src)


def _broadcast_object(obj, src: int = 0):
    if not _is_dist():
        return obj
    container = [obj if dist.get_rank() == src else None]
    dist.broadcast_object_list(container, src=src)
    return container[0]


def _pad_and_all_gather_sequences(seqs: torch.Tensor, pad_id: int, device: torch.device):
    rank, world_size, _ = _get_dist_info()
    if world_size == 1:
        return [seqs]

    local_len = torch.tensor([seqs.size(1)], device=device, dtype=torch.int64)
    max_len = local_len.clone()
    dist.all_reduce(max_len, op=dist.ReduceOp.MAX)
    max_len_val = int(max_len.item())

    if seqs.size(1) < max_len_val:
        pad_width = max_len_val - seqs.size(1)
        pad_tensor = torch.full((seqs.size(0), pad_width), pad_id, device=seqs.device, dtype=seqs.dtype)
        seqs = torch.cat([seqs, pad_tensor], dim=1)

    gather_list = [torch.empty_like(seqs) for _ in range(world_size)]
    dist.all_gather(gather_list, seqs)
    return gather_list


def _run_evaluation(
    model: torch.nn.Module,
    tokenizer,
    data_module,
    eval_cfg,
    *,
    tgt_lang_id: int,
    device: torch.device,
    max_new_tokens: int,
    rank: int,
):
    if eval_cfg is None:
        return {}

    requested_metrics = getattr(eval_cfg, "translation_metric", [])
    if isinstance(requested_metrics, str):
        requested_metrics = [requested_metrics]
    requested_metrics = [metric.lower() for metric in requested_metrics]

    # Only rank 0 performs evaluation to avoid duplicated work.
    if rank != 0:
        return {}

    was_training = model.training
    model.eval()

    val_loader = data_module.val_dataloader()
    predictions = []
    references = []
    sample_records = []

    # Progress bar setup (rank 0 only)
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
                encoder_inputs, ground_truth, sample_id = batch
                encoder_inputs = {k: v.to(device, non_blocking=True) for k, v in encoder_inputs.items()}
                generated = model.generate(
                    input_ids=encoder_inputs["input_ids"],
                    attention_mask=encoder_inputs.get("attention_mask"),
                    **generation_kwargs,
                )
                hypothesis = tokenizer.decode(generated[0], skip_special_tokens=True)
                reference = ground_truth if isinstance(ground_truth, str) else str(ground_truth)
                predictions.append(hypothesis)
                references.append(reference)
                sample_records.append((reference, hypothesis, sample_id))
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
    if predictions and "chrf" in requested_metrics:
        chrf_metric = evaluate.load("chrf")
        chrf_res = chrf_metric.compute(
            predictions=predictions,
            references=[[r] for r in references],
            char_order=6,
            word_order=2,
        )
        results["eval/chrf"] = float(chrf_res.get("score", 0.0))

    if results:
        print("Evaluation results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")

    # if wandb.run is not None and (results or sample_records):
    #     log_payload = results.copy()
    #     if sample_records:
    #         eval_table = wandb.Table(columns=["reference", "generated", "sample_id"], log_mode="MUTABLE")
    #         for reference, hypothesis, sample_id in sample_records:
    #             eval_table.add_data(reference, hypothesis, sample_id)
    #         log_payload["eval/Translations"] = eval_table
    #     wandb.log(log_payload)

    if was_training:
        model.train()

    return results


@hydra.main(version_base=None, config_path="./configs/", config_name="train")
def train(config: DictConfig):
    torch.manual_seed(config.seed)

    # Initialize process group if launched with torchrun
    if dist.is_available() and not dist.is_initialized() and os.environ.get("RANK") is not None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    rank, world_size, local_rank = _get_dist_info()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(f"{device_type}:{local_rank}" if device_type == "cuda" else device_type)
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)

    # Build model and tokenizer on rank 0, then broadcast weights
    model, tokenizer = get_model(config)
    model.to(device)

    eval_cfg = getattr(config.task, "eval", None)
    eval_only = bool(getattr(eval_cfg, "only", False)) if eval_cfg is not None else False
    run_eval = bool(getattr(eval_cfg, "run", False)) if eval_cfg is not None else False
    if eval_only:
        run_eval = True
    run_training = not eval_only

    # Initialize Weights & Biases on rank 0
    train_table = None
    if rank == 0:
        wandb.init(
            project="grpo-translation-nllb-multi-domain",
            name=time.strftime("%Y-%m-%d_%H-%M-%S"),
            config=OmegaConf.to_container(config, resolve=True),
            dir = "/root/wandb"
        )
        if run_training:
            train_table = wandb.Table(columns=["reference", "generated", "sample_id"], log_mode="MUTABLE")
    # Reference model (frozen) on rank 0
    reference_name = getattr(config.task.model, "reference_name", None)
    ref_model = None
    if run_training and rank == 0:
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

    # Make sure all ranks start from the same initial weights
    if _is_dist():
        dist.barrier()
    _broadcast_model_(model, src=0)
    if _is_dist():
        dist.barrier()

    # Data module: only rank 0 pulls batches; others receive via broadcast
    illegal_token_mask = None
    data = TranslationDataModule(
        tokenizer=tokenizer,
        illegal_token_mask=illegal_token_mask,
        data_path=config.task.data.path,
        dataset_config_name=config.task.data.dataset_config_name,
        source_lang=config.task.data.source_lang,
        target_lang=config.task.data.target_lang,
        sort_by_length=False,
    )
    data.setup("fit")

    # Training hyperparameters
    max_epochs = int(config.task.training.epochs)
    updates_per_batch = int(getattr(config.task.training, "updates_per_batch", 50))
    num_return_sequences = int(getattr(config.task.training, "num_return_sequences", 12))
    max_new_tokens = int(getattr(config.task.constraints, "max_sentence_len", 128))
    gen_temperature = float(getattr(config.task.training, "gen_temperature", 1.3))
    beta = float(getattr(config.task.training, "beta", 0.04))
    clip_param = float(getattr(config.task.training, "clip_param", 0.2))
    tgt_lang_id = tokenizer.convert_tokens_to_ids(config.task.data.target_lang)

    # Optimizer only on rank 0
    if run_training and rank == 0:
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
            if rank == 0:
                train_loader = data.train_dataloader()
                data_iter = iter(train_loader)

            step_idx = 0
            while True:
                # Rank 0 pulls next batch and broadcasts, or signals epoch end
                if rank == 0:
                    try:
                        batch = next(data_iter)
                        # Expect batch = (src_prompt_dict, ground_truth_str, sample_id)
                        src_prompt, ground_truth, sample_id = batch
                        # Move batch to CPU for pickling broadcast
                        src_prompt_cpu = {k: v.cpu() for k, v in src_prompt.items()}
                        payload = (src_prompt_cpu, ground_truth, sample_id)
                    except StopIteration:
                        payload = None
                else:
                    payload = None

                payload = _broadcast_object(payload, src=0)
                if payload is None:
                    break  # epoch finished

                src_prompt_cpu, ground_truth, sample_id = payload
                # Move tensors to this rank's device
                encoder_inputs = {k: v.to(device, non_blocking=True) for k, v in src_prompt_cpu.items()}

                for _ in range(updates_per_batch):
                    # Each rank generates sequences with current local model weights
                    generated_local = grpo_generate_sequences(
                        model,
                        tokenizer,
                        encoder_inputs,
                        tgt_lang_id,
                        max_new_tokens=max_new_tokens,
                        gen_temperature=gen_temperature,
                        num_return_sequences=num_return_sequences,
                        top_k=int(getattr(config.task.training, "top_k", 1000)),
                        top_p=float(getattr(config.task.training, "top_p", 0.95)),
                        end_of_sentence_token_id=tokenizer.eos_token_id,
                    )

                    # Pad and gather sequences to rank 0
                    gathered = _pad_and_all_gather_sequences(
                        generated_local, pad_id=tokenizer.pad_token_id, device=device
                    )

                    if rank == 0:
                        generated_all = torch.cat(gathered, dim=0)  # (world_size*K, L)

                        # Compute GRPO loss on rank 0 and update
                        loss, logs = grpo_compute_loss_and_logs(
                            model,
                            ref_model,
                            tokenizer,
                            encoder_inputs,
                            generated_all,
                            ground_truth if isinstance(ground_truth, str) else str(ground_truth),
                            end_of_sentence_token_id=tokenizer.eos_token_id,
                            beta=beta,
                            clip_param=clip_param,
                        )

                        optimizer.zero_grad()
                        loss.backward()

                        # Compute global gradient L2 norm (rank 0 only)
                        grad_norm = 0.0
                        with torch.no_grad():
                            grads = []
                            for p in model.parameters():
                                if p.requires_grad and p.grad is not None:
                                    grads.append(p.grad.detach().float().norm(2))
                            if grads:
                                grad_norm = torch.norm(torch.stack(grads), 2).item()

                        optimizer.step()

                        if step_idx % int(getattr(config.task.training, "log_every_n_steps", 10)) == 0:
                            print(
                                f"[epoch {epoch}] step {step_idx} | loss={logs['loss'].item():.4f} "
                                f"kl={logs['kl'].item():.4f} reward={logs['reward'].item():.4f} chrf={logs['chrf'].item():.4f} gradient_norm={grad_norm:.4f}"
                            )
                            # Print the reference and one generated sequence for inspection
                            # Decode one generated sequence (first in batch)
                            gen_text = tokenizer.decode(
                                generated_all[0], skip_special_tokens=True
                            )
                            print(f"Reference: {ground_truth if isinstance(ground_truth, str) else str(ground_truth)}")
                            print(f"Generated: {gen_text}")

                            # Log to Weights & Biases on rank 0
                            wandb.log(
                                {
                                    "train/loss": float(logs["loss"].item()),
                                    "train/kl": float(logs["kl"].item()),
                                    "train/chrf": float(logs["chrf"].item()),
                                    "train/gradient_norm": float(grad_norm),
                                }
                            )
                            if train_table is not None:
                                train_table.add_data(
                                    ground_truth if isinstance(ground_truth, str) else str(ground_truth),
                                    gen_text,
                                    sample_id,
                                )
                                wandb.log({"train/Translations": train_table}, step=step_idx)
                    # Synchronize and broadcast updated weights to all ranks
                    if _is_dist():
                        dist.barrier()
                    _broadcast_model_(model, src=0)
                    if _is_dist():
                        dist.barrier()

                    # Optional periodic evaluation every N updates (rank 0 performs work)
                    if run_eval:
                        eval_every_n_batches = int(getattr(eval_cfg, "every_n_batches", 0))
                        if eval_every_n_batches > 0 and step_idx > 0 and (step_idx % eval_every_n_batches == 0):
                            if _is_dist():
                                dist.barrier()
                            _run_evaluation(
                                model,
                                tokenizer,
                                data,
                                eval_cfg,
                                tgt_lang_id=tgt_lang_id,
                                device=device,
                                max_new_tokens=max_new_tokens,
                                rank=rank,
                            )
                            if _is_dist():
                                dist.barrier()

                    step_idx += 1

                # Refresh reference model on rank 0 at example boundaries
                if rank == 0:
                    ref_model = copy.deepcopy(model).to(device)
                    ref_model.eval()
                    for p in ref_model.parameters():
                        p.requires_grad_(False)
                if _is_dist():
                    dist.barrier()
                # Share refreshed weights so next epoch starts synchronized
                _broadcast_model_(model, src=0)
                if _is_dist():
                    dist.barrier()

    if run_eval:
        if _is_dist():
            dist.barrier()
        _run_evaluation(
            model,
            tokenizer,
            data,
            eval_cfg,
            tgt_lang_id=tgt_lang_id,
            device=device,
            max_new_tokens=max_new_tokens,
            rank=rank,
        )
        if _is_dist():
            dist.barrier()

    # Save final model from rank 0 only when training occurred
    if run_training and rank == 0:
        save_dir = os.path.join(os.getcwd(), "model")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    if rank == 0 and wandb.run is not None:
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
