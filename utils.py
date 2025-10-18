import torch
import evaluate

# =========================
# Distributed GRPO Utilities
# =========================

@torch.no_grad()
def grpo_generate_sequences(
    model,
    tokenizer,
    encoder_inputs,
    tgt_lang_id,
    *,
    max_new_tokens: int,
    gen_temperature: float,
    num_return_sequences: int,
    top_k: int = 100,
    top_p: float = 0.9,
    end_of_sentence_token_id: int = None,
):
    eos_id = (
        end_of_sentence_token_id
        if end_of_sentence_token_id is not None
        else tokenizer.eos_token_id
    )
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=gen_temperature,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_id,
        top_k=top_k,
        top_p=top_p,
    )
    gen = model.generate(
        input_ids=encoder_inputs["input_ids"],
        attention_mask=encoder_inputs.get("attention_mask", None),
        forced_bos_token_id=tgt_lang_id,
        **generation_kwargs,
    )
    return gen


def _gather_log_probs_from_logits_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    log_probs = logits.log_softmax(dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return gathered


def grpo_compute_decoder_per_token_logps(
    model,
    tokenizer,
    encoder_inputs,
    decoder_input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    # Repeat encoder inputs to match number of sequences
    batch_multiplier = decoder_input_ids.size(0)
    repeated_input_ids = encoder_inputs["input_ids"].repeat(batch_multiplier, 1)
    attention_mask = encoder_inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.repeat(batch_multiplier, 1)

    decoder_attention_mask = (decoder_input_ids != tokenizer.pad_token_id).long()

    outputs = model(
        input_ids=repeated_input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        use_cache=False,
    )
    logits = outputs.logits  # (B, L, V)
    per_token_logps = _gather_log_probs_from_logits_logits(logits, target_ids)  # (B, L)
    return per_token_logps


def grpo_compute_loss_and_logs(
    model,
    ref_model,
    tokenizer,
    encoder_inputs,
    generated_sequences: torch.Tensor,
    ground_truth: str,
    *,
    end_of_sentence_token_id: int,
    beta: float,
    clip_param: float,
):
    device = next(model.parameters()).device
    # Prepare decoder inputs/targets
    decoder_input_ids = generated_sequences[:, :-1]
    target_ids = generated_sequences[:, 1:]

    # Compute per-token logps under current and reference policies
    per_token_logps = grpo_compute_decoder_per_token_logps(
        model, tokenizer, encoder_inputs, decoder_input_ids, target_ids
    )
    with torch.no_grad():
        ref_per_token_logps = grpo_compute_decoder_per_token_logps(
            ref_model, tokenizer, encoder_inputs, decoder_input_ids, target_ids
        )

    # Completion mask to ignore pads and tokens after first EOS
    is_pad = target_ids == tokenizer.pad_token_id
    is_eos = target_ids == end_of_sentence_token_id
    eos_cumsum = is_eos.cumsum(dim=-1)
    after_eos = eos_cumsum >= 1
    completion_mask = (~is_pad) & (~after_eos)

    # Decode generated sequences without special tokens for reward computation
    generated_texts = tokenizer.batch_decode(
        torch.where(
            generated_sequences == tokenizer.pad_token_id,
            end_of_sentence_token_id,
            generated_sequences,
        ),
        skip_special_tokens=True,
    )
    references = [ground_truth for _ in range(len(generated_texts))]

    # Compute chrF with evaluate (character order=6) for rewards
    chrf_metric = evaluate.load("chrf")
    # evaluate returns an average score; we also approximate per-sample by recomputing individually
    rewards = []
    chrf_scores = []
    for hyp, ref in zip(generated_texts, references):
        sample = chrf_metric.compute(predictions=[hyp], references=[[ref]], char_order=6, word_order=2)["score"] / 100.0
        chrf_scores.append(sample)
        rewards.append(torch.log(torch.tensor(sample + 1e-8, device=device)))
    chrf_mean = torch.tensor(chrf_scores, device=device).mean()
    rewards = torch.stack(rewards, dim=0)

    # Normalize rewards -> advantages
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    advantages = advantages.unsqueeze(1).expand_as(per_token_logps)

    # PPO-style ratio (on-policy baseline trick)
    ratio = torch.exp(per_token_logps - per_token_logps.detach())
    clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
    per_token_adv_loss = torch.min(ratio * advantages, clipped_ratio * advantages)

    # KL penalty versus reference policy
    ref_minus_pi = ref_per_token_logps - per_token_logps
    per_token_kl = torch.exp(ref_minus_pi) - ref_minus_pi - 1.0

    per_token_obj = per_token_adv_loss - beta * per_token_kl
    per_token_obj = per_token_obj * completion_mask

    # Mean over valid tokens per sequence, then batch mean
    token_counts = completion_mask.sum(dim=-1).clamp_min(1)
    seq_losses = -per_token_obj.sum(dim=-1) / token_counts
    loss = seq_losses.mean()

    with torch.no_grad():
        kl_mean = (per_token_kl * completion_mask).sum() / token_counts.sum()

    logs = {
        "loss": loss.detach(),
        "kl": kl_mean.detach(),
        "reward": rewards.mean().detach(),
        "chrf": chrf_mean.detach(),
    }
    return loss, logs
