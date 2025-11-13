import math
from typing import Sequence

import torch
from sacrebleu.metrics import CHRF, BLEU

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
    device = next(model.parameters()).device

    # Repeat encoder inputs to match number of sequences
    base_batch_size = encoder_inputs["input_ids"].size(0)
    batch_multiplier = decoder_input_ids.size(0) // base_batch_size
    if batch_multiplier * base_batch_size != decoder_input_ids.size(0):
        raise ValueError(
            "decoder_input_ids size does not align with encoder batch size. "
            f"Got encoder batch {base_batch_size} and decoder batch {decoder_input_ids.size(0)}."
        )
    repeated_input_ids = encoder_inputs["input_ids"].repeat_interleave(batch_multiplier, dim=0)
    repeated_input_ids = repeated_input_ids.to(device)
    attention_mask = encoder_inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.repeat_interleave(batch_multiplier, dim=0)
        attention_mask = attention_mask.to(device)

    decoder_input_ids = decoder_input_ids.to(device)
    target_ids = target_ids.to(device)

    decoder_attention_mask = (decoder_input_ids != tokenizer.pad_token_id).long()
    decoder_attention_mask = decoder_attention_mask.to(device)

    outputs = model(
        input_ids=repeated_input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        use_cache=False,
    )
    logits = outputs.logits  # (B, L, V)
    del outputs
    per_token_logps = _gather_log_probs_from_logits_logits(logits, target_ids)  # (B, L)
    return per_token_logps

# Return the log-probability of the target text given the input text.
def score_text(model, tokenizer, input_text, target_text):
    loss = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
    # Prepare inputs.
    input_tokens = tokenizer([input_text], add_special_tokens=False)['input_ids'][0]
    target_tokens = tokenizer([target_text], add_special_tokens=False)['input_ids'][0]
    sequence_tokens = input_tokens + target_tokens
    # Prepend [CLS] to input sequence, to match training format.
    sequence_tokens.insert(0, tokenizer.cls_token_id)  # Start token.
    max_seq_len = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_seq_len, int) and max_seq_len > 0:
        assert len(sequence_tokens) <= max_seq_len
    sequence_tokens = torch.tensor([sequence_tokens])
    if torch.cuda.is_available():
        sequence_tokens = sequence_tokens.cuda()
    # Run model.
    # input_ids shape: (n_examples=1, seq_length). Sequence tokens includes
    # start of sequence token.
    outputs = model(input_ids=sequence_tokens,
                    output_hidden_states=False, return_dict=True)
    # Logits shape: (n_examples=1, seq_len, vocab_size).
    logits = outputs['logits'].detach()
    del outputs
    # Labels are the ground truth next token for each index.
    labels = sequence_tokens[:, 1:]  # Shape: (n_examples=1, seq_len-1).
    # Next token probabilities ignored for last token.
    logits = logits[:, :-1, :]
    # To apply loss, logits should be shape: (n_examples=1, vocab_size, seq_len-1).
    logits = torch.transpose(logits, 1, 2)
    # Loss shape: (n_examples=1, seq_len-1).
    # These are negative log probabilities (natural log), corresponding to each
    # token in sequence_tokens excluding the start token.
    losses = loss(logits, labels).cpu()
    # Only consider for the targets, not inputs.
    losses = losses[0, len(input_tokens):]
    logprobs = -1.0 * losses
    # Log-probability of entire target text is the sum of token log-probs.
    summed_logprobs = torch.sum(logprobs, dim=-1).item()
    return summed_logprobs

def grpo_compute_loss_and_logs(
    model,
    ref_model,
    tokenizer,
    encoder_inputs,
    generated_sequences: torch.Tensor,
    ground_truths: Sequence[str],
    *,
    end_of_sentence_token_id: int,
    beta: float,
    clip_param: float,
    tgt_lang_id: int,
    length_penalty_weight: float = 0.0,
    goldfish_model=None,
    goldfish_tokenizer=None,
    goldfish_reward_weight: float = 0.5,

):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    else:
        ground_truths = list(ground_truths)

    if generated_sequences.dim() != 3:
        raise ValueError(
            "generated_sequences must be a 3D tensor of shape (batch_size, num_candidates, seq_len). "
            f"Received tensor with shape {tuple(generated_sequences.shape)}."
        )

    device = next(model.parameters()).device
    generated_sequences = generated_sequences.to(device)

    batch_size, num_candidates, seq_len = generated_sequences.size()
    if batch_size != len(ground_truths):
        raise ValueError(
            f"Number of ground truths ({len(ground_truths)}) does not match generated batch size ({batch_size})."
        )
    flat_sequences = generated_sequences.reshape(batch_size * num_candidates, seq_len)

    # Prepare decoder inputs/targets
    decoder_input_ids = flat_sequences[:, :-1]
    target_ids = flat_sequences[:, 1:]

    # Compute per-token logps under current and reference policies
    per_token_logps = grpo_compute_decoder_per_token_logps(
        model, tokenizer, encoder_inputs, decoder_input_ids, target_ids
    )
    with torch.no_grad():
        ref_per_token_logps = grpo_compute_decoder_per_token_logps(
            ref_model, tokenizer, encoder_inputs, decoder_input_ids, target_ids
        ).to(device)

    # Completion mask to ignore pads and tokens after first EOS
    is_pad = target_ids == tokenizer.pad_token_id
    is_eos = target_ids == end_of_sentence_token_id
    is_lang_id = target_ids == tgt_lang_id
    eos_cumsum = is_eos.cumsum(dim=-1)
    after_eos = eos_cumsum >= 1
    completion_mask = (~is_pad) & (~after_eos) & (~is_lang_id)
    completion_mask = completion_mask.reshape(batch_size, num_candidates, -1)

    # Decode generated sequences without special tokens for reward computation
    generated_texts = tokenizer.batch_decode(
        torch.where(
            generated_sequences.reshape(-1, generated_sequences.size(-1)) == tokenizer.pad_token_id,
            end_of_sentence_token_id,
            generated_sequences.reshape(-1, generated_sequences.size(-1)),
        ).cpu(),
        skip_special_tokens=True,
    )
    references = [
        ground_truths[idx // num_candidates]
        for idx in range(len(generated_texts))
    ]

    # Compute chrF with evaluate (character order=6) for outcome rewards
    chrf_metric = CHRF(word_order=2, char_order=6)
    bleu_metric = BLEU(effective_order=True)
    chrf_scores = []
    bleu_scores = []
    for hyp, ref in zip(generated_texts, references):
        chrf_scores.append(chrf_metric.corpus_score(hypotheses=[hyp], references=[[ref]]).score / 100.0)
        bleu_scores.append(bleu_metric.corpus_score(hypotheses=[hyp], references=[[ref]]).score / 100.0)
    chrf_scores_tensor = torch.tensor(chrf_scores, device=device)
    bleu_scores_tensor = torch.tensor(bleu_scores, device=device)
    chrf_mean = chrf_scores_tensor.mean()
    bleu_mean = bleu_scores_tensor.mean()
    # Combine chrF and BLEU equally for rewards
    combined_scores_tensor = 0.5 * chrf_scores_tensor + 0.5 * bleu_scores_tensor
    rewards = combined_scores_tensor.reshape(batch_size, num_candidates)
    standardized_rewards = (rewards - rewards.mean(dim=1, keepdim=True)) / (
        rewards.std(dim=1, keepdim=True) + 1e-4
    )

    # Outcome-based advantages replicate the standardized reward over all tokens
    per_token_logps = per_token_logps.reshape(batch_size, num_candidates, -1)
    advantages = standardized_rewards.unsqueeze(-1).expand_as(per_token_logps)

    # PPO-style ratio (on-policy baseline trick)
    ratio = torch.exp(per_token_logps - per_token_logps.detach())
    clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
    per_token_adv_loss = torch.min(ratio * advantages, clipped_ratio * advantages)

    # KL penalty versus reference policy
    ref_minus_pi = ref_per_token_logps.reshape(batch_size, num_candidates, -1) - per_token_logps
    per_token_kl = torch.exp(ref_minus_pi) - ref_minus_pi - 1.0

    per_token_obj = per_token_adv_loss - beta * per_token_kl
    per_token_obj = per_token_obj * completion_mask

    # Mean over valid tokens per sequence, then batch mean
    token_counts = completion_mask.sum(dim=-1).clamp_min(1)
    # per_token
    seq_losses = -per_token_obj.sum(dim=-1) / token_counts
    loss = seq_losses.mean()

    with torch.no_grad():
        kl_mean = (per_token_kl * completion_mask).sum() / token_counts.sum()

    logs = {
        "loss": loss.detach(),
        "kl": kl_mean.detach(),
        "reward": rewards.mean().detach(),
        "chrf": chrf_mean.detach(),
        "bleu": bleu_mean.detach(),
    }
    return loss, logs
