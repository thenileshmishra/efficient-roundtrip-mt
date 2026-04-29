"""
Multi-language LoRA-GRPO training: eng→{Russian, Wolof, Aymara}
Model: facebook/nllb-200-distilled-600M + LoRA r=8 (0.19% trainable)
Steps: 500  |  Batch: 1  |  G: 2 candidates

Produces paper table:
  Language | Method | Trainable% | Peak MB | Steps | chrF++ pre | chrF++ post | Δ
                                                  | spBLEU pre | spBLEU post | Δ | s/step
"""

import sys, os, copy, gc, time
sys.stdout.reconfigure(line_buffering=True)   # real-time output in Colab / pipes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, TaskType, get_peft_model
from sacrebleu.metrics import CHRF, BLEU
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from utils import grpo_generate_sequences, grpo_compute_loss_and_logs
from dl import TranslationDataModule

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "facebook/nllb-200-distilled-600M"
SRC_LANG    = "eng_Latn"
DEVICE      = torch.device("cuda")
CPU         = torch.device("cpu")
DTYPE       = torch.bfloat16

LORA_R      = 8
LORA_ALPHA  = 16      # 2× rank
LORA_DROP   = 0.05

TRAIN_STEPS = 500
BATCH_SIZE  = 1
NUM_SEQS    = 2
MAX_TOKENS  = 63
GEN_TEMP    = 1.8
LR          = 2e-5
LOG_EVERY   = 50      # quick eval on 50 val samples
CLIP_NORM   = 1.0

LANGUAGES = [
    {"name": "Russian",  "tgt": "rus_Cyrl", "ds_config": "eng_Latn-rus_Cyrl"},
    {"name": "Wolof",    "tgt": "wol_Latn", "ds_config": "eng_Latn-wol_Latn"},
    {"name": "Aymara",   "tgt": "ayr_Latn", "ds_config": "eng_Latn-ayr_Latn"},
]

# ── Tiny val dataset for fast in-training monitoring ─────────────────────────
class ListDataset(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self):         return len(self.items)
    def __getitem__(self, i):  return self.items[i]

# ── Model ─────────────────────────────────────────────────────────────────────
def load_lora_model(tgt_lang: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.src_lang = SRC_LANG
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
    for p in model.parameters():
        p.requires_grad_(False)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROP,
        target_modules=["q_proj", "v_proj"], bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model = model.to(DEVICE)
    return model, tokenizer

# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, tokenizer, src_sents, tgt_sents, tgt_lang_id, batch_size=16):
    model.eval()
    preds = []
    ds    = ListDataset(src_sents)
    for i in range(0, len(ds), batch_size):
        batch = src_sents[i : i + batch_size]
        enc   = tokenizer(batch, return_tensors="pt",
                          padding=True, truncation=True).to(DEVICE)
        out   = model.generate(
            **enc,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    chrf  = CHRF(word_order=2, char_order=6).corpus_score(preds, [tgt_sents]).score
    bleu  = BLEU(tokenize="flores200").corpus_score(preds, [tgt_sents]).score
    model.train()
    return chrf, bleu, preds

# ── One language run ──────────────────────────────────────────────────────────
def run_language(lang_cfg: dict) -> dict:
    name       = lang_cfg["name"]
    tgt_lang   = lang_cfg["tgt"]
    ds_config  = lang_cfg["ds_config"]

    print(f"\n{'='*68}")
    print(f"  Language: {name}  ({SRC_LANG} → {tgt_lang})")
    print(f"  Model:    {MODEL_NAME}  |  LoRA r={LORA_R}  |  {TRAIN_STEPS} steps")
    print(f"{'='*68}")

    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats()

    # ── Load dataset ──────────────────────────────────────────────────────
    print("Loading dataset…")
    ds = load_dataset("breakend/nllb-multi-domain", ds_config, trust_remote_code=True)
    val_key  = "valid" if "valid" in ds else "validation"
    val_src  = ds[val_key][f"sentence_{SRC_LANG}"]
    val_tgt  = ds[val_key][f"sentence_{tgt_lang}"]
    test_src = ds["test"][f"sentence_{SRC_LANG}"]
    test_tgt = ds["test"][f"sentence_{tgt_lang}"]
    # Small held-out subset for quick mid-training eval
    quick_src = val_src[:50]
    quick_tgt = val_tgt[:50]
    print(f"  Train: {len(ds['train'])} | Val: {len(val_src)} | Test: {len(test_src)}")

    # ── Build model ───────────────────────────────────────────────────────
    model, tokenizer = load_lora_model(tgt_lang)
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    src_id = tokenizer.convert_tokens_to_ids(SRC_LANG)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} / {total:,}  ({trainable/total:.2%})")
    mem_load  = torch.cuda.memory_allocated() / 1024**2

    # Ref model on CPU to save VRAM
    ref_model = copy.deepcopy(model).to(CPU)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── Pretrained baseline (full val set) ────────────────────────────────
    print("Evaluating pretrained baseline…")
    chrf_pre, bleu_pre, _ = evaluate(model, tokenizer, val_src, val_tgt, tgt_id)
    print(f"  chrF++ pre: {chrf_pre:.2f}  |  spBLEU pre: {bleu_pre:.2f}")

    # ── Data module ───────────────────────────────────────────────────────
    dm = TranslationDataModule(
        tokenizer=tokenizer,
        illegal_token_mask=None,
        data_path="breakend/nllb-multi-domain",
        dataset_config_name=ds_config,
        source_lang=SRC_LANG,
        target_lang=tgt_lang,
        train_batch_size=BATCH_SIZE,
    )
    dm.setup("fit")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    train_iter = iter(dm.train_dataloader())

    step_times  = []
    log_records = []          # (step, quick_chrf)
    total_train_time = 0.0

    print(f"\n  Training {TRAIN_STEPS} steps…")
    print(f"  {'Step':>5}  {'loss':>8}  {'bwd_chrf':>9}  {'quick_chrF':>11}  {'s/step':>7}")
    print(f"  {'-'*55}")

    for step in range(1, TRAIN_STEPS + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(dm.train_dataloader())
            batch      = next(train_iter)

        src_prompt, ground_truths, source_texts, _ = batch
        enc = {k: v.to(DEVICE) for k, v in src_prompt.items()}

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Forward: src → tgt
        fwd = grpo_generate_sequences(
            model, tokenizer, enc, tgt_id,
            max_new_tokens=MAX_TOKENS, gen_temperature=GEN_TEMP,
            num_return_sequences=NUM_SEQS, top_k=100, top_p=0.95,
            end_of_sentence_token_id=tokenizer.eos_token_id,
        )
        fwd_texts = tokenizer.batch_decode(
            fwd.reshape(BATCH_SIZE * NUM_SEQS, fwd.size(1)),
            skip_special_tokens=True,
        )

        # Backward: tgt → src
        tokenizer.src_lang = tgt_lang
        bwd_enc = tokenizer(fwd_texts, return_tensors="pt",
                            padding=True, truncation=True)
        tokenizer.src_lang = SRC_LANG
        bwd_enc  = {k: v.to(DEVICE) for k, v in bwd_enc.items()}
        bwd_bsz  = bwd_enc["input_ids"].size(0)

        bwd = grpo_generate_sequences(
            model, tokenizer, bwd_enc, src_id,
            max_new_tokens=MAX_TOKENS, gen_temperature=GEN_TEMP,
            num_return_sequences=NUM_SEQS, top_k=100, top_p=0.95,
            end_of_sentence_token_id=tokenizer.eos_token_id,
        )
        bwd_all      = bwd.reshape(bwd_bsz, NUM_SEQS, bwd.size(1))
        expanded_src = [source_texts[i // NUM_SEQS] for i in range(bwd_bsz)]

        loss, logs = grpo_compute_loss_and_logs(
            model, ref_model, tokenizer, bwd_enc, bwd_all, expanded_src,
            end_of_sentence_token_id=tokenizer.eos_token_id,
            beta=0.04, clip_param=0.2, tgt_lang_id=src_id,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], CLIP_NORM
        )
        optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        step_times.append(elapsed)
        total_train_time += elapsed

        # Quick mid-training eval every LOG_EVERY steps
        if step % LOG_EVERY == 0:
            q_chrf, _, _ = evaluate(model, tokenizer, quick_src, quick_tgt, tgt_id)
            log_records.append((step, q_chrf))
            print(f"  {step:>5}  "
                  f"{logs['loss'].item():>8.4f}  "
                  f"{logs['chrf'].item():>9.3f}  "
                  f"{q_chrf:>11.2f}  "
                  f"{elapsed:>7.2f}s")

    peak_mem    = torch.cuda.max_memory_allocated() / 1024**2
    avg_step_s  = sum(step_times) / len(step_times)

    # ── Post-training eval — full validation set ──────────────────────────
    print("\n  Evaluating after training (full val set)…")
    chrf_post_val, bleu_post_val, _ = evaluate(
        model, tokenizer, val_src, val_tgt, tgt_id
    )
    # ── Post-training eval — test set (for paper) ─────────────────────────
    print("  Evaluating on test set…")
    chrf_post_test, bleu_post_test, _ = evaluate(
        model, tokenizer, test_src, test_tgt, tgt_id
    )
    # Also get pre-training test set score
    # Reload fresh model for clean pre-training test eval
    print("  Evaluating pretrained on test set (clean model)…")
    fresh_model, fresh_tok = load_lora_model(tgt_lang)
    fresh_tgt_id = fresh_tok.convert_tokens_to_ids(tgt_lang)
    chrf_pre_test, bleu_pre_test, _ = evaluate(
        fresh_model, fresh_tok, test_src, test_tgt, fresh_tgt_id
    )
    del fresh_model, fresh_tok
    torch.cuda.empty_cache(); gc.collect()

    print(f"\n  ── {name} ──")
    print(f"  Peak GPU mem      : {peak_mem:,.0f} MB")
    print(f"  Avg time/step     : {avg_step_s:.2f}s  ({total_train_time/60:.1f} min total)")
    print(f"  Val  chrF++  pre→post : {chrf_pre:.2f} → {chrf_post_val:.2f}  (Δ {chrf_post_val-chrf_pre:+.2f})")
    print(f"  Val  spBLEU  pre→post : {bleu_pre:.2f} → {bleu_post_val:.2f}  (Δ {bleu_post_val-bleu_pre:+.2f})")
    print(f"  Test chrF++  pre→post : {chrf_pre_test:.2f} → {chrf_post_test:.2f}  (Δ {chrf_post_test-chrf_pre_test:+.2f})")
    print(f"  Test spBLEU  pre→post : {bleu_pre_test:.2f} → {bleu_post_test:.2f}  (Δ {bleu_post_test-bleu_pre_test:+.2f})")
    print(f"  Progress (quick chrF per {LOG_EVERY} steps): "
          + "  ".join(f"s{s}:{c:.1f}" for s, c in log_records))

    result = dict(
        name            = name,
        tgt             = tgt_lang,
        trainable_pct   = trainable / total * 100,
        peak_mem_mb     = peak_mem,
        steps           = TRAIN_STEPS,
        avg_step_s      = avg_step_s,
        total_min       = total_train_time / 60,
        # Validation
        chrf_pre_val    = chrf_pre,
        chrf_post_val   = chrf_post_val,
        bleu_pre_val    = bleu_pre,
        bleu_post_val   = bleu_post_val,
        # Test
        chrf_pre_test   = chrf_pre_test,
        chrf_post_test  = chrf_post_test,
        bleu_pre_test   = bleu_pre_test,
        bleu_post_test  = bleu_post_test,
        progress        = log_records,
    )

    del model, ref_model, optimizer
    torch.cuda.empty_cache(); gc.collect()
    return result

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MB")
    print(f"Setup: {MODEL_NAME}  |  LoRA r={LORA_R}  |  {TRAIN_STEPS} steps  "
          f"|  batch={BATCH_SIZE}  G={NUM_SEQS}")

    results = []
    for lang in LANGUAGES:
        results.append(run_language(lang))

    # ── Paper table ───────────────────────────────────────────────────────────
    SEP = "=" * 100
    print(f"\n{SEP}")
    print("PAPER TABLE  —  LoRA r=8 GRPO  |  facebook/nllb-200-distilled-600M")
    print(f"{'─'*100}")

    # Validation set table
    print("\n[Validation set]")
    print(f"{'Language':<10} {'Method':<12} {'Train%':>7} {'PeakMB':>8} "
          f"{'Steps':>6} {'chrF pre':>9} {'chrF post':>10} {'ΔchrF':>7} "
          f"{'BLEU pre':>9} {'BLEU post':>10} {'ΔBLEU':>7} {'s/step':>8}")
    print(f"{'─'*100}")
    for r in results:
        print(f"{r['name']:<10} {'LoRA r=8':<12} {r['trainable_pct']:>6.2f}% "
              f"{r['peak_mem_mb']:>8,.0f} {r['steps']:>6} "
              f"{r['chrf_pre_val']:>9.2f} {r['chrf_post_val']:>10.2f} "
              f"{r['chrf_post_val']-r['chrf_pre_val']:>+7.2f} "
              f"{r['bleu_pre_val']:>9.2f} {r['bleu_post_val']:>10.2f} "
              f"{r['bleu_post_val']-r['bleu_pre_val']:>+7.2f} "
              f"{r['avg_step_s']:>8.2f}")

    # Test set table
    print(f"\n[Test set]")
    print(f"{'Language':<10} {'Method':<12} {'Train%':>7} {'PeakMB':>8} "
          f"{'Steps':>6} {'chrF pre':>9} {'chrF post':>10} {'ΔchrF':>7} "
          f"{'BLEU pre':>9} {'BLEU post':>10} {'ΔBLEU':>7} {'s/step':>8}")
    print(f"{'─'*100}")
    for r in results:
        print(f"{r['name']:<10} {'LoRA r=8':<12} {r['trainable_pct']:>6.2f}% "
              f"{r['peak_mem_mb']:>8,.0f} {r['steps']:>6} "
              f"{r['chrf_pre_test']:>9.2f} {r['chrf_post_test']:>10.2f} "
              f"{r['chrf_post_test']-r['chrf_pre_test']:>+7.2f} "
              f"{r['bleu_pre_test']:>9.2f} {r['bleu_post_test']:>10.2f} "
              f"{r['bleu_post_test']-r['bleu_pre_test']:>+7.2f} "
              f"{r['avg_step_s']:>8.2f}")

    print(f"\n[Training cost]")
    for r in results:
        print(f"  {r['name']:<10}  {r['steps']} steps  ×  {r['avg_step_s']:.2f}s/step  "
              f"=  {r['total_min']:.1f} min  |  Peak {r['peak_mem_mb']:,.0f} MB GPU")

    print(f"\n[Learning curve  —  quick chrF++ on 50 val samples every {LOG_EVERY} steps]")
    for r in results:
        curve = "  ".join(f"s{s}={c:.1f}" for s, c in r["progress"])
        print(f"  {r['name']:<10}  pre={r['chrf_pre_val']:.2f}  |  {curve}  |  post={r['chrf_post_val']:.2f}")

    print(SEP)
