"""
LoRA rank sweep: Full FT vs r=8/16/32 on eng→rus GRPO training.
Logs peak GPU memory, time/step, and chrF++ for each configuration.

Memory-efficient setup for limited free VRAM:
  - Ref model kept on CPU; only policy model lives on GPU.
  - Eval uses greedy decode (num_beams=1), batch_size=2.
  - Full FT is attempted but will likely OOM on the Adam optimizer states
    (~5 GB fp32) — itself a useful data point.
"""

import sys, os, copy, gc, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, TaskType, get_peft_model
from sacrebleu.metrics import CHRF
from datasets import load_dataset
from torch.utils.data import DataLoader

from utils import grpo_generate_sequences, grpo_compute_loss_and_logs
from dl import TranslationDataModule

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "facebook/nllb-200-distilled-600M"
SRC_LANG       = "eng_Latn"
TGT_LANG       = "rus_Cyrl"
DEVICE         = torch.device("cuda")
CPU            = torch.device("cpu")
DTYPE          = torch.bfloat16
TRAIN_STEPS    = 20
BATCH_SIZE     = 2
NUM_SEQS       = 4
MAX_NEW_TOKENS = 63
GEN_TEMP       = 1.8
VAL_SAMPLES    = 50
LR             = 2e-5

CONFIGS = [
    {"label": "Full FT",   "use_lora": False, "lora_r": None},
    {"label": "LoRA r=8",  "use_lora": True,  "lora_r": 8},
    {"label": "LoRA r=16", "use_lora": True,  "lora_r": 16},
    {"label": "LoRA r=32", "use_lora": True,  "lora_r": 32},
]

# ── Shared data ───────────────────────────────────────────────────────────────
def build_data():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.src_lang = SRC_LANG
    dm = TranslationDataModule(
        tokenizer=tok,
        illegal_token_mask=None,
        data_path="breakend/nllb-multi-domain",
        dataset_config_name=f"{SRC_LANG}-{TGT_LANG}",
        source_lang=SRC_LANG,
        target_lang=TGT_LANG,
        train_batch_size=BATCH_SIZE,
    )
    dm.setup("fit")
    return dm

def build_val_loader(tokenizer):
    raw = load_dataset("breakend/nllb-multi-domain",
                       f"{SRC_LANG}-{TGT_LANG}", trust_remote_code=True)
    val = raw["valid"].select(range(VAL_SAMPLES))
    def _collate(batch):
        srcs = [b[f"sentence_{SRC_LANG}"] for b in batch]
        tgts = [b[f"sentence_{TGT_LANG}"] for b in batch]
        enc  = tokenizer(srcs, return_tensors="pt", padding=True, truncation=True)
        return enc, tgts, srcs, list(range(len(srcs)))
    return DataLoader(val, batch_size=2, collate_fn=_collate)

# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(use_lora: bool, lora_r):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.src_lang = SRC_LANG
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
    if use_lora:
        for p in model.parameters():
            p.requires_grad_(False)
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_r,
            lora_alpha=lora_r * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
    model = model.to(DEVICE)
    return model, tokenizer

# ── Eval (greedy, small batch) ────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, tokenizer, val_loader, tgt_lang_id: int) -> float:
    model.eval()
    preds, refs = [], []
    for enc, gts, _, _ in val_loader:
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        out = model.generate(
            **enc,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,                   # greedy — low memory
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        refs.extend(list(gts))
    model.train()
    return CHRF(word_order=2, char_order=6).corpus_score(preds, [refs]).score

# ── One experiment ────────────────────────────────────────────────────────────
def run(cfg: dict, dm: TranslationDataModule) -> dict:
    label    = cfg["label"]
    use_lora = cfg["use_lora"]
    lora_r   = cfg["lora_r"]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    model, tokenizer = build_model(use_lora, lora_r)
    tgt_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    src_id = tokenizer.convert_tokens_to_ids(SRC_LANG)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} / {total:,}  ({trainable/total:.2%})")
    print(f"  GPU after model load: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

    # Ref model lives on CPU to save GPU VRAM
    ref_model = copy.deepcopy(model).to(CPU)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    val_loader = build_val_loader(tokenizer)

    chrf_before = evaluate(model, tokenizer, val_loader, tgt_id)
    print(f"  chrF++ (before): {chrf_before:.2f}")

    # Optimizer (tiny for LoRA; huge for Full FT — OOM expected there)
    try:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=LR
        )
    except torch.cuda.OutOfMemoryError:
        print("  OOM creating optimizer — skipping training.")
        del model, ref_model
        torch.cuda.empty_cache(); gc.collect()
        return dict(label=label, trainable_pct=trainable/total*100,
                    peak_mem_mb=None, avg_step_s=None,
                    chrf_before=chrf_before, chrf_after=None, oom=True)

    mem_after_opt = torch.cuda.memory_allocated() / 1024**2
    print(f"  GPU after optimizer: {mem_after_opt:.0f} MB")

    train_iter = iter(dm.train_dataloader())
    step_times = []
    oom_step   = None

    for step in range(TRAIN_STEPS):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(dm.train_dataloader())
            batch      = next(train_iter)

        src_prompt, ground_truths, source_texts, _ = batch
        enc = {k: v.to(DEVICE) for k, v in src_prompt.items()}

        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Forward: src → tgt
            fwd = grpo_generate_sequences(
                model, tokenizer, enc, tgt_id,
                max_new_tokens=MAX_NEW_TOKENS, gen_temperature=GEN_TEMP,
                num_return_sequences=NUM_SEQS, top_k=100, top_p=0.95,
                end_of_sentence_token_id=tokenizer.eos_token_id,
            )
            seq_len   = fwd.size(1)
            fwd_texts = tokenizer.batch_decode(
                fwd.reshape(BATCH_SIZE * NUM_SEQS, seq_len), skip_special_tokens=True
            )

            # Backward: tgt → src
            tokenizer.src_lang = TGT_LANG
            bwd_enc = tokenizer(fwd_texts, return_tensors="pt",
                                padding=True, truncation=True)
            tokenizer.src_lang = SRC_LANG
            bwd_enc  = {k: v.to(DEVICE) for k, v in bwd_enc.items()}
            bwd_bsz  = bwd_enc["input_ids"].size(0)

            bwd = grpo_generate_sequences(
                model, tokenizer, bwd_enc, src_id,
                max_new_tokens=MAX_NEW_TOKENS, gen_temperature=GEN_TEMP,
                num_return_sequences=NUM_SEQS, top_k=100, top_p=0.95,
                end_of_sentence_token_id=tokenizer.eos_token_id,
            )
            bwd_all      = bwd.reshape(bwd_bsz, NUM_SEQS, bwd.size(1))
            expanded_src = [source_texts[i // NUM_SEQS] for i in range(bwd_bsz)]

            # GRPO loss — ref model is on CPU; grpo fn moves its tensors there
            loss, logs = grpo_compute_loss_and_logs(
                model, ref_model, tokenizer, bwd_enc, bwd_all, expanded_src,
                end_of_sentence_token_id=tokenizer.eos_token_id,
                beta=0.04, clip_param=0.2, tgt_lang_id=src_id,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)

            if (step + 1) % 5 == 0:
                print(f"  step {step+1:2d}/{TRAIN_STEPS}  "
                      f"loss={logs['loss'].item():.4f}  "
                      f"bwd_chrf={logs['chrf'].item():.3f}  "
                      f"{step_times[-1]:.1f}s")

        except torch.cuda.OutOfMemoryError:
            oom_step = step + 1
            print(f"  OOM at step {oom_step} — stopping training.")
            torch.cuda.empty_cache()
            break

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    if step_times:
        avg_step_s  = sum(step_times) / len(step_times)
        chrf_after  = evaluate(model, tokenizer, val_loader, tgt_id)
    else:
        avg_step_s = None
        chrf_after = None

    print(f"\n  ── {label} ──")
    print(f"  Peak GPU mem  : {peak_mem:,.0f} MB")
    print(f"  Avg s/step    : {avg_step_s:.2f}s" if avg_step_s else "  Avg s/step    : OOM")
    print(f"  chrF++ before : {chrf_before:.2f}")
    print(f"  chrF++ after  : {chrf_after:.2f}" if chrf_after else "  chrF++ after  : N/A")

    del model, ref_model, optimizer
    torch.cuda.empty_cache(); gc.collect()

    return dict(
        label         = label,
        trainable_pct = trainable / total * 100,
        peak_mem_mb   = peak_mem,
        avg_step_s    = avg_step_s,
        chrf_before   = chrf_before,
        chrf_after    = chrf_after,
        oom           = oom_step is not None,
    )

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    free_vram = (torch.cuda.get_device_properties(0).total_memory
                 - torch.cuda.memory_reserved()) / 1024**2
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MB total"
          f"  |  {free_vram:.0f} MB free")
    print(f"Run  : {MODEL_NAME}  ·  {TRAIN_STEPS} steps  ·  batch={BATCH_SIZE}  ·  G={NUM_SEQS}")

    dm      = build_data()
    results = []
    for cfg in CONFIGS:
        results.append(run(cfg, dm))

    # ── Final table ───────────────────────────────────────────────────────
    ft = next((r for r in results if r["label"] == "Full FT"), None)
    base_mem = ft["peak_mem_mb"] if ft and ft["peak_mem_mb"] else None
    base_t   = ft["avg_step_s"]  if ft and ft["avg_step_s"]  else None

    print("\n" + "="*88)
    print(f"{'Config':<12} {'Trainable%':>10} {'Peak MB':>9} {'Mem↓%':>7} "
          f"{'s/step':>8} {'Time↓%':>7} {'chrF pre':>9} {'chrF post':>10}")
    print("-"*88)
    for r in results:
        mem_s  = f"{r['peak_mem_mb']:>9,.0f}" if r["peak_mem_mb"] else "     OOM"
        mdrop  = (f"{(base_mem - r['peak_mem_mb'])/base_mem*100:>6.1f}%"
                  if base_mem and r["peak_mem_mb"] else "      -")
        t_s    = f"{r['avg_step_s']:>8.2f}" if r["avg_step_s"] else "     OOM"
        tdrop  = (f"{(base_t - r['avg_step_s'])/base_t*100:>6.1f}%"
                  if base_t and r["avg_step_s"] else "      -")
        chrf_a = f"{r['chrf_after']:>10.2f}" if r["chrf_after"] else "       N/A"
        print(f"{r['label']:<12} {r['trainable_pct']:>9.2f}% "
              f"{mem_s} {mdrop} {t_s} {tdrop} "
              f"{r['chrf_before']:>9.2f} {chrf_a}")
    print("="*88)
