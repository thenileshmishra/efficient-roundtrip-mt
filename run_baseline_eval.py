"""
Quick baseline eval: pretrained facebook/nllb-200-distilled-600M on eng->rus.
Loads 50 validation samples, translates, reports chrF++.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from sacrebleu.metrics import CHRF

SRC_LANG = "eng_Latn"
TGT_LANG = "rus_Cyrl"
MODEL_NAME = "facebook/nllb-200-distilled-600M"
MAX_SAMPLES = 50
MAX_NEW_TOKENS = 63
BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

print(f"Loading dataset: breakend/nllb-multi-domain ({SRC_LANG}-{TGT_LANG})")
ds = load_dataset("breakend/nllb-multi-domain", f"{SRC_LANG}-{TGT_LANG}", trust_remote_code=True)
val_split = ds.get("valid") or ds.get("validation") or ds.get("val")
val_split = val_split.select(range(min(MAX_SAMPLES, len(val_split))))

src_sentences = val_split[f"sentence_{SRC_LANG}"]
tgt_sentences = val_split[f"sentence_{TGT_LANG}"]
print(f"Loaded {len(src_sentences)} validation pairs")

tgt_lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
tokenizer.src_lang = SRC_LANG

predictions = []
for i in range(0, len(src_sentences), BATCH_SIZE):
    batch = src_sentences[i : i + BATCH_SIZE]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            early_stopping=True,
        )
    predictions.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    print(f"  Translated {min(i + BATCH_SIZE, len(src_sentences))}/{len(src_sentences)}")

chrf = CHRF(word_order=2, char_order=6)
score = chrf.corpus_score(hypotheses=predictions, references=[tgt_sentences]).score
print(f"\nBaseline chrF++ ({SRC_LANG}->{TGT_LANG}): {score:.2f}")

# Print a couple examples
for i in range(min(3, len(predictions))):
    print(f"\n[{i}] SRC: {src_sentences[i]}")
    print(f"    REF: {tgt_sentences[i]}")
    print(f"    HYP: {predictions[i]}")
