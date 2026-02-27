#!/bin/bash

# Backtranslation Baseline Training Script
# This script trains the roundtrip backtranslation baseline on multiple language pairs.

# Language pairs to train (source-target)
LANG_PAIRS=(
    "eng_Latn-ayr_Latn"
    "eng_Latn-wol_Latn"
    "eng_Latn-fur_Latn"
    "eng_Latn-rus_Cyrl"
    "eng_Latn-dyu_Latn"
    "eng_Latn-bho_Deva"
)

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Output log file
LOG_FILE="${SCRIPT_DIR}/roundtrip-bt_output_$(date +%Y%m%d_%H%M%S).log"

echo "Roundtrip Backtranslation Baseline Training"
echo "Log file: ${LOG_FILE}"
echo ""

# Loop through each language pair
for LANG_PAIR in "${LANG_PAIRS[@]}"; do
    {
        echo "=========================================="
        echo "Starting roundtrip backtranslation training: ${LANG_PAIR}"
        echo "Time: $(date)"
        echo "=========================================="
        
        # Extract source and target languages
        SRC_LANG="${LANG_PAIR%%-*}"
        TGT_LANG="${LANG_PAIR##*-}"
        
        /workspace/miniconda3/envs/gfn/bin/python "${SCRIPT_DIR}/roundtrip-bt.py" \
            --dataset_config "${LANG_PAIR}" \
            --src_lang_nllb "${SRC_LANG}" \
            --tgt_lang_nllb "${TGT_LANG}" \
            --num_epochs 2 \
            --batch_size 2 \
            --bt_batch_size 2 \
            --learning_rate 3e-5 \
            --regenerate_bt_every 1
        
        # Check if training succeeded
        if [ $? -ne 0 ]; then
            echo "ERROR: Training failed for ${LANG_PAIR}"
            exit 1
        fi
        
        echo "Completed: ${LANG_PAIR} at $(date)"
        echo ""
    } >> "$LOG_FILE" 2>&1
done

echo "==========================================" | tee -a "$LOG_FILE"
echo "All roundtrip backtranslation training runs completed!" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
