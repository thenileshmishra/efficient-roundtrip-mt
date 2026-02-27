#!/bin/bash

# Language pairs to train (source-target)
LANG_PAIRS=(
    "eng_Latn-ayr_Latn"
    "eng_Latn-wol_Latn"
    "eng_Latn-rus_Cyrl"
    "eng_Latn-fur_Latn"
)

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Output log file
LOG_FILE="${SCRIPT_DIR}/training_output_$(date +%Y%m%d_%H%M%S).log"

# Loop through each language pair
for LANG_PAIR in "${LANG_PAIRS[@]}"; do
    {
        echo "=========================================="
        echo "Starting training: ${LANG_PAIR}"
        echo "Time: $(date)"
        echo "=========================================="
        
        # Extract source and target languages
        SRC_LANG="${LANG_PAIR%%-*}"
        TGT_LANG="${LANG_PAIR##*-}"
        
        /workspace/miniconda3/envs/gfn/bin/python "${SCRIPT_DIR}/umnmt.py" \
            --dataset_config "${LANG_PAIR}" \
            --src_lang_nllb "${SRC_LANG}" \
            --tgt_lang_nllb "${TGT_LANG}"
        
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
echo "All training runs completed!" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
