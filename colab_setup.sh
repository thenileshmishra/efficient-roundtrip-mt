#!/bin/bash
# Run this once at the top of your Colab notebook (Runtime → Run all, or paste in a cell).
# Everything downloads automatically — no manual dataset/model downloads needed.

set -e

# 1. Clone the repo (skip if already cloned)
if [ ! -d "MT-via-Round-Trip-RL" ]; then
  git clone https://github.com/YOUR_USERNAME/MT-via-Round-Trip-RL.git
fi
cd MT-via-Round-Trip-RL

# 2. Install Python dependencies
pip install -q \
  "torch==2.6.0" torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

pip uninstall -q -y torchao 2>/dev/null || true   # remove incompatible pre-installed version

pip install -q \
  transformers peft datasets==3.6.0 sacrebleu sentencepiece \
  accelerate hydra-core lightning pytorch-lightning \
  wandb torchmetrics torchdata \
  numpy Pillow>=9.1.0 editdistance loralib \
  matplotlib seaborn nltk spacy

# 3. Patch torchdata (MapDataPipe removed in 0.9+) — already in repo
# dl.py uses: from torch.utils.data import Dataset as MapDataPipe

echo ""
echo "Setup complete. GPU info:"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); \
           [print(f'  GPU {i}:', torch.cuda.get_device_name(i)) \
            for i in range(torch.cuda.device_count())]"
