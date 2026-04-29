# Improving Low-Resource Machine Translation via Round-Trip Reinforcement Learning 

## Overview

This repository contains the code for my MSc. thesis: "Improving Low-Resource Machine Translation via Round-Trip Reinforcement Learning". The paper is available [here](https://arxiv.org/abs/2601.12535).

Wandb Report Available [here](https://wandb.ai/ahmed-attia-mbzuai/grpo-translation-nllb-multi-domain/reports/MT-via-Round-Trip-RL-run-plots--VmlldzoxNjY1MjMwMg?accessToken=s8tssgnx0pwvzs96m9hd9tbf9z78odfixifcfo14xwv3bk9dh18opnz9erx37ndd).
## Abstract 
Low-resource machine translation (MT) has gained increasing attention as parallel data from low-resource language communities is collected, but many potential methods for improving low-resource MT remain unexplored. We investigate a self-supervised reinforcement-learning-based fine-tuning for translation in low-resource settings using round-trip bootstrapping with the No Language Left Behind (NLLB) family of models. Our approach translates English into a target low-resource language and then back into English, using a combination of chrF++ and BLEU as the reward function on the reconstructed English sentences. Using the NLLB-MD dataset, we evaluate both the 600M and 1.3B parameter NLLB models and observe consistent improvements for the following languages: Central Aymara, Friulian, Wolof and Russian. Qualitative inspection of translation outputs indicates increased fluency and semantic fidelity. We argue that our method can further benefit from scale, enabling models to increasingly leverage their pretrained knowledge and continue self-improving.


## Key Files

- `main.py`: Contains the main logic for loading the datasets, nllb models, training loop, evaluation, wandb logging, etc.
- `dl.py`: Contains the data loading logic for the NLLB-MD dataset.
- `utils.py`: Contains helpful utilities as well as the GRPO loss function.
- `configs/`: Contains the configuration files for the training and evaluation.
- `baselines/`: Contains the code for the backtranslation and UMNMT baselines used in the paper.
- `requirements.txt`: Contains the dependencies for the project.

## Usage
Specify the dataset and model in the configuration file and run the training script:

```bash
python main.py
```
Evaluation on the validation and test sets is done during and after training.

## Baselines
For running the baselines, you can directly run the bash scripts in the `baselines/` directory.
```bash
bash baselines/backtranslation/run_backtranslation.sh
bash baselines/UMNMT/run_training.sh
```

## Citation
If you find this work useful, please cite:
```bibtex
@misc{attia2026improvinglowresourcemachinetranslation,
      title={Improving Low-Resource Machine Translation via Round-Trip Reinforcement Learning}, 
      author={Ahmed Attia and Alham Fikri Aji},
      year={2026},
      eprint={2601.12535},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.12535}, 
}
```
# efficient-roundtrip-mt
