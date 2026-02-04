"""
Backtranslation Baseline for Unsupervised NMT

Backtranslation by Sennrich et al. (2016) is a simple and effective method for 
semi-supervised and unsupervised machine translation.

The approach:
1. Given monolingual data in language A and B
2. Use a pretrained model to translate A → B (creates synthetic parallel data: A, B')
3. Use a pretrained model to translate B → A (creates synthetic parallel data: A', B)
4. Train on synthetic parallel data:
   - Forward: A → B' teaches src→tgt direction
   - Backward: A' → B teaches tgt→src direction (backtranslation proper)
   
In unsupervised setting:
- We don't use the original parallel data
- We iteratively improve translations by regenerating backtranslations each epoch

Training objective:
    L = E_{x~D_src}[Δ(M(BT(x)), x)] + E_{y~D_tgt}[Δ(M(BT(y)), y)]
    
where:
    - BT(x) = M(x, tgt_lang) is the backtranslation of x to target language
    - M(BT(x)) = M(BT(x), src_lang) translates it back
    - Δ is cross-entropy loss
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Optional, Dict, Tuple
from datasets import load_dataset
import sacrebleu
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import generate_sequences


class MonolingualDataset(Dataset):
    """Simple dataset for monolingual sentences."""
    
    def __init__(self, sentences: List[str]):
        self.sentences = sentences
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return self.sentences[idx]


class ParallelDataset(Dataset):
    """Dataset for parallel sentence pairs (source, target)."""
    
    def __init__(self, src_sentences: List[str], tgt_sentences: List[str]):
        assert len(src_sentences) == len(tgt_sentences), \
            f"Source ({len(src_sentences)}) and target ({len(tgt_sentences)}) must have same length"
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        return self.src_sentences[idx], self.tgt_sentences[idx]


def load_nllb_multi_domain(
    dataset_config_name: str,
    source_lang: str,
    target_lang: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """
    Load monolingual sentences from breakend/nllb-multi-domain dataset.
    
    Args:
        dataset_config_name: Config name (e.g., "eng_Latn-wol_Latn", "eng_Latn-ayr_Latn")
        source_lang: Source language code (e.g., "eng_Latn", "ayr_Latn")
        target_lang: Target language code (e.g., "wol_Latn", "ayr_Latn")
        split: Dataset split ("train", "validation", "test")
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        Tuple of (source_sentences, target_sentences)
    """
    print(f"Loading dataset: breakend/nllb-multi-domain ({dataset_config_name})")
    dataset = load_dataset("breakend/nllb-multi-domain", dataset_config_name, trust_remote_code=True)
    
    # Resolve split name
    split_names = {
        "train": ["train", "training"],
        "validation": ["valid", "validation", "val"],
        "test": ["test", "test_final"],
    }
    
    data_split = None
    for split_name in split_names.get(split, [split]):
        if split_name in dataset:
            data_split = dataset[split_name]
            break
    
    if data_split is None:
        raise ValueError(f"Split '{split}' not found in dataset. Available: {list(dataset.keys())}")
    
    # Extract sentences using column naming convention: sentence_{lang}
    src_col = f"sentence_{source_lang}"
    tgt_col = f"sentence_{target_lang}"
    
    # Check if columns exist
    available_cols = data_split.column_names
    if src_col not in available_cols:
        raise ValueError(f"Source column '{src_col}' not found. Available: {available_cols}")
    if tgt_col not in available_cols:
        raise ValueError(f"Target column '{tgt_col}' not found. Available: {available_cols}")
    
    # Extract sentences
    if max_samples is not None:
        data_split = data_split.select(range(min(max_samples, len(data_split))))
    
    src_sentences = data_split[src_col]
    tgt_sentences = data_split[tgt_col]
    
    print(f"Loaded {len(src_sentences)} sentence pairs from '{split}' split")
    print(f"  Source column: {src_col}")
    print(f"  Target column: {tgt_col}")
    
    return src_sentences, tgt_sentences


class BacktranslationTrainer:
    """
    Trainer for Backtranslation-based Unsupervised NMT.
    
    Implements iterative backtranslation:
        1. Generate backtranslations using current model
        2. Train on synthetic parallel data
        3. Repeat for multiple epochs
    
    Training directions:
        - src_mono → translate to tgt → train tgt→src (reconstruct src_mono)
        - tgt_mono → translate to src → train src→tgt (reconstruct tgt_mono)
    """
    
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600m",
        src_lang: str = "eng_Latn",
        tgt_lang: str = "fra_Latn",
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Clone model for generating backtranslations (updated each epoch)
        self.bt_model = copy.deepcopy(self.model)
        self.bt_model.eval()
        
        # Language token IDs
        self.src_lang_id = self.tokenizer.convert_tokens_to_ids(src_lang)
        self.tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        
        print(f"Source language: {src_lang} (ID: {self.src_lang_id})")
        print(f"Target language: {tgt_lang} (ID: {self.tgt_lang_id})")
        print(f"Device: {self.device}")
    
    def _tokenize(self, sentences: List[str], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """Tokenize a batch of sentences."""
        tokens = self.tokenizer(
            sentences,
            return_tensors=return_tensors,
            padding=True,
            truncation=True,
            max_length=256,
        )
        return {k: v.to(self.device) for k, v in tokens.items()}
    
    @torch.no_grad()
    def generate_backtranslations(
        self,
        sentences: List[str],
        tgt_lang_id: int,
        batch_size: int = 16,
        max_new_tokens: int = 128,
    ) -> List[str]:
        """
        Generate backtranslations for a list of sentences using the BT model.
        
        Args:
            sentences: List of source sentences to translate
            tgt_lang_id: Target language token ID
            batch_size: Batch size for generation
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of translated sentences
        """
        self.bt_model.eval()
        translations = []
        
        loader = DataLoader(
            MonolingualDataset(sentences),
            batch_size=batch_size,
            shuffle=False,
        )
        
        for batch in loader:
            inputs = self._tokenize(batch)
            
            generated = generate_sequences(
                self.bt_model,
                self.tokenizer,
                inputs,
                tgt_lang_id=tgt_lang_id,
                max_new_tokens=max_new_tokens,
                gen_temperature=1.0,
                num_return_sequences=1,
                do_sample=False,  # Greedy decoding for BT
            )
            
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            translations.extend(decoded)
        
        return translations
    
    def compute_translation_loss(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        tgt_lang_id: int,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for translation.
        
        Args:
            src_sentences: Source sentences (encoder input)
            tgt_sentences: Target sentences (decoder target)
            tgt_lang_id: Target language token ID
            
        Returns:
            loss: Cross-entropy loss (sum over tokens)
        """
        # Tokenize source (encoder input)
        encoder_inputs = self._tokenize(src_sentences)
        
        # Tokenize target (decoder target)
        target_tokens = self._tokenize(tgt_sentences)
        target_ids = target_tokens["input_ids"]
        
        batch_size = target_ids.size(0)
        
        # Create decoder input: [lang_token, target_ids[:-1]]
        lang_tokens = torch.full(
            (batch_size, 1), tgt_lang_id, dtype=torch.long, device=self.device
        )
        decoder_input_ids = torch.cat([lang_tokens, target_ids[:, :-1]], dim=1)
        labels = target_ids
        
        # Forward pass
        outputs = self.model(
            input_ids=encoder_inputs["input_ids"],
            attention_mask=encoder_inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids,
        )
        
        logits = outputs.logits
        
        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
            reduction='sum',
        )
        
        return loss
    
    def train_step(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        bt_src_sentences: List[str],  # Backtranslated source sentences
        bt_tgt_sentences: List[str],  # Backtranslated target sentences
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Single training step using backtranslation.
        
        We train on two directions:
        1. bt_tgt → src (BT of src to tgt, then reconstruct src)
        2. bt_src → tgt (BT of tgt to src, then reconstruct tgt)
        
        Args:
            src_sentences: Original source sentences (reconstruction target)
            tgt_sentences: Original target sentences (reconstruction target)
            bt_src_sentences: Backtranslations of tgt_sentences to source language
            bt_tgt_sentences: Backtranslations of src_sentences to target language
            optimizer: Model optimizer
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        
        # Direction 1: bt_tgt → src (tgt→src direction)
        # Input: backtranslated target sentences (in tgt lang)
        # Output: original source sentences
        loss_tgt_to_src = self.compute_translation_loss(
            bt_tgt_sentences, src_sentences, self.src_lang_id
        )
        
        # Direction 2: bt_src → tgt (src→tgt direction)
        # Input: backtranslated source sentences (in src lang)
        # Output: original target sentences
        loss_src_to_tgt = self.compute_translation_loss(
            bt_src_sentences, tgt_sentences, self.tgt_lang_id
        )
        
        loss_src_to_tgt = torch.tensor(0.0, device=self.device)
        # Total loss
        total_loss = loss_tgt_to_src
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            "loss_total": total_loss.item(),
            "loss_tgt_to_src": loss_tgt_to_src.item(),
            "loss_src_to_tgt": loss_src_to_tgt.item(),
        }
    
    @torch.no_grad()
    def translate(
        self,
        sentences: List[str],
        tgt_lang_id: int,
        max_new_tokens: int = 128,
    ) -> List[str]:
        """
        Translate a batch of sentences using greedy decoding.
        
        Args:
            sentences: List of source sentences
            tgt_lang_id: Target language token ID
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of translated sentences
        """
        self.model.eval()
        inputs = self._tokenize(sentences)
        
        generated_sequences = generate_sequences(
            self.model,
            self.tokenizer,
            inputs,
            tgt_lang_id=tgt_lang_id,
            max_new_tokens=max_new_tokens,
            gen_temperature=1.0,
            num_return_sequences=1,
            do_sample=False,
        )
        
        translations = self.tokenizer.batch_decode(
            generated_sequences, skip_special_tokens=True
        )
        return translations
    
    def run_evaluation(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        batch_size: int = 32,
        split_name: str = "Validation",
    ) -> Dict[str, float]:
        """
        Run evaluation using chrF++ and spBLEU metrics.
        
        Args:
            src_sentences: List of source language sentences
            tgt_sentences: List of target language sentences
            batch_size: Batch size for translation
            split_name: Name of the split for logging
            
        Returns:
            Dictionary with chrF++ and spBLEU scores
        """
        self.model.eval()
        
        # Collect all translations
        all_src_to_tgt = []
        all_tgt_to_src = []
        
        src_loader = DataLoader(
            MonolingualDataset(src_sentences),
            batch_size=batch_size,
            shuffle=False,
        )
        tgt_loader = DataLoader(
            MonolingualDataset(tgt_sentences),
            batch_size=batch_size,
            shuffle=False,
        )
        
        print(f"\n{split_name}: Translating {len(src_sentences)} sentence pairs...")
        
        # Translate src → tgt in batches
        for src_batch in src_loader:
            translations = self.translate(src_batch, self.tgt_lang_id)
            all_src_to_tgt.extend(translations)
        
        # Translate tgt → src in batches
        for tgt_batch in tgt_loader:
            translations = self.translate(tgt_batch, self.src_lang_id)
            all_tgt_to_src.extend(translations)
        
        # Compute corpus-level chrF++ and spBLEU
        chrf_src_to_tgt = sacrebleu.corpus_chrf(
            all_src_to_tgt,
            [list(tgt_sentences)],
            word_order=2,
        )
        
        chrf_tgt_to_src = sacrebleu.corpus_chrf(
            all_tgt_to_src,
            [list(src_sentences)],
            word_order=2,
        )
        
        bleu_src_to_tgt = sacrebleu.corpus_bleu(
            all_src_to_tgt,
            [list(tgt_sentences)],
            tokenize="flores200",
        )
        
        bleu_tgt_to_src = sacrebleu.corpus_bleu(
            all_tgt_to_src,
            [list(src_sentences)],
            tokenize="flores200",
        )
        
        metrics = {
            "chrf++_src_to_tgt": chrf_src_to_tgt.score,
            "chrf++_tgt_to_src": chrf_tgt_to_src.score,
            "spbleu_src_to_tgt": bleu_src_to_tgt.score,
            "spbleu_tgt_to_src": bleu_tgt_to_src.score,
            "chrf++_avg": (chrf_src_to_tgt.score + chrf_tgt_to_src.score) / 2,
            "spbleu_avg": (bleu_src_to_tgt.score + bleu_tgt_to_src.score) / 2,
        }
        
        # Print summary
        print(f"{'-'*70}")
        print(f"{split_name} Results ({len(src_sentences)} pairs):")
        print(f"  chrF++ (src→tgt): {metrics['chrf++_src_to_tgt']:.2f}")
        print(f"  chrF++ (tgt→src): {metrics['chrf++_tgt_to_src']:.2f}")
        print(f"  chrF++ (avg):     {metrics['chrf++_avg']:.2f}")
        print(f"  spBLEU (src→tgt): {metrics['spbleu_src_to_tgt']:.2f}")
        print(f"  spBLEU (tgt→src): {metrics['spbleu_tgt_to_src']:.2f}")
        print(f"  spBLEU (avg):     {metrics['spbleu_avg']:.2f}")
        print(f"{'-'*70}")
        
        return metrics
    
    def train(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        bt_batch_size: int = 32,
        log_interval: int = 10,
        val_src_sentences: Optional[List[str]] = None,
        val_tgt_sentences: Optional[List[str]] = None,
        test_src_sentences: Optional[List[str]] = None,
        test_tgt_sentences: Optional[List[str]] = None,
        val_batch_size: int = 16,
        regenerate_bt_every: int = 1,
    ):
        """
        Training loop for backtranslation.
        
        Args:
            src_sentences: List of source language sentences (train)
            tgt_sentences: List of target language sentences (train)
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            bt_batch_size: Batch size for backtranslation generation
            log_interval: Log every N steps
            val_src_sentences: Validation source sentences (optional)
            val_tgt_sentences: Validation target sentences (optional)
            test_src_sentences: Test source sentences (optional)
            test_tgt_sentences: Test target sentences (optional)
            val_batch_size: Batch size for validation/test
            regenerate_bt_every: Regenerate backtranslations every N epochs
        """
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Check if validation/test data is provided
        has_validation = val_src_sentences is not None and val_tgt_sentences is not None
        has_test = test_src_sentences is not None and test_tgt_sentences is not None
        
        print(f"\nStarting Backtranslation Training:")
        print(f"  - Train source sentences: {len(src_sentences)}")
        print(f"  - Train target sentences: {len(tgt_sentences)}")
        if has_validation:
            print(f"  - Val source sentences: {len(val_src_sentences)}")
            print(f"  - Val target sentences: {len(val_tgt_sentences)}")
        if has_test:
            print(f"  - Test source sentences: {len(test_src_sentences)}")
            print(f"  - Test target sentences: {len(test_tgt_sentences)}")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - BT batch size: {bt_batch_size}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Regenerate BT every: {regenerate_bt_every} epoch(s)")
        print()
        
        # Run initial validation
        if has_validation:
            print("Running initial validation...")
            self.run_evaluation(
                val_src_sentences, val_tgt_sentences,
                batch_size=val_batch_size,
                split_name="Validation (Initial)",
            )
        
        # Initialize backtranslations
        bt_tgt_sentences = None  # BT of src to tgt
        bt_src_sentences = None  # BT of tgt to src
        
        global_step = 0
        for epoch in range(num_epochs):
            # Regenerate backtranslations if needed
            if epoch % regenerate_bt_every == 0:
                print(f"\nEpoch {epoch + 1}: Generating backtranslations...")
                
                # Update BT model with current model weights
                self.bt_model = copy.deepcopy(self.model)
                self.bt_model.eval()
                
                # # Generate BT: src → tgt (for training tgt→src direction)
                print(f"  Translating {len(src_sentences)} source sentences to target...")
                bt_tgt_sentences = self.generate_backtranslations(
                    src_sentences, self.tgt_lang_id, batch_size=bt_batch_size
                )
                
                # Generate BT: tgt → src (for training src→tgt direction)
                print(f"  Translating {len(tgt_sentences)} target sentences to source...")
                bt_src_sentences = self.generate_backtranslations(
                    tgt_sentences, self.src_lang_id, batch_size=bt_batch_size
                )
                
                print(f"  Backtranslations generated.")
            
            # Create data loaders for this epoch
            # We pair: (src, bt_tgt) for tgt→src training
            #          (tgt, bt_src) for src→tgt training
            src_dataset = ParallelDataset(src_sentences, bt_tgt_sentences)
            tgt_dataset = ParallelDataset(tgt_sentences, bt_src_sentences)
            
            src_loader = DataLoader(src_dataset, batch_size=batch_size, shuffle=True)
            tgt_loader = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=True)
            
            epoch_losses = []
            
            # Iterate through both loaders
            for (src_batch, bt_tgt_batch), (tgt_batch, bt_src_batch) in zip(src_loader, tgt_loader):
                losses = self.train_step(
                    src_batch, tgt_batch,
                    bt_src_batch, bt_tgt_batch,
                    optimizer,
                )
                epoch_losses.append(losses)
                global_step += 1
                
                if global_step % log_interval == 0:
                    print(
                        f"Step {global_step:4d} | "
                        f"L_total: {losses['loss_total']:.4f} | "
                        f"L_tgt→src: {losses['loss_tgt_to_src']:.4f} | "
                        f"L_src→tgt: {losses['loss_src_to_tgt']:.4f}"
                    )
            
            # Epoch summary
            avg_loss = sum(l["loss_total"] for l in epoch_losses) / len(epoch_losses)
            avg_loss_tgt_to_src = sum(l["loss_tgt_to_src"] for l in epoch_losses) / len(epoch_losses)
            avg_loss_src_to_tgt = sum(l["loss_src_to_tgt"] for l in epoch_losses) / len(epoch_losses)
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{num_epochs} - Training Summary")
            print(f"  Avg Loss (total): {avg_loss:.4f}")
            print(f"  Avg Loss (tgt→src): {avg_loss_tgt_to_src:.4f}")
            print(f"  Avg Loss (src→tgt): {avg_loss_src_to_tgt:.4f}")
            
            # Run validation at the end of each epoch
            if has_validation:
                self.run_evaluation(
                    val_src_sentences, val_tgt_sentences,
                    batch_size=val_batch_size,
                    split_name=f"Validation (Epoch {epoch + 1})",
                )
            
            print(f"{'='*70}\n")
        
        # Run test evaluation after training completes
        if has_test:
            print("\n" + "="*70)
            print("FINAL TEST EVALUATION")
            print("="*70)
            test_metrics = self.run_evaluation(
                test_src_sentences, test_tgt_sentences,
                batch_size=val_batch_size,
                split_name="Test (Final)",
            )
            return test_metrics
        
        return None

# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtranslation Baseline for Unsupervised NMT")
    parser.add_argument("--dataset_config", type=str, default="eng_Latn-ayr_Latn",
                        help="Dataset config name (e.g., 'eng_Latn-wol_Latn', 'eng_Latn-ayr_Latn')")
    parser.add_argument("--src_lang_nllb", type=str, default="eng_Latn",
                        help="NLLB source language token (e.g., 'eng_Latn')")
    parser.add_argument("--tgt_lang_nllb", type=str, default="ayr_Latn",
                        help="NLLB target language token (e.g., 'ayr_Latn')")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to load (None for all)")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training")
    parser.add_argument("--bt_batch_size", type=int, default=2,
                        help="Batch size for backtranslation generation")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--val_batch_size", type=int, default=2,
                        help="Validation/test batch size")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Maximum validation samples (None for all)")
    parser.add_argument("--max_test_samples", type=int, default=None,
                        help="Maximum test samples (None for all)")
    parser.add_argument("--skip_validation", action="store_true",
                        help="Skip validation during training")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip test evaluation after training")
    parser.add_argument("--regenerate_bt_every", type=int, default=1,
                        help="Regenerate backtranslations every N epochs")
    args = parser.parse_args()
    
    # Load training data from breakend/nllb-multi-domain
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70)
    
    src_sentences, tgt_sentences = load_nllb_multi_domain(
        dataset_config_name=args.dataset_config,
        source_lang=args.src_lang_nllb,
        target_lang=args.tgt_lang_nllb,
        split="train",
        max_samples=args.max_samples,
    )
    
    # Load validation data
    val_src_sentences, val_tgt_sentences = None, None
    if not args.skip_validation:
        try:
            val_src_sentences, val_tgt_sentences = load_nllb_multi_domain(
                dataset_config_name=args.dataset_config,
                source_lang=args.src_lang_nllb,
                target_lang=args.tgt_lang_nllb,
                split="validation",
                max_samples=args.max_val_samples,
            )
        except ValueError as e:
            print(f"Warning: Could not load validation split: {e}")
    
    # Load test data
    test_src_sentences, test_tgt_sentences = None, None
    if not args.skip_test:
        try:
            test_src_sentences, test_tgt_sentences = load_nllb_multi_domain(
                dataset_config_name=args.dataset_config,
                source_lang=args.src_lang_nllb,
                target_lang=args.tgt_lang_nllb,
                split="test",
                max_samples=args.max_test_samples,
            )
        except ValueError as e:
            print(f"Warning: Could not load test split: {e}")
    
    # Initialize trainer
    trainer = BacktranslationTrainer(
        model_name="facebook/nllb-200-distilled-1.3b",
        src_lang=args.src_lang_nllb,
        tgt_lang=args.tgt_lang_nllb,
    )
    
    # Run training
    print("\n" + "="*70)
    print("TRAINING (Backtranslation Baseline)")
    print("="*70)
    
    test_metrics = trainer.train(
        src_sentences=src_sentences,
        tgt_sentences=tgt_sentences,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        bt_batch_size=args.bt_batch_size,
        log_interval=args.log_interval,
        val_src_sentences=val_src_sentences,
        val_tgt_sentences=val_tgt_sentences,
        test_src_sentences=test_src_sentences,
        test_tgt_sentences=test_tgt_sentences,
        val_batch_size=args.val_batch_size,
        regenerate_bt_every=args.regenerate_bt_every,
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    if test_metrics:
        print(f"\nFinal Test Results:")
        print(f"  chrF++ (src→tgt): {test_metrics['chrf++_src_to_tgt']:.2f}")
        print(f"  chrF++ (tgt→src): {test_metrics['chrf++_tgt_to_src']:.2f}")
        print(f"  chrF++ (avg):     {test_metrics['chrf++_avg']:.2f}")
        print(f"  spBLEU (src→tgt): {test_metrics['spbleu_src_to_tgt']:.2f}")
        print(f"  spBLEU (tgt→src): {test_metrics['spbleu_tgt_to_src']:.2f}")
        print(f"  spBLEU (avg):     {test_metrics['spbleu_avg']:.2f}")
