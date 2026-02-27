"""
Implementation of the baseline from the paper: Unsupervised Machine Translation Using Monolingual Corpora Only by Lample et al. (2018).

Denoising Auto-Encoding + Cross-Domain + Adversarial Training for Unsupervised NMT

This script implements:

1. Auto-Encoding Loss (Equation 1):
    L_auto(θ_enc, θ_dec, Z, ℓ) = E_{x~D_ℓ, x̂~d(e(C(x),ℓ),ℓ)} [Δ(x̂, x)]
    Pipeline: x → C(x) → encode → decode → x̂ (same language)

2. Cross-Domain Loss (Equation 2):
    L_cd(θ_enc, θ_dec, Z, ℓ_src, ℓ_tgt) = E_{x~D_src}[Δ(x̂, x)]
    Pipeline: x_src → translate to tgt → C(tgt) → translate back to src → x̂_src
    Also: x_tgt → translate to src → C(src) → translate back to tgt → x̂_tgt

3. Discriminator Loss:
    L_D(θ_D|θ,Z) = -E_{(x_i, ℓ_i)}[log p_D(ℓ_i|e(x_i, ℓ_i))]
    
4. Adversarial Loss (Equation 3):
    L_adv(θ_enc, Z|θ_D) = -E_{(x_i, ℓ_i)}[log p_D(ℓ_j|e(x_i, ℓ_i))]
    where ℓ_j = ℓ_1 if ℓ_i = ℓ_2 (flipped labels - encoder fools discriminator)

Training objective:
    L_enc_dec = λ_auto * [L_auto(src) + L_auto(tgt)] 
              + λ_cd * [L_cd(src→tgt→src) + L_cd(tgt→src→tgt)] 
              + λ_adv * [L_adv + L_adv_cd]
    where:
        - L_adv: adversarial loss on original sentence encoder outputs
        - L_adv_cd: adversarial loss on cross-domain intermediate encoder outputs
    L_disc = L_D (trained separately)
"""
import copy
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Optional, Dict, Tuple
from datasets import load_dataset
from sacrebleu.metrics import CHRF, BLEU

# Add parent directory to path for imports when running as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import noise_model_batch, generate_sequences

class Discriminator(nn.Module):
    """
    Discriminator for adversarial training in unsupervised NMT.
    
    The discriminator operates on the encoder output (z1, ..., zm) and produces
    a binary prediction about the language:
    
        p_D(ℓ|z1,...,zm) ∝ ∏_{j=1}^{m} p_D(ℓ|z_j)
    
    In log space: log p_D(ℓ|z1,...,zm) = Σ_{j=1}^{m} log p_D(ℓ|z_j)
    
    Architecture: MLP with 3 hidden layers (1024 units), LeakyReLU activations.
    
    Args:
        input_dim: Dimension of encoder hidden states
        hidden_dim: Dimension of hidden layers (default: 1024)
        smoothing: Label smoothing coefficient (default: 0.1)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 1024, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        
        # MLP that processes each token embedding z_j → p_D(ℓ|z_j)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),  # Output: log p_D(ℓ=1|z_j) before sigmoid
        )
    
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass: compute p_D(ℓ|z1,...,zm) for each sentence.
        
        Args:
            encoder_hidden_states: (batch, seq_len, hidden_dim) - encoder outputs
            attention_mask: (batch, seq_len) - mask for valid tokens
            
        Returns:
            probs: (batch, 1) - probability that input is from target language
        """
        batch_size, seq_len, hidden_dim = encoder_hidden_states.shape
        
        # Apply MLP to each token: (batch, seq_len, hidden_dim) → (batch, seq_len, 1)
        token_logits = self.network(encoder_hidden_states)  # (batch, seq_len, 1)
        token_logits = token_logits.squeeze(-1)  # (batch, seq_len)
        
        # Aggregate across sequence: log p_D(ℓ|z1,...,zm) = Σ log p_D(ℓ|z_j)
        # Use attention mask to only sum over valid tokens
        if attention_mask is not None:
            # Mask out padding tokens
            token_logits = token_logits.masked_fill(attention_mask == 0, 0.0)
            # Sum log probabilities and normalize by sequence length
            seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
            aggregated_logits = token_logits.sum(dim=1, keepdim=True) / seq_lengths
        else:
            aggregated_logits = token_logits.mean(dim=1, keepdim=True)
        
        # Convert to probability
        probs = torch.sigmoid(aggregated_logits)  # (batch, 1)
        
        return probs
    
    def compute_loss(
        self,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        language_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute discriminator loss with label smoothing.
        
        L_D = -E[log p_D(ℓ_i|e(x_i))] = BCE(preds, smoothed_labels)
        
        Args:
            encoder_hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len)
            language_labels: (batch,) - 0 for source, 1 for target
            
        Returns:
            loss: Discriminator loss (scalar)
        """
        preds = self.forward(encoder_hidden_states, attention_mask)
        
        # Ensure labels have correct shape
        if language_labels.dim() == 1:
            language_labels = language_labels.unsqueeze(1)
        language_labels = language_labels.float()
        
        # Label smoothing: 1 → (1-s), 0 → s
        smoothed_labels = language_labels * (1.0 - 2 * self.smoothing) + self.smoothing
        
        loss = F.binary_cross_entropy(preds, smoothed_labels)
        return loss


class MonolingualDataset(Dataset):
    """Simple dataset for monolingual sentences."""
    
    def __init__(self, sentences: List[str]):
        self.sentences = sentences
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return self.sentences[idx]


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


class UMNMTTrainer:
    """
    Trainer for Denoising Auto-Encoding + Cross-Domain + Adversarial Training.
    
    Implements:
        1. Auto-encoding: src → C(src) → encode → decode → src_hat (same lang)
        2. Auto-encoding: tgt → C(tgt) → encode → decode → tgt_hat (same lang)
        3. Cross-domain: src → translate to tgt → C(tgt) → translate back to src
        4. Cross-domain: tgt → translate to src → C(src) → translate back to tgt
        5. Adversarial: encoder fools discriminator (language-agnostic representations)
    """
    
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600m",
        src_lang: str = "eng_Latn",
        tgt_lang: str = "fra_Latn",
        noise_pwd: float = 0.1,
        noise_k: int = 3,
        lambda_auto: float = 1.0,
        lambda_cd: float = 1.0,
        lambda_adv: float = 1.0,
        disc_hidden_dim: int = 1024,
        disc_smoothing: float = 0.1,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.noise_pwd = noise_pwd
        self.noise_k = noise_k
        self.lambda_auto = lambda_auto
        self.lambda_cd = lambda_cd
        self.lambda_adv = lambda_adv
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # clone the model to self.prev_model
        self.prev_model = copy.deepcopy(self.model)
        # eval only
        self.prev_model.eval()
        # Language token IDs
        self.src_lang_id = self.tokenizer.convert_tokens_to_ids(src_lang)
        self.tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        
        # Initialize discriminator
        hidden_dim = self.model.config.d_model
        self.discriminator = Discriminator(
            input_dim=hidden_dim,
            hidden_dim=disc_hidden_dim,
            smoothing=disc_smoothing,
        ).to(self.device)
        
        print(f"Source language: {src_lang} (ID: {self.src_lang_id})")
        print(f"Target language: {tgt_lang} (ID: {self.tgt_lang_id})")
        print(f"Discriminator: input_dim={hidden_dim}, hidden_dim={disc_hidden_dim}")
        print(f"Lambda auto-encoding: {lambda_auto}")
        print(f"Lambda cross-domain: {lambda_cd}")
        print(f"Lambda adversarial: {lambda_adv}")
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
    
    def compute_reconstruction_loss(
        self,
        encoder_inputs: Dict[str, torch.Tensor],
        target_ids: torch.Tensor,
        tgt_lang_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction loss: Δ(x̂, x) = sum of token-level cross-entropy.
        
        Uses teacher forcing: encoder gets noisy input, decoder gets shifted target
        with target language token prepended.
        
        Args:
            encoder_inputs: Tokenized sentences for encoder (noisy or translated)
            target_ids: Original sentence token IDs (reconstruction target)
            tgt_lang_id: Target language token ID (prepended to decoder input)
            
        Returns:
            loss: Reconstruction loss
            logits: Model output logits
        """
        batch_size = target_ids.size(0)
        
        # Create decoder input: [lang_token, target_ids[:-1]]
        # Prepend target language token for NLLB
        lang_tokens = torch.full(
            (batch_size, 1), tgt_lang_id, dtype=torch.long, device=self.device
        )
        # Decoder input: [lang_id, tok1, tok2, ..., tok_{n-1}]
        decoder_input_ids = torch.cat([lang_tokens, target_ids[:, :-1]], dim=1)
        # Labels: [tok1, tok2, ..., tok_n] (shifted by 1)
        labels = target_ids
        
        # Forward pass with teacher forcing
        outputs = self.model(
            input_ids=encoder_inputs["input_ids"],
            attention_mask=encoder_inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids,
        )
        
        logits = outputs.logits
        
        # Compute cross-entropy loss (mean over non-padding tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
            reduction='sum',
        )
        
        return loss, logits
    
    def compute_auto_encoding_loss(
        self,
        sentences: List[str],
        lang_id: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Denoising Auto-Encoding Loss (Equation 1):
        
        L_auto(θ_enc, θ_dec, Z, ℓ) = E_{x~D_ℓ, x̂~d(e(C(x),ℓ),ℓ)} [Δ(x̂, x)]
        
        Steps:
            1. Sample sentence x from monolingual data
            2. Apply noise: C(x)
            3. Encode: e(C(x), ℓ)
            4. Decode with teacher forcing: use original x as target
            5. Compute loss: Δ(decoder_output, x)
        
        Args:
            sentences: Batch of original sentences
            lang_id: Language token ID for this language
        
        Returns:
            loss: Auto-encoding loss
            metrics: Dictionary with additional metrics
        """
        # Step 1-2: Apply noise model C(x)
        noisy_sentences = noise_model_batch(
            sentences, pwd=self.noise_pwd, k=self.noise_k
        )
        
        # Tokenize noisy sentences (encoder input)
        noisy_inputs = self._tokenize(noisy_sentences)
        
        # Tokenize original sentences (reconstruction target)
        target_tokens = self._tokenize(sentences)
        target_ids = target_tokens["input_ids"]
    
        # Step 3-5: Encode noisy, decode with teacher forcing to reconstruct original
        # This is differentiable - gradients flow through encoder and decoder
        loss, logits = self.compute_reconstruction_loss(noisy_inputs, target_ids, lang_id)
        
        # Compute accuracy (for monitoring)
        preds = logits.argmax(dim=-1)
        labels = target_ids
        mask = labels != self.tokenizer.pad_token_id
        accuracy = ((preds == labels) & mask).sum().float() / mask.sum().float()
        
        metrics = {
            "accuracy": accuracy.item(),
            "num_tokens": mask.sum().item(),
        }
        
        return loss, metrics
    
    def compute_cross_domain_loss(
        self,
        sentences: List[str],
        src_lang_id: int,
        tgt_lang_id: int,
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor, torch.Tensor]:
        """
        Compute Cross-Domain Loss (Equation 2):
        
        L_cd(θ_enc, θ_dec, Z, ℓ_src, ℓ_tgt) = E_{x~D_src}[Δ(x̂, x)]
        
        Pipeline: x → translate to tgt (prev_model) → C(tgt) → encode → decode back to src (teacher forcing)
        
        Steps:
            1. Sample sentence x from source monolingual data D_src
            2. Translate to target language using prev_model (no gradient): y = M_prev(x, tgt_lang)
            3. Apply noise: C(y)
            4. Encode C(y) with current model (gradients flow here)
            5. Decode back to source with teacher forcing using original x as target
            6. Compute reconstruction loss: Δ(decoder_output, x)
        
        Note: This is differentiable because we use teacher forcing, not generation,
        for the back-translation step.
        
        Args:
            sentences: Batch of original sentences (in src_lang)
            src_lang_id: Language token ID for source language (decoder target)
            tgt_lang_id: Language token ID for target language (intermediate translation)
            
        Returns:
            loss: Cross-domain reconstruction loss
            metrics: Dictionary with additional metrics
            encoder_hidden_states: Encoder outputs from encoding C(y) - for adversarial loss
            attention_mask: Attention mask for encoder outputs
        """
        # Step 1: Tokenize original sentences
        original_inputs = self._tokenize(sentences)
        
        # Step 2: Translate to target language using prev_model (no gradient)
        # This creates pseudo-parallel data: (x_src, y_tgt)
        with torch.no_grad():
            translated_sequences = generate_sequences(
                self.prev_model,
                self.tokenizer,
                original_inputs,
                tgt_lang_id=tgt_lang_id,
                max_new_tokens=128,
                gen_temperature=0.7,
                num_return_sequences=1,
                do_sample=False,  # greedy decoding as in the paper
            )
        translated_texts = self.tokenizer.batch_decode(
            translated_sequences, skip_special_tokens=True
        )
        
        # Step 3: Apply noise to translated sentences C(y)
        noisy_translated = noise_model_batch(
            translated_texts, pwd=self.noise_pwd, k=self.noise_k
        )
        noisy_translated_inputs = self._tokenize(noisy_translated)
        
        # Step 4: Encode C(y) - gradients flow through encoder
        encoder_outputs = self.model.get_encoder()(
            input_ids=noisy_translated_inputs["input_ids"],
            attention_mask=noisy_translated_inputs["attention_mask"],
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        attention_mask = noisy_translated_inputs["attention_mask"]
        
        # Step 5-6: Decode back to source with teacher forcing
        # Encoder input: C(y) (noisy translation in target language)
        # Decoder target: x (original sentence in source language)
        # This is differentiable - gradients flow through both encoder and decoder
        target_tokens = self._tokenize(sentences)
        target_ids = target_tokens["input_ids"]
        
        loss, logits = self.compute_reconstruction_loss(
            noisy_translated_inputs, target_ids, src_lang_id
        )
        
        # Compute accuracy (for monitoring)
        preds = logits.argmax(dim=-1)
        labels = target_ids
        mask = labels != self.tokenizer.pad_token_id
        accuracy = ((preds == labels) & mask).sum().float() / mask.sum().float()
        
        metrics = {
            "accuracy": accuracy.item(),
            "num_tokens": mask.sum().item(),
        }
        
        return loss, metrics, encoder_hidden_states, attention_mask
    
    def _encode_sentences(
        self,
        sentences: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sentences and return encoder hidden states.
        
        Args:
            sentences: List of sentences to encode
        
        Returns:
            encoder_hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len)
        """
        inputs = self._tokenize(sentences)
        
        encoder_outputs = self.model.get_encoder()(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        
        return encoder_outputs.last_hidden_state, inputs["attention_mask"]
    
    def compute_adversarial_loss_from_hidden(
        self,
        src_hidden: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_hidden: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Adversarial Loss from pre-computed encoder hidden states.
        
        L_adv(θ_enc, Z|θ_D) = -E_{(x_i, ℓ_i)}[log p_D(ℓ_j|e(x_i, ℓ_i))]
        
        where ℓ_j = ℓ_1 if ℓ_i = ℓ_2 (FLIPPED labels)
        
        Args:
            src_hidden: Encoder hidden states for source-domain text (batch, seq, hidden)
            src_mask: Attention mask for source hidden states
            tgt_hidden: Encoder hidden states for target-domain text (batch, seq, hidden)
            tgt_mask: Attention mask for target hidden states
            
        Returns:
            adv_loss: Adversarial loss for encoder
        """
        # Source: true label = 0, flipped label = 1
        src_disc_preds = self.discriminator(src_hidden, src_mask)
        src_flipped_labels = torch.ones(src_hidden.size(0), 1, device=self.device)
        loss_adv_src = F.binary_cross_entropy(src_disc_preds, src_flipped_labels)
        
        # Target: true label = 1, flipped label = 0
        tgt_disc_preds = self.discriminator(tgt_hidden, tgt_mask)
        tgt_flipped_labels = torch.zeros(tgt_hidden.size(0), 1, device=self.device)
        loss_adv_tgt = F.binary_cross_entropy(tgt_disc_preds, tgt_flipped_labels)
        
        adv_loss = (loss_adv_src + loss_adv_tgt) / 2
        
        return adv_loss
    
    def compute_adversarial_loss(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute Adversarial Loss (Equation 3):
        
        L_adv(θ_enc, Z|θ_D) = -E_{(x_i, ℓ_i)}[log p_D(ℓ_j|e(x_i, ℓ_i))]
        
        where ℓ_j = ℓ_1 if ℓ_i = ℓ_2 (FLIPPED labels)
        
        The encoder is trained to FOOL the discriminator by making it predict
        the WRONG language → language-agnostic representations.
        
        Args:
            src_sentences: Sentences from source domain (label = 0)
            tgt_sentences: Sentences from target domain (label = 1)
            
        Returns:
            adv_loss: Adversarial loss for encoder
            hidden_states: Tuple of (src_hidden, tgt_hidden) encoder outputs
            attention_masks: Tuple of (src_mask, tgt_mask) attention masks
        """
        # Encode source sentences
        src_hidden, src_mask = self._encode_sentences(src_sentences)
        
        # Encode target sentences
        tgt_hidden, tgt_mask = self._encode_sentences(tgt_sentences)
        
        # Compute adversarial loss using helper method
        adv_loss = self.compute_adversarial_loss_from_hidden(
            src_hidden, src_mask, tgt_hidden, tgt_mask
        )
        
        return adv_loss, (src_hidden, tgt_hidden), (src_mask, tgt_mask)
    
    def compute_discriminator_loss(
        self,
        hidden_states: Tuple[torch.Tensor, torch.Tensor],
        attention_masks: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute Discriminator Loss:
        
        L_D(θ_D|θ,Z) = -E_{(x_i, ℓ_i)}[log p_D(ℓ_i|e(x_i, ℓ_i))]
        
        The discriminator is trained to CORRECTLY classify languages.
        
        Args:
            hidden_states: Tuple of (src_hidden, tgt_hidden) encoder outputs
            attention_masks: Tuple of (src_mask, tgt_mask) attention masks
            
        Returns:
            disc_loss: Discriminator loss
        """
        src_hidden, tgt_hidden = hidden_states
        src_mask, tgt_mask = attention_masks
        
        # DETACH embeddings so discriminator training doesn't affect encoder
        src_hidden_detached = src_hidden.detach()
        tgt_hidden_detached = tgt_hidden.detach()
        
        # Source: true label = 0
        src_labels = torch.zeros(src_hidden.size(0), device=self.device)
        loss_disc_src = self.discriminator.compute_loss(
            src_hidden_detached, src_mask, src_labels
        )
        
        # Target: true label = 1
        tgt_labels = torch.ones(tgt_hidden.size(0), device=self.device)
        loss_disc_tgt = self.discriminator.compute_loss(
            tgt_hidden_detached, tgt_mask, tgt_labels
        )
        
        # Average the two losses
        disc_loss = (loss_disc_src + loss_disc_tgt) / 2
        
        return disc_loss
    
    def train_step(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        enc_dec_optimizer: torch.optim.Optimizer,
        disc_optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Single training step for auto-encoding + adversarial training.
        
        Computes:
            L_enc_dec = L_auto(src) + L_auto(tgt) + λ_adv * L_adv
            L_disc = L_D (discriminator loss)
        
        Args:
            src_sentences: Batch of source language sentences
            tgt_sentences: Batch of target language sentences
            enc_dec_optimizer: Optimizer for encoder/decoder parameters
            disc_optimizer: Optimizer for discriminator parameters
            
        Returns:
            Dictionary of loss values and metrics
        """
        self.model.train()
        self.discriminator.train()
        
        # =====================================================================
        # STEP 1: Compute Auto-Encoding Losses
        # =====================================================================
        
        # L_auto(src): src → C(src) → src_hat
        loss_auto_src, metrics_auto_src = self.compute_auto_encoding_loss(
            src_sentences, self.src_lang_id
        )
        
        # L_auto(tgt): tgt → C(tgt) → tgt_hat
        loss_auto_tgt, metrics_auto_tgt = self.compute_auto_encoding_loss(
            tgt_sentences, self.tgt_lang_id
        )
        
        # =====================================================================
        # STEP 2: Compute Cross-Domain Losses
        # =====================================================================
        
        # L_cd(src→tgt→src): src → translate to tgt → C(tgt) → translate back to src
        # Also returns encoder hidden states from encoding C(tgt) for adversarial loss
        loss_cd_src, metrics_cd_src, cd_tgt_hidden, cd_tgt_mask = self.compute_cross_domain_loss(
            src_sentences, self.src_lang_id, self.tgt_lang_id
        )
        
        # L_cd(tgt→src→tgt): tgt → translate to src → C(src) → translate back to tgt
        # Also returns encoder hidden states from encoding C(src) for adversarial loss
        loss_cd_tgt, metrics_cd_tgt, cd_src_hidden, cd_src_mask = self.compute_cross_domain_loss(
            tgt_sentences, self.tgt_lang_id, self.src_lang_id
        )
        
        # =====================================================================
        # STEP 3: Compute Adversarial Loss (encoder fools discriminator)
        # =====================================================================
        
        # Adversarial loss on original sentences
        loss_adv, hidden_states, attention_masks = self.compute_adversarial_loss(
            src_sentences, tgt_sentences
        )
        
        # Adversarial loss on cross-domain encoder outputs
        # cd_src_hidden: encoder output when encoding C(src) (in tgt→src→tgt pipeline)
        # cd_tgt_hidden: encoder output when encoding C(tgt) (in src→tgt→src pipeline)
        loss_adv_cd = self.compute_adversarial_loss_from_hidden(
            cd_src_hidden, cd_src_mask, cd_tgt_hidden, cd_tgt_mask
        )
        
        # =====================================================================
        # STEP 4: Update Encoder/Decoder
        # =====================================================================
        
        # Total encoder/decoder loss (Equation from paper)
        # L_enc_dec = λ_auto * [L_auto(src) + L_auto(tgt)] 
        #           + λ_cd * [L_cd(src,tgt) + L_cd(tgt,src)] 
        #           + λ_adv * [L_adv + L_adv_cd]
        # where L_adv is on original sentences, L_adv_cd is on cross-domain encoder outputs
        loss_enc_dec = (
            self.lambda_auto * (loss_auto_src + loss_auto_tgt) +
            self.lambda_cd * (loss_cd_src + loss_cd_tgt) +
            self.lambda_adv * (loss_adv + loss_adv_cd)
        )
        
        enc_dec_optimizer.zero_grad()
        loss_enc_dec.backward()
        enc_dec_optimizer.step()
        
        # =====================================================================
        # STEP 5: Update Discriminator (trained to classify correctly)
        # =====================================================================
        
        loss_disc = self.compute_discriminator_loss(hidden_states, attention_masks)
        
        disc_optimizer.zero_grad()
        loss_disc.backward()
        disc_optimizer.step()
        
        # =====================================================================
        # Compute discriminator accuracy for monitoring
        # =====================================================================
        src_hidden, tgt_hidden = hidden_states
        src_mask, tgt_mask = attention_masks
        with torch.no_grad():
            src_preds = self.discriminator(src_hidden.detach(), src_mask)
            tgt_preds = self.discriminator(tgt_hidden.detach(), tgt_mask)
            src_acc = (src_preds < 0.5).float().mean()  # src should predict 0
            tgt_acc = (tgt_preds > 0.5).float().mean()  # tgt should predict 1
            disc_acc = (src_acc + tgt_acc) / 2
        
        return {
            "loss_total": loss_enc_dec.item(),
            "loss_auto_src": loss_auto_src.item(),
            "loss_auto_tgt": loss_auto_tgt.item(),
            "loss_cd_src": loss_cd_src.item(),
            "loss_cd_tgt": loss_cd_tgt.item(),
            "loss_adv": loss_adv.item(),
            "loss_adv_cd": loss_adv_cd.item(),
            "loss_disc": loss_disc.item(),
            "acc_auto_src": metrics_auto_src["accuracy"],
            "acc_auto_tgt": metrics_auto_tgt["accuracy"],
            "acc_cd_src": metrics_cd_src["accuracy"],
            "acc_cd_tgt": metrics_cd_tgt["accuracy"],
            "disc_acc": disc_acc.item(),
        }
    
    @torch.no_grad()
    def translate(
        self,
        sentences: List[str],
        tgt_lang_id: int,
        max_new_tokens: int = 63,
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
            gen_temperature=1.8,
            num_return_sequences=1,
            do_sample=False,  # Greedy decoding
        )
        
        translations = self.tokenizer.batch_decode(
            generated_sequences, skip_special_tokens=True
        )
        return translations
    
    @torch.no_grad()
    def evaluate(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate translation quality using chrF++ and spBLEU metrics.
        
        Uses greedy decoding for translations.
        
        Args:
            src_sentences: Source language sentences (used as references for tgt→src)
            tgt_sentences: Target language sentences (used as references for src→tgt)
            
        Returns:
            Dictionary with chrF++ and spBLEU scores for both directions
        """
        self.model.eval()
        
        # Translate src → tgt (greedy decoding)
        src_to_tgt_translations = self.translate(
            src_sentences, self.tgt_lang_id
        )
        
        # Translate tgt → src (greedy decoding)
        tgt_to_src_translations = self.translate(
            tgt_sentences, self.src_lang_id
        )
        chrf_metric = CHRF(word_order=2, char_order=6)
        bleu_metric = BLEU(tokenize="flores200")
        # Compute chrF++ for src → tgt
        chrf_src_to_tgt = chrf_metric.corpus_score(
            hypotheses=src_to_tgt_translations,
            references=[tgt_sentences],
        )
        
        # Compute chrF++ for tgt → src
        chrf_tgt_to_src = chrf_metric.corpus_score(
            hypotheses=tgt_to_src_translations,
            references=[src_sentences],
        )
        
        # Compute spBLEU for src → tgt (using flores200 tokenizer for NLLB)
        bleu_src_to_tgt = bleu_metric.corpus_score(
            hypotheses=src_to_tgt_translations,
            references=[tgt_sentences],
        )
        
        # Compute spBLEU for tgt → src
        bleu_tgt_to_src = bleu_metric.corpus_score(
            hypotheses=tgt_to_src_translations,
            references=[src_sentences],
        )
        
        return {
            "chrf++_src_to_tgt": chrf_src_to_tgt.score,
            "chrf++_tgt_to_src": chrf_tgt_to_src.score,
            "spbleu_src_to_tgt": bleu_src_to_tgt.score,
            "spbleu_tgt_to_src": bleu_tgt_to_src.score,
            "chrf++_avg": (chrf_src_to_tgt.score + chrf_tgt_to_src.score) / 2,
            "spbleu_avg": (bleu_src_to_tgt.score + bleu_tgt_to_src.score) / 2,
        }
    
    def run_evaluation(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        batch_size: int = 32,
        split_name: str = "Validation",
    ) -> Dict[str, float]:
        """
        Run evaluation on a full dataset split using chrF++ and spBLEU.
        
        Translates in batches but computes corpus-level metrics on all translations.
        
        Args:
            src_sentences: List of source language sentences
            tgt_sentences: List of target language sentences
            batch_size: Batch size for translation
            split_name: Name of the split for logging (e.g., "Validation", "Test")
            
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
        chrf_metric = CHRF(word_order=2, char_order=6)
        bleu_metric = BLEU(tokenize="flores200")
        chrf_src_to_tgt = chrf_metric.corpus_score(
            hypotheses=all_src_to_tgt,
            references=[tgt_sentences],
        )
        chrf_tgt_to_src = chrf_metric.corpus_score(
            hypotheses=all_tgt_to_src,
            references=[src_sentences],
        )
        
        bleu_src_to_tgt = bleu_metric.corpus_score(
            hypotheses=all_src_to_tgt,
            references=[tgt_sentences],
        )
        
        bleu_tgt_to_src = bleu_metric.corpus_score(
            hypotheses=all_tgt_to_src,
            references=[src_sentences],
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
        enc_dec_lr: float = 1e-4,
        disc_lr: float = 1e-4,
        log_interval: int = 10,
        val_src_sentences: Optional[List[str]] = None,
        val_tgt_sentences: Optional[List[str]] = None,
        test_src_sentences: Optional[List[str]] = None,
        test_tgt_sentences: Optional[List[str]] = None,
        val_batch_size: int = 16,
    ):
        """
        Training loop for auto-encoding + adversarial training.
        
        Args:
            src_sentences: List of source language sentences (train)
            tgt_sentences: List of target language sentences (train)
            num_epochs: Number of training epochs
            batch_size: Batch size
            enc_dec_lr: Learning rate for encoder/decoder
            disc_lr: Learning rate for discriminator
            log_interval: Log every N steps
            val_src_sentences: Validation source sentences (optional)
            val_tgt_sentences: Validation target sentences (optional)
            test_src_sentences: Test source sentences (optional)
            test_tgt_sentences: Test target sentences (optional)
            val_batch_size: Batch size for validation/test
        """
        # Create data loaders
        src_loader = DataLoader(
            MonolingualDataset(src_sentences),
            batch_size=batch_size,
            shuffle=True,
        )
        tgt_loader = DataLoader(
            MonolingualDataset(tgt_sentences),
            batch_size=batch_size,
            shuffle=True,
        )
        
        # Setup optimizers (separate for encoder/decoder and discriminator)
        enc_dec_optimizer = torch.optim.RMSprop(self.model.parameters(), lr=enc_dec_lr, alpha=0.5)
        disc_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=disc_lr, alpha=0.5)
        
        # Check if validation/test data is provided
        has_validation = val_src_sentences is not None and val_tgt_sentences is not None
        has_test = test_src_sentences is not None and test_tgt_sentences is not None
        
        print(f"\nStarting training:")
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
        print(f"  - Val/Test batch size: {val_batch_size}")
        print(f"  - Enc/Dec LR: {enc_dec_lr}")
        print(f"  - Disc LR: {disc_lr}")
        print(f"  - Lambda auto: {self.lambda_auto}")
        print(f"  - Lambda cd: {self.lambda_cd}")
        print(f"  - Lambda adv: {self.lambda_adv}")
        print(f"  - Noise dropout: {self.noise_pwd}")
        print(f"  - Noise shuffle k: {self.noise_k}")
        print()
        print(f"Running initial validation...")
        self.run_evaluation(
                    val_src_sentences, val_tgt_sentences,
                    batch_size=val_batch_size,
                    split_name=f"Validation (Initial)",
        )
        global_step = 0
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Iterate through both loaders (zip to align batches)
            for src_batch, tgt_batch in zip(src_loader, tgt_loader):
                losses = self.train_step(
                    src_batch, tgt_batch,
                    enc_dec_optimizer, disc_optimizer
                )
                epoch_losses.append(losses)
                global_step += 1
                
                if global_step % log_interval == 0:
                    print(
                        f"Step {global_step:4d} | "
                        f"L_total: {losses['loss_total']:.4f} "
                        f"(auto: {losses['loss_auto_src'] + losses['loss_auto_tgt']:.4f}, "
                        f"cd: {losses['loss_cd_src'] + losses['loss_cd_tgt']:.4f}, "
                        f"adv: {losses['loss_adv']:.4f}, "
                        f"adv_cd: {losses['loss_adv_cd']:.4f}) | "
                        f"L_disc: {losses['loss_disc']:.4f} | "
                        f"Disc acc: {losses['disc_acc']:.2%}"
                    )
            
            # Epoch summary
            avg_loss = sum(l["loss_total"] for l in epoch_losses) / len(epoch_losses)
            avg_loss_auto = sum(l["loss_auto_src"] + l["loss_auto_tgt"] for l in epoch_losses) / len(epoch_losses)
            avg_loss_cd = sum(l["loss_cd_src"] + l["loss_cd_tgt"] for l in epoch_losses) / len(epoch_losses)
            avg_loss_adv = sum(l["loss_adv"] for l in epoch_losses) / len(epoch_losses)
            avg_loss_adv_cd = sum(l["loss_adv_cd"] for l in epoch_losses) / len(epoch_losses)
            avg_loss_disc = sum(l["loss_disc"] for l in epoch_losses) / len(epoch_losses)
            avg_acc_auto_src = sum(l["acc_auto_src"] for l in epoch_losses) / len(epoch_losses)
            avg_acc_auto_tgt = sum(l["acc_auto_tgt"] for l in epoch_losses) / len(epoch_losses)
            avg_acc_cd_src = sum(l["acc_cd_src"] for l in epoch_losses) / len(epoch_losses)
            avg_acc_cd_tgt = sum(l["acc_cd_tgt"] for l in epoch_losses) / len(epoch_losses)
            avg_disc_acc = sum(l["disc_acc"] for l in epoch_losses) / len(epoch_losses)
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{num_epochs} - Training Summary")
            print(f"  Avg Loss (enc/dec): {avg_loss:.4f}")
            print(f"  Avg L_auto: {avg_loss_auto:.4f}")
            print(f"  Avg L_cd: {avg_loss_cd:.4f}")
            print(f"  Avg L_adv (original): {avg_loss_adv:.4f}")
            print(f"  Avg L_adv (cross-domain): {avg_loss_adv_cd:.4f}")
            print(f"  Avg L_disc: {avg_loss_disc:.4f}")
            print(f"  Avg Acc (auto-enc src): {avg_acc_auto_src:.2%}")
            print(f"  Avg Acc (auto-enc tgt): {avg_acc_auto_tgt:.2%}")
            print(f"  Avg Acc (cross-domain src→tgt→src): {avg_acc_cd_src:.2%}")
            print(f"  Avg Acc (cross-domain tgt→src→tgt): {avg_acc_cd_tgt:.2%}")
            print(f"  Avg Discriminator Acc: {avg_disc_acc:.2%}")
            
            # Run validation at the end of each epoch
            if has_validation:
                self.run_evaluation(
                    val_src_sentences, val_tgt_sentences,
                    batch_size=val_batch_size,
                    split_name=f"Validation (Epoch {epoch + 1})",
                )
            
            # Update prev_model at end of epoch (for stable cross-domain translations)
            self.prev_model = copy.deepcopy(self.model)
            self.prev_model.eval()
            
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
    
    parser = argparse.ArgumentParser(description="Unsupervised NMT Training")
    parser.add_argument("--dataset_config", type=str, default="eng_Latn-ayr_Latn",
                        help="Dataset config name (e.g., 'eng_Latn-wol_Latn', 'eng_Latn-ayr_Latn')")
    parser.add_argument("--src_lang_nllb", type=str, default="eng_Latn",
                        help="NLLB source language token (e.g., 'eng_Latn')")
    parser.add_argument("--tgt_lang_nllb", type=str, default="ayr_Latn",
                        help="NLLB target language token (e.g., 'fra_Latn')")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to load (None for all)")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--enc_dec_lr", type=float, default=3e-5,
                        help="Encoder/decoder learning rate")
    parser.add_argument("--disc_lr", type=float, default=5e-5,
                        help="Discriminator learning rate")
    parser.add_argument("--lambda_auto", type=float, default=1.0,
                        help="Auto-encoding loss weight")
    parser.add_argument("--lambda_cd", type=float, default=1.0,
                        help="Cross-domain loss weight")
    parser.add_argument("--lambda_adv", type=float, default=1.0,
                        help="Adversarial loss weight")
    parser.add_argument("--noise_pwd", type=float, default=0.1,
                        help="Word dropout probability")
    parser.add_argument("--noise_k", type=int, default=3,
                        help="Max shuffle distance")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--val_batch_size", type=int, default=16,
                        help="Validation/test batch size")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Maximum validation samples (None for all)")
    parser.add_argument("--max_test_samples", type=int, default=None,
                        help="Maximum test samples (None for all)")
    parser.add_argument("--skip_validation", action="store_true",
                        help="Skip validation during training")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip test evaluation after training")
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
    
    # Initialize trainer with auto-encoding, cross-domain, and adversarial components
    trainer = UMNMTTrainer(
        model_name="facebook/nllb-200-distilled-600M",
        src_lang=args.src_lang_nllb,
        tgt_lang=args.tgt_lang_nllb,
        noise_pwd=args.noise_pwd,
        noise_k=args.noise_k,
        lambda_auto=args.lambda_auto,
        lambda_cd=args.lambda_cd,
        lambda_adv=args.lambda_adv,
        disc_hidden_dim=1024,
        disc_smoothing=0.1,
    )
    
    # Full training with validation and test
    print("\n" + "="*70)
    print("TRAINING (Auto-Encoding + Cross-Domain + Adversarial)")
    print("="*70)
    
    test_metrics = trainer.train(
        src_sentences=src_sentences,
        tgt_sentences=tgt_sentences,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        enc_dec_lr=args.enc_dec_lr,
        disc_lr=args.disc_lr,
        log_interval=args.log_interval,
        val_src_sentences=val_src_sentences,
        val_tgt_sentences=val_tgt_sentences,
        test_src_sentences=test_src_sentences,
        test_tgt_sentences=test_tgt_sentences,
        val_batch_size=args.val_batch_size,
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
