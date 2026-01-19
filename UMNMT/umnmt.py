"""
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Optional, Dict, Tuple
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


class AutoEncodingTrainer:
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
        noisy_inputs: Dict[str, torch.Tensor],
        target_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction loss: Δ(x̂, x) = sum of token-level cross-entropy.
        
        Args:
            noisy_inputs: Tokenized noisy sentences (encoder input)
            target_ids: Original sentence token IDs (reconstruction target)
            
        Returns:
            loss: Reconstruction loss
            logits: Model output logits
        """
        # Shift for decoder input (teacher forcing)
        decoder_input_ids = target_ids[:, :-1]
        labels = target_ids[:, 1:]
        
        # Forward pass
        outputs = self.model(
            input_ids=noisy_inputs["input_ids"],
            attention_mask=noisy_inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids,
        )
        
        logits = outputs.logits
        
        # Compute cross-entropy loss
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
            4. Decode: x̂ ~ d(e(C(x), ℓ), ℓ)
            5. Compute loss: Δ(x̂, x)
        
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
        
        # Generate sequence ids 
        generated_sequences = generate_sequences(
            self.model,
            self.tokenizer,
            noisy_inputs,
            tgt_lang_id=lang_id,
            max_new_tokens=128,
            gen_temperature=0.7,
            num_return_sequences=1,
            do_sample=False, # greedy decoding as in the paper 
        )
        # detokenize and tokenize again the generated sequences
        generated_sequences = self.tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
        generated_inputs = self._tokenize(generated_sequences)
        # Tokenize original sentences (reconstruction target)
        target_tokens = self._tokenize(sentences)
        target_ids = target_tokens["input_ids"]
    
        # Step 3-5: Encode, decode, compute loss
        loss, logits = self.compute_reconstruction_loss(generated_inputs, target_ids)
        
        # Compute accuracy (for monitoring)
        preds = logits.argmax(dim=-1)
        labels = target_ids[:, 1:]
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
        
        Pipeline: x → translate to tgt → C(tgt) → translate back to src → compare with x
        
        Steps:
            1. Sample sentence x from source monolingual data D_src
            2. Translate to target language: y = M(x, tgt_lang)
            3. Apply noise: C(y)
            4. Encode C(y) and get encoder hidden states (for adversarial loss)
            5. Translate back to source: x̂ = M(C(y), src_lang)
            6. Compute reconstruction loss: Δ(x̂, x)
        
        Args:
            sentences: Batch of original sentences (in src_lang)
            src_lang_id: Language token ID for source language (to translate back to)
            tgt_lang_id: Language token ID for target language (intermediate translation)
            
        Returns:
            loss: Cross-domain reconstruction loss
            metrics: Dictionary with additional metrics
            encoder_hidden_states: Encoder outputs from encoding C(y) - for adversarial loss
            attention_mask: Attention mask for encoder outputs
        """
        # Step 1: Tokenize original sentences
        original_inputs = self._tokenize(sentences)
        
        # Step 2: Translate to target language (src → tgt)
        with torch.no_grad():
            translated_sequences = generate_sequences(
                self.model,
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
        
        # Step 4: Get encoder hidden states for adversarial loss
        # These are the representations of the noisy translated text (in tgt_lang domain)
        encoder_outputs = self.model.get_encoder()(
            input_ids=noisy_translated_inputs["input_ids"],
            attention_mask=noisy_translated_inputs["attention_mask"],
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        attention_mask = noisy_translated_inputs["attention_mask"]
        
        # Step 5: Translate back to source language (C(y) → src)
        back_translated_sequences = generate_sequences(
            self.model,
            self.tokenizer,
            noisy_translated_inputs,
            tgt_lang_id=src_lang_id,  # translate back to source
            max_new_tokens=128,
            gen_temperature=0.7,
            num_return_sequences=1,
            do_sample=False,
        )
        back_translated_texts = self.tokenizer.batch_decode(
            back_translated_sequences, skip_special_tokens=True
        )
        back_translated_inputs = self._tokenize(back_translated_texts)
        
        # Step 6: Compute reconstruction loss against original sentences
        target_tokens = self._tokenize(sentences)
        target_ids = target_tokens["input_ids"]
        
        loss, logits = self.compute_reconstruction_loss(back_translated_inputs, target_ids)
        
        # Compute accuracy (for monitoring)
        preds = logits.argmax(dim=-1)
        labels = target_ids[:, 1:]
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
    def evaluate(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate losses without gradient computation.
        
        Args:
            src_sentences: Batch of source language sentences
            tgt_sentences: Batch of target language sentences
            
        Returns:
            Dictionary of loss values and metrics
        """
        self.model.eval()
        self.discriminator.eval()
        
        # Auto-encoding losses
        loss_auto_src, metrics_auto_src = self.compute_auto_encoding_loss(
            src_sentences, self.src_lang_id
        )
        loss_auto_tgt, metrics_auto_tgt = self.compute_auto_encoding_loss(
            tgt_sentences, self.tgt_lang_id
        )
        
        # Cross-domain losses (also returns encoder hidden states for adversarial loss)
        loss_cd_src, metrics_cd_src, cd_tgt_hidden, cd_tgt_mask = self.compute_cross_domain_loss(
            src_sentences, self.src_lang_id, self.tgt_lang_id
        )
        loss_cd_tgt, metrics_cd_tgt, cd_src_hidden, cd_src_mask = self.compute_cross_domain_loss(
            tgt_sentences, self.tgt_lang_id, self.src_lang_id
        )
        
        # Adversarial loss on original sentences
        loss_adv, hidden_states, attention_masks = self.compute_adversarial_loss(
            src_sentences, tgt_sentences
        )
        
        # Adversarial loss on cross-domain encoder outputs
        loss_adv_cd = self.compute_adversarial_loss_from_hidden(
            cd_src_hidden, cd_src_mask, cd_tgt_hidden, cd_tgt_mask
        )
        
        # Discriminator accuracy
        src_hidden, tgt_hidden = hidden_states
        src_mask, tgt_mask = attention_masks
        src_preds = self.discriminator(src_hidden, src_mask)
        tgt_preds = self.discriminator(tgt_hidden, tgt_mask)
        src_acc = (src_preds < 0.5).float().mean()  # src should predict 0
        tgt_acc = (tgt_preds > 0.5).float().mean()  # tgt should predict 1
        disc_acc = (src_acc + tgt_acc) / 2
        
        # Total loss (same as training)
        total_loss = (
            self.lambda_auto * (loss_auto_src + loss_auto_tgt) +
            self.lambda_cd * (loss_cd_src + loss_cd_tgt) +
            self.lambda_adv * (loss_adv + loss_adv_cd)
        )
        
        return {
            "loss_total": total_loss.item(),
            "loss_auto_src": loss_auto_src.item(),
            "loss_auto_tgt": loss_auto_tgt.item(),
            "loss_cd_src": loss_cd_src.item(),
            "loss_cd_tgt": loss_cd_tgt.item(),
            "loss_adv": loss_adv.item(),
            "loss_adv_cd": loss_adv_cd.item(),
            "acc_auto_src": metrics_auto_src["accuracy"],
            "acc_auto_tgt": metrics_auto_tgt["accuracy"],
            "acc_cd_src": metrics_cd_src["accuracy"],
            "acc_cd_tgt": metrics_cd_tgt["accuracy"],
            "disc_acc": disc_acc.item(),
        }
    
    def train(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        num_epochs: int = 10,
        batch_size: int = 8,
        enc_dec_lr: float = 1e-4,
        disc_lr: float = 1e-4,
        log_interval: int = 10,
    ):
        """
        Training loop for auto-encoding + adversarial training.
        
        Args:
            src_sentences: List of source language sentences
            tgt_sentences: List of target language sentences
            num_epochs: Number of training epochs
            batch_size: Batch size
            enc_dec_lr: Learning rate for encoder/decoder
            disc_lr: Learning rate for discriminator
            log_interval: Log every N steps
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
        enc_dec_optimizer = torch.optim.AdamW(self.model.parameters(), lr=enc_dec_lr)
        disc_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=disc_lr)
        
        print(f"\nStarting training:")
        print(f"  - Source sentences: {len(src_sentences)}")
        print(f"  - Target sentences: {len(tgt_sentences)}")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Enc/Dec LR: {enc_dec_lr}")
        print(f"  - Disc LR: {disc_lr}")
        print(f"  - Lambda auto: {self.lambda_auto}")
        print(f"  - Lambda cd: {self.lambda_cd}")
        print(f"  - Lambda adv: {self.lambda_adv}")
        print(f"  - Noise dropout: {self.noise_pwd}")
        print(f"  - Noise shuffle k: {self.noise_k}")
        print()
        
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
            print(f"Epoch {epoch + 1}/{num_epochs} completed")
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
            print(f"{'='*70}\n")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    # Initialize trainer with auto-encoding, cross-domain, and adversarial components
    trainer = AutoEncodingTrainer(
        model_name="facebook/nllb-200-distilled-600m",
        src_lang="eng_Latn",
        tgt_lang="fra_Latn",
        noise_pwd=0.1,      # 10% word dropout
        noise_k=3,          # Max shuffle distance
        lambda_auto=1.0,    # Auto-encoding loss weight
        lambda_cd=1.0,      # Cross-domain loss weight
        lambda_adv=1.0,     # Adversarial loss weight
        disc_hidden_dim=1024,
        disc_smoothing=0.1,
    )
    
    # Example monolingual sentences (replace with your data)
    src_sentences = [
        "Hello, how are you?",
        "The weather is nice today.",
        "I like to read books.",
        "Machine learning is interesting.",
        "The cat sat on the mat.",
        "This is a test sentence.",
        "Python is a great programming language.",
        "Natural language processing is fun.",
    ]
    
    tgt_sentences = [
        "Bonjour, comment allez-vous?",
        "Le temps est beau aujourd'hui.",
        "J'aime lire des livres.",
        "L'apprentissage automatique est intéressant.",
        "Le chat était assis sur le tapis.",
        "Ceci est une phrase de test.",
        "Python est un excellent langage de programmation.",
        "Le traitement du langage naturel est amusant.",
    ]
    
    # Single training step example
    print("\n" + "="*70)
    print("SINGLE STEP EXAMPLE")
    print("="*70)
    
    enc_dec_optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-4)
    disc_optimizer = torch.optim.AdamW(trainer.discriminator.parameters(), lr=1e-4)
    
    losses = trainer.train_step(
        src_sentences[:4], tgt_sentences[:4],
        enc_dec_optimizer, disc_optimizer
    )
    
    print(f"\nSingle step results:")
    print(f"  Total Enc/Dec Loss: {losses['loss_total']:.4f}")
    print(f"  L_auto(src): {losses['loss_auto_src']:.4f}")
    print(f"  L_auto(tgt): {losses['loss_auto_tgt']:.4f}")
    print(f"  L_cd(src→tgt→src): {losses['loss_cd_src']:.4f}")
    print(f"  L_cd(tgt→src→tgt): {losses['loss_cd_tgt']:.4f}")
    print(f"  L_adv (original): {losses['loss_adv']:.4f}")
    print(f"  L_adv (cross-domain): {losses['loss_adv_cd']:.4f}")
    print(f"  L_disc: {losses['loss_disc']:.4f}")
    print(f"  Auto-enc Acc (src): {losses['acc_auto_src']:.2%}")
    print(f"  Auto-enc Acc (tgt): {losses['acc_auto_tgt']:.2%}")
    print(f"  Cross-domain Acc (src→tgt→src): {losses['acc_cd_src']:.2%}")
    print(f"  Cross-domain Acc (tgt→src→tgt): {losses['acc_cd_tgt']:.2%}")
    print(f"  Discriminator Acc: {losses['disc_acc']:.2%}")
    
    # Full training example
    print("\n" + "="*70)
    print("FULL TRAINING EXAMPLE (Auto-Encoding + Cross-Domain + Adversarial)")
    print("="*70)
    
    trainer.train(
        src_sentences=src_sentences,
        tgt_sentences=tgt_sentences,
        num_epochs=3,
        batch_size=4,
        enc_dec_lr=1e-4,
        disc_lr=1e-4,
        log_interval=1,
    )
    
    print("\nAuto-encoding + Cross-Domain + Adversarial training completed!")
