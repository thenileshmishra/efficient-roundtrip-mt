"""
Unsupervised Neural Machine Translation Training System

Based on the paper's three-component training:
1. Denoising Auto-Encoding (L_auto)
2. Cross-Domain Training (L_cd)
3. Adversarial Training (L_adv)

Final objective:
L = λ_auto * [L_auto(src) + L_auto(tgt)] + 
    λ_cd * [L_cd(src,tgt) + L_cd(tgt,src)] + 
    λ_adv * L_adv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import generate_sequences, Discriminator, noise_model_batch


class UNMTTrainer:
    """
    Unsupervised Neural Machine Translation Trainer.
    
    Implements the three training objectives:
    - Denoising Auto-Encoding
    - Cross-Domain Training  
    - Adversarial Training
    """
    
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600m",
        src_lang: str = "eng_Latn",
        tgt_lang: str = "fra_Latn",
        lambda_auto: float = 1.0,
        lambda_cd: float = 1.0,
        lambda_adv: float = 1.0,
        noise_pwd: float = 0.1,
        noise_k: int = 3,
        max_new_tokens: int = 128,
        gen_temperature: float = 0.7,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.lambda_auto = lambda_auto
        self.lambda_cd = lambda_cd
        self.lambda_adv = lambda_adv
        self.noise_pwd = noise_pwd
        self.noise_k = noise_k
        self.max_new_tokens = max_new_tokens
        self.gen_temperature = gen_temperature
        
        # Load model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Language token IDs
        self.src_lang_id = self.tokenizer.convert_tokens_to_ids(src_lang)
        self.tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        
        # Initialize discriminator
        hidden_dim = self.model.config.d_model
        self.discriminator = Discriminator(input_dim=hidden_dim).to(device)
        
    def _tokenize(self, sentences, return_tensors="pt"):
        """Tokenize sentences."""
        tokens = self.tokenizer(
            sentences, 
            return_tensors=return_tensors, 
            padding=True, 
            truncation=True
        )
        return {k: v.to(self.device) for k, v in tokens.items()}
    
    def _generate(self, inputs, tgt_lang_id):
        """Generate translations using the current model."""
        return generate_sequences(
            self.model,
            self.tokenizer,
            inputs,
            tgt_lang_id=tgt_lang_id,
            max_new_tokens=self.max_new_tokens,
            gen_temperature=self.gen_temperature,
            num_return_sequences=1,
            end_of_sentence_token_id=self.tokenizer.eos_token_id,
        )
    
    def _compute_reconstruction_loss(self, inputs, target_ids):
        """
        Compute reconstruction loss (sum of token-level cross-entropy).
        
        Δ(x̂, x) = sum of token-level cross-entropy losses
        """
        # Shift for decoder input
        decoder_input_ids = target_ids[:, :-1]
        labels = target_ids[:, 1:]
        
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            output_hidden_states=True,
        )
        
        # Sum of token-level cross-entropy (as per paper)
        logits = outputs.logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
            reduction='sum',
        )
        
        return loss, outputs
    
    # =========================================================================
    # LOSS 1: DENOISING AUTO-ENCODING (Equation 1)
    # =========================================================================
    def compute_auto_encoding_loss(self, sentences, lang_id):
        """
        Denoising Auto-Encoding Loss (Equation 1):
        
        L_auto(θ_enc, θ_dec, Z, ℓ) = E_{x~D_ℓ, x̂~d(e(C(x),ℓ),ℓ)} [Δ(x̂, x)]
        
        1. Sample sentence x from monolingual data D_ℓ
        2. Apply noise: C(x) = noisy version of x
        3. Encode noisy input: e(C(x), ℓ)
        4. Decode to reconstruct: x̂ ~ d(e(C(x), ℓ), ℓ)
        5. Compute loss: Δ(x̂, x) = sum of token-level CE
        
        Args:
            sentences: List of sentences from monolingual data D_ℓ
            lang_id: Language token ID for encoding/decoding
            
        Returns:
            L_auto loss (scalar)
        """
        # Apply noise model C(x): word dropout + shuffling
        noisy_sentences = noise_model_batch(sentences, pwd=self.noise_pwd, k=self.noise_k)
        
        # Tokenize noisy input C(x)
        noisy_inputs = self._tokenize(noisy_sentences)
        
        # Tokenize original sentences x (targets for reconstruction)
        with self.tokenizer.as_target_tokenizer():
            target_tokens = self._tokenize(sentences)
            target_ids = target_tokens["input_ids"]
        
        # Compute reconstruction loss: encode C(x), decode to x̂, compare with x
        loss, outputs = self._compute_reconstruction_loss(noisy_inputs, target_ids)
        
        return loss, outputs
    
    # =========================================================================
    # LOSS 2: CROSS-DOMAIN TRAINING (Equation 2)
    # =========================================================================
    def compute_cross_domain_loss(self, sentences, src_lang_id, tgt_lang_id):
        """
        Cross-Domain Training Loss (Equation 2):
        
        L_cd(θ_enc, θ_dec, Z, ℓ1, ℓ2) = E_{x~D_ℓ1, x̂~d(e(C(M(x)),ℓ2),ℓ1)} [Δ(x̂, x)]
        
        1. Sample sentence x from source domain D_ℓ1
        2. Translate to target domain: y = M(x)
        3. Apply noise: C(y)
        4. Encode noisy translation: e(C(y), ℓ2)
        5. Decode back to source: x̂ ~ d(e(C(y), ℓ2), ℓ1)
        6. Compute loss: Δ(x̂, x) = sum of token-level CE
        
        Args:
            sentences: List of sentences from source domain D_ℓ1
            src_lang_id: Source language token ID (ℓ1)
            tgt_lang_id: Target language token ID (ℓ2)
            
        Returns:
            L_cd loss (scalar)
        """
        # Step 1: Tokenize source sentences x
        src_inputs = self._tokenize(sentences)
        
        # Step 2: Translate x → y using current model M
        # y = M(x): src → tgt translation
        with torch.no_grad():
            translated_ids = self._generate(src_inputs, tgt_lang_id)
        
        # Decode translated tokens to text
        translated_sentences = self.tokenizer.batch_decode(
            translated_ids, skip_special_tokens=True
        )
        
        # Step 3: Apply noise C(y)
        noisy_translations = noise_model_batch(
            translated_sentences, pwd=self.noise_pwd, k=self.noise_k
        )
        
        # Step 4: Tokenize noisy translations C(y)
        noisy_inputs = self._tokenize(noisy_translations)
        
        # Step 5-6: Encode C(y), decode to x̂, compute loss Δ(x̂, x)
        # Target is original source sentence x
        with self.tokenizer.as_target_tokenizer():
            target_tokens = self._tokenize(sentences)
            target_ids = target_tokens["input_ids"]
        
        loss, outputs = self._compute_reconstruction_loss(noisy_inputs, target_ids)
        
        return loss, outputs
    
    # =========================================================================
    # LOSS 3: ADVERSARIAL TRAINING (Equation 3)
    # =========================================================================
    def compute_adversarial_loss(self, src_sentences, tgt_sentences):
        """
        Adversarial Loss (Equation 3):
        
        L_adv(θ_enc, Z | θ_D) = -E_{(x_i, ℓ_i)} [log p_D(ℓ_j | e(x_i, ℓ_i))]
        
        where ℓ_j = ℓ_1 if ℓ_i = ℓ_2, and vice versa (FLIPPED labels!)
        
        The encoder is trained to FOOL the discriminator by making it predict
        the WRONG language.
        
        Args:
            src_sentences: Sentences from source domain (ℓ = 0)
            tgt_sentences: Sentences from target domain (ℓ = 1)
            
        Returns:
            L_adv loss for encoder (scalar)
        """
        # Encode source sentences
        src_inputs = self._tokenize(src_sentences)
        src_outputs = self.model.get_encoder()(
            input_ids=src_inputs["input_ids"],
            attention_mask=src_inputs["attention_mask"],
        )
        src_embeddings = src_outputs.last_hidden_state.mean(dim=1)  # (batch, hidden)
        
        # Encode target sentences
        tgt_inputs = self._tokenize(tgt_sentences)
        tgt_outputs = self.model.get_encoder()(
            input_ids=tgt_inputs["input_ids"],
            attention_mask=tgt_inputs["attention_mask"],
        )
        tgt_embeddings = tgt_outputs.last_hidden_state.mean(dim=1)  # (batch, hidden)
        
        # Combine embeddings
        all_embeddings = torch.cat([src_embeddings, tgt_embeddings], dim=0)
        
        # TRUE labels: src = 0, tgt = 1
        true_labels = torch.cat([
            torch.zeros(src_embeddings.size(0)),
            torch.ones(tgt_embeddings.size(0)),
        ]).to(self.device)
        
        # FLIPPED labels for adversarial loss: ℓ_j = 1 - ℓ_i
        # Encoder wants discriminator to predict WRONG language
        flipped_labels = 1.0 - true_labels
        
        # Get discriminator predictions
        disc_preds = self.discriminator(all_embeddings)
        
        # L_adv = -E[log p_D(ℓ_j | e)] = BCE with flipped labels
        adv_loss = F.binary_cross_entropy(
            disc_preds, flipped_labels.unsqueeze(1)
        )
        
        return adv_loss, all_embeddings, true_labels
    
    def compute_discriminator_loss(self, encoder_embeddings, true_labels):
        """
        Discriminator Loss:
        
        L_D(θ_D | θ, Z) = -E_{(x_i, ℓ_i)} [log p_D(ℓ_i | e(x_i, ℓ_i))]
        
        The discriminator is trained to correctly classify languages.
        
        Args:
            encoder_embeddings: Encoder outputs (detached from encoder graph)
            true_labels: True language labels (0 for src, 1 for tgt)
            
        Returns:
            L_D loss for discriminator (scalar)
        """
        # Detach embeddings so discriminator training doesn't affect encoder
        embeddings_detached = encoder_embeddings.detach()
        
        return self.discriminator.compute_loss(embeddings_detached, true_labels)
    
    # =========================================================================
    # FINAL OBJECTIVE (Equation 4)
    # =========================================================================
    def compute_total_loss(self, src_sentences, tgt_sentences):
        """
        Final Objective Function (Equation 4):
        
        L(θ_enc, θ_dec, Z) = 
            λ_auto * [L_auto(src) + L_auto(tgt)] +
            λ_cd * [L_cd(src,tgt) + L_cd(tgt,src)] +
            λ_adv * L_adv
            
        Returns encoder/decoder loss and discriminator loss separately.
        """
        losses = {}
        
        # =====================================================================
        # DENOISING AUTO-ENCODING: L_auto(src) + L_auto(tgt)
        # =====================================================================
        l_auto_src, _ = self.compute_auto_encoding_loss(src_sentences, self.src_lang_id)
        l_auto_tgt, _ = self.compute_auto_encoding_loss(tgt_sentences, self.tgt_lang_id)
        l_auto = l_auto_src + l_auto_tgt
        losses['l_auto_src'] = l_auto_src.item()
        losses['l_auto_tgt'] = l_auto_tgt.item()
        
        # =====================================================================
        # CROSS-DOMAIN TRAINING: L_cd(src,tgt) + L_cd(tgt,src)
        # =====================================================================
        # src → tgt → src (back-translation)
        l_cd_src_tgt, _ = self.compute_cross_domain_loss(
            src_sentences, self.src_lang_id, self.tgt_lang_id
        )
        # tgt → src → tgt (back-translation)
        l_cd_tgt_src, _ = self.compute_cross_domain_loss(
            tgt_sentences, self.tgt_lang_id, self.src_lang_id
        )
        l_cd = l_cd_src_tgt + l_cd_tgt_src
        losses['l_cd_src_tgt'] = l_cd_src_tgt.item()
        losses['l_cd_tgt_src'] = l_cd_tgt_src.item()
        
        # =====================================================================
        # ADVERSARIAL TRAINING: L_adv (encoder fools discriminator)
        # =====================================================================
        l_adv, encoder_embeddings, true_labels = self.compute_adversarial_loss(
            src_sentences, tgt_sentences
        )
        losses['l_adv'] = l_adv.item()
        
        # =====================================================================
        # FINAL ENCODER/DECODER LOSS
        # =====================================================================
        encoder_decoder_loss = (
            self.lambda_auto * l_auto +
            self.lambda_cd * l_cd +
            self.lambda_adv * l_adv
        )
        losses['encoder_decoder_loss'] = encoder_decoder_loss.item()
        
        # =====================================================================
        # DISCRIMINATOR LOSS (trained separately)
        # =====================================================================
        discriminator_loss = self.compute_discriminator_loss(
            encoder_embeddings, true_labels
        )
        losses['discriminator_loss'] = discriminator_loss.item()
        
        return encoder_decoder_loss, discriminator_loss, losses
    
    def train_step(self, src_sentences, tgt_sentences, enc_dec_optimizer, disc_optimizer):
        """
        Single training step.
        
        Args:
            src_sentences: Batch of source language sentences
            tgt_sentences: Batch of target language sentences
            enc_dec_optimizer: Optimizer for encoder/decoder
            disc_optimizer: Optimizer for discriminator
            
        Returns:
            Dictionary of loss values
        """
        # Compute all losses
        enc_dec_loss, disc_loss, losses = self.compute_total_loss(
            src_sentences, tgt_sentences
        )
        
        # Update encoder/decoder
        enc_dec_optimizer.zero_grad()
        enc_dec_loss.backward()
        enc_dec_optimizer.step()
        
        # Update discriminator (separate optimization)
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()
        
        return losses


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    # Initialize trainer
    trainer = UNMTTrainer(
        model_name="facebook/nllb-200-distilled-600m",
        src_lang="eng_Latn",
        tgt_lang="fra_Latn",
        lambda_auto=1.0,
        lambda_cd=1.0,
        lambda_adv=1.0,
    )
    
    # Example sentences
    src_sentences = ["Hello, how are you?", "The weather is nice today."]
    tgt_sentences = ["Bonjour, comment allez-vous?", "Le temps est beau aujourd'hui."]
    
    # Setup optimizers
    enc_dec_optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-4)
    disc_optimizer = torch.optim.Adam(trainer.discriminator.parameters(), lr=1e-4)
    
    # Single training step
    losses = trainer.train_step(
        src_sentences, tgt_sentences,
        enc_dec_optimizer, disc_optimizer
    )
    
    print("Training step completed!")
    print(f"  L_auto (src): {losses['l_auto_src']:.4f}")
    print(f"  L_auto (tgt): {losses['l_auto_tgt']:.4f}")
    print(f"  L_cd (src→tgt→src): {losses['l_cd_src_tgt']:.4f}")
    print(f"  L_cd (tgt→src→tgt): {losses['l_cd_tgt_src']:.4f}")
    print(f"  L_adv: {losses['l_adv']:.4f}")
    print(f"  Total Enc/Dec Loss: {losses['encoder_decoder_loss']:.4f}")
    print(f"  Discriminator Loss: {losses['discriminator_loss']:.4f}")
