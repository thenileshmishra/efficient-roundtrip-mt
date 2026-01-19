import random
import numpy as np
from typing import List, Union
import torch
import torch.nn as nn

def noise_model(sentence: Union[str, List[str]], pwd: float = 0.1, k: int = 3) -> Union[str, List[str]]:
    """
    Apply noise to a sentence for denoising autoencoder training.
    
    This implements the noise model C(x) which applies two types of noise:
    1. Word dropout: Drop each word with probability pwd
    2. Word shuffling: Permute words such that no word moves more than k positions
    
    Args:
        sentence: Input sentence as a string or list of words
        pwd: Word dropout probability (default: 0.1)
        k: Maximum distance a word can move during shuffling (default: 3)
    
    Returns:
        Noised sentence in the same format as input (string or list)
    """
    # Convert to list of words if string
    is_string = isinstance(sentence, str)
    if is_string:
        words = sentence.split()
    else:
        words = list(sentence)
    
    n = len(words)
    if n == 0:
        return sentence
    
    # Step 1: Word dropout - drop each word with probability pwd
    words = [word for word in words if random.random() >= pwd]
    
    # If all words were dropped, keep at least one random word from original
    if len(words) == 0:
        if is_string:
            words = sentence.split()
        else:
            words = list(sentence)
        words = [random.choice(words)]
    
    n = len(words)
    
    # Step 2: Word shuffling with constraint |σ(i) - i| ≤ k
    # Generate q_i = i + U(0, α) where α = k + 1
    alpha = k + 1
    q = np.arange(n) + np.random.uniform(0, alpha, n)
    
    # Get permutation σ by sorting q (argsort gives indices that would sort q)
    # We want the inverse: for each position, where should it go
    # argsort gives us: new_position -> old_position
    # So words[argsort] gives us the shuffled sentence
    permutation = np.argsort(q)
    words = [words[i] for i in permutation]
    
    # Return in same format as input
    if is_string:
        return ' '.join(words)
    return words


def noise_model_batch(sentences: List[str], pwd: float = 0.1, k: int = 3) -> List[str]:
    """
    Apply noise model to a batch of sentences.
    
    Args:
        sentences: List of input sentences
        pwd: Word dropout probability (default: 0.1)
        k: Maximum distance a word can move during shuffling (default: 3)
    
    Returns:
        List of noised sentences
    """
    return [noise_model(s, pwd, k) for s in sentences]


@torch.no_grad()
def generate_sequences(
    model,
    tokenizer,
    encoder_inputs,
    tgt_lang_id,
    max_new_tokens: int,
    gen_temperature: float,
    num_return_sequences: int,
    top_k: int = 100,
    top_p: float = 0.9,
    end_of_sentence_token_id: int = None,
    do_sample: bool = True,
):
    eos_id = (
        end_of_sentence_token_id
        if end_of_sentence_token_id is not None
        else tokenizer.eos_token_id
    )
    if not do_sample:
        generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=gen_temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_id,
            top_k=top_k,
            top_p=top_p,
        )
    else:
        generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_id,
        )
    gen = model.generate(
        input_ids=encoder_inputs["input_ids"],
        attention_mask=encoder_inputs.get("attention_mask", None),
        forced_bos_token_id=tgt_lang_id,
        **generation_kwargs,
    )
    return gen


class Discriminator(nn.Module):
    """
    Discriminator for adversarial training in unsupervised NMT.
    
    Architecture: MLP with 3 hidden layers of size 1024, Leaky-ReLU activations,
    and an output logistic unit (sigmoid).
    
    Following Goodfellow (2016), includes label smoothing coefficient s=0.1
    in the discriminator predictions.
    
    Args:
        input_dim: Dimension of input features (encoder hidden size)
        hidden_dim: Dimension of hidden layers (default: 1024)
        smoothing: Label smoothing coefficient (default: 0.1)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 1024, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        
        self.network = nn.Sequential(
            # Hidden layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            
            # Hidden layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            
            # Hidden layer 3
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            
            # Output logistic unit
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               Typically the encoder's sentence embedding (mean pooled hidden states)
        
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        return self.network(x)
    
    def compute_loss(
        self,
        encoder_outputs: torch.Tensor,
        language_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute discriminator loss:
        
        L_D(θ_D | θ, Z) = -E_{(x_i, ℓ_i)}[log p_D(ℓ_i | e(x_i, ℓ_i))]
        
        This is binary cross-entropy with label smoothing (s=0.1).
        
        Args:
            encoder_outputs: Encoder hidden states e(x_i, ℓ_i), shape (batch_size, hidden_dim)
                            Typically mean-pooled encoder last hidden state
            language_labels: Binary language labels ℓ_i, shape (batch_size,) or (batch_size, 1)
                            0 for language L1, 1 for language L2
        
        Returns:
            Discriminator loss (scalar) - discriminator MINIMIZES this to classify languages
        """
        # Get discriminator predictions: p_D(ℓ=1 | e(x))
        preds = self.forward(encoder_outputs)  # (batch, 1)
        
        # Ensure labels have correct shape
        if language_labels.dim() == 1:
            language_labels = language_labels.unsqueeze(1)
        language_labels = language_labels.float()
        
        # Apply label smoothing: 1 -> (1 - s), 0 -> s
        smoothed_labels = language_labels * (1.0 - 2 * self.smoothing) + self.smoothing
        
        # L_D = -E[log p_D(ℓ_i | e)] = BCE(preds, labels)
        loss = nn.functional.binary_cross_entropy(preds, smoothed_labels)
        
        return loss