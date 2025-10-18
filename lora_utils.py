import math
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Lightweight replacement for nn.Linear that adds a low-rank adapter.
    The original weights remain frozen while the LoRA matrices learn the update.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0.")

        self.base = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Freeze original weights
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        weight = self.base.weight
        factory_kwargs = {"device": weight.device, "dtype": weight.dtype}
        self.lora_A = nn.Parameter(torch.zeros((r, self.in_features), **factory_kwargs))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r), **factory_kwargs))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x)
        lora_out = F.linear(lora_out, self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return base_out + lora_out * self.scaling

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor:
        return self.base.bias


def _is_target_module(module_name: str, target_suffixes: Sequence[str]) -> bool:
    return any(module_name.endswith(suffix) for suffix in target_suffixes)


def _resolve_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    """
    Returns the parent module and attribute name for the provided dotted path.
    """
    components = module_name.split(".")
    parent = model
    for comp in components[:-1]:
        if comp.isdigit():
            parent = parent[int(comp)]
        else:
            parent = getattr(parent, comp)
    return parent, components[-1]


def apply_lora(
    model: nn.Module,
    target_modules: Sequence[str],
    r: int,
    lora_alpha: float,
    lora_dropout: float,
) -> Iterable[str]:
    """
    Replaces each matching Linear module with LoRALinear.

    Returns an iterable with the names of modules that were adapted.
    """
    replaced = []

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _is_target_module(module_name, target_modules):
            continue

        parent, attribute = _resolve_module(model, module_name)
        lora_linear = LoRALinear(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        if attribute.isdigit():
            parent[int(attribute)] = lora_linear
        else:
            setattr(parent, attribute, lora_linear)

        replaced.append(module_name)

    if not replaced:
        raise ValueError(
            f"No Linear modules matched the provided target modules: {target_modules}"
        )

    return replaced
