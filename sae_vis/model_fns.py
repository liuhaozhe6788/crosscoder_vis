import re
from dataclasses import dataclass
from typing import Literal, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from nnsight import LanguageModel
from torch import Tensor
import einops

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

@dataclass
class CrossCoderConfig:
    """Class for storing configuration parameters for the CrossCoder"""

    d_in: int
    d_hidden: int | None = None
    dict_mult: int | None = None

    l1_coeff: float = 3e-4

    apply_b_dec_to_input: bool = False

    def __post_init__(self):
        assert (
            int(self.d_hidden is None) + int(self.dict_mult is None) == 1
        ), "Exactly one of d_hidden or dict_mult must be provided"
        if (self.d_hidden is None) and isinstance(self.dict_mult, int):
            self.d_hidden = self.d_in * self.dict_mult
        elif (self.dict_mult is None) and isinstance(self.d_hidden, int):
            #assert self.d_hidden % self.d_in == 0, "d_hidden must be a multiple of d_in"
            self.dict_mult = self.d_hidden // self.d_in


class CrossCoder(nn.Module):
    def __init__(self, cfg: CrossCoderConfig):
        super().__init__()
        self.cfg = cfg

        assert isinstance(cfg.d_hidden, int)

        # W_enc has shape (d_in, d_encoder), where d_encoder is a multiple of d_in (cause dictionary learning; overcomplete basis)
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(2, cfg.d_in, cfg.d_hidden))
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_hidden, 2, cfg.d_in))
        )
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(2, cfg.d_in))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor):
        # TODO: lots of this stuff is legacy SAE stuff that probably is wrong / unnecessary
        x_cent = x - self.b_dec * self.cfg.apply_b_dec_to_input
        x_enc = einops.einsum(
            x_cent,
            self.W_enc,
            "... n_layers d_model, n_layers d_model d_hidden -> ... d_hidden",
        )
        acts = F.relu(x_enc + self.b_enc)
        #x_reconstruct = acts @ self.W_dec + self.b_dec
        x_reconstruct = einops.einsum(
            acts,
            self.W_dec,
            "... d_hidden, d_hidden n_layers d_model -> ... n_layers d_model",
        )
        diff = x_reconstruct.float() - x.float()
        squared_diff = diff.pow(2)
        l2_per_batch = einops.reduce(squared_diff, 'batch n_layers d_model -> batch', 'sum')
        l2_loss = l2_per_batch.mean()
        #l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        decoder_norms = self.W_dec.norm(dim=-1)
        # decoder_norms: [d_hidden, n_layers]
        total_decoder_norm = einops.reduce(decoder_norms, 'd_hidden n_layers -> d_hidden', 'sum')
        l1_loss = (acts * total_decoder_norm[None, :]).sum(-1).mean(0)
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj

    def __repr__(self) -> str:
        return f"CrossCoder(d_in={self.cfg.d_in}, dict_mult={self.cfg.dict_mult})"


# # ==============================================================
# # ! TRANSFORMERS
# # This returns the activations & resid_pre as well (optionally)
# # ==============================================================


class LanguageModelWrapper(nn.Module):
    """
    This class wraps around & extends the LanguageModel model, so that we can make sure things like the forward
    function have a standardized signature.
    """

    def __init__(self, model: LanguageModel, hook_layer: int):
        super().__init__()
        self.model = model
        self.hook_layer = hook_layer

    @overload
    def forward(
        self,
        tokens: Tensor,
        return_logits: Literal[True],
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        tokens: Tensor,
        return_logits: Literal[False],
    ) -> tuple[Tensor, Tensor]: ...

    def forward(
        self,
        texts: list[str],
        return_logits: bool = True,
    ):
        """
        Inputs:
            texts: list[str]
                The input texts
            return_logits: bool
                If True, returns (logits, residual, activation)
                If False, returns (residual, activation)
        """
        with self.model.trace(texts) as tracer:
            ## only save the activation of the output of the mlp layer
            activation = self.model.model.layers[self.hook_layer].mlp.output.save()

        residual = activation
        return residual, activation

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def W_U(self):
        return self.model.W_U

def to_resid_dir(dir: Float[Tensor, "feats d_in"], model: LanguageModelWrapper):
    """
    Takes a direction (eg. in the post-ReLU MLP activations) and returns the corresponding dir in the residual stream.

    Args:
        dir:
            The direction in the activations, i.e. shape (feats, d_in) where d_in could be d_model, d_mlp, etc.
        model:
            The model, which should be a HookedTransformerWrapper or similar.
    """
    return dir
