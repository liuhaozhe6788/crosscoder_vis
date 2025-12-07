import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Union
from huggingface_hub import hf_hub_download, login
import json
import einops
import os
from typing import NamedTuple
from sae_vis.data_config_classes import SaeVisConfig
from nnsight import LanguageModel
import pandas as pd


HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)
torch.set_grad_enabled(False)

base_model = LanguageModel('mistralai/Mistral-7B-Instruct-v0.3', device_map='cuda:0', dtype=torch.bfloat16)
chat_model = LanguageModel('liuhaozhe6788/mistralai_Mistral-7B-Instruct-v0.3-FinQA-lora', device_map='cuda:0', dtype=torch.bfloat16)

def prepare_text_data():
    data = pd.read_csv("data/finqa_test_generated_filtered.csv")
    full_text_data = data.apply(lambda x: x["prompt"] + x["generated_code"], axis=1)

    all_text_data = full_text_data.tolist()
    return all_text_data

all_texts = prepare_text_data()


from sae_vis.model_fns import CrossCoderConfig, CrossCoder_vis

base_estimated_scaling_factor = 27.489933013916016
chat_estimated_scaling_factor = 27.12582778930664

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

class LossOutput(NamedTuple):
    # loss: torch.Tensor
    l2_loss: torch.Tensor
    l1_loss: torch.Tensor
    l0_loss: torch.Tensor
    explained_variance: torch.Tensor
    explained_variance_A: torch.Tensor
    explained_variance_B: torch.Tensor

class CrossCoder_demo(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_hidden = self.cfg["dict_size"]
        d_in = self.cfg["d_in"]
        self.dtype = DTYPES[self.cfg["enc_dtype"]]
        torch.manual_seed(self.cfg["seed"])
        # hardcoding n_models to 2
        self.W_enc = nn.Parameter(
            torch.empty(2, d_in, d_hidden, dtype=self.dtype)
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(
                    d_hidden, 2, d_in, dtype=self.dtype
                )
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(
                    d_hidden, 2, d_in, dtype=self.dtype
                )
            )
        )
        # Make norm of W_dec 0.1 for each column, separate per layer
        self.W_dec.data = (
            self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True) * self.cfg["dec_init_norm"]
        )
        # Initialise W_enc to be the transpose of W_dec
        self.W_enc.data = einops.rearrange(
            self.W_dec.data.clone(),
            "d_hidden n_models d_model -> n_models d_model d_hidden",
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=self.dtype))
        self.b_dec = nn.Parameter(
            torch.zeros((2, d_in), dtype=self.dtype)
        )
        self.d_hidden = d_hidden

        self.to(self.cfg["device"])
        self.save_dir = None
        self.save_version = 0

    def encode(self, x, apply_relu=True):
        # x: [batch, n_models, d_model]
        x_enc = einops.einsum(
            x,
            self.W_enc,
            "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden",
        )
        if apply_relu:
            acts = F.relu(x_enc + self.b_enc)
        else:
            acts = x_enc + self.b_enc
        return acts

    def decode(self, acts):
        # acts: [batch, d_hidden]
        acts_dec = einops.einsum(
            acts,
            self.W_dec,
            "batch d_hidden, d_hidden n_models d_model -> batch n_models d_model",
        )
        return acts_dec + self.b_dec

    def forward(self, x):
        # x: [batch, n_models, d_model]
        acts = self.encode(x)
        return self.decode(acts)

    def get_losses(self, x):
        # x: [batch, n_models, d_model]
        x = x.to(self.dtype)
        acts = self.encode(x)
        # acts: [batch, d_hidden]
        x_reconstruct = self.decode(acts)
        diff = x_reconstruct.float() - x.float()
        squared_diff = diff.pow(2)
        l2_per_batch = einops.reduce(squared_diff, 'batch n_models d_model -> batch', 'sum')
        l2_loss = l2_per_batch.mean()

        total_variance = einops.reduce((x - x.mean(0)).pow(2), 'batch n_models d_model -> batch', 'sum')
        explained_variance = 1 - l2_per_batch / total_variance

        per_token_l2_loss_A = (x_reconstruct[:, 0, :] - x[:, 0, :]).pow(2).sum(dim=-1).squeeze()
        total_variance_A = (x[:, 0, :] - x[:, 0, :].mean(0)).pow(2).sum(-1).squeeze()
        explained_variance_A = 1 - per_token_l2_loss_A / total_variance_A

        per_token_l2_loss_B = (x_reconstruct[:, 1, :] - x[:, 1, :]).pow(2).sum(dim=-1).squeeze()
        total_variance_B = (x[:, 1, :] - x[:, 1, :].mean(0)).pow(2).sum(-1).squeeze()
        explained_variance_B = 1 - per_token_l2_loss_B / total_variance_B

        decoder_norms = self.W_dec.norm(dim=-1)
        # decoder_norms: [d_hidden, n_models]
        total_decoder_norm = einops.reduce(decoder_norms, 'd_hidden n_models -> d_hidden', 'sum')
        l1_loss = (acts * total_decoder_norm[None, :]).sum(-1).mean(0)

        l0_loss = (acts>0).float().sum(-1).mean()

        return LossOutput(l2_loss=l2_loss, l1_loss=l1_loss, l0_loss=l0_loss, explained_variance=explained_variance, explained_variance_A=explained_variance_A, explained_variance_B=explained_variance_B)

    @classmethod
    def load_from_hf(
        cls,
        repo_id: str = "liuhaozhe6788/crosscoder-model-diff-mistral-7b-instruct-v0.3_finQA_lora_topk_100",
        device: Optional[Union[str, torch.device]] = None
    ) -> "CrossCoder_demo":
        """
        Load CrossCoder_demo weights and config from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID
            path: Path within the repo to the weights/config
            model: The transformer model instance needed for initialization
            device: Device to load the model to (defaults to cfg device if not specified)

        Returns:
            Initialized CrossCoder_demo instance
        """

        # Download config and weights
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"cfg.json"
        )
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"model.pt"
        )

        # Load config
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        # Override device if specified
        if device is not None:
            cfg["device"] = str(device)

        # Initialize CrossCoder_demo with config
        instance = cls(cfg)

        # Load weights
        state_dict = torch.load(weights_path, map_location=cfg["device"])
        instance.load_state_dict(state_dict["model_state_dict"])

        return instance

cross_coder = CrossCoder_demo.load_from_hf()

import copy
folded_cross_coder = copy.deepcopy(cross_coder)

def fold_activation_scaling_factor(cross_coder, base_scaling_factor, chat_scaling_factor):
    cross_coder.W_enc.data[0, :, :] = cross_coder.W_enc.data[0, :, :] * base_scaling_factor
    cross_coder.W_enc.data[1, :, :] = cross_coder.W_enc.data[1, :, :] * chat_scaling_factor

    # cross_coder.W_dec.data[:, 0, :] = cross_coder.W_dec.data[:, 0, :] / base_scaling_factor
    # cross_coder.W_dec.data[:, 1, :] = cross_coder.W_dec.data[:, 1, :] / chat_scaling_factor

    # cross_coder.b_dec.data[0, :] = cross_coder.b_dec.data[0, :] / base_scaling_factor
    # cross_coder.b_dec.data[1, :] = cross_coder.b_dec.data[1, :] / chat_scaling_factor
    return cross_coder

folded_cross_coder = fold_activation_scaling_factor(folded_cross_coder, base_estimated_scaling_factor, chat_estimated_scaling_factor)

encoder_cfg = CrossCoderConfig(d_in=base_model.config.hidden_size, d_hidden=cross_coder.cfg["dict_size"], apply_b_dec_to_input=False)
sae_vis_cross_coder = CrossCoder_vis(encoder_cfg)
sae_vis_cross_coder.load_state_dict(folded_cross_coder.state_dict())
sae_vis_cross_coder = sae_vis_cross_coder.to("cuda:0")
sae_vis_cross_coder = sae_vis_cross_coder.to(torch.bfloat16)
test_feature_idx = [2325,12698,15]
sae_vis_config = SaeVisConfig(
    hook_layer = 16,
    features = test_feature_idx,
    verbose = True,
    minibatch_size_texts=1,
    minibatch_size_features=16,
)

from sae_vis.data_storing_fns import SaeVisData
sae_vis_data = SaeVisData.create(
    encoder = sae_vis_cross_coder,
    encoder_B = None,
    model_A = base_model,
    model_B = chat_model,
    texts = all_texts[:128], # in practice, better to use more data
    cfg = sae_vis_config,
)

filename = "_feature_vis_demo.html"
sae_vis_data.save_feature_centric_vis(filename)