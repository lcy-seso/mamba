import pdb

import context

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.models.config_mamba import MambaConfig

from transformers import AutoTokenizer
import transformers
import torch.nn.functional as F
import torch
from torch import nn


def generate(model,
             tokenizer,
             prompt: str,
             n_tokens_to_gen: int = 5,
             sample: bool = True,
             top_k: int = 40,
             device: str = "cuda"):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    for token_n in range(n_tokens_to_gen):
        with torch.no_grad():
            indices_to_input = input_ids
            model(indices_to_input)


if __name__ == "__main__":
    pretrained = "state-spaces/mamba-370m"
    max_length = 2048
    batch_size = None
    device = "cuda"
    dtype = torch.float16

    # config_data = load_config_hf(pretrained)
    # config = MambaConfig(**config_data)
    # model = MambaLMHeadModel(config, device=device, dtype=dtype)

    model = MambaLMHeadModel.from_pretrained(
        pretrained, device=device, dtype=dtype).to(device)

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    generate(model, tokenizer, 'The quick brown fox jumps over the lazy dog')
