import time
import heapq
import math
import torch
import random

import torch.nn as nn
from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT
# from .layerwrapper1 import WrappedGPT
# from .layerwrapper2 import WrappedGPT
from .data import get_loaders
from .ablate import AblateGPT
from transformers import AutoConfig

from .opt.configuration_opt import OPTConfig
from .model_opt import OPTAttention, OPTDecoderLayer

config = AutoConfig.from_pretrained("./opt-6.7b/config.json")
optconfig = OPTConfig()


def find_layers(module, layers=[nn.Linear], name=''):
    # 递归查找模块中某种类型的层，并返回字典
    """
    Recursively find the layers of a certain type in a module.
    递归查找模块中某种类型的层
    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask


def lord(args, model, tokenizer, device=torch.device("cuda:0")):
    # lord
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device)

    layers = model.model.decoder.layers
    layer_lst = []
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        new_subset = {}
        for name in subset:
            print(f"pruning layer {i} name {name}")

            _, Q = torch.linalg.eigh(wrapped_layers[name].out_cov.to(device))
            new_subset[name + 'mean'] = wrapped_layers[name].out_mean.to('cpu')
            new_subset[name + 'Q'] = Q.half().to('cpu')
        print(new_subset.keys())
        layer_lst.append(new_subset)
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    return layer_lst

def print_model_data_types(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Data Type: {param.dtype}")

def prune_lord(model, tokenizer, layer_lst, ratios, device):
    layers = model.model.decoder.layers

    for i in range(len(layers)):
        # print(f"pruning layer {i}")
        layer = layers[i]
        subset = find_layers(layer)
        # print(subset)
        new_subset = {}
        new_subset['self_attn_layer_norm'] = layer.self_attn_layer_norm
        new_subset['final_layer_norm'] = layer.final_layer_norm
        # ratios = all_ratios[5 * (i//8):5 * (i//8) + 5]
        for name in subset:
            new_subset[name] = subset[name].weight.data.half()
            new_subset[name+'bias'] = subset[name].bias.data.half()
            # print(f"pruning layer {i} name {name}")
            parameter_ratio = 1
            if 'q_proj' in name:
                parameter_ratio = ratios[0]
            elif 'k_proj' in name:
                parameter_ratio = ratios[1]
            elif 'v_proj' in name:
                parameter_ratio = ratios[2]
            elif 'out_proj' in name:
                parameter_ratio = ratios[3]
            elif 'fc1' in name:
                parameter_ratio = ratios[4]
            elif 'fc2' in name:
                parameter_ratio = ratios[5]
            weight = new_subset[name].float().to('cuda')
            bias = new_subset[name+'bias'].float().to('cuda')
            out_mean = layer_lst[i][name + 'mean'].float().to('cuda')
            Q = layer_lst[i][name + 'Q'].float().to('cuda')
            H, W = weight.size()
            reduced_rank = math.ceil(parameter_ratio * (H * W) / (H + W))
            reduced_rank = (reduced_rank // 8) * 8

            len_Q = len(Q)
            L = Q[:, len_Q - reduced_rank:].T @ weight  # r X d1
            R = Q[:, len_Q - reduced_rank:]  # d2 X r
            b1 = Q[:, len_Q - reduced_rank:].T @ bias
            b2 = (out_mean - Q[:, len_Q - reduced_rank:] @ Q[:, len_Q - reduced_rank:].T @ out_mean).squeeze(-1)
            # print(L.size(), R.size(), b1.size(), b2.size())
            new_subset[name + '.0'] = L.half().to('cuda')
            new_subset[name + '.1'] = R.half().to('cuda')
            new_subset[name + '.0bias'] = b1.half().to('cuda')
            new_subset[name + '.1bias'] = b2.half().to('cuda')

        # print(new_subset.keys())
        # layer.self_attn_layer_norm = new_subset['self_attn_layer_norm'].to(device)
        # layer.final_layer_norm = new_subset['final_layer_norm'].to(device)
        # layer.self_attn = OPTAttention(config=config, ratios=ratios, is_decoder=True)
        fc1_rank = math.ceil(ratios[4] * (layer.embed_dim * config.ffn_dim) / (layer.embed_dim + config.ffn_dim))
        fc1_rank = (fc1_rank // 8) * 8
        layer.fc1 = nn.Sequential(nn.Linear(layer.embed_dim, fc1_rank, bias=config.enable_bias), nn.Linear(fc1_rank, config.ffn_dim, bias=True))
        # if ratios[5] == 1:
        #     layer.fc2 = nn.Linear(config.ffn_dim, layer.embed_dim, bias=config.enable_bias)
        # else:
        #     layer.fc2 = nn.Sequential(nn.Linear(config.ffn_dim, fc2_rank, bias=config.enable_bias), nn.Linear(fc2_rank, layer.embed_dim, bias=True))
        subset = find_layers(layer)
        # print(subset) layer.embed_dim
        for name in subset:
            if 'fc1' in name:
                subset[name].weight.data = new_subset[name].to(device)
                if subset[name].bias is not None:
                    subset[name].bias.data = new_subset[name + 'bias'].to(device)
# self.embed_dim, config.ffn_dim
# print_model_data_types(model)


