import time
import heapq
import math
import torch
import random

import torch.nn as nn
# from .layerwrapper import WrappedGPT
from .layerwrapper2 import WrappedGPT
from .data import get_loaders
from transformers import AutoConfig

# from .llama_7b.configuration_llama import LlamaConfig
# from .llama_13b.configuration_llama import LlamaConfig
from .llama2_7b.configuration_llama import LlamaConfig
# from .llama2_13b.configuration_llama import LlamaConfig
from .model_llama_1 import LlamaDecoderLayer
# from .model_llama import LlamaDecoderLayer

# config = AutoConfig.from_pretrained("./llama-7b-hf/config.json")
# config = AutoConfig.from_pretrained("./llama-13b-hf/config.json")
config = AutoConfig.from_pretrained("./Llama2-7b-hf/config.json")
# config = AutoConfig.from_pretrained("./Llama2-13b-hf/config.json")
llamaconfig = LlamaConfig()


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
    # 准备校准输入
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    # inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps = torch.zeros((1024, model.seqlen, model.config.hidden_size), dtype=dtype, device='cpu')
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
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        # print(batch[0])
        input_ids = batch[0]
        try:
            model(input_ids.to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    # 返回给定的alpha
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def print_model_data_types(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Data Type: {param.dtype}")


def lord(args, model, tokenizer, device=torch.device("cuda:0")):
    # lord
    use_cache = model.config.use_cache
    model.config.use_cache = False
    print("loading calibdation data")

    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)

    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers

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

        # 对每个(mlp或attention层)添加钩子函数
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0).to(device), attention_mask=attention_mask.to(device), position_ids=position_ids.to(device))[0]
        for h in handles:
            h.remove()

        new_subset = {}
        for name in subset:
            print(f"pruning layer {i} name {name}")

            # _, Q = torch.linalg.eigh(wrapped_layers[name].out_cov.to(device))
            _, Q = torch.linalg.eigh(wrapped_layers[name].out_avgcov.to(device))
            # new_subset[name + 'mean'] = wrapped_layers[name].out_mean.to('cpu')
            new_subset[name + 'mean'] = wrapped_layers[name].out_avgmean.to('cpu')
            new_subset[name + 'Q'] = Q.half().to('cpu')
        print(new_subset.keys())
        layer_lst.append(new_subset)
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    return layer_lst


def prune_lord(model, tokenizer, layer_lst, ratios, device):
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        new_subset = {}
        new_subset['input_layernorm'] = layer.input_layernorm.weight.data
        new_subset['post_attention_layernorm'] = layer.post_attention_layernorm.weight.data
        # ratios = all_ratios[5 * (i//8):5 * (i//8) + 5]

        for name in subset:
            new_subset[name] = subset[name].weight.data.half()
            # print(f"pruning layer {i} name {name}")
            parameter_ratio = 1
            if 'q_proj' in name:
                parameter_ratio = ratios[0]
            elif 'k_proj' in name:
                parameter_ratio = ratios[1]
            elif 'v_proj' in name:
                parameter_ratio = ratios[2]
            elif 'o_proj' in name:
                parameter_ratio = ratios[3]
            elif 'gate_proj' in name:
                parameter_ratio = ratios[4]
            elif 'up_proj' in name:
                parameter_ratio = ratios[5]
            elif 'down_proj' in name:
                parameter_ratio = ratios[6]
            weight = new_subset[name].float().to('cuda')
            out_mean = layer_lst[i][name + 'mean'].float().to('cuda')
            Q = layer_lst[i][name + 'Q'].float().to('cuda')
            H, W = weight.size()
            reduced_rank = math.ceil(parameter_ratio * (H * W) / (H + W))
            reduced_rank = (reduced_rank // 8) * 8

            len_Q = len(Q)
            L = Q[:, len_Q - reduced_rank:].T @ weight  # r X d1
            R = Q[:, len_Q - reduced_rank:]  # d2 X r
            b = (out_mean - Q[:, len_Q - reduced_rank:] @ Q[:, len_Q - reduced_rank:].T @ out_mean).squeeze(-1)

            new_subset[name + '.0'] = L.half().to('cuda')
            new_subset[name + '.1'] = R.half().to('cuda')
            new_subset[name + '.1bias'] = b.half().to('cuda')
            # print(new_subset.keys())
        layers[i] = LlamaDecoderLayer(config, llamaconfig, ratios)
        layer = layers[i]
        layer.input_layernorm.weight.data = new_subset['input_layernorm']
        layer.post_attention_layernorm.weight.data = new_subset['post_attention_layernorm']

        subset = find_layers(layer)
        # print(subset)

        for name in subset:
            subset[name].weight.data = new_subset[name].to(device)
            if subset[name].bias is not None:
                subset[name].bias.data = new_subset[name + 'bias'].to(device)

