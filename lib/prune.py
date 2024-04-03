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

# from .llama_7b.configuration_llama import LlamaConfig
# from .llama_13b.configuration_llama import LlamaConfig
from .llama2_7b.configuration_llama import LlamaConfig
# from .llama2_13b.configuration_llama import LlamaConfig
from .model_llama import LlamaMLP, LlamaAttention, LlamaDecoderLayer

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


def check_sparsity(model):
    # 检查模型的稀疏度
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()    # count统计W中为0的个数
            total_params += W.numel()       # .numel()返回W中元素的总数

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 


def check_ratio(layer, rank_set):
    # 检查一层的现权重/原权重
    subset = find_layers(layer)
    orginal_paras = 0
    current_paras = 0
    for name in subset:
        H, W = subset[name].weight.data.size()
        orginal_paras += H * W
        r = rank_set[name]
        current_paras += (H + W) * r
    return current_paras / orginal_paras


def prepare_calibration_input(model, dataloader, device):
    # 准备校准输入
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    # inps = torch.zeros((256, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
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
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def low_rank_decomposition(weight, parameter_ratio = 0.8):
    # SVD
    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"
    H, W = weight.size()
    print(H,W)
    # svd
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    # print(U.size(),S.size(),Vh.size())
    rank = torch.count_nonzero(S)
    is_full_rank = rank == min(H, W)
    reduced_rank = math.ceil(parameter_ratio * (H * W) / (H + W))

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh
    # L = torch.linalg.inv(torch.diag(torch.sqrt(lst))) @ U[:, 0:reduced_rank] @ (torch.diag(S)[0:reduced_rank, 0:reduced_rank])
    # L = U[:, 0:reduced_rank] @ (torch.diag(S)[0:reduced_rank, 0:reduced_rank])
    # R = Vh[0:reduced_rank, :]

    return L, R


def EIG_Decomposition(weight, out_cov, out_mean, parameter_ratio):
    # 特征值分解
    H, W = weight.size()

    # out_cov = out_t - out_mean @ out_mean.T
    
    # r = int(torch.linalg.matrix_rank(out_cov))
    # print(out_cov.size(), r)
    reduced_rank = math.ceil(parameter_ratio * (H * W) / (H + W))
    reduced_rank = (reduced_rank // 8) * 8

    # 特征值分解
    _, Q = torch.linalg.eigh(out_cov)
    len_Q = len(Q)
    L = Q[:, len_Q-reduced_rank:].T @ weight.float() # r X d1

    R = Q[:, len_Q-reduced_rank:].float() # d2 X r
    
    b = (out_mean - Q[:, len_Q-reduced_rank:] @ Q[:, len_Q-reduced_rank:].T @ out_mean).squeeze(-1)
    print(L.size(), R.size(), b.size())
    # return L, R, b, reduced_rank, weight_ratio
    return L, R, b


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            
            thresh1 = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*0.1)].cpu()
            # thresh2 = torch.sort(W_metric.flatten().cuda())[0][W.numel() - int(W.numel()*0.1)].cpu()
            W_mask = (W_metric<=thresh1)

            W[W_mask] = 0


def prune_SVD(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # SVD
    layers = model.model.layers 
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        new_subset = {}
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.float()
            # print(W.size())
            L, R = low_rank_decomposition(W.T, 0.8)
            # L, R = lplr_svd(W.T, 1)
            print(L.T.size(), R.T.size())
            new_subset[name+'1'] = L.T.half()
            new_subset[name+'2'] = R.T.half()
            new_subset[name] = W.half()

        # print(new_subset)
        # layers[i] = LlamaDecoderLayer(config, llamaconfig)
        # layer = layers[i]
        layer.mlp = LlamaMLP(config, 0.5)
        layer.self_attn = LlamaAttention(llamaconfig, 0.5)
        subset = find_layers(layer)
        # print(subset)
        for name in subset:
            # if 'mlp' in name:
            subset[name].weight.data = new_subset[name]
    
    print(model.model.layers)
    

def print_model_data_types(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Data Type: {param.dtype}")


def prune_LORD(args, model, tokenizer, device=torch.device("cuda")):
    # lord
    use_cache = model.config.use_cache 
    model.config.use_cache = False
    print("loading calibdation data")
    
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    # dataloader = get_loaders("bookcorpus", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen,tokenizer=tokenizer)
    # dataloader, _ = get_loaders("wikitext2", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen,tokenizer=tokenizer)

    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    layers = model.model.layers
    # out_cov_rank = {}
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        # print(subset)
        
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
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        new_subset = {}
        new_subset['input_layernorm'] = layer.input_layernorm.weight.data.to(device)
        new_subset['post_attention_layernorm'] = layer.post_attention_layernorm.weight.data.to(device)
        # ratios = [0.385, 0.671, 0.857, 0.879, 0.946]
        ratios = [0.7, 0.8, 0.8, 0.8, 0.8]
        # ratios = [0.3700159759394165, 0.7537840832935749, 0.7702711036342362, 1.0, 0.8938073818519294]
        for name in subset:
            print(f"pruning layer {i} name {name}")

            new_subset[name] = subset[name].weight.data.half().to(device)
            parameter_ratio = 1
            if 'v_proj' not in name:
                if 'q_proj' in name or 'k_proj' in name:
                    parameter_ratio = ratios[0]
                elif 'o_proj' in name:
                    parameter_ratio = ratios[1]
                elif 'gate_proj' in name:
                    parameter_ratio = ratios[2]
                elif 'up_proj' in name:
                    parameter_ratio = ratios[3]
                elif 'down_proj' in name:
                    parameter_ratio = ratios[4]

                if parameter_ratio != 1:
                    L, R, b = EIG_Decomposition(subset[name].weight.data.to(device), wrapped_layers[name].out_cov.to(device), wrapped_layers[name].out_mean.to(device), parameter_ratio)
                    # L, R, b = EIG_Decomposition(subset[name].weight.data.to(device), wrapped_layers[name].out_avgcov.to(device), wrapped_layers[name].out_avgmean.to(device), parameter_ratio)
                    # if name not in out_cov_rank:
                    #     out_cov_rank[name] = []
                    # out_cov_rank[name].append(out_r)

                    # 将分解后的weight放入集合中
                    new_subset[name+'.0'] = L.half().to(device)
                    new_subset[name+'.1'] = R.half().to(device)
                    new_subset[name+'.1bias'] = b.half().to(device)

        # 对mlp和attention层做替换
        layers[i] = LlamaDecoderLayer(config, llamaconfig, ratios)
        layer = layers[i]
        layer.input_layernorm.weight.data = new_subset['input_layernorm']
        layer.post_attention_layernorm.weight.data = new_subset['post_attention_layernorm']
        
        subset = find_layers(layer)
        # print(subset)
        
        for name in subset:
            subset[name].weight.data = new_subset[name].to(device)
            if subset[name].bias is not None:
                subset[name].bias.data = new_subset[name+'bias'].to(device)

        # for j in range(args.nsamples):
        #     with d.no_grad():
        #         outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    # print(out_cov_rank)
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

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
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
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
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
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
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()