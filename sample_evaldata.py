import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model_llama import LlamaForCausalLM
from importlib.metadata import version
import copy
import json
from tqdm import tqdm

from lib.prune_bbo import lord, prune_lord
from openbox import Optimizer, space as sp
from lib.datasets.ppl_dataset import get_loaders

torch.set_printoptions(threshold=torch.inf)
print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
    #model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings
    return model

@torch.no_grad()
def llama_eval(model, test_loader, device):
    ppls = []
    n = 0
    logits_lst = []
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        #print(batch)
        output = model(batch)
        lm_logits = output.logits  #[batch, seqlen, vab]
        
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        #loss = loss_fct(shift_logits, shift_labels.view(-1))
        loss = loss.reshape(lm_logits.size(0),-1)
        
        ppl = torch.exp(loss.mean(dim=1))
        #ppls.append(ppl)
        ppls+=ppl.cpu().numpy().tolist()
        #print(ppls)
        #exit()
        #nlls.append(loss)
    #print(torch.cat(nlls, dim=-1).mean())
    #ppl = np.exp(torch.cat(nlls, dim=-1).mean().item()).item()
    return ppls

@torch.no_grad()
def gemma_eval(model, test_lodaer, device):
    nlls = []
    n = 0
    for batch in tqdm(test_lodaer):
        
        batch = batch.to(device)
        batch[:, 0] = 2
        target_ids = batch.clone()
        output = model(batch,labels=target_ids)
        
        neg_log_likelihood = output.loss

        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).mean()).item()

    return ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="path/to/model", help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--remain_ratio', type=float, default=0.8, help='Sparsity level')
    parser.add_argument("--cache_dir", default="llm_weights", type=str)

    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    
    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    #print(model)
    #exit()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model or "70b" in args.model:  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    layer_lst = torch.load('path/to/pca/result')

    def sample_condition(config):
        s = 0
        if '7b' in model_name:
            s += 2.6875 * (config[f'gate_ratio'] + config[f'up_ratio'] + config[f'down_ratio']) + config[f'qk_ratio'] * 2 + 1 + config[f'o_ratio']
            # for i in range(4):
            #     s += (2.6875 * (config[f'gate_ratio_{i}'] + config[f'up_ratio_{i}'] + config[f'down_ratio_{i}']) + config[f'qk_ratio_{i}'] * 2 + 1 + config[f'o_ratio_{i}']) * 0.25
            # print(s)
            if round(s, 4) > args.remain_ratio * 12.0625:
                return False
            if round(s, 4) < (args.remain_ratio - 0.1) * 12.0625:
                return False
        if '13b' in model_name:
            s += 2.7 * (config[f'gate_ratio'] + config[f'up_ratio'] + config[f'down_ratio']) + config[f'qk_ratio'] * 2 + 1 + config[f'o_ratio']
            # for i in range(4):
            #     s += (2.7 * (config[f'gate_ratio_{i}'] + config[f'up_ratio_{i}'] + config[f'down_ratio_{i}']) + config[f'qk_ratio_{i}'] * 2 + 1 + config[f'o_ratio_{i}']) * 0.25
            # print(s)
            if round(s, 4) > args.remain_ratio * 12.1:
                return False
            if round(s, 4) < (args.remain_ratio - 0.1) * 12.1:
                return False
        
        return True

    space = sp.ConditionedSpace(seed=42)
    variables = []
    '''
    #for i in range(4):
    x1 = sp.Real(f"qk_ratio", 0.2, 1.001, default_value=0.5)
    # x2 = sp.Real(f"v_ratio_{i}", 0.2, 1.001, default_value=1.0)
    x2 = sp.Real(f"o_ratio", 0.2, 1.001, default_value=0.5)
    x3 = sp.Real(f"gate_ratio", 0.2, 1.001, default_value=0.8)
    x4 = sp.Real(f"up_ratio", 0.2, 1.001, default_value=0.8)
    x5 = sp.Real(f"down_ratio", 0.2, 1.001, default_value=0.8)
    variables += [x1, x2, x3, x4, x5]
    '''
    for i in range(4):
        if args.remain_ratio==0.8:
            x1 = sp.Real(f"qk_ratio_{i}", 0.2, 1.001, default_value=0.5)
            x2 = sp.Real(f"o_ratio_{i}", 0.2, 1.001, default_value=0.5)
            x3 = sp.Real(f"gate_ratio_{i}", 0.2, 1.001, default_value=0.8)
            x4 = sp.Real(f"up_ratio_{i}", 0.2, 1.001, default_value=0.8)
            x5 = sp.Real(f"down_ratio_{i}", 0.2, 1.001, default_value=0.8)
        else:
            x1 = sp.Real(f"qk_ratio_{i}", 0.2, 0.8, default_value=0.5)
            x2 = sp.Real(f"o_ratio_{i}", 0.2, 0.8, default_value=0.5)
            x3 = sp.Real(f"gate_ratio_{i}", 0.2, 0.8, default_value=0.7)
            x4 = sp.Real(f"up_ratio_{i}", 0.2, 0.8, default_value=0.7)
            x5 = sp.Real(f"down_ratio_{i}", 0.2, 0.8, default_value=0.7)

        variables += [x1, x2, x3, x4, x5]
    
    # print(variables)
    space.add_variables(variables)
    space.set_sample_condition(sample_condition)


    
    def ppl_wikipedia_metric(config, model=model):
        prune_model = copy.deepcopy(model)
        #ratios = [config[i] for i in ["qk_ratio", "o_ratio", "gate_ratio", "up_ratio", "down_ratio"]]
        ratios = []
        for i in range(4):
            ratios += [config[j] for j in [f"qk_ratio_{i}", f"o_ratio_{i}", f"gate_ratio_{i}", f"up_ratio_{i}", f"down_ratio_{i}"]]

        for i in range(len(ratios)):
            if ratios[i]>=1:
                ratios[i]=1.0
        
        prune_lord(prune_model, tokenizer, layer_lst, ratios)
        prune_model.to(device)
        
        prune_model.eval()
        test_loader = get_loaders('wikipedia', tokenizer, seq_len=4096, batch_size = 1)
        
        ppl = llama_eval(prune_model, test_loader, device)
        
        return ppl

    
    candidate = []
    for i in range(20):
        print("The iter ", i)
        config = space.sample_configuration()
        #print(config)
        #exit()
        ppl = ppl_wikipedia_metric(config)
        
        candidate.append(ppl)
        #candidate = np.array(candidate).T
    std = np.std(np.array(candidate), axis=0)
    #var = np.var(np.array(candidate), axis=0)
    #mean = np.mean(np.array(candidate), axis=0)
    #cv = std/mean
    #cv = var/mean
    k=100
    top_indices = np.argpartition(std, -k)[-k:]
    least_indices = np.argpartition(std, k)[:k]
    print(top_indices)
    print(least_indices)
            

    
    
if __name__ == '__main__':
    main()
