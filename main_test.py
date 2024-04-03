import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from lib.model_llama import LlamaForCausalLM
from importlib.metadata import version
import copy
import gc

# from lib.prune_bbo import lord, prune_lord
from lib.prune_test import lord, prune_lord
from openbox import Optimizer, space as sp
from lib.eval import eval_ppl, eval_zero_shot
# from lib.ppl_test import PPLMetric
from lib.evaluator import PPLMetric

torch.set_printoptions(threshold=torch.inf)
print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        # model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--save_model', action='store_true', help='if save model')

    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda")

    # layer_lst = lord(args, model, tokenizer, device)
    # torch.save(layer_lst, './output/llama2_chat_7b_1024_avg1024_parameters.pth')
    layer_lst = torch.load('./output/llama2_chat_7b_1024_avg1024_parameters.pth')

    ratio_ppl = {}
    layer_name = ["q_ratio", "k_ratio", "v_ratio", "o_ratio", "gate_ratio", "up_ratio", "down_ratio"]
    for i in range(4,7):
        for j in range(100, 29, -5):
            gc.collect()
            ratios = [1, 1, 1, 1, 1, 1, 1]
            ratios[i] = j/100
            prune_model = copy.deepcopy(model)
            # prune_model = get_llm(args.model, args.cache_dir)
            print(ratios)
            prune_lord(prune_model, tokenizer, layer_lst, ratios, device)
            # prune_model.to(device)
            prune_model.eval()
            ppl = PPLMetric(prune_model, tokenizer, ['wikitext2'], 4096, batch_size=1, device="cuda")['wikitext2']
            if layer_name[i] not in ratio_ppl:
                ratio_ppl[layer_name[i]] = []
            ratio_ppl[layer_name[i]].append(ppl)
            print(ratio_ppl)
    print(ratio_ppl)


if __name__ == '__main__':
    main()