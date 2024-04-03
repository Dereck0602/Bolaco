import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from lib.model_llama import LlamaForCausalLM
from importlib.metadata import version
import copy

from lib.prune_opt import lord, prune_lord
from openbox import Optimizer, space as sp
from lib.eval import eval_ppl, eval_zero_shot

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
    # print(model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")

    # layer_lst = lord(args, model, tokenizer, device)
    # torch.save(layer_lst, './output/opt_6.7b_parameters.pth')

    layer_lst = torch.load('./output/opt_6.7b_parameters.pth')

    ratios = [1, 1, 1, 1, 1, 1]
    # print(ratios)
# q_proj, k_proj, v_proj, out_proj, fc1, fc2
    # before_pruning_parameters = sum(p.numel() for p in model.parameters())
    ratio_ppl = {}
    layer_name = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    for i in range(4,5):
        for j in range(100, 29, -5):
            model = get_llm(args.model, args.cache_dir)
            model.eval()
            ratios = [1, 1, 1, 1, 1, 1]
            ratios[i] = j/100
            print(ratios)
            prune_lord(model, tokenizer, layer_lst, ratios, device)
            ppl_test = eval_ppl(model, tokenizer, device)
            print(f"wikitext perplexity {ppl_test}")
            if layer_name[i] not in ratio_ppl:
                ratio_ppl[layer_name[i]] = []
            ratio_ppl[layer_name[i]].append(ppl_test)
        print(ratio_ppl)

    # prune_lord(model, tokenizer, layer_lst, ratios, device)
    # # print(model)
    # after_pruning_parameters = sum(p.numel() for p in model.parameters())
    # print("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters, 100.0 * after_pruning_parameters / before_pruning_parameters))
    #
    # ppl_test = eval_ppl(model, tokenizer, device)
    # print(f"wikitext perplexity {ppl_test}")

    # accelerate = False
    # task_list = ["boolq", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa", "piqa"]
    # # task_list = ["hellaswag", "arc_challenge", "boolq"]
    # num_shot = 0
    # results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
    # print("********************************")
    # print("zero_shot evaluation results")
    # print(results)
    #
    # if args.save_model:
    #     model.half()
    #     torch.save({
    #         'model': model,
    #         'tokenizer': tokenizer,
    #     }, './output/llama2_7b_90.pt')

if __name__ == '__main__':
    main()