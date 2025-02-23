import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from lib.model_llama import LlamaForCausalLM
from importlib.metadata import version
import copy

from lib.prune_bbo import lord, prune_lord
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
    parser.add_argument('--remain_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', action='store_true', help='if save model')

    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model:  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    layer_lst = lord(args, model, tokenizer, device)
    torch.save(layer_lst, './output/llama2_7b_1024_avg32_parameters.pth')

    layer_lst = torch.load('./output/llama2_7b_1024_avg32_parameters.pth')

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

    space = sp.ConditionedSpace()
    variables = []
    if args.remain_ratio==0.8:
        ratios = [0.5, 0.5, 0.8, 0.8, 0.8]
        # for i in range(4):
        x1 = sp.Real(f"qk_ratio", 0.2, 1.001, default_value=ratios[0])
        x2 = sp.Real(f"o_ratio", 0.2, 1.001, default_value=ratios[1])
        x3 = sp.Real(f"gate_ratio", 0.2, 1.001, default_value=ratios[2])
        x4 = sp.Real(f"up_ratio", 0.2, 1.001, default_value=ratios[3])
        x5 = sp.Real(f"down_ratio", 0.2, 1.001, default_value=ratios[4])
    if args.remain_ratio==0.7:
        ratios = [0.5, 0.5, 0.7, 0.7, 0.7]
        # for i in range(4):
        x1 = sp.Real(f"qk_ratio", 0.2, 1.001, default_value=ratios[0])
        x2 = sp.Real(f"o_ratio", 0.2, 1.001, default_value=ratios[1])
        x3 = sp.Real(f"gate_ratio", 0.2, 1.001, default_value=ratios[2])
        x4 = sp.Real(f"up_ratio", 0.2, 1.001, default_value=ratios[3])
        x5 = sp.Real(f"down_ratio", 0.2, 1.001, default_value=ratios[4])
    variables += [x1, x2, x3, x4, x5]
    print(variables)
    space.add_variables(variables)
    space.set_sample_condition(sample_condition)

    unprune_ppl, logits = PPLMetric(model, tokenizer, ['wikipedia'], 4096, 1, device='cuda', out_logits=True)

    def ppl_metric(config, model=model):
        prune_model = copy.deepcopy(model)
        ratios = [config[i] for i in ["qk_ratio", "o_ratio", "gate_ratio", "up_ratio", "down_ratio"]]
        # ratios = []
        # for i in range(4):
        #     ratios += [config[j] for j in [f"qk_ratio_{i}", f"o_ratio_{i}", f"gate_ratio_{i}", f"up_ratio_{i}", f"down_ratio_{i}"]]
        for i in range(len(ratios)):
            if ratios[i] > 1.0:
                ratios[i] = 1.0
        prune_lord(prune_model, tokenizer, layer_lst, ratios, device)
        prune_model.to(device)
        prune_model.eval()
        # ppl_test = eval_ppl(prune_model, tokenizer, device)
        ppl_test = PPLMetric(prune_model, tokenizer, ['wikipedia'], 4096, 1, device='cuda', target=logits)['wikipedia']
        # ppl_test = PPLMetric(prune_model, tokenizer, ['wikitext2'], 4096, batch_size=1, device="cuda")['wikitext2']
        result = dict()
        result['objectives'] = [ppl_test, ]
        return result

    opt = Optimizer(
        ppl_metric,
        space,
        num_objectives=1,
        surrogate_type='gp',
        # acq_optimizer_type='random_scipy',
        acq_optimizer_type='local_random',
        max_runs=50,
        task_id='lowrank',
    )

    opt.run()

    history = opt.get_history()
    print(history)
    lst = history.observations

    # print(ratios)
    # print(history.observations[0].objectives)

    lst.sort(key=lambda x: x.objectives[0])

    ratio_dict = lst[0].config.get_dictionary()
    for i in ratio_dict:
        if ratio_dict[i] > 1:
            ratio_dict[i] = 1.0
    ratios = [ratio_dict[j] for j in ["qk_ratio", "o_ratio", "gate_ratio", "up_ratio", "down_ratio"]]
    # ratios = []
    # for i in range(4):
    #     ratios += [ratio_dict[j] for j in [f"qk_ratio_{i}", f"o_ratio_{i}", f"gate_ratio_{i}", f"up_ratio_{i}", f"down_ratio_{i}"]]
    print(ratios)

    before_pruning_parameters = sum(p.numel() for p in model.parameters())

    prune_lord(model, tokenizer, layer_lst, ratios, device)

    after_pruning_parameters = sum(p.numel() for p in model.parameters())
    print("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters, 100.0 * after_pruning_parameters / before_pruning_parameters))

    ppl_test = eval_ppl(model, tokenizer, device)
    # ppl_test = PPLMetric(model, tokenizer, ['wikitext2'], 4096, batch_size=1, device="cuda")
    print(f"wikitext perplexity {ppl_test}")
    # print(f"wikitext perplexity(2048) {ppl}")
    # ppl = PPLMetric(model, tokenizer, ['wikitext2'], 4096, batch_size=1, device="cuda")
    # ppl = PPLMetric(model, tokenizer, ['wikitext2'], 4096, batch_size=1, device="cuda")
    # ppl = PPLMetric(model, tokenizer, ['ptb'], 256, batch_size=4, device="cuda")

    accelerate = False
    task_list = ["boolq", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa", "piqa"]
    # task_list = ["hellaswag", "arc_challenge", "boolq"]
    num_shot = 0
    results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
    print("********************************")
    print("zero_shot evaluation results")
    print(results)

    if args.save_model:
        model.half()
        torch.save({
            'model': model,
            'tokenizer': tokenizer,
        }, '/output/llama2_7b_80.pt')


if __name__ == '__main__':
    main()
