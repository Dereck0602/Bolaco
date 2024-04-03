import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_LORD, prune_SVD, prune_sparsegpt, prune_ablate, check_sparsity, prune_wanda, prune_magnitude
from lib.eval import eval_ppl, eval_zero_shot

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

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
    parser.add_argument('--model', type=str, default='./llama-7b-hf', help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default = 'unstructured', choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, default='lord', choices=["magnitude", "wanda", "sparsegpt", "lord", "SVD", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default='out/llama_7b/unstructured/wanda/', help='Path to save results.')
    # parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--save_model', action='store_true', help='if save model')

    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    before_pruning_parameters = sum(p.numel() for p in model.parameters())

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "lord":
            prune_LORD(args, model, tokenizer, device)
        elif args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "SVD":
            prune_SVD(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    # print("*"*30)
    # sparsity_ratio = check_sparsity(model)
    # print(f"sparsity sanity check {sparsity_ratio:.4f}")
    # print("*"*30)
    ################################################################
    after_pruning_parameters = sum(p.numel() for p in model.parameters())
    print("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters, 100.0 * after_pruning_parameters / before_pruning_parameters))

    ppl_test = eval_ppl(model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "hellaswag", "winogrande", "arc_easy","arc_challenge", "openbookqa", "piqa"]
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
        }, './output/llama2_13b_bbo_80.pt')

if __name__ == '__main__':
    main()