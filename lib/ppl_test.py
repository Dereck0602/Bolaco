import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch
import fnmatch
from datasets import load_dataset
from torch.utils.data.dataset import Dataset


def get_wikitext2(seq_len, tokenizer):
    traindata = load_dataset("parquet", data_files={"train": '/public/home/ljt/xy/prune_llm/Bolaco/wikitext-2-raw-v1/train-00000-of-00001.parquet'}, split='train')
    testdata = load_dataset("parquet", data_files={"test": '/public/home/ljt/xy/prune_llm/Bolaco/wikitext-2-raw-v1/test-00000-of-00001.parquet'}, split='test')
    return traindata, testdata

def get_ptb(seq_len, tokenizer):
    traindata = load_dataset('./ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('./ptb_text_only', 'penn_treebank', split='validation')
    return traindata, valdata

def get_c4(seq_len, tokenizer):
    traindata = load_dataset('json', data_files={'train': './c4/c4-train.00000-of-01024.json'}, split='train')
    valdata = load_dataset('json', data_files={'validation': './c4/c4-validation.00000-of-00008.json'}, split='validation')
    valdata = valdata.select(range(5000))
    return traindata, valdata

def get_wikipedia(seq_len, tokenizer):
    dataset = load_dataset('parquet',data_files='/public/home/ljt/xy/prune_llm/Bolaco/wikipedia/train-00000-of-00041.parquet')
    testdata = dataset['train'].select(range(500))
    return testdata, testdata

class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

def process_data(samples, tokenizer, seq_len, split=None, field_name=None, choose=False, idx=None):
    if split=='train':
        test_ids = tokenizer("\n\n".join(samples['train'][field_name]), return_tensors='pt').input_ids[0]
    else:
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
    
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        if choose:
            if i in idx:
                batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
                test_ids_batch.append(batch)
        else:
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)

def get_loaders(name, tokenizer, seq_len=2048, batch_size=8):
    if 'wikitext2' in name:
        train_data, test_data = get_wikitext2(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, None, 'text')
    if 'ptb' in name:
        train_data, test_data = get_ptb(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    if 'c4' in name:
        train_data, test_data = get_c4(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'wikipedia' in name:
        idx = [583,367,574,818,518,656,608,225,611,731,710,4,426,77,447,684,261,228,740,705,788,262,812,667,717,209,657,530,819,799,771,156,10,222,808,669,809,380,671,68,257,396,627,437,631,793,415,208,820,516,770,678,672,52,379,480,496,521,529,537,685,45,553,554,821,742,315,733,662,810,713,815,510,269,172,11,12,243,368,435,444,112,91,565,741,593,594,619,218,632,722,661,716,714,676,677,680,694,51]
        train_data, test_data = get_wikipedia(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, split=None, field_name='text', choose=True, idx=idx)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_data, test_loader


def PPLMetric(model, tokenizer, datasets, seq_len=128, batch_size=4, device="cuda"):
    metric = {}
    for dataset in datasets:
        _, test_loader = get_loaders(dataset, tokenizer, seq_len=seq_len, batch_size=batch_size)
        ppl = llama_eval(model, test_loader, device)
        metric[dataset] = ppl
    print(metric)
    return metric


@torch.no_grad()
def llama_eval(model, test_lodaer, device):
    nlls = []
    n_samples = 0
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
    # print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()