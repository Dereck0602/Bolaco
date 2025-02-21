'''
Some of the code refer to
https://github.com/IST-DASLab/gptq/blob/main/datautils.py
'''

import random
import numpy as np
import torch

from datasets import load_dataset, load_from_disk
from torch.utils.data.dataset import Dataset

def get_wikitext2(seq_len, tokenizer):
    #traindata = load_dataset('text',data_files='/ossfs/workspace/data/wikitext-2-raw/wiki.train.txt')
    #testdata = load_dataset('text',data_files='/ossfs/workspace/data/wikitext-2-raw/wiki.test.txt')
    #traindata = load_dataset('json', data_files='/ossfs/workspace/datacube-nas/yixin_llm/data/wikitext-2-raw/train.jsonl')
    #testdata = load_dataset('json',data_files='/ossfs/workspace/datacube-nas/yixin_llm/data/wikitext-2-raw/test.jsonl')
    traindata = load_dataset('parquet', data_files='/ossfs/workspace/datacube-nas/yixin_llm/data/wikitext-2-raw-v1/train-00000-of-00001.parquet')
    testdata = load_dataset('parquet',data_files='/ossfs/workspace/datacube-nas/yixin_llm/data/wikitext-2-raw-v1/test-00000-of-00001.parquet')
    #testdata = load_dataset('json', data_files='/ossfs/workspace/data/wikitext-2-raw/text.jsonl')
    #traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    #testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    return traindata, testdata

def get_ptb(seq_len, tokenizer):
    traindata = load_dataset('text',data_files='/ossfs/workspace/datacube-nas/yixin_llm/data/ptb/ptb.train.txt')
    valdata = load_dataset('text',data_files='/ossfs/workspace/datacube-nas/yixin_llm/data/ptb/ptb.valid.txt')
    #traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    #valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    return traindata, valdata

def get_wikipedia(seq_len, tokenizer):
    dataset = load_dataset('parquet',data_files='data/wikipedia/train-00000-of-00041.parquet')
    testdata = dataset['train'].select(range(500))
    return testdata

def get_arxiv(seq_len, tokenizer):
    dataset = load_dataset('json',data_files='/ossfs/workspace/datacube-nas/yixin_llm/data/arxiv/arxiv_001.jsonl')
    testdata = dataset['train'].select(range(20))
    return testdata

def get_github(seq_len, tokenizer):
    dataset = load_dataset('json',data_files='/ossfs/workspace/datacube-nas/yixin_llm/data/github/github_001.jsonl')
    testdata = dataset['train'].select(range(200))
    return testdata

def get_c4(seq_len, tokenizer):
    testdata = load_from_disk('/ossfs/workspace/datacube-nas/yixin_llm/data/c4_validation')
    testdata = testdata.select(range(2000,3000))  #5000
    return testdata

def get_dclm(seq_len, tokenizer):
    dataset = load_dataset('parquet',data_files='/ossfs/workspace/yixin.jyx/data/dclm-micro/output_1.parquet')
    testdata = dataset['train'].select(range(3000))
    return testdata

def get_magpie(seq_len, tokenizer):
    dataset = data = load_dataset('json', data_files='/ossfs/workspace/yixin.jyx/data/magpie_llama3-8b_300k.json')
    testdata = dataset['train'].select(range(1000))
    return testdata

class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

def process_data(samples, tokenizer, seq_len, split=None, field_name=None, choose=False, idx=None):
    #print(samples['train'])
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
       

def get_loaders(name, tokenizer, seq_len=2048, batch_size = 8):
    if 'wikitext2' in name:
        train_data, test_data = get_wikitext2(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len,'train', 'text')
    if 'ptb' in name:
        train_data, test_data = get_ptb(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len,'train', 'text')
    if 'wikipedia' in name:
        test_data = get_wikipedia(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len,split=None,field_name='text',choose=True)
    if 'arxiv' in name:
        test_data = get_arxiv(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len,split=None,field_name='text')
    if 'github' in name:
        test_data = get_github(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len,split=None,field_name='text')
    if 'c4' in name:
        test_data = get_c4(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len,split=None,field_name='text')
    if 'dclm' in name:
        test_data = get_dclm(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len,split=None,field_name='text')
    if 'magpie' in name:
        test_data = get_magpie(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len,split=None,field_name='text')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
