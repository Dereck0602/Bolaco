import torch
import numpy as np
from tqdm import tqdm

from .ppl_test import get_loaders


def PPLMetric(model, tokenizer, datasets, seq_len=128, batch_size=1, device="cuda", out_logits=False, target=None):
    metric = {}
    for dataset in datasets:
        if dataset == 'ptb':
            _, test_loader = get_loaders(dataset, tokenizer, seq_len=256, batch_size=batch_size)
        else:
            _, test_loader = get_loaders(dataset, tokenizer, seq_len=seq_len, batch_size=batch_size)
        if out_logits:
            ppl, logits = llama_eval(model, test_loader, device, out_logits=out_logits, target=target)
        else:
            ppl, _ = llama_eval(model, test_loader, device, out_logits=out_logits, target=target)
        metric[dataset] = ppl
    print(metric)
    if out_logits:
        return metric, logits
    else:
        return metric


@torch.no_grad()
def llama_eval(model, test_lodaer, device, out_logits=False, target=None):
    nlls = []
    n = 0
    logits_lst = []
    kl = []
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits  # [batch, seqlen, vab]

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
        if target:
            # p_tec = torch.softmax(lm_logits.view(-1, lm_logits.shape[2]), dim=-1)
            # q = torch.log_softmax(target[n].view(-1, lm_logits.shape[2]).to(device), dim=-1)
            p_tec = torch.softmax(lm_logits, dim=-1)
            q = torch.log_softmax(target[n].to(device), dim=-1)
            kl_loss = torch.nn.functional.kl_div(q, p_tec, reduction='batchmean')

            n += 1
            kl.append(kl_loss.item())
            # print(len(target))
        if out_logits:
            logits_lst.append(lm_logits.to('cpu'))
    # print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item()).item()
    if target:
        kl_loss = sum(kl) / len(kl)
        ppl = ppl + 1e-3 * kl_loss

    return ppl, logits_lst