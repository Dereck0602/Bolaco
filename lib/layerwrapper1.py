import torch
import torch.nn as nn

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        # self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.out = torch.zeros((self.rows, self.rows), device=self.dev)
        self.inp = torch.zeros((2048, self.columns), device=self.dev)
        self.out_cov = torch.zeros((self.rows, self.rows), device=self.dev)
        # self.out_t = torch.zeros((self.rows, self.rows), device=self.dev)
        self.out_mean = torch.zeros((self.rows, 1), device=self.dev)

        self.nsamples = 0
        self.navg = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):

        if len(out.shape) == 2:
            out = out.unsqueeze(0)

        tmp = out.shape[0]  # 1
        out = out.squeeze(0).T.type(torch.float32)
        if self.nsamples == 0:
            self.out = out
        else:
            self.out = torch.cat((self.out, out), dim=-1)
        self.nsamples += tmp
        if self.nsamples % 2 == 0:
            tmp = 1
            out_cov = torch.cov(self.out)
            out_mean = torch.mean(self.out, dim=-1).unsqueeze(1)

            self.out_cov *= self.navg / (self.navg + tmp)
            self.out_mean *= self.navg / (self.navg + tmp)
            self.navg += 1
            self.out_mean += out_mean / self.navg
            self.out_cov += out_cov / self.navg

            self.nsamples = 0


