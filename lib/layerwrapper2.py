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

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.out = torch.zeros((self.rows, 2048), device=self.dev)
        self.inp = torch.zeros((2048, self.columns), device=self.dev)
        self.out_cov = torch.zeros((self.rows, self.rows), device=self.dev)  # 记录之前sample的所有cov
        self.out_mean = torch.zeros((self.rows, 1), device=self.dev)

        self.out_avgcov = torch.zeros((self.rows, self.rows), device=self.dev)
        self.out_avgmean = torch.zeros((self.rows, 1), device=self.dev)
        self.nsamples = 0
        self.avg_step = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):

        if len(out.shape) == 2:
            out = out.unsqueeze(0)

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        out = out.squeeze(0).T.type(torch.float32)

        out_cov = torch.cov(out, correction=1)
        out_mean = torch.mean(out, dim=-1).unsqueeze(1)


        tmp = out.shape[1]

        mean = (self.nsamples * self.out_mean + tmp * out_mean) / (self.nsamples + tmp)
        # cov = (self.nsamples * self.out_cov + out_cov + self.nsamples* (self.out_mean-mean)@(self.out_mean-mean).T + (out_mean-mean)@(out_mean-mean).T)/(self.nsamples+tmp)
        # cov = (self.nsamples * self.out_cov + out_cov + self.nsamples/(self.nsamples+tmp)*(out_mean-mean)@(out_mean-mean).T)/(self.nsamples+tmp)
        cov = ((self.nsamples - 1) * self.out_cov + (tmp - 1) * out_cov + self.nsamples * tmp / (
                    self.nsamples + tmp) * (out_mean - mean) @ (out_mean - mean).T) / (
                          self.nsamples + tmp - 1)  # unbias
        self.out_mean = mean
        self.out_cov = cov
        self.nsamples += tmp
        if self.nsamples % (1024 * tmp) == 0:
            tmp = 1
            self.out_avgcov = (self.avg_step * self.out_avgcov + tmp * self.out_cov) / (self.avg_step + tmp)
            self.out_avgmean = (self.avg_step * self.out_avgmean + tmp * self.out_mean) / (self.avg_step + tmp)
            self.out_cov = torch.zeros((self.rows, self.rows), device=self.dev)
            self.out_mean = torch.zeros((self.rows, 1), device=self.dev)
            self.avg_step += tmp
            self.nsamples = 0