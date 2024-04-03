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
        self.out = torch.zeros((self.rows, 2048), device=self.dev)
        self.inp = torch.zeros((2048, self.columns), device=self.dev)
        self.out_cov = torch.zeros((self.rows, self.rows), device=self.dev)
        # self.out_t = torch.zeros((self.rows, self.rows), device=self.dev)
        self.out_mean = torch.zeros((self.rows, 1), device=self.dev)
        
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):

        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = out.shape[0]  # 1

        # if isinstance(self.layer, nn.Linear):
        #     if len(inp.shape) == 3:
        #         inp = inp.reshape((-1, inp.shape[-1]))
        #     inp = inp.t()

        out = out.squeeze(0).T.type(torch.float32)
        out_cov = torch.cov(out)
        # out_cov = torch.cov(out, correction = 0)
        out_mean = torch.mean(out, dim = -1).unsqueeze(1)
        # out_t = out @ out.T / out.shape[-1]
        # self.out_t *= self.nsamples / (self.nsamples+tmp)
        # self.out *= self.nsamples / (self.nsamples+tmp)
        # self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.out_cov *= self.nsamples / (self.nsamples+tmp)
        self.out_mean *= self.nsamples / (self.nsamples+tmp)

        self.nsamples += tmp
        
        # self.out_t += out_t / self.nsamples
        # self.out += out / self.nsamples
        self.out_mean += out_mean / self.nsamples
        self.out_cov += out_cov / self.nsamples

        # inp = inp.type(torch.float32)
        # self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
        # out = out.squeeze(0).type(torch.float32)
        # self.nsamples += 1

        
        
        
        