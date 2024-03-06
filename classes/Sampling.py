import torch

class Sampling:
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = z_mean.shape[0]
        dim = z_mean.shape[1]
        epsilon = torch.randn(batch, dim)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon
        