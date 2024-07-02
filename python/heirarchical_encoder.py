import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import HashEmbedderNative

def coherent_prime_hash(coords):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = torch.tensor([1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737])

    xor_result = torch.zeros_like(coords[..., 0].to(torch.int32))
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i].to(torch.int32) * primes[i]

    return xor_result

class HeirarchicalHashEmbedderNative(nn.Module):
    def __init__(self, n_pos_dims=2,
                 partition_size=2,
                 individual_encoding_config={}):
        super(HeirarchicalHashEmbedderNative, self).__init__()
        
        # only support 2 or 3 dimensional inputs for now
        assert (n_pos_dims == 2 or n_pos_dims == 3)

        self.n_pos_dims = n_pos_dims
        self.partition_size = partition_size

        individual_encoding_config["otype"]="Grid"
        individual_encoding_config["type"]="Hash"
        self.individual_encoding_config = individual_encoding_config
        
        self.n_output_dims = self.individual_encoding_config["n_levels"] * self.individual_encoding_config["n_features_per_level"]
        
        self.encoders = nn.ModuleList([HashEmbedderNative(n_pos_dims=self.n_pos_dims, encoding_config=individual_encoding_config)
                                            for _ in range(self.partition_size ** self.n_pos_dims)])    
    
    def forward(self, coords:torch.Tensor):
        with torch.no_grad():
            coords = coords.contiguous().to(torch.float32)

            enc_idx = coords * torch.tensor([self.partition_size, self.partition_size, self.partition_size], device=coords.device).float()
            enc_idx = enc_idx.long()
            
            # TODO: this part should revise for 2D image
            x_idx = enc_idx[:,0].clamp(min=0, max=self.partition_size - 1)
            y_idx = enc_idx[:,1].clamp(min=0, max=self.partition_size - 1)
            z_idx = enc_idx[:,2].clamp(min=0, max=self.partition_size - 1)
            flat_idx = x_idx * self.partition_size ** 2 + y_idx * self.partition_size + z_idx
        
        # Initialize the tensor for encoded results
        encoded_results = torch.zeros(coords.shape[0], self.n_output_dims, device=coords.device).float()
        
        # Group inputs by partition index and send each group into each encoder respectively
        for i in range(self.partition_size ** self.n_pos_dims):
            group_idx = torch.nonzero(flat_idx == i).squeeze()
            encoded_results[group_idx] = self.encoders[i](coords[group_idx])

        # combine the encoded results together
        return encoded_results
