import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


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

class HashEmbedderNative(nn.Module):
    def __init__(self, n_pos_dims=2,
                 # encoder parameters
                #  n_levels=16, 
                #  n_features_per_level=4,
                #  log2_hashmap_size=19,
                #  base_resolution=16,
                #  per_level_scale=2.0):
                encoding_config={}):
        super(HashEmbedderNative, self).__init__()

        # only support 2 or 3 dimensional inputs for now
        assert (n_pos_dims == 2 or n_pos_dims == 3)

        self.n_pos_dims = n_pos_dims

        encoding_config["otype"]="Grid"
        encoding_config["type"]="Hash"

        self.encoding_config = encoding_config

        self.n_levels = encoding_config["n_levels"]
        self.n_features_per_level = encoding_config["n_features_per_level"]
        self.log2_hashmap_size = encoding_config["log2_hashmap_size"]
        self.base_resolution = encoding_config["base_resolution"]
        self.per_level_scale = encoding_config["per_level_scale"]

        self.n_output_dims = self.n_levels * self.n_features_per_level
        # instead concatenate feature vectors, try to multiply them 
        # self.n_output_dims = self.n_features_per_level

        embedding_offsets = []
        embedding_lengths = []
        offset = 0
        for i in range(self.n_levels):
            scale = self.grid_scale(i, self.per_level_scale, self.base_resolution)
            resolution = self.grid_resolution(scale)
            length = resolution ** n_pos_dims
            length = (length + 8 - 1) // 8 * 8  # Make sure memory accesses will be aligned
            length = min(length, 1 << self.log2_hashmap_size)
            embedding_offsets.append(offset)
            embedding_lengths.append(length)
            offset += length
        self.embedding_offsets = embedding_offsets
        self.embedding_lengths = embedding_lengths

        # https://github.com/NVlabs/tiny-cuda-nn/blob/v1.6/include/tiny-cuda-nn/encodings/grid.h#L1355
        scale = 1.0
        # create parameters for this model
        self.params = nn.Parameter(data=torch.zeros(offset * self.n_features_per_level, dtype=torch.float32))
        # register these parameters to this model (nn.Module)
        self.register_parameter("params", self.params)
        # initialize these parameters
        nn.init.uniform_(self.params, -1e-4 * scale, 1e-4 * scale)

    def increase_embedding_size_by_two(self):
        new_log2_hashmap_size = self.log2_hashmap_size + 1
        # exceed increasing limit, return immediately
        if new_log2_hashmap_size > 24:
            return
        
        new_embedding_offsets = []
        new_embedding_lengths = []
        offset = 0
        new_params = []
        for i in range(self.n_levels):
            scale = self.grid_scale(i, self.per_level_scale, self.base_resolution)
            resolution = self.grid_resolution(scale)
            length = resolution ** self.n_pos_dims
            length = (length + 8 - 1) // 8 * 8  # Make sure memory accesses will be aligned
            # case 1: dense grid -> dense grid (just copy the table in that level)
            if length <= (1 << self.log2_hashmap_size):
                new_params.append(self.get_params()[self.embedding_offsets[i] * self.n_features_per_level:
                    self.embedding_offsets[i] * self.n_features_per_level + self.embedding_lengths[i] * self.n_features_per_level])
                # use dense grid size as length
                length = length
            # case 2: hash grid -> dense grid (convert hash grid to dense grid)
            elif length > (1 << self.log2_hashmap_size) and length <= (1 << new_log2_hashmap_size):
                HASH = coherent_prime_hash # TCNN provides multiple hash functions
                # traverse all grid points on dense grid and
                # hash them to query the weights in original hash table
                # and then copy those queried weights to new hash table
                with torch.no_grad():
                    # create dense grid points
                    x = torch.arange(resolution, dtype=torch.float32)
                    y = torch.arange(resolution, dtype=torch.float32)
                    z = torch.arange(resolution, dtype=torch.float32)
                    zv, yv, xv = torch.meshgrid([z, y, x])
                    xyz = torch.stack((zv.flatten(), yv.flatten(), xv.flatten())).t()
                    xyz = xyz[:, [2, 1, 0]]
                    # xyz = xyz.to("xpu")
                    
                # hash dense grid points (xyz) to get hashed indices
                hashed_indices = HASH(xyz) % self.embedding_lengths[i]
                # offset indices to the head of each feature vector (because each vector has n_features_per_level dimensions)
                hashed_indices = hashed_indices * self.n_features_per_level
                # offset indices to the begining of that level
                hashed_indices = hashed_indices + self.embedding_offsets[i] * self.n_features_per_level
                # expand hashed indices to access all elements in each feature vector
                hashed_indices = torch.stack([hashed_indices + ith for ith in range(self.n_features_per_level)], dim=1)
                hashed_indices = hashed_indices.view(-1)
                hashed_indices = hashed_indices.to("xpu")
                # use indices to query the original hash table and store them as the new table
                # must cast hashed_indices to int64 to avoid error (don't know why now)
                # also need to pad new params if xyz.shape is not aligned with length
                old_params = self.get_params()
                new_params.append(torch.cat([old_params[hashed_indices.to(torch.int64)], 
                                             torch.zeros((length - xyz.shape[0]) * self.n_features_per_level,
                                                         dtype=old_params.dtype, device=old_params.device)], dim=0))
                # use dense grid size as length
                length = length
            # case 3: hash grid -> hash grid (duplicate original hash table once)
            elif length > (1 << new_log2_hashmap_size):
                # duplicate original hash table once
                new_params.append(self.get_params()[self.embedding_offsets[i] * self.n_features_per_level:
                    self.embedding_offsets[i] * self.n_features_per_level + self.embedding_lengths[i] * self.n_features_per_level])
                new_params.append(self.get_params()[self.embedding_offsets[i] * self.n_features_per_level:
                    self.embedding_offsets[i] * self.n_features_per_level + self.embedding_lengths[i] * self.n_features_per_level])
                # use new hash grid size as length
                length = (1 << new_log2_hashmap_size)
            new_embedding_offsets.append(offset)
            new_embedding_lengths.append(length)
            offset += length
            
        # update new parameters in class
        self.embedding_offsets = new_embedding_offsets
        self.embedding_lengths = new_embedding_lengths
        self.log2_hashmap_size = new_log2_hashmap_size
        
        new_params = torch.cat(new_params, dim=0)
        self.params = nn.Parameter(data=new_params)
        # register these parameters to this model (nn.Module)
        self.register_parameter("params", self.params)

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.load_state_dict({ 'params': params })

    # def access_embeddings(self, level:int, inputs:torch.Tensor):
    #     offset = self.embedding_offsets[level] * self.n_features_per_level
    #     length = self.embedding_lengths[level] * self.n_features_per_level
    #     weights = self.params[offset:offset+length].reshape((-1, self.n_features_per_level))
    #     return F.embedding(inputs, weights)

    # def compute_resolution(self, i:int):
    #     scale = self.grid_scale(i, self.per_level_scale, self.base_resolution)
    #     return self.grid_resolution(scale)

    @staticmethod
    @torch.no_grad()
    def trilinear_interp_weights(weights):
        c0 = (1-weights[...,0]) * (1-weights[...,1]) * (1-weights[...,2]) # c00 c0
        c1 = (1-weights[...,0]) * (1-weights[...,1]) *    weights[...,2]  # c01 c1
        c2 = (1-weights[...,0]) *    weights[...,1]  * (1-weights[...,2]) # c10 c0
        c3 = (1-weights[...,0]) *    weights[...,1]  *    weights[...,2]  # c11 c1
        c4 =    weights[...,0]  * (1-weights[...,1]) * (1-weights[...,2]) # c00 c0
        c5 =    weights[...,0]  * (1-weights[...,1]) *    weights[...,2]  # c01 c1
        c6 =    weights[...,0]  *    weights[...,1]  * (1-weights[...,2]) # c10 c0
        c7 =    weights[...,0]  *    weights[...,1]  *    weights[...,2]  # c11 c1
        return torch.stack([c0,c1,c2,c3,c4,c5,c6,c7], dim=-1)
    
    @staticmethod
    @torch.no_grad()
    def bilinear_interp_weights(weights):
        # shape of weights: [Batch size, 16(number of levels), 2(weights along different axes)]
        c0 = (1-weights[...,0]) * (1-weights[...,1]) # c00 c0
        c1 = (1-weights[...,0]) *    weights[...,1]  # c01 c1
        c2 =    weights[...,0]  * (1-weights[...,1]) # c10 c0
        c3 =    weights[...,0]  *    weights[...,1]  # c11 c1
        # shape of torch.stack([c0,c1,c2,c3], dim=-1): [Batch size, 16(number of levels), 4(number of closest grid points)]
        # return coffecients of each grid point in interpolation
        return torch.stack([c0,c1,c2,c3], dim=-1)

    # @staticmethod
    # def trilinear_interp(embedds, weights):
    #     c00 = embedds[:,0]*(1-weights[:,0][:,None]) + embedds[:,4]*weights[:,0][:,None] # y0z0
    #     c01 = embedds[:,1]*(1-weights[:,0][:,None]) + embedds[:,5]*weights[:,0][:,None] # y0z1
    #     c10 = embedds[:,2]*(1-weights[:,0][:,None]) + embedds[:,6]*weights[:,0][:,None] # y1z0
    #     c11 = embedds[:,3]*(1-weights[:,0][:,None]) + embedds[:,7]*weights[:,0][:,None] # y1z1
    #     c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]        # z0
    #     c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]        # z1
    #     c  = c0 *(1-weights[:,2][:,None]) + c1 *weights[:,2][:,None]
    #     return c

    @staticmethod
    @torch.no_grad()
    def grid_scale(level:int, per_level_scale:float, base_resolution:float):
        return np.power(np.float32(2), np.float32(level) * np.log2(np.float32(per_level_scale))) * np.float32(base_resolution) - np.float32(1.0)

    @staticmethod
    @torch.no_grad()
    def grid_resolution(scale:float):
        return np.int32(np.ceil(np.float32(scale))) + 1

    # https://github.com/NVlabs/tiny-cuda-nn/blob/v1.6/include/tiny-cuda-nn/common_device.h#L403
    # @staticmethod
    @torch.no_grad()
    def grid_indices(self, scale:int, coords:torch.Tensor):
        positions = (coords * scale + 0.5).to(torch.float32)
        indices = torch.floor(positions).to(torch.int32) # shape => [B, 2] or [B, 3]
        positions = positions - indices # fractional part
        if self.n_pos_dims == 2:
            # fix to support operation in vmap
            offsets = torch.tensor([
                [0,0],[0,1],[1,0],[1,1]
            ], device=coords.device, dtype=torch.int32)
            # offsets = coords.new_tensor([
            #     [0,0],[0,1],[1,0],[1,1]
            # ], dtype=torch.int32) # shape => [4, 3]
        elif self.n_pos_dims == 3:
            # fix to support operation in vmap
            offsets = torch.tensor([
                [0,0,0],[0,0,1],[0,1,0],[0,1,1],
                [1,0,0],[1,0,1],[1,1,0],[1,1,1]
            ], device=coords.device, dtype=torch.int32)
            # offsets = coords.new_tensor([
            #     [0,0,0],[0,0,1],[0,1,0],[0,1,1],
            #     [1,0,0],[1,0,1],[1,1,0],[1,1,1]
            # ], dtype=torch.int32) # shape => [8, 3]
        # Calculate indices to store closest 4 or 8 grid points for this coordinates
        # shape for dim 2: [Batch size, 4, 2] = [Batch size, 1, 2] + [1, 4, 2]
        # shape for dim 3: [Batch size, 8, 3] = [Batch size, 1, 3] + [1, 8, 3]
        # Search Broadcasting in pytorch for more details
        indices = indices.unsqueeze(-2) + offsets.unsqueeze(0)
        return indices, positions

    # @staticmethod
    @torch.no_grad()
    def hash_it(self, hashmap_size:int, resolution:int, indices:torch.Tensor):
        '''It is possible that the coordinate is larger than the domain size.'''
        HASH = coherent_prime_hash # TCNN provides multiple hash functions
        assert (indices.shape[-1] == 2 or indices.shape[-1] == 3)
        resolution = np.uint32(resolution)
        stride = np.uint32(1)
        output = torch.zeros_like(indices[...,0])
        for dim in range(self.n_pos_dims):
            output += indices[...,dim] * stride
            stride *= resolution  # --> expecting integer overflow in scalar multiply
            if stride > hashmap_size: break
        if hashmap_size < stride: output = HASH(indices)
        return output % hashmap_size

    @torch.no_grad()
    def access(self, coords:torch.Tensor, level:int):
        scale = self.grid_scale(level, self.per_level_scale, self.base_resolution)
        resolution = self.grid_resolution(scale)
        hashmap_size = self.embedding_lengths[level]
        indices, fractions = self.grid_indices(scale, coords)
        offsets = self.hash_it(hashmap_size, resolution, indices)
        return offsets, fractions

    def forward(self, coords:torch.Tensor):
        coords = coords.contiguous().to(torch.float32)

        # out = []
        # for i in range(self.n_levels):
        #     offsets, fractions = self.access(coords, i)
        #     h = self.access_embeddings(i, offsets).to(torch.float32)
        #     o = self.trilinear_interp(h, fractions)
        #     out.append(o)
        # out = torch.cat(out, dim=-1)
        # return out

        with torch.no_grad():
            weights_arr = []
            offsets_arr = []
            for i in range(self.n_levels):
                # shape of offsets: [Batch size, 4]
                offsets, weights = self.access(coords, i)
                # might need to offset hashed indices, so it can access the hashmap of its level
                offsets += self.embedding_offsets[i]
                # shape of offsets.unsqueeze(1): [Batch size, 1, 4(number of closest grid points)]
                offsets_arr.append(offsets.unsqueeze(1))
                weights_arr.append(weights.unsqueeze(1))
            # shape of offsets_arr: [Batch size, 16(number of levels), 4(number of closest grid points)]
            offsets_arr = torch.cat(offsets_arr, dim=1)
            weights_arr = torch.cat(weights_arr, dim=1)
            if self.n_pos_dims == 2:
                # shape of weights_arr: [Batch size, 16(number of levels), 4(number of closest grid points)]
                weights_arr = self.bilinear_interp_weights(weights_arr)
            elif self.n_pos_dims == 3:
                weights_arr = self.trilinear_interp_weights(weights_arr)     

        # shape of embeds_arr: [Batch size, 16, 4, 2(number of dimensions of feature vectors)]
        # query hash table to get each grid points' corresponding feature vectors and store in embeds_arr
        embeds_arr = F.embedding(offsets_arr, self.params.reshape((-1, self.n_features_per_level)))
        # shape of weights_arr.unsqueeze(-1): [Batch size, 16, 4, 1]
        # weights_arr.unsqueeze(-1) would broadcast (repeat the elements in last dim) to [Batch size, 16, 4, 2]
        # and then do element-wise multiplication with embeds_arr
        # (weights_arr.unsqueeze(-1) * embeds_arr) calculate the feature vectors on the input coordinates
        # shape of out: [Batch size, 16, 2]
        out = (weights_arr.unsqueeze(-1) * embeds_arr).sum(dim=-2)
        # instead concatenate feature vectors, try to multiply them 
        # out = out.prod(dim=-2)
        return out.reshape(-1, self.n_output_dims)

    def extra_repr(self):
        return f"hyperparams={self.encoding_config}"