import torch
import torch.nn as nn
import torch.nn.functional as F

def find_activation(activation):
    if activation == "None":
        return lambda x : x
    if activation == "ReLU":
        return lambda x : F.relu(x, inplace=True)
    elif activation == "Sigmoid":
        return F.sigmoid

class MLP_Native(torch.nn.Module):
    def __init__(self, n_input_dims,
                 n_output_dims=1,
                 network_config={}
                #  bias=False,
                #  n_hidden_layers=3,
                #  n_neurons=64,
                #  activation="ReLU",
                #  output_activation="None",
                #  feedback_alignment=False
                 ):
        super(MLP_Native, self).__init__()

        self.n_input_dims  = n_input_dims
        self.n_output_dims = n_output_dims

        # network_config = {
        #     "otype": "MLP_Native",
        #     "activation": activation,
        #     "output_activation": output_activation,
        #     "n_neurons": n_neurons,
        #     "n_hidden_layers": n_hidden_layers,
        #     "feedback_alignment": feedback_alignment
        # }

        self.n_hidden_layers = network_config["n_hidden_layers"]
        self.n_neurons = network_config["n_neurons"]
        # self.bias = bias

        self.network_config = network_config

        assert self.n_hidden_layers >= 0, "expect at least one hidden layer"

        # self.first = nn.Linear(self.n_input_dims, self.n_neurons, bias=self.bias)
        self.first = nn.Linear(self.n_input_dims, self.n_neurons)
        self.hidden = nn.ModuleList([
            # nn.Linear(self.n_neurons, self.n_neurons, bias=self.bias) for _ in range(self.n_hidden_layers-1)
            nn.Linear(self.n_neurons, self.n_neurons) for _ in range(self.n_hidden_layers-1)
        ])
        # self.last = nn.Linear(self.n_neurons, self.n_output_dims, bias=self.bias)
        self.last = nn.Linear(self.n_neurons, self.n_output_dims)

        self.activation = find_activation(network_config["activation"])
        self.output_activation = find_activation(network_config["output_activation"])

        # initialize weights
        # https://github.com/NVlabs/tiny-cuda-nn/blob/v1.6/src/fully_fused_mlp.cu#L882
        scale = 1.0
        nn.init.xavier_uniform_(self.first.weight, scale)
        for layer in self.hidden:
            nn.init.xavier_uniform_(layer.weight, scale)
        nn.init.xavier_uniform_(self.last.weight, scale)

    def forward(self, x):
        x = self.activation(self.first(x))
        for layer in self.hidden:
            x = self.activation(layer(x))
        return self.output_activation(self.last(x))

    def get_params(self):
        params = []
        for k, w in self.state_dict().items():
            params.append(w.flatten())
        return torch.cat(params)

    def get_transpose_params(self):
        params = []
        for k, w in self.state_dict().items():
            params.append(w.transpose(0,1).flatten())
        return torch.cat(params)

    def set_params(self, params):
        offset = 0
        states = {}
        for k, w in self.state_dict().items():
            states[k] = params[offset:offset+w.numel()].reshape(w.shape)
            offset += w.numel()
        self.load_state_dict(states)