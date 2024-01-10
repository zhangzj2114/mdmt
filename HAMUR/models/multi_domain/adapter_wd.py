

import torch
from torch import nn
from torch.nn.parameter import Parameter
from ...basic.layers import LR, MLP, EmbeddingLayer


class Dice(nn.Module):
    """The Dice activation function mentioned in the `DIN paper
    https://arxiv.org/abs/1706.06978`
    """

    def __init__(self, epsilon=1e-3):
        super(Dice, self).__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor):
        # x: N * num_neurons
        avg = x.mean(dim=1)  # N
        avg = avg.unsqueeze(dim=1)  # N * 1
        var = torch.pow(x - avg, 2) + self.epsilon  # N * num_neurons
        var = var.sum(dim=1).unsqueeze(dim=1)  # N * 1

        ps = (x - avg) / torch.sqrt(var)  # N * 1

        ps = nn.Sigmoid()(ps)  # N * 1
        return ps * x + (1 - ps) * self.alpha * x


def activation_layer(act_name):
    """Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function
    
    Returns:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'dice':
            act_layer = Dice()
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
        elif act_name.lower() == "softmax":
            act_layer = nn.Softmax(dim=1)
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError
    return act_layer


class MLP_adap_2_layer_1_adp(nn.Module):
    # 2 layers MLP with 1 adapter cells
    def __init__(self, features, domain_num, fcn_dims, hyper_dims, k ):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1
        self.fcn_dim = [self.input_dim] + fcn_dims
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)

        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")

        self.layer_list = nn.ModuleList()
        for d in range(domain_num):
            domain_specific = nn.ModuleList()
            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())
            domain_specific.append(nn.Linear(self.fcn_dim[2], 1))

            self.layer_list.append(domain_specific)

        # instance matrix initiation
        self.k = k
        self.u = nn.ParameterList()
        self.v = nn.ParameterList()

        # u,v initiation
        self.u.append(Parameter(torch.ones((self.fcn_dim[-1], self.k)), requires_grad=True))
        self.u.append(Parameter(torch.ones((32, self.k)), requires_grad=True))

        self.v.append(Parameter(torch.ones((self.k, 32)), requires_grad=True))
        self.v.append(Parameter(torch.ones((self.k, self.fcn_dim[-1])), requires_grad=True))

        # hypernwt work
        hyper_dims += [self.k * self.k]
        input_dim = self.input_dim
        hyper_layers = []
        for i_dim in hyper_dims:
            hyper_layers.append(nn.Linear(input_dim, i_dim))
            hyper_layers.append(nn.BatchNorm1d(i_dim))
            hyper_layers.append(nn.ReLU())
            hyper_layers.append(nn.Dropout(p=0))
            input_dim = i_dim
        self.hyper_net = nn.Sequential(*hyper_layers)

        # Adapter parameters
        self.b_list = nn.ParameterList()
        self.b_list.append(Parameter(torch.zeros((32)), requires_grad=True))
        self.b_list.append(Parameter(torch.zeros((self.fcn_dim[-1])), requires_grad=True))

        self.gamma1 = nn.Parameter(torch.ones(self.fcn_dim[-1]))
        self.bias1 = nn.Parameter(torch.zeros(self.fcn_dim[-1]))
        self.eps = 1e-5

    def forward(self, x):

        domain_id = x[-1, :].clone().detach()
        # domain_id = x["domain_indicator"].clone().detach()

        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        mask = []

        out_l = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = emb

            # hyper-network output
            hyper_out_full = self.hyper_net(domain_input)  # B * (k * k)
            # Representation matrix
            hyper_out = hyper_out_full.reshape(-1, self.k, self.k)  # B * k * k

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear

            domain_input = model_list[1](domain_input)  # bn

            domain_input = model_list[2](domain_input)  # relu



            domain_input = model_list[3](domain_input)  # linear

            domain_input = model_list[4](domain_input)  # bn

            domain_input = model_list[5](domain_input)  # relu

            # Adapter cell
            # Adapter layer-1: Down projection
            w1 = torch.einsum('mi,bij,jn->bmn', self.u[0], hyper_out, self.v[0])
            b1 = self.b_list[0]
            tmp_out = torch.einsum('bf,bfj->bj', domain_input, w1)
            tmp_out += b1

            # Adapter layer-2: Non-linear
            tmp_out = self.sig(tmp_out)

            # Adapter layer-3: Up projection
            w2 = torch.einsum('mi,bij,jn->bmn', self.u[1], hyper_out, self.v[1])
            b2 = self.b_list[1]
            tmp_out = torch.einsum('bf,bfj->bj', tmp_out, w2)
            tmp_out += b2

            # Adapter layer-4: Domain norm
            mean = tmp_out.mean(dim=0)
            var = tmp_out.var(dim=0)
            x_norm = (tmp_out - mean) / torch.sqrt(var + self.eps)
            out = self.gamma1 * x_norm + self.bias1

            # Adapter: Short-cut
            domain_input = out + domain_input

            domain_input = model_list[6](domain_input)
            domain_input = self.sig(domain_input)

            out_l.append(domain_input)

        final = torch.zeros_like(out_l[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out_l[d], final)

        return final


class WideDeep_MD(torch.nn.Module):
    """
    Multi-domain Wide & Deep Learning model.
    """

    def __init__(self, wide_features, num_domains, deep_features, mlp_params):
        super(WideDeep_MD, self).__init__()
        self.num_domains = num_domains
        self.wide_features = wide_features
        self.deep_features = deep_features
        self.wide_dims = sum([fea.embed_dim for fea in wide_features])
        self.deep_dims = sum([fea.embed_dim for fea in deep_features])
        self.linear = LR(self.wide_dims)
        self.embedding = EmbeddingLayer(wide_features + deep_features)
        self.mlp = MLP(self.deep_dims, **mlp_params)

    def forward(self, x):
        domain_id = x["domain_indicator"].clone().detach()

        input_wide = self.embedding(x, self.wide_features, squeeze_dim=True)  #[batch_size, wide_dims]
        input_deep = self.embedding(x, self.deep_features, squeeze_dim=True)  #[batch_size, deep_dims]

        # mask 存储每个batch中的domain id
        mask = []
        # out 存储
        out = []

        for d in range(self.num_domains):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input_wide = input_wide
            domain_input_deep = input_deep

            y_wide = self.linear(domain_input_wide)  #[batch_size, 1]
            y_deep = self.mlp(domain_input_deep)  #[batch_size, 1]
            y = y_wide + y_deep
            out.append(torch.sigmoid(y))

        final = torch.zeros_like(out[0])
        for d in range(self.num_domains):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)
        return final.squeeze(1)

class WideDeep_MD_adp(torch.nn.Module):
    """
    Multi-domain Wide & Deep Learning model with adapter.
    """

    def __init__(self, wide_features, num_domains, deep_features, k, mlp_params, hyper_dims):
        super(WideDeep_MD_adp, self).__init__()
        self.num_domains = num_domains
        self.wide_features = wide_features
        self.deep_features = deep_features
        self.wide_dims = sum([fea.embed_dim for fea in wide_features])
        self.deep_dims = sum([fea.embed_dim for fea in deep_features])
        self.linear = LR(self.wide_dims)
        self.embedding = EmbeddingLayer(wide_features + deep_features)
        self.mlp = MLP(self.deep_dims, **mlp_params, output_layer = False)
        self.mlp_final = LR(mlp_params["dims"][-1])



        # instance representation matrix initiation
        self.k = k

        # u,v initiation
        self.u = nn.ParameterList()
        self.v = nn.ParameterList()

        self.u.append(Parameter(torch.ones((mlp_params["dims"][-1], self.k)), requires_grad=True))
        self.u.append(Parameter(torch.ones((32, self.k)), requires_grad=True))

        self.v.append(Parameter(torch.ones((self.k, 32)), requires_grad=True))
        self.v.append(Parameter(torch.ones((self.k, mlp_params["dims"][-1])), requires_grad=True))

        # hyper-network initiation
        hyper_dims += [self.k * self.k]
        input_dim = self.wide_dims +self.deep_dims
        hyper_layers = []
        for i_dim in hyper_dims:
            hyper_layers.append(nn.Linear(input_dim, i_dim))
            hyper_layers.append(nn.BatchNorm1d(i_dim))
            hyper_layers.append(nn.ReLU())
            hyper_layers.append(nn.Dropout(p=0))
            input_dim = i_dim
        self.hyper_net = nn.Sequential(*hyper_layers)

        # adapter parameters initiation
        self.b_list = nn.ParameterList()
        self.b_list.append(Parameter(torch.zeros((32)), requires_grad=True))
        self.b_list.append(Parameter(torch.zeros(mlp_params["dims"][-1]), requires_grad=True))
        self.gamma1 = nn.Parameter(torch.ones(mlp_params["dims"][-1]))
        self.bias1 = nn.Parameter(torch.zeros(mlp_params["dims"][-1]))
        self.eps = 1e-5


    def forward(self, x):
        domain_id = x["domain_indicator"].clone().detach()

        input_wide = self.embedding(x, self.wide_features, squeeze_dim=True)  #[batch_size, wide_dims]
        input_deep = self.embedding(x, self.deep_features, squeeze_dim=True)  #[batch_size, deep_dims]

        hyper_out_full = self.hyper_net(torch.cat((input_wide,input_deep),dim=1))  # B * (k * k)
        hyper_out = hyper_out_full.reshape(-1, self.k, self.k)  # B * k * k

        # mask
        mask = []
        # out
        out_l = []

        for d in range(self.num_domains):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input_wide = input_wide
            domain_input_deep = input_deep

            y_wide = self.linear(domain_input_wide)  #[batch_size, 1]
            y_deep = self.mlp(domain_input_deep)  #[batch_size, f]

            # Adapter cell
            # Adapter layer1: Down projection
            w1 = torch.einsum('mi,bij,jn->bmn', self.u[0], hyper_out, self.v[0])
            b1 = self.b_list[0]
            tmp_out = torch.einsum('bf,bfj->bj', y_deep, w1)
            tmp_out += b1
            # Adapter layer2: Non-linear
            tmp_out = torch.sigmoid(tmp_out)
            # Adapter layer3: Up projection
            w2 = torch.einsum('mi,bij,jn->bmn', self.u[1], hyper_out, self.v[1])
            b2 = self.b_list[1]
            tmp_out = torch.einsum('bf,bfj->bj', tmp_out, w2)
            tmp_out += b2
            # Adapter layer4: Domain Norm
            mean = tmp_out.mean(dim=0)
            var = tmp_out.var(dim=0)
            x_norm = (tmp_out - mean) / torch.sqrt(var + self.eps)
            out = self.gamma1 * x_norm + self.bias1
            # Adapter: short-cut
            mlp_out = out + y_deep

            mlp_out = self.mlp_final(mlp_out) # linear

            y = y_wide + mlp_out
            out_l.append(torch.sigmoid(y))

        final = torch.zeros_like(out_l[0])
        for d in range(self.num_domains):
            final = torch.where(mask[d].unsqueeze(1), out_l[d], final)
        return final.squeeze(1)

