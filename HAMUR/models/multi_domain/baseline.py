import torch
import torch.nn as nn
from ...basic.layers import EmbeddingLayer


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
        elif act_name.lower() == "leakyrelu":
            act_layer = nn.LeakyReLU(0.1)
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError
    return act_layer


class MLP(nn.Module):
    """Multi Layer Perceptron Module, it is the most widely used module for 
    learning feature. Note we default add `BatchNorm1d` and `Activation` 
    `Dropout` for each `Linear` Module.

    Args:
        input dim (int): input size of the first Linear Layer.
        output_layer (bool): whether this MLP module is the output layer. If `True`, then append one Linear(*,1) module. 
        dims (list): output size of Linear Layer (default=[]).
        dropout (float): probability of an element to be zeroed (default = 0.5).
        activation (str): the activation function, support `[sigmoid, relu, prelu, dice, softmax]` (default='relu').

    Shape:
        - Input: `(batch_size, input_dim)`
        - Output: `(batch_size, 1)` or `(batch_size, dims[-1])`
    """

    def __init__(self, input_dim, output_layer=True, dims=None, dropout=0, activation="relu"):
        super().__init__()
        if dims is None:
            dims = []
        layers = list()
        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(activation_layer(activation))
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# class EmbeddingLayer(nn.Module):
#     """General Embedding Layer.
#     We save all the feature embeddings in embed_dict: `{feature_name : embedding table}`.

    
#     Args:
#         features (list): the list of `Feature Class`. It is means all the features which we want to create a embedding table.

#     Shape:
#         - Input: 
#             x (dict): {feature_name: feature_value}, sequence feature value is a 2D tensor with shape:`(batch_size, seq_len)`,\
#                       sparse/dense feature value is a 1D tensor with shape `(batch_size)`.
#             features (list): the list of `Feature Class`. It is means the current features which we want to do embedding lookup.
#             squeeze_dim (bool): whether to squeeze dim of output (default = `False`).
#         - Output: 
#             - if input Dense: `(batch_size, num_features_dense)`.
#             - if input Sparse: `(batch_size, num_features, embed_dim)` or  `(batch_size, num_features * embed_dim)`.
#             - if input Sequence: same with input sparse or `(batch_size, num_features_seq, seq_length, embed_dim)` when `pooling=="concat"`.
#             - if input Dense and Sparse/Sequence: `(batch_size, num_features_sparse * embed_dim)`. Note we must squeeze_dim for concat dense value with sparse embedding.
#     """

#     def __init__(self, features):
#         super().__init__()
#         self.features = features
#         self.embed_dict = nn.ModuleDict()
#         self.n_dense = 0

#         # for fea in features:
#         #     if fea.name in self.embed_dict:  #exist
#         #         continue
#         #     if isinstance(fea, SparseFeature) and fea.shared_with == None:
#         #         self.embed_dict[fea.name] = fea.get_embedding_layer()
#         #     elif isinstance(fea, SequenceFeature) and fea.shared_with == None:
#         #         self.embed_dict[fea.name] = fea.get_embedding_layer()
#         #     elif isinstance(fea, DenseFeature):
#         #         self.n_dense += 1

#     def forward(self, x, features, squeeze_dim=False):
#         sparse_emb, dense_values = [], []
#         sparse_exists, dense_exists = False, False
#         emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]
#         for fea in features:
#             if isinstance(fea, SparseFeature):
#                 if fea.shared_with == None:
#                     sparse_emb.append(self.embed_dict[fea.name](x[fea.name].long()).unsqueeze(1))
#                 else:
#                     sparse_emb.append(self.embed_dict[fea.shared_with](x[fea.name].long()).unsqueeze(1))
#             elif isinstance(fea, SequenceFeature):
#                 if fea.pooling == "sum":
#                     pooling_layer = SumPooling()
#                 elif fea.pooling == "mean":
#                     pooling_layer = AveragePooling()
#                 elif fea.pooling == "concat":
#                     pooling_layer = ConcatPooling()
#                 else:
#                     raise ValueError("Sequence pooling method supports only pooling in %s, got %s." %
#                                      (["sum", "mean"], fea.pooling))
#                 fea_mask = InputMask()(x, fea)
#                 if fea.shared_with == None:
#                     sparse_emb.append(pooling_layer(self.embed_dict[fea.name](x[fea.name].long()), fea_mask).unsqueeze(1))
#                 else:
#                     sparse_emb.append(pooling_layer(self.embed_dict[fea.shared_with](x[fea.name].long()), fea_mask).unsqueeze(1))  #shared specific sparse feature embedding
#             else:
#                 dense_values.append(x[fea.name].float().unsqueeze(1))  #.unsqueeze(1).unsqueeze(1)

#         if len(dense_values) > 0:
#             dense_exists = True
#             dense_values = torch.cat(dense_values, dim=1)
#         if len(sparse_emb) > 0:
#             sparse_exists = True
#             sparse_emb = torch.cat(sparse_emb, dim=1)  #[batch_size, num_features, embed_dim]

#         if squeeze_dim:  #Note: if the emb_dim of sparse features is different, we must squeeze_dim
#             if dense_exists and not sparse_exists:  #only input dense features
#                 return dense_values
#             elif not dense_exists and sparse_exists:
#                 return sparse_emb.flatten(start_dim=1)  #squeeze dim to : [batch_size, num_features*embed_dim]
#             elif dense_exists and sparse_exists:
#                 return torch.cat((sparse_emb.flatten(start_dim=1), dense_values),
#                                  dim=1)  #concat dense value with sparse embedding
#             else:
#                 raise ValueError("The input features can note be empty")
#         else:
#             if sparse_exists:
#                 return sparse_emb  #[batch_size, num_features, embed_dim]
#             else:
#                 raise ValueError(
#                     "If keep the original shape:[batch_size, num_features, embed_dim], expected %s in feature list, got %s" %
#                     ("SparseFeatures", features))

class PLE(nn.Module):
    """Progressive Layered Extraction model.

    Args:
        features (list): the list of `Feature Class`, training by the expert and tower module.
        domain_num (int): domain numbers.
        n_level (int): the  number of CGC layer.
        n_expert_specific (int): the number of task-specific expert net.
        n_expert_shared (int): the number of task-shared expert net.
        expert_params (dict): the params of all the expert module, keys include:`{"dims":list, "activation":str, "dropout":float}.
        tower_params_list (list): the list of tower params dict, the keys same as expert_params.
    """

    def __init__(self, features, domain_num, task_num, n_level=1, n_expert_specific=2, n_expert_shared=1, expert_params={'dims':[16]},
                 tower_params={"dims": [8]}):
        super().__init__()
        self.features = features
        self.domain_num = domain_num
        self.task_num = task_num
        self.n_level = n_level
        self.input_dims = sum([fea.embed_dim for fea in features])
        self.embedding = EmbeddingLayer(features)
        self.cgc_layers = nn.ModuleList(
            CGC(i + 1, n_level, self.domain_num*self.task_num, n_expert_specific, n_expert_shared, self.input_dims, expert_params)
            for i in range(n_level))
        self.towers = nn.ModuleList(
            MLP(expert_params["dims"][-1], output_layer=True, **tower_params) for i in range(self.domain_num*self.task_num))
        # self.predict_layers = nn.ModuleList(PredictionLayer(task_type) for task_type in task_types)

    def forward(self, x):
        # domain_id = x["domain_indicator"].clone().detach()
        domain_id = x[-1, :].clone().detach()
        embed_x = self.embedding(x, self.features, squeeze_dim=True)  #[batch_size, input_dims]
        ple_inputs = [embed_x] * (self.domain_num*self.task_num + 1)
        ple_outs = []
        for i in range(self.n_level):
            ple_outs = self.cgc_layers[i](ple_inputs)  #ple_outs[i]: [batch_size, expert_dims[-1]]
            ple_inputs = ple_outs

        ys = []
        mask = []
        d =0
        for ple_out, tower in zip(ple_outs, self.towers):
            domain_mask = (domain_id == d)
            mask.append(domain_mask.cpu())
            d+=1

            tower_out = tower(ple_out)  #[batch_size, 1]
            y = torch.sigmoid(tower_out)  #logit -> proba
            ys.append(y.squeeze(1))
        result = torch.zeros((len(ys[0]), self.task_num))
        for t in range(self.task_num):
            for d in range(self.domain_num):
                result[:, t] = torch.where(mask[d], ys[t*self.domain_num+d].cpu(), result[:, t])
        return result
        


class CGC(nn.Module):
    """Customized Gate Control (CGC) Model mentioned in PLE paper.

    Args:
        cur_level (int): the current level of CGC in PLE.
        n_level (int): the  number of CGC layer.
        domain_num (int): the number of domain_num.
        n_expert_specific (int): the number of task-specific expert net.
        n_expert_shared (int): the number of task-shared expert net.
        input_dims (int): the input dims of the xpert module in current CGC layer.
        expert_params (dict): the params of all the expert module, keys include:`{"dims":list, "activation":str, "dropout":float}.
    """

    def __init__(self, cur_level, n_level, domain_num, n_expert_specific, n_expert_shared, input_dims, expert_params):
        super().__init__()
        self.cur_level = cur_level  # the CGC level of PLE
        self.n_level = n_level
        self.domain_num = domain_num
        self.n_expert_specific = n_expert_specific
        self.n_expert_shared = n_expert_shared
        self.n_expert_all = n_expert_specific * self.domain_num + n_expert_shared
        input_dims = input_dims if cur_level == 1 else expert_params["dims"][
            -1]  #the first layer expert dim is the input data dim other expert dim
        self.experts_specific = nn.ModuleList(
            MLP(input_dims, output_layer=False, **expert_params) for _ in range(self.domain_num * self.n_expert_specific))
        self.experts_shared = nn.ModuleList(
            MLP(input_dims, output_layer=False, **expert_params) for _ in range(self.n_expert_shared))
        self.gates_specific = nn.ModuleList(
            MLP(
                input_dims, **{
                    "dims": [self.n_expert_specific + self.n_expert_shared],
                    "activation": "softmax",
                    "output_layer": False
                }) for _ in range(self.domain_num))  #n_gate_specific = n_task
        if cur_level < n_level:
            self.gate_shared = MLP(input_dims, **{
                "dims": [self.n_expert_all],
                "activation": "softmax",
                "output_layer": False
            })  #n_gate_specific = n_task

    def forward(self, x_list):
        expert_specific_outs = []  #expert_out[i]: [batch_size, 1, expert_dims[-1]]
        for i in range(self.domain_num):
            expert_specific_outs.extend([
                expert(x_list[i]).unsqueeze(1)
                for expert in self.experts_specific[i * self.n_expert_specific:(i + 1) * self.n_expert_specific]
            ])
        expert_shared_outs = [expert(x_list[-1]).unsqueeze(1) for expert in self.experts_shared
                             ]  #x_list[-1]: the input for shared experts
        gate_specific_outs = [gate(x_list[i]).unsqueeze(-1) for i, gate in enumerate(self.gates_specific)
                             ]  #gate_out[i]: [batch_size, n_expert_specific+n_expert_shared, 1]
        cgc_outs = []
        for i, gate_out in enumerate(gate_specific_outs):
            cur_expert_list = expert_specific_outs[i * self.n_expert_specific:(i + 1) *
                                                   self.n_expert_specific] + expert_shared_outs
            expert_concat = torch.cat(cur_expert_list,
                                      dim=1)  #[batch_size, n_expert_specific+n_expert_shared, expert_dims[-1]]
            expert_weight = torch.mul(gate_out,
                                      expert_concat)  #[batch_size, n_expert_specific+n_expert_shared, expert_dims[-1]]
            expert_pooling = torch.sum(expert_weight, dim=1)  #[batch_size, expert_dims[-1]]
            cgc_outs.append(expert_pooling)  #length: n_task
        if self.cur_level < self.n_level:  #not the last layer
            gate_shared_out = self.gate_shared(x_list[-1]).unsqueeze(-1)  #[batch_size, n_expert_all, 1]
            expert_concat = torch.cat(expert_specific_outs + expert_shared_outs,
                                      dim=1)  #[batch_size, n_expert_all, expert_dims[-1]]
            expert_weight = torch.mul(gate_shared_out, expert_concat)  #[batch_size, n_expert_all, expert_dims[-1]]
            expert_pooling = torch.sum(expert_weight, dim=1)  #[batch_size, expert_dims[-1]]
            cgc_outs.append(expert_pooling)  #length: n_task+1

        return cgc_outs


