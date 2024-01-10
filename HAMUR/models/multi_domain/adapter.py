import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
from torch.nn.parameter import Parameter
from ...basic.layers import EmbeddingLayer, CrossNetwork, MLP, LR
from ...basic.activation import activation_layer

DOMAIN_SET = set([0, 1, 2, 3, 4])
class Mlp_7_Layer(nn.Module):
    # 7-layers Mlp model
    def __init__(self, features, domain_num, task_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims  
        self.domain_num = domain_num
        self.task_num = task_num
        self.embedding = EmbeddingLayer(features)

        self.sig = activation_layer("sigmoid")

        self.layer_list = nn.ModuleList()
        self.head_list = nn.ModuleList()
        for d in range(domain_num*task_num):
            self.layer_list.append(MLP_N(self.fcn_dim))
            self.head_list.append(nn.Sequential(
                nn.Linear(self.fcn_dim[-1], self.fcn_dim[-1]),
                # nn.BatchNorm1d(self.fcn_dim[-1]),
                nn.LayerNorm(self.fcn_dim[-1]),
                nn.ReLU(),
                nn.Linear(self.fcn_dim[-1], 1)
            ))

    def forward(self, x):

        domain_id = x[-1, :].clone().detach()
        assert set(domain_id.cpu().numpy().flatten()).issubset(DOMAIN_SET), set(domain_id.cpu().numpy().flatten())

        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        mask = []

        out = []
        for t in range(self.task_num):
            for d in range(self.domain_num):
                domain_mask = (domain_id == d)
                mask.append(domain_mask)
    
                domain_input = emb
    
                model_ = self.layer_list[d+self.task_num*t]
                head_ = self.head_list[d+self.task_num*t]

                domain_input = model_(domain_input)
                domain_input = head_(domain_input)
                domain_input = self.sig(domain_input)
                out.append(domain_input)

        final = torch.zeros_like(out[0])
        output = torch.zeros((len(out[0]), self.task_num))
        for t in range(self.task_num):
            for d in range(self.domain_num):
                final = torch.where(mask[d+self.task_num*t].unsqueeze(1), out[d+self.task_num*t], final)
            output[:, t] = final.squeeze(1)
        return output
        
class MLP_N(nn.Module):
    '''
    fcn_dim: list of dimensions of mlp layers, e.g., [1024, 512, 512, 256, 256, 64] 
    return [linear, bn1d, relu]*n
    '''
    def __init__(self, fcn_dim):
        super().__init__()
        self.fcn_dim = fcn_dim
        self.n = len(fcn_dim)
        
        self.domain_specific = nn.ModuleList()
        for (i) in range(self.n-1):
            self.domain_specific.append(nn.Linear(self.fcn_dim[i], self.fcn_dim[i+1]))
            # self.domain_specific.append(nn.BatchNorm1d(self.fcn_dim[i+1]))
            self.domain_specific.append(nn.LayerNorm(self.fcn_dim[i+1]))
            self.domain_specific.append(nn.ReLU())            
        # self.domain_specific.append(nn.Linear(self.fcn_dim[-1], 1))
        
    def forward(self, x):
        output = x
        for f in self.domain_specific:
            output = f(output)
        return output
        
class MMOE(nn.Module):
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims 
        self.domain_num = domain_num
        self.task_num = task_num
        self.expert_num = expert_num
        self.embedding = EmbeddingLayer(features)
        self.expert = nn.ModuleList()
        for d in range(expert_num):
            self.expert.append(MLP_N(self.fcn_dim))
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.input_dim, expert_num), torch.nn.Softmax(dim=1)) for i in range(domain_num*task_num)])
        
        self.tower = nn.ModuleList()
        for d in range(domain_num*task_num):
            domain_specific = nn.Sequential(
                nn.Linear(self.fcn_dim[-1], self.fcn_dim[-1]),
                nn.BatchNorm1d(self.fcn_dim[-1]),
                nn.ReLU(),
                nn.Linear(self.fcn_dim[-1], 1)
            )
            self.tower.append(domain_specific)
            
    def forward(self, x):
        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]
        gate_value = [self.gate[i](emb).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, expert_num]

        out = [] # batch_size, expert_num, embedding_size
        for i in range(self.expert_num):
            model_list = self.expert[i]
            domain_input = model_list(emb)
            out.append(domain_input)        
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, 1
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num*self.domain_num)]

        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num*self.domain_num)]
        output = torch.zeros((len(results[0]), len(results)))
        for i in range(len(results)):
            output[:, i] = results[i]

        domain_id = x[-1, :].clone().detach()
        assert set(domain_id.cpu().numpy().flatten()).issubset(DOMAIN_SET), set(domain_id.cpu().numpy().flatten())
        mask = []
        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask.cpu())

        result = torch.zeros((len(results[0]), self.task_num))
        for t in range(self.task_num):
            for d in range(self.domain_num):
                result[:, t] = torch.where(mask[d], output[:, t*self.domain_num+d], result[:, t])
        return result

class STAR(nn.Module):
    def __init__(self, features, domain_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims 
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)

        assert len(self.fcn_dim) > 3, f'too few layers assigned, must larger than 3. Star owns 3 layers, mmoe owns the rest.'
        self.star_dim = self.fcn_dim[:3]
        self.fcn_dim = self.fcn_dim[3:]
        self.skip_conn = MLP_N([self.star_dim[0], self.star_dim[2]]) 
        self.shared_weight = nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1]))
        self.shared_bias = nn.Parameter(torch.zeros(self.star_dim[1]))
        self.sig = activation_layer("sigmoid")

        # multi-domain fusion: star
        self.slot_weight = nn.ParameterList([nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1])) for i in range(self.domain_num)])
        self.slot_bias = nn.ParameterList([nn.Parameter(torch.zeros(self.star_dim[1])) for i in range(self.domain_num)])
        
        self.star_mlp = MLP_N([self.star_dim[1], self.star_dim[2]])

        self.main_mlp = MLP_N(self.fcn_dim)
        self.head = nn.Sequential(
            nn.Linear(self.fcn_dim[-1], self.fcn_dim[-1]),
            # nn.BatchNorm1d(self.fcn_dim[-1]),
            nn.LayerNorm(self.fcn_dim[-1]),
            nn.ReLU(),
            nn.Linear(self.fcn_dim[-1], 1)
        )
        
        # for m in ([self.shared_weight]+ self.slot_weight):
        torch.nn.init.xavier_uniform_(self.shared_weight.data)
        for m in (self.slot_weight):
            torch.nn.init.xavier_uniform_(m.data)

    def forward(self, x):
        _device = x.device
        domain_id = x[-1, :].clone().detach()
        assert set(domain_id.cpu().numpy().flatten()).issubset(DOMAIN_SET), set(domain_id.cpu().numpy().flatten())
        mask = []
        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask.cpu())

        input_emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        skip = self.skip_conn(input_emb)
        emb = torch.zeros((input_emb.shape[0], self.star_dim[1])).to(_device)
        for i, (_weight, _bias) in enumerate(zip(self.slot_weight, self.slot_bias)):
            _output = torch.matmul(input_emb, torch.multiply(_weight, self.shared_weight))+_bias+self.shared_bias
            emb = torch.where(mask[i].unsqueeze(1).to(_device), _output, emb)
        emb = self.star_mlp(emb)+skip
        emb = self.main_mlp(emb)
        emb = self.sig(self.head(emb))
        
        return emb
        

class MTMD(nn.Module):
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims 
        self.domain_num = domain_num
        self.task_num = task_num
        self.expert_num = expert_num
        self.embedding = EmbeddingLayer(features)

        assert len(self.fcn_dim) > 3, f'too few layers assigned, must larger than 3. Star owns 3 layers, mmoe owns the rest.'
        self.star_dim = self.fcn_dim[:3]
        self.fcn_dim = self.fcn_dim[3:]
        self.skip_conn = MLP_N([self.star_dim[0], self.star_dim[2]]) 
        self.shared_weight = nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1]))
        self.shared_bias = nn.Parameter(torch.zeros(self.star_dim[1]))

        # multi-domain fusion: star
        self.slot_weight = nn.ParameterList([nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1])) for i in range(self.domain_num)])
        self.slot_bias = nn.ParameterList([nn.Parameter(torch.zeros(self.star_dim[1])) for i in range(self.domain_num)])
        
        self.star_mlp = MLP_N([self.star_dim[1], self.star_dim[2]])
        
        # for m in ([self.shared_weight]+ self.slot_weight):
        torch.nn.init.xavier_uniform_(self.shared_weight.data)
        for m in (self.slot_weight):
            torch.nn.init.xavier_uniform_(m.data)

        # multi-task balance: mmoe
        self.expert = nn.ModuleList()
        for d in range(expert_num):
            self.expert.append(MLP_N(self.fcn_dim))
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], expert_num), torch.nn.Softmax(dim=1)) for i in range(domain_num*task_num)])
        
        self.tower = nn.ModuleList()
        for d in range(domain_num*task_num):
            domain_specific = nn.Sequential(
                nn.Linear(self.fcn_dim[-1], self.fcn_dim[-1]),
                nn.BatchNorm1d(self.fcn_dim[-1]),
                nn.ReLU(),
                nn.Linear(self.fcn_dim[-1], 1)
            )
            self.tower.append(domain_specific)
            
    def forward(self, x, test_flag=False):
        _device = x.device
        domain_id = x[-1, :].clone().detach()
        assert set(domain_id.cpu().numpy().flatten()).issubset(DOMAIN_SET), set(domain_id.cpu().numpy().flatten())
        mask = []
        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask.cpu())

        input_emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        # star: w_shared*w_specific(input_emb) -> mlp + skip
        skip = self.skip_conn(input_emb)
        emb = torch.zeros((input_emb.shape[0], self.star_dim[1])).to(_device)
        for i, (_weight, _bias) in enumerate(zip(self.slot_weight, self.slot_bias)):
            _output = torch.matmul(input_emb, torch.multiply(_weight, self.shared_weight))+_bias+self.shared_bias
            emb = torch.where(mask[i].unsqueeze(1).to(_device), _output, emb)
        emb = self.star_mlp(emb)+skip
        
        gate_value = [self.gate[i](emb.detach()).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, expert_num]
        # gate_value = [self.gate[i](emb).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, expert_num]

        # save test data gate values for visualization
        if test_flag:
            _gate_value_list = [i.detach().cpu().numpy() for i in gate_value]
            _gate_value_df = list()
            for i, g in enumerate(_gate_value_list):
                g = g.squeeze(1)
                df = pd.DataFrame(data=g, columns=[f'e{i}' for i in range(self.expert_num)])
                df['domain_task'] = i
                _gate_value_df.append(df)
            _gate_value_df = pd.concat(_gate_value_df)

        # mmoe: expert(emb)*gate(emb) -> tower(task_fea)
        out = [] # batch_size, expert_num, embedding_size
        for i in range(self.expert_num):
            model_list = self.expert[i]
            domain_input = model_list(emb)
            out.append(domain_input)        
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, 1
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num*self.domain_num)]

        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num*self.domain_num)]

        # save output in result [bs, task_num], all domains are integrated with bs length
        output = torch.zeros((len(results[0]), len(results)))
        for i in range(len(results)):
            output[:, i] = results[i]

        result = torch.zeros((len(results[0]), self.task_num))
        for t in range(self.task_num):
            for d in range(self.domain_num):
                result[:, t] = torch.where(mask[d], output[:, t*self.domain_num+d], result[:, t])
        if test_flag:
            return result, _gate_value_df
        else:
            return result 

class MTMD_ADV(nn.Module):
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims 
        self.domain_num = domain_num
        self.task_num = task_num
        self.expert_num = expert_num
        self.embedding = EmbeddingLayer(features)

        assert len(self.fcn_dim) > 3, f'too few layers assigned, must larger than 3. Star owns 3 layers, mmoe owns the rest.'
        self.star_dim = self.fcn_dim[:3]
        self.fcn_dim = self.fcn_dim[3:]
        self.skip_conn = MLP_N([self.star_dim[0], self.star_dim[2]]) 
        self.shared_weight = nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1]))
        self.shared_bias = nn.Parameter(torch.zeros(self.star_dim[1]))

        # multi-domain fusion: star
        self.slot_weight = nn.ParameterList([nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1])) for i in range(self.domain_num)])
        self.slot_bias = nn.ParameterList([nn.Parameter(torch.zeros(self.star_dim[1])) for i in range(self.domain_num)])
        
        self.star_mlp = MLP_N([self.star_dim[1], self.star_dim[2]])
        
        # for m in ([self.shared_weight]+ self.slot_weight):
        torch.nn.init.xavier_uniform_(self.shared_weight.data)
        for m in (self.slot_weight):
            torch.nn.init.xavier_uniform_(m.data)

        # multi-task balance: mmoe
        self.expert = nn.ModuleList()
        for d in range(expert_num):
            self.expert.append(MLP_N(self.fcn_dim))
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], expert_num), torch.nn.Softmax(dim=1)) for i in range(domain_num*task_num)])
        
        self.tower = nn.ModuleList()
        for d in range(domain_num*task_num):
            domain_specific = nn.Sequential(
                nn.Linear(self.fcn_dim[-1], self.fcn_dim[-1]),
                nn.BatchNorm1d(self.fcn_dim[-1]),
                nn.ReLU(),
                nn.Linear(self.fcn_dim[-1], 1)
            )
            self.tower.append(domain_specific)

        self.user_w = MLP_N([self.star_dim[0]] + [2 * self.star_dim[0]] + [self.domain_num*self.task_num])
            
    def forward(self, x, test_flag=False):
        _device = x.device
        domain_id = x[-1, :].clone().detach()
        assert set(domain_id.cpu().numpy().flatten()).issubset(DOMAIN_SET), set(domain_id.cpu().numpy().flatten())
        mask = []
        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask.cpu())

        input_emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]
        self.adv_weight = self.user_w(input_emb)

        # star: w_shared*w_specific(input_emb) -> mlp + skip
        skip = self.skip_conn(input_emb)
        emb = torch.zeros((input_emb.shape[0], self.star_dim[1])).to(_device)
        for i, (_weight, _bias) in enumerate(zip(self.slot_weight, self.slot_bias)):
            _output = torch.matmul(input_emb, torch.multiply(_weight, self.shared_weight))+_bias+self.shared_bias
            emb = torch.where(mask[i].unsqueeze(1).to(_device), _output, emb)
        emb = self.star_mlp(emb)+skip
        
        gate_value = [self.gate[i](emb.detach()).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, expert_num]
        # gate_value = [self.gate[i](emb).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, expert_num]

        # save test data gate values for visualization
        if test_flag:
            _gate_value_list = [i.detach().cpu().numpy() for i in gate_value]
            _gate_value_df = list()
            for i, g in enumerate(_gate_value_list):
                g = g.squeeze(1)
                df = pd.DataFrame(data=g, columns=[f'e{i}' for i in range(self.expert_num)])
                df['domain_task'] = i
                _gate_value_df.append(df)
            _gate_value_df = pd.concat(_gate_value_df)

        # mmoe: expert(emb)*gate(emb) -> tower(task_fea)
        out = [] # batch_size, expert_num, embedding_size
        for i in range(self.expert_num):
            model_list = self.expert[i]
            domain_input = model_list(emb)
            out.append(domain_input)        
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, 1
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num*self.domain_num)]

        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num*self.domain_num)]

        # save output in result [bs, task_num], all domains are integrated with bs length
        output = torch.zeros((len(results[0]), len(results)))
        for i in range(len(results)):
            output[:, i] = results[i]

        result = torch.zeros((len(results[0]), self.task_num))
        for t in range(self.task_num):
            for d in range(self.domain_num):
                result[:, t] = torch.where(mask[d], output[:, t*self.domain_num+d], result[:, t])
        if test_flag:
            return result, _gate_value_df
        else:
            return result 


class DCN_MMOE(torch.nn.Module):
    """
    multi-domain Deep & Cross Network and MMOE
    """

    def __init__(self, features, num_domains, task_num, expert_num, fcn_dims, n_cross_layers, mlp_params):
        super().__init__()
        self.fcn_dim = fcn_dims
        self.features = features
        self.dims = sum([fea.embed_dim for fea in features])
        self.domain_num = num_domains
        self.expert_num = expert_num
        self.task_num = task_num
        self.embedding = EmbeddingLayer(features)
        self.cn = CrossNetwork(self.dims, n_cross_layers)
        self.mlp = MLP(self.dims, output_layer=False, **mlp_params)
        self.output_shape = mlp_params["dims"][-1]
        self.linear = LR(self.dims + self.output_shape)

        # multi-task balance: mmoe
        self.expert = nn.ModuleList()
        for d in range(expert_num):
            self.expert.append(MLP_N(self.fcn_dim))
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], self.expert_num), torch.nn.Softmax(dim=1)) for i in range(self.domain_num*self.task_num)])
        
        self.tower = nn.ModuleList()
        for d in range(self.domain_num*self.task_num):
            domain_specific = nn.Sequential(
                nn.Linear(self.fcn_dim[-1], self.fcn_dim[-1]),
                nn.BatchNorm1d(self.fcn_dim[-1]),
                nn.ReLU(),
                nn.Linear(self.fcn_dim[-1], 1)
            )
            self.tower.append(domain_specific)
            
    def forward(self, x):
        domain_id = x[-1, :].clone().detach()
        _device = x.device
        # assert set(domain_id.cpu().numpy().flatten()).issubset(set([0, 1, 2])), set(domain_id.cpu().numpy().flatten())

        embed_x = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        # mask list
        mask = []
        # out list
        emb = torch.zeros((embed_x.shape[0], self.output_shape)).to(_device)
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask.cpu())

            domain_input = embed_x
            cn_out = self.cn(domain_input)
            mlp_out = self.mlp(domain_input)
            x_stack = torch.cat([cn_out, mlp_out], dim=1)
            y = self.linear(x_stack)
            emb = torch.where(mask[d].unsqueeze(1).to(_device), y, emb)
            # out.append(torch.sigmoid(y))

        gate_value = [self.gate[i](emb.detach()).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, expert_num]
        # gate_value = [self.gate[i](emb).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, expert_num]

        # mmoe: expert(emb)*gate(emb) -> tower(task_fea)
        out = [] # batch_size, expert_num, embedding_size
        for i in range(self.expert_num):
            model_list = self.expert[i]
            domain_input = model_list(emb)
            out.append(domain_input)  
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, 1
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num*self.domain_num)]

        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num*self.domain_num)]

        # save output in result [bs, task_num], all domains are integrated with bs length
        output = torch.zeros((len(results[0]), len(results)))
        for i in range(len(results)):
            output[:, i] = results[i]

        result = torch.zeros((len(results[0]), self.task_num))
        for t in range(self.task_num):
            for d in range(self.domain_num):
                result[:, t] = torch.where(mask[d], output[:, t*self.domain_num+d], result[:, t])
                
        return result 

class DCN_MTMD(torch.nn.Module):
    """
    multi-domain Deep & Cross Network and MMOE
    """

    def __init__(self, features, num_domains, task_num, expert_num, fcn_dims, n_cross_layers, mlp_params):
        super().__init__()
        self.fcn_dim = fcn_dims
        self.features = features
        self.dims = sum([fea.embed_dim for fea in features])
        self.domain_num = num_domains
        self.expert_num = expert_num
        self.task_num = task_num
        self.embedding = EmbeddingLayer(features)
        self.cn = CrossNetwork(self.dims, n_cross_layers)
        self.mlp = MLP(self.dims, output_layer=False, **mlp_params)
        self.output_shape = mlp_params["dims"][-1]
        self.linear = LR(self.dims + self.output_shape)

        # multi-task balance: mmoe
        self.expert = nn.ModuleList()
        for d in range(self.expert_num):
            self.expert.append(MLP_N(self.fcn_dim))
            
        self.domain_expert = nn.ModuleList()
        for d in range(self.domain_num):
            self.domain_expert.append(MLP_N(self.fcn_dim))
            
        self.task_expert = nn.ModuleList()
        for d in range(self.task_num):
            self.task_expert.append(MLP_N(self.fcn_dim))
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], self.expert_num + 2), torch.nn.Softmax(dim=1)) for i in range(self.domain_num*self.task_num)])
        
        self.tower = nn.ModuleList()
        for d in range(self.domain_num*self.task_num):
            domain_specific = nn.Sequential(
                nn.Linear(self.fcn_dim[-1], self.fcn_dim[-1]),
                nn.BatchNorm1d(self.fcn_dim[-1]),
                nn.ReLU(),
                nn.Linear(self.fcn_dim[-1], 1)
            )
            self.tower.append(domain_specific)
            
    def forward(self, x):
        domain_id = x[-1, :].clone().detach()
        _device = x.device
        # assert set(domain_id.cpu().numpy().flatten()).issubset(set([0, 1, 2])), set(domain_id.cpu().numpy().flatten())

        embed_x = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        # mask list
        mask = []
        # out list
        emb = torch.zeros((embed_x.shape[0], self.output_shape)).to(_device)
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask.cpu())

            domain_input = embed_x
            cn_out = self.cn(domain_input)
            mlp_out = self.mlp(domain_input)
            x_stack = torch.cat([cn_out, mlp_out], dim=1)
            y = self.linear(x_stack)
            emb = torch.where(mask[d].unsqueeze(1).to(_device), y, emb)
            # out.append(torch.sigmoid(y))

        gate_value = [self.gate[i](emb.detach()).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, expert_num]
        # gate_value = [self.gate[i](emb).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, expert_num]

        # mmoe: expert(emb)*gate(emb) -> tower(task_fea)
        out = [] # batch_size, expert_num, embedding_size
        for i in range(self.expert_num):
            model_list = self.expert[i]
            domain_input = model_list(emb)
            out.append(domain_input)     
        domain_exp_out = []
        for i in range(self.domain_num):
            domain_input = self.domain_expert[i](emb)
            domain_exp_out.append(domain_input)  
        task_exp_out = []
        for i in range(self.task_num):
            task_input = self.task_expert[i](emb)
            task_exp_out.append(task_input)          
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, 1
        domain_fea = torch.cat([domain_exp_out[i].unsqueeze(1) for i in range(self.domain_num)], dim = 1) # batch_size, domain_num, 1
        for i in range(self.domain_num):
            domain_fea[:, i, :][~mask[i]] = 0
        task_fea = torch.cat([task_exp_out[i].unsqueeze(1) for i in range(self.task_num)], dim = 1) # batch_size, task_num, 1
        # concat fea, domain_fea[i%domain_num], and task_fea[i%task_num] for gate[i]
        fused_fea = [torch.bmm(gate_value[i], torch.cat((fea, 
                                                         domain_fea[:, i%self.domain_num, :].unsqueeze(1), 
                                                         task_fea[:, i%self.task_num, :].unsqueeze(1)), 
                                                        dim = 1)).squeeze(1) 
                     for i in range(self.task_num*self.domain_num)]

        results = [torch.sigmoid(self.tower[i](fused_fea[i]).squeeze(1)) for i in range(self.task_num*self.domain_num)]

        # save output in result [bs, task_num], all domains are integrated with bs length
        output = torch.zeros((len(results[0]), len(results)))
        for i in range(len(results)):
            output[:, i] = results[i]

        result = torch.zeros((len(results[0]), self.task_num))
        for t in range(self.task_num):
            for d in range(self.domain_num):
                result[:, t] = torch.where(mask[d], output[:, t*self.domain_num+d], result[:, t])
                
        return result 
        

class MTMD_Skip(nn.Module):
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num, skip_weight):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims 
        self.domain_num = domain_num
        self.task_num = task_num
        self.expert_num = expert_num
        self.embedding = EmbeddingLayer(features)

        assert len(self.fcn_dim) > 3, f'too few layers assigned, must larger than 3. Star owns 3 layers, mmoe owns the rest.'
        self.star_dim = self.fcn_dim[:3]
        self.fcn_dim = self.fcn_dim[3:]
        self.skip_conn = MLP_N([self.star_dim[0], self.star_dim[2]]) 
        self.shared_weight = nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1]))
        self.shared_bias = nn.Parameter(torch.zeros(self.star_dim[1]))

        # multi-domain fusion: star
        self.slot_weight = nn.ParameterList([nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1])) for i in range(self.domain_num)])
        self.slot_bias = nn.ParameterList([nn.Parameter(torch.zeros(self.star_dim[1])) for i in range(self.domain_num)])
        
        self.star_mlp = MLP_N([self.star_dim[1], self.star_dim[2]])
        
        # for m in ([self.shared_weight]+ self.slot_weight):
        torch.nn.init.xavier_uniform_(self.shared_weight.data)
        for m in (self.slot_weight):
            torch.nn.init.xavier_uniform_(m.data)

        # domain&task-specific skip conn from expert input to tower input
        self.domain_skip = nn.ModuleList()
        for d in range(domain_num*task_num):
            self.domain_skip.append(MLP_N([self.star_dim[-1], self.fcn_dim[-1]]))
        self.skip_weight = skip_weight
            
        # multi-task balance: mmoe
        self.expert = nn.ModuleList()
        for d in range(expert_num):
            self.expert.append(MLP_N(self.fcn_dim))
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], expert_num), torch.nn.Softmax(dim=1)) for i in range(domain_num*task_num)])
        
        self.tower = nn.ModuleList()
        for d in range(domain_num*task_num):
            domain_specific = nn.Sequential(
                nn.Linear(self.fcn_dim[-1], self.fcn_dim[-1]),
                # nn.BatchNorm1d(self.fcn_dim[-1]),
                nn.LayerNorm(self.fcn_dim[-1]),
                nn.ReLU(),
                nn.Linear(self.fcn_dim[-1], 1)
            )
            self.tower.append(domain_specific)
            
    def forward(self, x, test_flag=False):
        _device = x.device
        domain_id = x[-1, :].clone().detach()
        assert set(domain_id.cpu().numpy().flatten()).issubset(DOMAIN_SET), set(domain_id.cpu().numpy().flatten())
        mask = []
        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask.cpu())

        input_emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        # star: w_shared*w_specific(input_emb) -> mlp + skip
        skip = self.skip_conn(input_emb)
        emb = torch.zeros((input_emb.shape[0], self.star_dim[1])).to(_device)
        for i, (_weight, _bias) in enumerate(zip(self.slot_weight, self.slot_bias)):
            _output = torch.matmul(input_emb, torch.multiply(_weight, self.shared_weight))+_bias+self.shared_bias
            emb = torch.where(mask[i].unsqueeze(1).to(_device), _output, emb)
        emb = self.star_mlp(emb)+skip
        
        gate_value = [self.gate[i](emb.detach()).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, expert_num]

        # save test data gate values for visualization
        if test_flag:
            _gate_value_list = [i.detach().cpu().numpy() for i in gate_value]
            _gate_value_df = list()
            for i, g in enumerate(_gate_value_list):
                g = g.squeeze(1)
                df = pd.DataFrame(data=g, columns=[f'e{i}' for i in range(self.expert_num)])
                df['domain_task'] = i
                _gate_value_df.append(df)
            _gate_value_df = pd.concat(_gate_value_df)

        # mmoe: expert(emb)*gate(emb) -> tower(task_fea)
        out = [] # batch_size, expert_num, embedding_size
        for i in range(self.expert_num):
            model_list = self.expert[i]
            domain_input = model_list(emb)
            out.append(domain_input)        
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, 1
        # add skip connection according to domains
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) +
                    self.skip_weight * self.domain_skip[i](emb)
                    for i in range(self.task_num*self.domain_num)]

        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num*self.domain_num)]

        # save output in result [bs, task_num], all domains are integrated with bs length
        output = torch.zeros((len(results[0]), len(results)))
        for i in range(len(results)):
            output[:, i] = results[i]

        result = torch.zeros((len(results[0]), self.task_num))
        for t in range(self.task_num):
            for d in range(self.domain_num):
                result[:, t] = torch.where(mask[d], output[:, t*self.domain_num+d], result[:, t])
        if test_flag:
            return result, _gate_value_df
        else:
            return result 

# concat shared and domain expert output, gate proccesses 2*d features
class MTMD_Domain_Expert(nn.Module):
    # add domain-specific expert based on MTMD
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims 
        self.domain_num = domain_num
        self.task_num = task_num
        self.expert_num = expert_num
        self.embedding = EmbeddingLayer(features)

        assert len(self.fcn_dim) > 3, f'too few layers assigned, must larger than 3. Star owns 3 layers, mmoe owns the rest.'
        self.star_dim = self.fcn_dim[:3]
        self.fcn_dim = self.fcn_dim[3:]
        self.skip_conn = MLP_N([self.star_dim[0], self.star_dim[2]]) 
        self.shared_weight = nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1]))
        self.shared_bias = nn.Parameter(torch.zeros(self.star_dim[1]))

        # multi-domain fusion: star
        self.slot_weight = nn.ParameterList([nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1])) for i in range(self.domain_num)])
        self.slot_bias = nn.ParameterList([nn.Parameter(torch.zeros(self.star_dim[1])) for i in range(self.domain_num)])
        
        self.star_mlp = MLP_N([self.star_dim[1], self.star_dim[2]])
        
        # for m in ([self.shared_weight]+ self.slot_weight):
        torch.nn.init.xavier_uniform_(self.shared_weight.data)
        for m in (self.slot_weight):
            torch.nn.init.xavier_uniform_(m.data)

        # multi-task balance: mmoe
        self.expert = nn.ModuleList()
        for d in range(expert_num):
            self.expert.append(MLP_N(self.fcn_dim))
            
        self.domain_expert = nn.ModuleList()
        for d in range(domain_num):
            self.domain_expert.append(MLP_N(self.fcn_dim))
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], expert_num + 1), torch.nn.Softmax(dim=1)) for i in range(domain_num*task_num)])
        
        self.tower = nn.ModuleList()
        for d in range(domain_num*task_num):
            domain_specific = nn.Sequential(
                nn.Linear(self.fcn_dim[-1], self.fcn_dim[-1]),
                nn.BatchNorm1d(self.fcn_dim[-1]),
                nn.ReLU(),
                nn.Linear(self.fcn_dim[-1], 1)
            )
            self.tower.append(domain_specific)
            
    def forward(self, x, test_flag=False):
        _device = x.device
        domain_id = x[-1, :].clone().detach()
        assert set(domain_id.cpu().numpy().flatten()).issubset(set([0, 1, 2])), set(domain_id.cpu().numpy().flatten())
        mask = []
        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask.cpu())

        input_emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        # star: w_shared*w_specific(input_emb) -> mlp + skip
        skip = self.skip_conn(input_emb)
        emb = torch.zeros((input_emb.shape[0], self.star_dim[1])).to(_device)
        for i, (_weight, _bias) in enumerate(zip(self.slot_weight, self.slot_bias)):
            _output = torch.matmul(input_emb, torch.multiply(_weight, self.shared_weight))+_bias+self.shared_bias
            emb = torch.where(mask[i].unsqueeze(1).to(_device), _output, emb)
        emb = self.star_mlp(emb)+skip
        
        gate_value = [self.gate[i](emb.detach()).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, expert_num]

        # save test data gate values for visualization
        if test_flag:
            _gate_value_df = None
            # _gate_value_list = [i.detach().cpu().numpy() for i in gate_value]
            # _gate_value_df = list()
            # for i, g in enumerate(_gate_value_list):
            #     g = g.squeeze(1)
            #     df = pd.DataFrame(data=g, columns=[f'e{i}' for i in range(self.expert_num)])
            #     df['domain_task'] = i
            #     _gate_value_df.append(df)
            # _gate_value_df = pd.concat(_gate_value_df)

        # mmoe: [expert(emb)||domain_expert[i](emb)]*gate(emb) -> tower(task_fea)
        out = [] # batch_size, expert_num, embedding_size
        for i in range(self.expert_num):
            domain_input = self.expert[i](emb)
            out.append(domain_input)       
        domain_exp_out = []
        for i in range(self.domain_num):
            domain_input = self.domain_expert[i](emb)
            domain_exp_out.append(domain_input)        
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, 1
        domain_fea = torch.cat([domain_exp_out[i].unsqueeze(1) for i in range(self.domain_num)], dim = 1) # batch_size, expert_num, 1
        # concat fea and domain_fea[i%domain_num] for gate[i]
        for i in range(self.domain_num):
            domain_fea[:, i, :][~mask[i]] = 0
        task_fea = [torch.bmm(gate_value[i], torch.cat((fea, domain_fea[:, i%self.domain_num, :].unsqueeze(1)), dim = 1)).squeeze(1) 
                    for i in range(self.task_num*self.domain_num)]

        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num*self.domain_num)]

        # save output in result [bs, task_num], all domains are integrated with bs length
        output = torch.zeros((len(results[0]), len(results)))
        for i in range(len(results)):
            output[:, i] = results[i]

        result = torch.zeros((len(results[0]), self.task_num))
        for t in range(self.task_num):
            for d in range(self.domain_num):
                result[:, t] = torch.where(mask[d], output[:, t*self.domain_num+d], result[:, t])
        if test_flag:
            return result, _gate_value_df
        else:
            return result 
            
class MTMD_Domain_Task_Expert(nn.Module):
    # add domain-specific expert based on MTMD
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims 
        self.domain_num = domain_num
        self.task_num = task_num
        self.expert_num = expert_num
        self.embedding = EmbeddingLayer(features)

        assert len(self.fcn_dim) > 3, f'too few layers assigned, must larger than 3. Star owns 3 layers, mmoe owns the rest.'
        self.star_dim = self.fcn_dim[:3]
        self.fcn_dim = self.fcn_dim[3:]
        self.skip_conn = MLP_N([self.star_dim[0], self.star_dim[2]]) 
        self.shared_weight = nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1]))
        self.shared_bias = nn.Parameter(torch.zeros(self.star_dim[1]))

        # multi-domain fusion: star
        self.slot_weight = nn.ParameterList([nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1])) for i in range(self.domain_num)])
        self.slot_bias = nn.ParameterList([nn.Parameter(torch.zeros(self.star_dim[1])) for i in range(self.domain_num)])
        
        self.star_mlp = MLP_N([self.star_dim[1], self.star_dim[2]])
        
        # for m in ([self.shared_weight]+ self.slot_weight):
        torch.nn.init.xavier_uniform_(self.shared_weight.data)
        for m in (self.slot_weight):
            torch.nn.init.xavier_uniform_(m.data)

        # multi-task balance: mmoe
        self.expert = nn.ModuleList()
        for d in range(expert_num):
            self.expert.append(MLP_N(self.fcn_dim))
            
        self.domain_expert = nn.ModuleList()
        for d in range(domain_num):
            self.domain_expert.append(MLP_N(self.fcn_dim))
            
        self.task_expert = nn.ModuleList()
        for d in range(task_num):
            self.task_expert.append(MLP_N(self.fcn_dim))
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], expert_num + 1 + 1), torch.nn.Softmax(dim=1)) for i in range(domain_num*task_num)])
        
        self.tower = nn.ModuleList()
        for d in range(domain_num*task_num):
            domain_specific = nn.Sequential(
                nn.Linear(self.fcn_dim[-1], self.fcn_dim[-1]),
                nn.BatchNorm1d(self.fcn_dim[-1]),
                nn.ReLU(),
                nn.Linear(self.fcn_dim[-1], 1)
            )
            self.tower.append(domain_specific)
            
    def forward(self, x, test_flag=False):
        _device = x.device
        domain_id = x[-1, :].clone().detach()
        assert set(domain_id.cpu().numpy().flatten()).issubset(DOMAIN_SET), set(domain_id.cpu().numpy().flatten())
        mask = []
        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask.cpu())

        input_emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        # star: w_shared*w_specific(input_emb) -> mlp + skip
        skip = self.skip_conn(input_emb)
        emb = torch.zeros((input_emb.shape[0], self.star_dim[1])).to(_device)
        for i, (_weight, _bias) in enumerate(zip(self.slot_weight, self.slot_bias)):
            _output = torch.matmul(input_emb, torch.multiply(_weight, self.shared_weight))+_bias+self.shared_bias
            emb = torch.where(mask[i].unsqueeze(1).to(_device), _output, emb)
        emb = self.star_mlp(emb)+skip
        
        gate_value = [self.gate[i](emb.detach()).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, expert_num]

        # save test data gate values for visualization
        if test_flag:
            _gate_value_df = None
        #     _gate_value_list = [i.detach().cpu().numpy() for i in gate_value]
        #     _gate_value_df = list()
        #     for i, g in enumerate(_gate_value_list):
        #         g = g.squeeze(1)
        #         df = pd.DataFrame(data=g, columns=[f'e{i}' for i in range(self.expert_num)])
        #         df['domain_task'] = i
        #         _gate_value_df.append(df)
        #     _gate_value_df = pd.concat(_gate_value_df)

        # mmoe: [expert(emb)||domain_expert[i](emb)||task_expert[i](emb)]*gate(emb) -> tower(task_fea)
        out = [] # batch_size, expert_num, embedding_size
        for i in range(self.expert_num):
            domain_input = self.expert[i](emb)
            out.append(domain_input)       
        domain_exp_out = []
        for i in range(self.domain_num):
            domain_input = self.domain_expert[i](emb)
            domain_exp_out.append(domain_input)  
        task_exp_out = []
        for i in range(self.task_num):
            task_input = self.task_expert[i](emb)
            task_exp_out.append(task_input)       
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, 1
        domain_fea = torch.cat([domain_exp_out[i].unsqueeze(1) for i in range(self.domain_num)], dim = 1) # batch_size, domain_num, 1
        for i in range(self.domain_num):
            domain_fea[:, i, :][~mask[i]] = 0
        task_fea = torch.cat([task_exp_out[i].unsqueeze(1) for i in range(self.task_num)], dim = 1) # batch_size, task_num, 1
        # concat fea, domain_fea[i%domain_num], and task_fea[i%task_num] for gate[i]
        fused_fea = [torch.bmm(gate_value[i], torch.cat((fea, 
                                                         domain_fea[:, i%self.domain_num, :].unsqueeze(1), 
                                                         task_fea[:, i%self.task_num, :].unsqueeze(1)), 
                                                        dim = 1)).squeeze(1) 
                     for i in range(self.task_num*self.domain_num)]

        results = [torch.sigmoid(self.tower[i](fused_fea[i]).squeeze(1)) for i in range(self.task_num*self.domain_num)]

        # save output in result [bs, task_num], all domains are integrated with bs length
        output = torch.zeros((len(results[0]), len(results)))
        for i in range(len(results)):
            output[:, i] = results[i]

        result = torch.zeros((len(results[0]), self.task_num))
        for t in range(self.task_num):
            for d in range(self.domain_num):
                result[:, t] = torch.where(mask[d], output[:, t*self.domain_num+d], result[:, t])
        if test_flag:
            return result, _gate_value_df
        else:
            return result 

# add domain-task-specific expert. In each layer we have shared exp and T*D domain-task-specific expert
class MTMD_DT_Expert(nn.Module):
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims 
        self.domain_num = domain_num
        self.task_num = task_num
        self.expert_num = expert_num
        self.embedding = EmbeddingLayer(features)

        assert len(self.fcn_dim) > 3, f'too few layers assigned, must larger than 3. Star owns 3 layers, mmoe owns the rest.'
        self.star_dim = self.fcn_dim[:3]
        self.fcn_dim = self.fcn_dim[3:]
        self.skip_conn = MLP_N([self.star_dim[0], self.star_dim[2]]) 
        self.shared_weight = nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1]))
        self.shared_bias = nn.Parameter(torch.zeros(self.star_dim[1]))

        # multi-domain fusion: star
        self.slot_weight = nn.ParameterList([nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1])) for i in range(self.domain_num)])
        self.slot_bias = nn.ParameterList([nn.Parameter(torch.zeros(self.star_dim[1])) for i in range(self.domain_num)])
        
        self.star_mlp = MLP_N([self.star_dim[1], self.star_dim[2]])
        
        # for m in ([self.shared_weight]+ self.slot_weight):
        torch.nn.init.xavier_uniform_(self.shared_weight.data)
        for m in (self.slot_weight):
            torch.nn.init.xavier_uniform_(m.data)

        # multi-task balance: mmoe
        self.expert = nn.ModuleList()
        for d in range(expert_num):
            self.expert.append(MLP_N(self.fcn_dim))
            
        self.dt_expert = nn.ModuleList()
        for d in range(self.domain_num*self.task_num):
            self.dt_expert.append(MLP_N(self.fcn_dim))
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], expert_num), torch.nn.Softmax(dim=1)) for i in range(domain_num*task_num)])
        
        self.tower = nn.ModuleList()
        for d in range(domain_num*task_num):
            domain_specific = nn.Sequential(
                nn.Linear(self.fcn_dim[-1], self.fcn_dim[-1]),
                nn.BatchNorm1d(self.fcn_dim[-1]),
                nn.ReLU(),
                nn.Linear(self.fcn_dim[-1], 1)
            )
            self.tower.append(domain_specific)
            
    def forward(self, x, test_flag=False):
        _device = x.device
        domain_id = x[-1, :].clone().detach()
        assert set(domain_id.cpu().numpy().flatten()).issubset(DOMAIN_SET), set(domain_id.cpu().numpy().flatten())
        mask = []
        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask.cpu())

        input_emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        # star: w_shared*w_specific(input_emb) -> mlp + skip
        skip = self.skip_conn(input_emb)
        emb = torch.zeros((input_emb.shape[0], self.star_dim[1])).to(_device)
        for i, (_weight, _bias) in enumerate(zip(self.slot_weight, self.slot_bias)):
            _output = torch.matmul(input_emb, torch.multiply(_weight, self.shared_weight))+_bias+self.shared_bias
            emb = torch.where(mask[i].unsqueeze(1).to(_device), _output, emb)
        emb = self.star_mlp(emb)+skip
        
        gate_value = [self.gate[i](emb.detach()).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, expert_num]
        # gate_value = [self.gate[i](emb).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, expert_num]

        # save test data gate values for visualization
        if test_flag:
            _gate_value_list = [i.detach().cpu().numpy() for i in gate_value]
            _gate_value_df = list()
            for i, g in enumerate(_gate_value_list):
                g = g.squeeze(1)
                df = pd.DataFrame(data=g, columns=[f'e{i}' for i in range(self.expert_num)])
                df['domain_task'] = i
                _gate_value_df.append(df)
            _gate_value_df = pd.concat(_gate_value_df)

        # mmoe: expert(emb)*gate(emb) -> tower(task_fea)
        out = [] # batch_size, expert_num, embedding_size
        for i in range(self.expert_num):
            model_list = self.expert[i]
            domain_input = model_list(emb)
            out.append(domain_input)        
        dt_exp_out = []
        for i in range(self.domain_num*self.task_num):
            dt_input = self.dt_expert[i](emb)
            dt_exp_out.append(dt_input)       
            
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, 1
        dt_fea = [dt_exp_out[i].unsqueeze(1) for i in range(self.domain_num*self.task_num)]
        fea = [(torch.bmm(gate_value[i], fea) + dt_fea[i]).squeeze(1) for i in range(self.task_num*self.domain_num)]

        results = [torch.sigmoid(self.tower[i](fea[i]).squeeze(1)) for i in range(self.task_num*self.domain_num)]

        # save output in result [bs, task_num], all domains are integrated with bs length
        output = torch.zeros((len(results[0]), len(results)))
        for i in range(len(results)):
            output[:, i] = results[i]

        result = torch.zeros((len(results[0]), self.task_num))
        for t in range(self.task_num):
            for d in range(self.domain_num):
                result[:, t] = torch.where(mask[d], output[:, t*self.domain_num+d], result[:, t])
        if test_flag:
            return result, _gate_value_df
        else:
            return result 

class MTMD_Domain_Task_Expert_Gate_Shared(nn.Module):
    # add domain-specific expert based on MTMD
    # gate(shared expert) + task-specific + domain-specifics
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims 
        self.domain_num = domain_num
        self.task_num = task_num
        self.expert_num = expert_num
        self.embedding = EmbeddingLayer(features)

        assert len(self.fcn_dim) > 3, f'too few layers assigned, must larger than 3. Star owns 3 layers, mmoe owns the rest.'
        self.star_dim = self.fcn_dim[:3]
        self.fcn_dim = self.fcn_dim[3:]
        self.skip_conn = MLP_N([self.star_dim[0], self.star_dim[2]]) 
        self.shared_weight = nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1]))
        self.shared_bias = nn.Parameter(torch.zeros(self.star_dim[1]))

        # multi-domain fusion: star
        self.slot_weight = nn.ParameterList([nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1])) for i in range(self.domain_num)])
        self.slot_bias = nn.ParameterList([nn.Parameter(torch.zeros(self.star_dim[1])) for i in range(self.domain_num)])
        
        self.star_mlp = MLP_N([self.star_dim[1], self.star_dim[2]])
        
        # for m in ([self.shared_weight]+ self.slot_weight):
        torch.nn.init.xavier_uniform_(self.shared_weight.data)
        for m in (self.slot_weight):
            torch.nn.init.xavier_uniform_(m.data)

        # multi-task balance: mmoe
        self.expert = nn.ModuleList()
        for d in range(expert_num):
            self.expert.append(MLP_N(self.fcn_dim))
            
        self.domain_expert = nn.ModuleList()
        for d in range(domain_num):
            self.domain_expert.append(MLP_N(self.fcn_dim))
            
        self.task_expert = nn.ModuleList()
        for d in range(task_num):
            self.task_expert.append(MLP_N(self.fcn_dim))
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], expert_num), torch.nn.Softmax(dim=1)) for i in range(domain_num*task_num)])
        
        self.tower = nn.ModuleList()
        for d in range(domain_num*task_num):
            domain_specific = nn.Sequential(
                nn.Linear(self.fcn_dim[-1], self.fcn_dim[-1]),
                # nn.BatchNorm1d(self.fcn_dim[-1]),
                nn.LayerNorm(self.fcn_dim[-1]),
                nn.ReLU(),
                nn.Linear(self.fcn_dim[-1], 1)
            )
            self.tower.append(domain_specific)
        
    def forward(self, x, test_flag=False):
        _device = x.device
        domain_id = x[-1, :].clone().detach()
        assert set(domain_id.cpu().numpy().flatten()).issubset(DOMAIN_SET), set(domain_id.cpu().numpy().flatten())
        mask = []
        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask.cpu())

        input_emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        # star: w_shared*w_specific(input_emb) -> mlp + skip
        skip = self.skip_conn(input_emb)
        emb = torch.zeros((input_emb.shape[0], self.star_dim[1])).to(_device)
        for i, (_weight, _bias) in enumerate(zip(self.slot_weight, self.slot_bias)):
            _output = torch.matmul(input_emb, torch.multiply(_weight, self.shared_weight))+_bias+self.shared_bias
            emb = torch.where(mask[i].unsqueeze(1).to(_device), _output, emb)
        emb = self.star_mlp(emb)+skip
        
        gate_value = [self.gate[i](emb.detach()).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, expert_num]

        # save test data gate values for visualization
        if test_flag:
            _gate_value_df = None

        # mmoe: [expert(emb)||domain_expert[i](emb)||task_expert[i](emb)]*gate(emb) -> tower(task_fea)
        out = [] # batch_size, expert_num, embedding_size
        for i in range(self.expert_num):
            domain_input = self.expert[i](emb)
            out.append(domain_input)       
        domain_exp_out = []
        for i in range(self.domain_num):
            domain_input = self.domain_expert[i](emb)
            domain_exp_out.append(domain_input)  
        task_exp_out = []
        for i in range(self.task_num):
            task_input = self.task_expert[i](emb)
            task_exp_out.append(task_input)       
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, 1
        domain_fea = torch.cat([domain_exp_out[i].unsqueeze(1) for i in range(self.domain_num)], dim = 1) # batch_size, domain_num, 1
        # for i in range(self.domain_num):
        #     domain_fea[:, i, :][~mask[i]] = 0
        task_fea = torch.cat([task_exp_out[i].unsqueeze(1) for i in range(self.task_num)], dim = 1) # batch_size, task_num, 1
        # concat fea, domain_fea[i%domain_num], and task_fea[i%task_num] for gate[i]
        
        fused_fea = [torch.bmm(gate_value[i], fea).squeeze(1) + 
                     domain_fea[:, i%self.domain_num, :] + 
                     task_fea[:, i%self.task_num, :]
                     for i in range(self.task_num*self.domain_num)]
        
        torch.save(fea, 'fea.pt')
        torch.save(domain_fea, 'domain_fea.pt')
        torch.save(task_fea, 'task_fea.pt')
        
        results = [torch.sigmoid(self.tower[i](fused_fea[i]).squeeze(1)) for i in range(self.task_num*self.domain_num)]

        # save output in result [bs, task_num], all domains are integrated with bs length
        output = torch.zeros((len(results[0]), len(results)))
        for i in range(len(results)):
            output[:, i] = results[i]

        result = torch.zeros((len(results[0]), self.task_num))
        for t in range(self.task_num):
            for d in range(self.domain_num):
                result[:, t] = torch.where(mask[d], output[:, t*self.domain_num+d], result[:, t])
        if test_flag:
            return result, _gate_value_df
        else:
            return result 
            