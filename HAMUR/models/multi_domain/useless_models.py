import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
from torch.nn.parameter import Parameter
from ...basic.layers import EmbeddingLayer
from ...basic.activation import activation_layer



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

class MTMD_Skip_Lora(nn.Module):
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
        self.skip_weight = skip_weight

        # domain&task-specific skip conn from expert input to tower input
        lora_size = 16
        self.domain_skip_a = nn.ParameterList()
        self.domain_skip_b = nn.ParameterList()
        for d in range(domain_num*task_num):
            self.domain_skip_a.append(nn.Parameter(torch.empty(self.star_dim[-1], lora_size)))
            self.domain_skip_b.append(nn.Parameter(torch.empty(lora_size, self.fcn_dim[-1])))
            
        for m in (self.domain_skip_a):
            torch.nn.init.xavier_uniform_(m.data)
        for m in (self.domain_skip_b):
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
                # nn.BatchNorm1d(self.fcn_dim[-1]),
                nn.LayerNorm(self.fcn_dim[-1]),
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
                    self.skip_weight * torch.matmul(emb, torch.mm(self.domain_skip_a[i], self.domain_skip_b[i]))
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

            
# add shared and domain expert output, gate proccesses d features, mask domain expert output with 0 for other domain samples
class MTMD_Domain_Expert_Add(nn.Module):
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
        for i in range(self.domain_num):
            domain_fea[:, i, :][~mask[i]] = 0
        # add fea and domain_fea[i%domain_num] for gate[i]
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) + 
                    domain_fea[:, i%self.domain_num, :] 
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
            
class MTMD_Domain_Expert_wo_Shared(nn.Module):
    # add domain-specific expert based on MTMD
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num=0):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims 
        self.domain_num = domain_num
        self.task_num = task_num
        # self.expert_num = expert_num
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
        self.domain_expert = nn.ModuleList()
        for d in range(domain_num):
            self.domain_expert.append(MLP_N(self.fcn_dim))
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], 1), torch.nn.Softmax(dim=1)) for i in range(domain_num*task_num)])
        
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
        domain_exp_out = []
        for i in range(self.domain_num):
            domain_input = self.domain_expert[i](emb)
            domain_exp_out.append(domain_input)        
        domain_fea = torch.cat([domain_exp_out[i].unsqueeze(1) for i in range(self.domain_num)], dim = 1) # batch_size, expert_num, 1
        # concat fea and domain_fea[i%domain_num] for gate[i]
        task_fea = [torch.bmm(gate_value[i], domain_fea[:, i%self.domain_num, :].unsqueeze(1)).squeeze(1) 
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


class MTMD_Task_Expert(nn.Module):
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
            
        self.task_expert = nn.ModuleList()
        for d in range(task_num):
            self.task_expert.append(MLP_N(self.fcn_dim))
        
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
        #     _gate_value_list = [i.detach().cpu().numpy() for i in gate_value]
        #     _gate_value_df = list()
        #     for i, g in enumerate(_gate_value_list):
        #         g = g.squeeze(1)
        #         df = pd.DataFrame(data=g, columns=[f'e{i}' for i in range(self.expert_num)])
        #         df['domain_task'] = i
        #         _gate_value_df.append(df)
        #     _gate_value_df = pd.concat(_gate_value_df)

        # mmoe: [expert(emb)||task_expert[i](emb)]*gate(emb) -> tower(task_fea)
        out = [] # batch_size, expert_num, embedding_size
        for i in range(self.expert_num):
            domain_input = self.expert[i](emb)
            out.append(domain_input)       
        task_exp_out = []
        for i in range(self.task_num):
            task_input = self.task_expert[i](emb)
            task_exp_out.append(task_input)        
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, 1
        task_fea = torch.cat([task_exp_out[i].unsqueeze(1) for i in range(self.task_num)], dim = 1) # batch_size, expert_num, 1
        # concat fea and task_fea[i%task_num] for gate[i]
        fused_fea = [torch.bmm(gate_value[i], torch.cat((fea, task_fea[:, i%self.task_num, :].unsqueeze(1)), dim = 1)).squeeze(1) 
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


class MTMD_Task_Expert_wo_Shared(nn.Module):
    # add domain-specific expert based on MTMD
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num=0):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims 
        self.domain_num = domain_num
        self.task_num = task_num
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
        self.task_expert = nn.ModuleList()
        for d in range(task_num):
            self.task_expert.append(MLP_N(self.fcn_dim))
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], 1), torch.nn.Softmax(dim=1)) for i in range(domain_num*task_num)])
        
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
        #     _gate_value_list = [i.detach().cpu().numpy() for i in gate_value]
        #     _gate_value_df = list()
        #     for i, g in enumerate(_gate_value_list):
        #         g = g.squeeze(1)
        #         df = pd.DataFrame(data=g, columns=[f'e{i}' for i in range(self.expert_num)])
        #         df['domain_task'] = i
        #         _gate_value_df.append(df)
        #     _gate_value_df = pd.concat(_gate_value_df)

        # mmoe: [expert(emb)||task_expert[i](emb)]*gate(emb) -> tower(task_fea)
        task_exp_out = []
        for i in range(self.task_num):
            task_input = self.task_expert[i](emb)
            task_exp_out.append(task_input)        
        task_fea = torch.cat([task_exp_out[i].unsqueeze(1) for i in range(self.task_num)], dim = 1) # batch_size, expert_num, 1
        # concat fea and task_fea[i%task_num] for gate[i]
        fused_fea = [torch.bmm(gate_value[i], task_fea[:, i%self.task_num, :].unsqueeze(1)).squeeze(1) 
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


class MTMD_Domain_Task_Expert_Gate_Shared_Skip(nn.Module):
    # add domain-specific expert based on MTMD
    # gate(shared expert) + task-specific + domain-specifics
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
        self.skip_weight = skip_weight
        
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
        
        # domain&task-specific skip conn from expert input to tower input
        self.domain_skip = nn.ModuleList()
        for d in range(domain_num*task_num):
            self.domain_skip.append(MLP_N([self.star_dim[-1], self.fcn_dim[-1]]))
            
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
        task_fea = torch.cat([task_exp_out[i].unsqueeze(1) for i in range(self.task_num)], dim = 1) # batch_size, task_num, 1
        # concat fea, domain_fea[i%domain_num], and task_fea[i%task_num] for gate[i]
        fused_fea = [torch.bmm(gate_value[i], fea).squeeze(1) + 
                     domain_fea[:, i%self.domain_num, :] + 
                     task_fea[:, i%self.task_num, :] +
                     self.skip_weight * self.domain_skip[i](emb)
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
            
class MTMD_Domain_Task_Expert_Gate_Shared_Concat(nn.Module):
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
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], expert_num), torch.nn.Softmax(dim=1)) for i in range(domain_num*task_num)])
        
        self.tower = nn.ModuleList()
        for d in range(domain_num*task_num):
            domain_specific = nn.Sequential(
                nn.Linear(self.fcn_dim[-1]*3, self.fcn_dim[-1]),
                # nn.BatchNorm1d(self.fcn_dim[-1]),
                nn.LayerNorm(self.fcn_dim[-1]),
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
        task_fea = torch.cat([task_exp_out[i].unsqueeze(1) for i in range(self.task_num)], dim = 1) # batch_size, task_num, 1
        # concat fea, domain_fea[i%domain_num], and task_fea[i%task_num] for gate[i]
        fused_fea = [torch.concat((torch.bmm(gate_value[i], fea).squeeze(1), 
                                   domain_fea[:, i%self.domain_num, :], 
                                   task_fea[:, i%self.task_num, :]
                                  ), -1)
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

class MTMD_Domain_Task_Expert_Individual_Gate(nn.Module):
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
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], expert_num), torch.nn.Softmax(dim=1)) for i in range(domain_num*task_num)])
        self.domain_gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], task_num), torch.nn.Softmax(dim=1)) for i in range(domain_num)])
        self.task_gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], domain_num), torch.nn.Softmax(dim=1)) for i in range(task_num)])
        
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
        
        gate_value = [self.gate[i](emb.detach()).unsqueeze(1) for i in range(self.task_num*self.domain_num)] 
        # [domain_num*task_num, batch_size, 1, expert_num]
        domain_gate_value = [self.domain_gate[i](emb.detach()).unsqueeze(1) for i in range(self.domain_num)] 
        # [domain_num, batch_size, 1, task_num]
        task_gate_value = [self.task_gate[i](emb.detach()).unsqueeze(1) for i in range(self.task_num)] 
        # [task_num, batch_size, 1, domain_num]

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
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, d
        domain_fea = torch.cat([domain_exp_out[i].unsqueeze(1) for i in range(self.domain_num)], dim = 1) # batch_size, domain_num, d
        task_fea = torch.cat([task_exp_out[i].unsqueeze(1) for i in range(self.task_num)], dim = 1) # batch_size, task_num, d
        # concat fea, domain_fea[i%domain_num], and task_fea[i%task_num] for gate[i]
        fused_fea = [torch.bmm(gate_value[i], fea).squeeze(1)
                     for i in range(self.task_num*self.domain_num)]
        # domain_num * [b, task_num, d]
        domain_fused_fea = []
        for d in range(self.domain_num):
            # [b, task_num, 1] * [b, 1, d] -> [b, task_num, d]
            fea_per_domain = torch.bmm(domain_gate_value[d].transpose(1, 2), domain_fea[:, d, :].unsqueeze(1))
            domain_fused_fea.append(fea_per_domain)

        # task_num * [b, domain_num, d]
        task_fused_fea = []
        for t in range(self.task_num):
            # [b, domain_num, 1] * [b, 1, d] -> [b, domain_num, d]
            fea_per_task = torch.bmm(task_gate_value[t].transpose(1, 2), task_fea[:, t, :].unsqueeze(1))
            task_fused_fea.append(fea_per_task)

        for i in range(self.domain_num*self.task_num):
            fused_fea[i] = (fused_fea[i] + 
                            domain_fused_fea[i//self.task_num][:, i%self.task_num, :] + 
                            task_fused_fea[i//self.domain_num][:, i%self.domain_num, :]
                           )
        
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

class MTMD_Domain_Task_Expert_wo_shared(nn.Module):
    # add domain-specific expert based on MTMD
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num=0):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims 
        self.domain_num = domain_num
        self.task_num = task_num
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
        self.domain_expert = nn.ModuleList()
        for d in range(domain_num):
            self.domain_expert.append(MLP_N(self.fcn_dim))
            
        self.task_expert = nn.ModuleList()
        for d in range(task_num):
            self.task_expert.append(MLP_N(self.fcn_dim))
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], 1 + 1), torch.nn.Softmax(dim=1)) for i in range(domain_num*task_num)])
        
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
        
        gate_value = [self.gate[i](emb.detach()).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, 1+1]

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
        domain_exp_out = []
        for i in range(self.domain_num):
            domain_input = self.domain_expert[i](emb)
            domain_exp_out.append(domain_input)  
        task_exp_out = []
        for i in range(self.task_num):
            task_input = self.task_expert[i](emb)
            task_exp_out.append(task_input)       
        domain_fea = torch.cat([domain_exp_out[i].unsqueeze(1) for i in range(self.domain_num)], dim = 1) # batch_size, domain_num, 1
        task_fea = torch.cat([task_exp_out[i].unsqueeze(1) for i in range(self.task_num)], dim = 1) # batch_size, task_num, 1
        # concat fea, domain_fea[i%domain_num], and task_fea[i%task_num] for gate[i]
        fused_fea = [torch.bmm(gate_value[i], torch.cat((domain_fea[:, i%self.domain_num, :].unsqueeze(1), 
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



class MTMD_Simple_Star(nn.Module):
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num):
        super().__init__()
        fcn_dims = [256, 256, 64]
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims 
        self.domain_num = domain_num
        self.task_num = task_num
        self.expert_num = expert_num
        self.embedding = EmbeddingLayer(features)

        assert len(self.fcn_dim) > 2, f'too few layers assigned, must larger than 2. Star owns 2 layers, mmoe owns the rest.'
        self.star_dim = self.fcn_dim[:2]
        self.fcn_dim = self.fcn_dim[1:]
        self.skip_conn = MLP_N([self.star_dim[0], self.star_dim[1]]) 
        self.shared_weight = nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1]))
        self.shared_bias = nn.Parameter(torch.zeros(self.star_dim[1]))

        # multi-domain fusion: star
        self.slot_weight = nn.ParameterList([nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1])) for i in range(self.domain_num)])
        self.slot_bias = nn.ParameterList([nn.Parameter(torch.zeros(self.star_dim[1])) for i in range(self.domain_num)])
        
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
        emb = emb+skip
        
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


class MTMD_Domain_Task_Expert_MI_Base(nn.Module):
    
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
            self.expert.append(MI_Expert(self.fcn_dim))
            
        self.domain_expert = nn.ModuleList()
        for d in range(domain_num):
            self.domain_expert.append(MI_Expert(self.fcn_dim))
            
        self.task_expert = nn.ModuleList()
        for d in range(task_num):
            self.task_expert.append(MI_Expert(self.fcn_dim))
        
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
            

    def _kdl_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """Using std to compute KLD"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
        # sigma_1 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_1))
        # sigma_2 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_2))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl
        
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
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, d
        domain_fea = torch.cat([domain_exp_out[i].unsqueeze(1) for i in range(self.domain_num)], dim = 1) # batch_size, domain_num, d
        task_fea = torch.cat([task_exp_out[i].unsqueeze(1) for i in range(self.task_num)], dim = 1) # batch_size, task_num, d
        
        # concat fea, domain_fea[i%domain_num], and task_fea[i%task_num]
        shared_fea = torch.bmm(gate_value[i], fea).squeeze(1)
        fused_fea = [shared_fea + 
                     domain_fea[:, i%self.domain_num, :] + 
                     task_fea[:, i%self.task_num, :]
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


class MTMD_Domain_Task_Expert_MI_Close_SD_ST(MTMD_Domain_Task_Expert_MI_Base):
    # add domain/task-specific expert based on MTMD, only keep one shared expert and remove gate
    
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num):
        super().__init__(features, domain_num, task_num, fcn_dims, expert_num)

    def kdl_sd_st(self, shared_mean, shared_cov, domain_mean_list, domain_cov_list, task_mean_list, task_cov_list, mask):
        kdl_sd = 0
        kdl_st = 0

        # seperate shared/domain matrices by domain_id
        # domain_sample_size, d
        for d in range(self.domain_num):
            kdl_sd += self._kdl_gauss(shared_mean[mask[d]], shared_cov[mask[d]], domain_mean_list[d][mask[d]], domain_cov_list[d][mask[d]])
        # calculate mutual information between shared with task
        for t in range(self.task_num):
            kdl_st += self._kdl_gauss(shared_mean, shared_cov, task_mean_list[t], task_cov_list[t])
        return kdl_sd + kdl_st
        
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
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, d
        domain_fea = torch.cat([domain_exp_out[i].unsqueeze(1) for i in range(self.domain_num)], dim = 1) # batch_size, domain_num, d
        task_fea = torch.cat([task_exp_out[i].unsqueeze(1) for i in range(self.task_num)], dim = 1) # batch_size, task_num, d
        
        # concat fea, domain_fea[i%domain_num], and task_fea[i%task_num]
        shared_fea = torch.bmm(gate_value[i], fea).squeeze(1)
        fused_fea = [shared_fea + 
                     domain_fea[:, i%self.domain_num, :] + 
                     task_fea[:, i%self.task_num, :]
                     for i in range(self.task_num*self.domain_num)]

        # calculate mutual information
        shared_mean_list, shared_cov_list = [], [] 
        for e in self.expert:
            shared_mean_list.append(e.output_mean.unsqueeze(1))
            shared_cov_list.append(e.output_cov.unsqueeze(1))
        shared_mean = torch.bmm(gate_value[i], torch.cat(shared_mean_list, dim = 1)).squeeze(1) # batch_size, d
        shared_cov = torch.bmm(gate_value[i], torch.cat(shared_cov_list, dim = 1)).squeeze(1) # batch_size, d
        
        domain_mean_list, domain_cov_list = [], [] # domain_num, batch_size, d
        for e in self.domain_expert:
            domain_mean_list.append(e.output_mean)
            domain_cov_list.append(e.output_cov)
            
        task_mean_list, task_cov_list = [], [] # task_num, batch_size, d
        for e in self.task_expert:
            task_mean_list.append(e.output_mean)
            task_cov_list.append(e.output_cov)

        self.kdl_loss_sd_st = self.kdl_sd_st(shared_mean, shared_cov, domain_mean_list, domain_cov_list, task_mean_list, task_cov_list, mask)

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

class MTMD_Domain_Task_Expert_MI_Push_T(MTMD_Domain_Task_Expert_MI_Base):
    # add domain/task-specific expert based on MTMD, only keep one shared expert and remove gate
    
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num):
        super().__init__(features, domain_num, task_num, fcn_dims, expert_num)

    def kdl_push_t(self, task_mean_list, task_cov_list, mask):
        kdl_t = 0
        
        # seperate shared/domain matrices by domain_id
        # domain_sample_size, d
        # calculate kl(t0, t1)
        for t1 in range(self.task_num):
            for t2 in range(t1 + 1, self.task_num):
                kdl_t += self._kdl_gauss(task_mean_list[t1], task_cov_list[t1], 
                                         task_mean_list[t2], task_cov_list[t2])
        return _KL_Weight * _KL_Weight * kdl_t
        
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
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, d
        domain_fea = torch.cat([domain_exp_out[i].unsqueeze(1) for i in range(self.domain_num)], dim = 1) # batch_size, domain_num, d
        task_fea = torch.cat([task_exp_out[i].unsqueeze(1) for i in range(self.task_num)], dim = 1) # batch_size, task_num, d
        
        # concat fea, domain_fea[i%domain_num], and task_fea[i%task_num]
        shared_fea = torch.bmm(gate_value[i], fea).squeeze(1)
        fused_fea = [shared_fea + 
                     domain_fea[:, i%self.domain_num, :] + 
                     task_fea[:, i%self.task_num, :]
                     for i in range(self.task_num*self.domain_num)]

        # calculate mutual information
        shared_mean_list, shared_cov_list = [], [] 
        for e in self.expert:
            shared_mean_list.append(e.output_mean.unsqueeze(1))
            shared_cov_list.append(e.output_cov.unsqueeze(1))
        shared_mean = torch.bmm(gate_value[i], torch.cat(shared_mean_list, dim = 1)).squeeze(1) # batch_size, d
        shared_cov = torch.bmm(gate_value[i], torch.cat(shared_cov_list, dim = 1)).squeeze(1) # batch_size, d
        
        domain_mean_list, domain_cov_list = [], [] # domain_num, batch_size, d
        for e in self.domain_expert:
            domain_mean_list.append(e.output_mean)
            domain_cov_list.append(e.output_cov)
            
        task_mean_list, task_cov_list = [], [] # task_num, batch_size, d
        for e in self.task_expert:
            task_mean_list.append(e.output_mean)
            task_cov_list.append(e.output_cov)

        self.kdl_loss_push_t = self.kdl_push_t(task_mean_list, task_cov_list, mask)

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


class MTMD_Domain_Task_Expert_MI_Close_SDST_Push_T(MTMD_Domain_Task_Expert_MI_Base):
    # add domain/task-specific expert based on MTMD, only keep one shared expert and remove gate
    
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num):
        super().__init__(features, domain_num, task_num, fcn_dims, expert_num)

    def kdl_sd_st(self, shared_mean, shared_cov, domain_mean_list, domain_cov_list, task_mean_list, task_cov_list, mask):
        kdl_sd = 0
        kdl_st = 0

        # seperate shared/domain matrices by domain_id
        # domain_sample_size, d
        for d in range(self.domain_num):
            kdl_sd += self._kdl_gauss(shared_mean[mask[d]], shared_cov[mask[d]], domain_mean_list[d][mask[d]], domain_cov_list[d][mask[d]])
        # calculate mutual information between shared with task
        for t in range(self.task_num):
            kdl_st += self._kdl_gauss(shared_mean, shared_cov, task_mean_list[t], task_cov_list[t])
        return kdl_sd + kdl_st
        
    def kdl_push_t(self, task_mean_list, task_cov_list, mask):
        kdl_t = 0
        
        # seperate shared/domain matrices by domain_id
        # domain_sample_size, d
        # calculate kl(t0, t1)
        for t1 in range(self.task_num):
            for t2 in range(t1 + 1, self.task_num):
                kdl_t += self._kdl_gauss(task_mean_list[t1], task_cov_list[t1], 
                                         task_mean_list[t2], task_cov_list[t2])
        return _KL_Weight * _KL_Weight * kdl_t
        
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
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, d
        domain_fea = torch.cat([domain_exp_out[i].unsqueeze(1) for i in range(self.domain_num)], dim = 1) # batch_size, domain_num, d
        task_fea = torch.cat([task_exp_out[i].unsqueeze(1) for i in range(self.task_num)], dim = 1) # batch_size, task_num, d
        
        # concat fea, domain_fea[i%domain_num], and task_fea[i%task_num]
        shared_fea = torch.bmm(gate_value[i], fea).squeeze(1)
        fused_fea = [shared_fea + 
                     domain_fea[:, i%self.domain_num, :] + 
                     task_fea[:, i%self.task_num, :]
                     for i in range(self.task_num*self.domain_num)]

        # calculate mutual information
        shared_mean_list, shared_cov_list = [], [] 
        for e in self.expert:
            shared_mean_list.append(e.output_mean.unsqueeze(1))
            shared_cov_list.append(e.output_cov.unsqueeze(1))
        shared_mean = torch.bmm(gate_value[i], torch.cat(shared_mean_list, dim = 1)).squeeze(1) # batch_size, d
        shared_cov = torch.bmm(gate_value[i], torch.cat(shared_cov_list, dim = 1)).squeeze(1) # batch_size, d
        
        domain_mean_list, domain_cov_list = [], [] # domain_num, batch_size, d
        for e in self.domain_expert:
            domain_mean_list.append(e.output_mean)
            domain_cov_list.append(e.output_cov)
            
        task_mean_list, task_cov_list = [], [] # task_num, batch_size, d
        for e in self.task_expert:
            task_mean_list.append(e.output_mean)
            task_cov_list.append(e.output_cov)

        self.kdl_loss_sd_st = self.kdl_sd_st(shared_mean, shared_cov, domain_mean_list, domain_cov_list, task_mean_list, task_cov_list, mask)

        self.kdl_loss_push_t = self.kdl_push_t(task_mean_list, task_cov_list, mask)

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


