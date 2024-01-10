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

# _KL_Weight = 0.1

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
        
class MI_Expert(nn.Module):
    # generate mean and covariance matrix with an mlp respectively
    def __init__(self, fcn_dim, sigma_w):
        super().__init__()
        # self.half_fcn_dim = [fcn_dim[0]] + [int(i/2) for i in fcn_dim[1:-1]] + [fcn_dim[-1]]
        self.half_fcn_dim = fcn_dim
        self.n = len(fcn_dim)
        self.mean_expert = nn.ModuleList()
        self.cov_expert = nn.ModuleList()
        self.sigma_w = sigma_w

        for i in range(self.n-1):
            self.mean_expert.append(nn.Linear(self.half_fcn_dim[i], self.half_fcn_dim[i+1]))
            self.mean_expert.append(nn.LayerNorm(self.half_fcn_dim[i+1]))
            self.mean_expert.append(nn.ReLU())   
            
            self.cov_expert.append(nn.Linear(self.half_fcn_dim[i], self.half_fcn_dim[i+1]))
            self.cov_expert.append(nn.LayerNorm(self.half_fcn_dim[i+1]))
            self.cov_expert.append(nn.ReLU())

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
    
    def reparameters(self, mean, logstd):
        # sigma = 0.1 + 0.9 * F.softplus(torch.exp(logstd))
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        gaussian_noise = torch.randn(mean.size(0), self.half_fcn_dim[-1]).to(mean.device)
        if self.training:
            sampled_z = self.sigma_w*gaussian_noise * torch.exp(sigma) + mean
        else:
            sampled_z = mean
        kld_loss = self._kdl_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        # return sampled_z, _KL_Weight * kld_loss
        return sampled_z, kld_loss
        
    def forward(self, x):
        output_mean = x
        output_cov = x
        for f_mean, f_cov in zip(self.mean_expert, self.cov_expert):
            output_mean = f_mean(output_mean)
            output_cov = f_cov(output_cov)
        self.output_mean, self.output_cov = output_mean, output_cov
        output, self.kld_loss = self.reparameters(output_mean, output_cov)
        return output


# concat of three types of experts
class MTMD_Domain_Task_Expert_Wo_Gate_Base(nn.Module):
    # add domain/task-specific expert based on MTMD, only keep one shared expert and remove gate
    
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num, sigma_w):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims 
        self.domain_num = domain_num
        self.task_num = task_num
        self.expert_num = 1
        self.embedding = EmbeddingLayer(features)
        self.sigma_w = sigma_w

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
        for d in range(self.expert_num):
            self.expert.append(MI_Expert(self.fcn_dim, self.sigma_w))
            
        self.domain_expert = nn.ModuleList()
        for d in range(domain_num):
            self.domain_expert.append(MI_Expert(self.fcn_dim, self.sigma_w))
            
        self.task_expert = nn.ModuleList()
        for d in range(task_num):
            self.task_expert.append(MI_Expert(self.fcn_dim, self.sigma_w))
        
        # self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], expert_num), torch.nn.Softmax(dim=1)) for i in range(domain_num*task_num)])
        
        self.tower = nn.ModuleList()
        for d in range(domain_num*task_num):
            domain_specific = nn.Sequential(
                nn.Linear(self.fcn_dim[-1]*3, self.fcn_dim[-1]),
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
        
        # gate_value = [self.gate[i](emb.detach()).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, expert_num]

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
        # shared_fea = torch.bmm(gate_value[i], fea).squeeze(1)
        shared_fea = fea.squeeze(1)
        fused_fea = [torch.concat((shared_fea, 
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

# concat of three types of experts
class MTMD_Domain_Task_Expert_Concat_MI_STAR_MI_Base(nn.Module):
    # add domain/task-specific expert based on MTMD, only keep one shared expert and remove gate
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num, sigma_w):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims 
        self.domain_num = domain_num
        self.task_num = task_num
        self.expert_num = expert_num
        self.embedding = EmbeddingLayer(features)
        self.sigma_w = sigma_w

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

        self.star_cov = MI_Expert(self.star_dim, self.sigma_w*self.sigma_w)
        # for m in ([self.shared_weight]+ self.slot_weight):
        torch.nn.init.xavier_uniform_(self.shared_weight.data)
        for m in (self.slot_weight):
            torch.nn.init.xavier_uniform_(m.data)
            
        # assert len(self.fcn_dim) > 3, f'too few layers assigned, must larger than 3. Star owns 3 layers, mmoe owns the rest.'
        # self.star_dim = self.fcn_dim[:3]
        # self.fcn_dim = self.fcn_dim[3:]
        # self.skip_conn = MLP_N([self.star_dim[0], self.star_dim[2]]) 
        
        # # self.shared_weight = nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1]))
        # # self.shared_bias = nn.Parameter(torch.zeros(self.star_dim[1]))

        # # multi-domain fusion: star
        # # self.slot_weight = nn.ParameterList([nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1])) for i in range(self.domain_num)])
        # self.slot_bias = nn.ParameterList([nn.Parameter(torch.zeros(self.star_dim[1])) for i in range(self.domain_num)])
        # # torch.nn.init.xavier_uniform_(self.shared_weight.data)
        # # for m in (self.slot_weight):
        # #     torch.nn.init.xavier_uniform_(m.data)
        
        # self.star_mlp = MLP_N([self.star_dim[1], self.star_dim[2]])

        # self.shared_weight = nn.Parameter(MI_Expert(self.star_dim[:2], self.sigma_w))
        # self.shared_bias = nn.Parameter(torch.zeros(self.star_dim[1]))
        # self.slot_weight = nn.ParameterList([MI_Expert(self.star_dim[:2], self.sigma_w) for i in range(self.domain_num)])
        
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
        
        # skip = self.skip_conn(input_emb)
        # emb = torch.zeros((input_emb.shape[0], self.star_dim[1])).to(_device)
        # for i, (_weight, _bias) in enumerate(zip(self.slot_weight, self.slot_bias)):
        #     # _output = torch.matmul(input_emb, torch.multiply(_weight, self.shared_weight))+_bias+self.shared_bias
        #     _output = torch.matmul(input_emb, torch.multiply(_weight(input_emb), self.shared_weight(input_emb)))+_bias+self.shared_bias
        #     emb = torch.where(mask[i].unsqueeze(1).to(_device), _output, emb)
        # emb = self.star_mlp(emb)+skip
        
        cov_emb = input_emb
        for f_cov in self.star_cov.cov_expert:
            cov_emb = f_cov(cov_emb)
        emb, _ = self.star_cov.reparameters(emb, cov_emb)
        
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
        fused_fea = [torch.concat((shared_fea, 
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

# concat of three types of experts
class MTMD_Domain_Task_Expert_Concat_MI_Base(nn.Module):
    # add domain/task-specific expert based on MTMD, only keep one shared expert and remove gate
    
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num, sigma_w):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims 
        self.domain_num = domain_num
        self.task_num = task_num
        self.expert_num = expert_num
        self.embedding = EmbeddingLayer(features)
        self.sigma_w = sigma_w

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
            self.expert.append(MI_Expert(self.fcn_dim, self.sigma_w))
            
        self.domain_expert = nn.ModuleList()
        for d in range(domain_num):
            self.domain_expert.append(MI_Expert(self.fcn_dim, self.sigma_w))
            
        self.task_expert = nn.ModuleList()
        for d in range(task_num):
            self.task_expert.append(MI_Expert(self.fcn_dim, self.sigma_w))
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], expert_num), torch.nn.Softmax(dim=1)) for i in range(domain_num*task_num)])
        
        self.tower = nn.ModuleList()
        for d in range(domain_num*task_num):
            domain_specific = nn.Sequential(
                nn.Linear(self.fcn_dim[-1]*3, self.fcn_dim[-1]),
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
        fused_fea = [torch.concat((shared_fea, 
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

class MTMD_Domain_Task_Expert_Concat_MI_Base_Negative(MTMD_Domain_Task_Expert_Concat_MI_Base):
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num):
        super().__init__(features, domain_num, task_num, fcn_dims, expert_num)

class MTMD_Domain_Task_Expert_Concat_MI_Close_SD_ST(MTMD_Domain_Task_Expert_Concat_MI_Base):
    # add domain/task-specific expert based on MTMD, only keep one shared expert and remove gate
    
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num, sigma_w):
        super().__init__(features, domain_num, task_num, fcn_dims, expert_num, sigma_w)

    def kdl_sd_st(self, shared_mean, shared_cov, domain_mean_list, domain_cov_list, task_mean_list, task_cov_list, mask):
        self.kdl_loss_sd = 0
        self.kdl_loss_st = 0

        # seperate shared/domain matrices by domain_id
        # domain_sample_size, d
        for d in range(self.domain_num):
            self.kdl_loss_sd += self._kdl_gauss(shared_mean[mask[d]], shared_cov[mask[d]], domain_mean_list[d][mask[d]], domain_cov_list[d][mask[d]])
        # calculate mutual information between shared with task
        for t in range(self.task_num):
            self.kdl_loss_st += self._kdl_gauss(shared_mean, shared_cov, task_mean_list[t], task_cov_list[t])
        return self.kdl_loss_sd + self.kdl_loss_st
        
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
        fused_fea = [torch.concat((shared_fea, 
                                   domain_fea[:, i%self.domain_num, :], 
                                   task_fea[:, i%self.task_num, :]
                                  ), -1)
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

class MTMD_Domain_Task_Expert_Concat_MI_Close_SD(MTMD_Domain_Task_Expert_Concat_MI_Close_SD_ST):
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num, sigma_w):
        super().__init__(features, domain_num, task_num, fcn_dims, expert_num, sigma_w)
class MTMD_Domain_Task_Expert_Concat_MI_Close_ST(MTMD_Domain_Task_Expert_Concat_MI_Close_SD_ST):
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num, sigma_w):
        super().__init__(features, domain_num, task_num, fcn_dims, expert_num, sigma_w)

class MTMD_Domain_Task_Expert_Concat_MI_Pull_SD_ST(MTMD_Domain_Task_Expert_Concat_MI_Close_SD_ST):
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num, sigma_w):
        super().__init__(features, domain_num, task_num, fcn_dims, expert_num, sigma_w)
class MTMD_Domain_Task_Expert_Concat_MI_Pull_SD(MTMD_Domain_Task_Expert_Concat_MI_Close_SD_ST):
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num, sigma_w):
        super().__init__(features, domain_num, task_num, fcn_dims, expert_num, sigma_w)
class MTMD_Domain_Task_Expert_Concat_MI_Pull_ST(MTMD_Domain_Task_Expert_Concat_MI_Close_SD_ST):
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num, sigma_w):
        super().__init__(features, domain_num, task_num, fcn_dims, expert_num, sigma_w)

class MTMD_Domain_Task_Expert_Concat_MI_Push_Domain(MTMD_Domain_Task_Expert_Concat_MI_Base):
    # add domain/task-specific expert based on MTMD, only keep one shared expert and remove gate
    
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num, sigma_w):
        super().__init__(features, domain_num, task_num, fcn_dims, expert_num, sigma_w)

    def orth_domain(self, domain_fea, mask):
        self.orth_loss_domain = 0
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        # seperate shared/domain matrices by domain_id, calculate cos(d1, d2), cos(d1, d3), and cos(d2, d3) for each domain expert output, where d1 is the average of domain 1 samples from one domain expert
        # domain_sample_size, d
        for di in range(self.domain_num):
            di_ave = torch.mean(domain_fea[:, di, :][mask[di]], dim=0)
            dj_list = list(range(self.domain_num))
            dj_list.remove(di)
            for dj in dj_list:
                dj_ave = torch.mean(domain_fea[:, di, :][mask[dj]], dim=0)
                self.orth_loss_domain += cos(di_ave, dj_ave)
        return self.orth_loss_domain
        
        # seperate shared/domain matrices by domain_id, calculate cos(d1, d2), cos(d1, d3), and cos(d2, d3) for each domain expert output, where d1 is the average of domain 1 samples from one domain expert
        # domain_sample_size, d
        # for di in range(self.domain_num-1):
        #     for dj in range(di+1, self.domain_num):
        #         di_ave = torch.mean(domain_fea[:, di, :][mask[di]], dim=0)
        #         dj_ave = torch.mean(domain_fea[:, dj, :][mask[dj]], dim=0)
        #         self.orth_loss_domain += cos(di_ave, dj_ave)
        # return self.orth_loss_domain
        
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
        fused_fea = [torch.concat((shared_fea, 
                                   domain_fea[:, i%self.domain_num, :], 
                                   task_fea[:, i%self.task_num, :]
                                  ), -1)
                     for i in range(self.task_num*self.domain_num)]

        orth_loss_domain = self.orth_domain(domain_fea, mask)

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
            
class MTMD_Domain_Task_Expert_Concat_MI_Push_T(MTMD_Domain_Task_Expert_Concat_MI_Base):
    # add domain/task-specific expert based on MTMD, only keep one shared expert and remove gate
    
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num, sigma_w):
        super().__init__(features, domain_num, task_num, fcn_dims, expert_num, sigma_w)

    def kdl_push_t(self, task_mean_list, task_cov_list, mask):
        kdl_t = 0
        
        # seperate shared/domain matrices by domain_id
        # domain_sample_size, d
        # calculate kl(t0, t1)
        for t1 in range(self.task_num):
            for t2 in range(t1 + 1, self.task_num):
                kdl_t += self._kdl_gauss(task_mean_list[t1], task_cov_list[t1], 
                                         task_mean_list[t2], task_cov_list[t2])
        # return _KL_Weight * _KL_Weight * kdl_t
        return kdl_t
    def orth_push_t(self, task_fea):
        orth_loss = 0
        cov = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for t1 in range(self.task_num):
            for t2 in range(t1 + 1, self.task_num):
                ti_ave = torch.mean(task_fea[:, t1, :], dim=0)
                tj_ave = torch.mean(task_fea[:, t2, :], dim=0)
                orth_loss += cov(ti_ave, tj_ave)
        return orth_loss
        
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
        fused_fea = [torch.concat((shared_fea, 
                                   domain_fea[:, i%self.domain_num, :], 
                                   task_fea[:, i%self.task_num, :]
                                  ), -1)
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
        self.orth_loss_push_t = self.orth_push_t(task_fea)

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


class MTMD_Domain_Task_Expert_Concat_MI_Close_SDST_Push_T(MTMD_Domain_Task_Expert_Concat_MI_Base):
    # add domain/task-specific expert based on MTMD, only keep one shared expert and remove gate
    
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num, sigma_w):
        super().__init__(features, domain_num, task_num, fcn_dims, expert_num, sigma_w)

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
        # return _KL_Weight * _KL_Weight * kdl_t
        return kdl_t
        
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
        fused_fea = [torch.concat((shared_fea, 
                                   domain_fea[:, i%self.domain_num, :], 
                                   task_fea[:, i%self.task_num, :]
                                  ), -1)
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

