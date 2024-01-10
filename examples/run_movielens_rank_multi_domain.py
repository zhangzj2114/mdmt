import sys
import os
sys.path.append("..")
import os 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from HAMUR.basic.features import DenseFeature, SparseFeature
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from HAMUR.trainers import CTRTrainer
from HAMUR.utils.data import DataGenerator
from HAMUR.models.multi_domain import *

# Mlp_7_Layer, DCN_MD, MMOE, MTMD,STAR,MTMD_Domain_Expert,MTMD_Task_Expert,MTMD_Domain_Task_Expert,MTMD_Simple_Star,MTMD_Skip,MTMD_Domain_Task_Expert_wo_shared,MTMD_Domain_Task_Expert_Gate_Shared, MTMD_Domain_Task_Expert_Individual_Gate,MTMD_Task_Expert_wo_Shared,MTMD_Domain_Expert_wo_Shared
import warnings

# Filter UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_movielens_data_rank_multidomain(task, domain, data_path="examples/ranking/data/ml-1m"):
    data = pd.read_csv(data_path+"/ml-1m.csv")
    data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])
    del data["genres"]

    x_used_cols = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', 'cate_id', 'domain_indicator']
    
    group1 = {1, 18}
    group2 = {25}
    group3 = {35, 45, 50, 56}

    domain_num = 3

    data["domain_indicator"] = data["age"].apply(lambda x: map_group_indicator(x, [group1, group2, group3]))

    # 选择域：0, 1, 2, all
    if domain != 'all':
        data = data[data['domain_indicator']==int(domain)]
        assert len(data)!=0, 'wrong domain indicator'
    else:
        data = data
        
    useless_features = ['title', 'timestamp']

    dense_features = ['age']
    domain_split_feature = ['age']
    sparse_features = ['user_id', 'movie_id', 'gender', 'occupation', 'zip', "cate_id", "domain_indicator"]

    for feature in dense_features:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_features:
        sca = MinMaxScaler()  # scaler dense feature
        data[dense_features] = sca.fit_transform(data[dense_features])

    for feature in useless_features:
        del data[feature]
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1

    data['click'] = data['rating'].apply(lambda x: convert_target(x, 'click'))
    data['like'] = data['rating'].apply(lambda x: convert_target(x, 'like'))

    for feat in tqdm(sparse_features):  # encode sparse feature
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    dense_feas = [DenseFeature(x_used_cols.index(feature_name)) for feature_name in dense_features]
    sparse_feas = [SparseFeature(x_used_cols.index(feature_name), vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name
                   in sparse_features]

    # 选择任务：click, purchase, all
    if task == 'all':
        y = data[['click', 'like']]
    else:
        y = data[task]
    del data['click']
    del data['like']
    del data['rating']
    
    x, y = data.values, y.values

    if task != 'all':
        y = np.expand_dims(y, axis=1)

    train_idx, val_idx = int(len(data)*0.8), int(len(data)*0.9)
    x_train, y_train = x[:train_idx, :], y[:train_idx]
    x_val, y_val = x[train_idx:val_idx, :], y[train_idx:val_idx]
    x_test, y_test = x[val_idx:, :], y[val_idx:]
    print(f'train: {(x_train.shape)}, {(y_train.shape)}')
    print(f'val: {(x_val.shape)}, {(y_val.shape)}')
    print(f'test: {(x_test.shape)}, {(y_test.shape)}')
    # return dense_feas, sparse_feas,  data, y, domain_num
    return dense_feas, sparse_feas, x_train, y_train, x_val, y_val, x_test, y_test, domain_num


def map_group_indicator(age, list_group):
    l = len(list(list_group))
    for i in range(l):
        if age in list_group[i]:
            return i


def convert_target(val, target):
    v = int(val)
    if target == 'click':
        thre = 3 
    elif target == 'like':
        thre = 4
    else:
        assert 0, 'wrong target'
    if v > thre:
        return int(1)
    else:
        return int(0)


def convert_numeric(val):
    """
    Forced conversion
    """
    return int(val)


def df_to_dict_multi_domain(data, columns):
    """
    Convert the array to a dict type input that the network can accept
    Args:
        data (array): 3D datasets of type DataFrame (Length * Domain_num * feature_num)
        columns (list): feature name list
    Returns:
        The converted dict, which can be used directly into the input network
    """

    data_dict = {}
    for i in range(len(columns)):
        data_dict[columns[i]] = data[:, :, i]
    return data_dict
    
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main(args):
    setup_seed(args.seed)
    # dense_feas, sparse_feas, x, y ,domain_num= get_movielens_data_rank_multidomain(task, domain, dataset_path)
    dense_feas, sparse_feas, x_train, y_train, x_val, y_val, x_test, y_test, domain_num = get_movielens_data_rank_multidomain(args.task, args.domain, args.dataset_path)
    if args.domain != 'all':
        domain_num = 1
    task_num = 2 if args.task == 'all' else 1
    # fcn_dims = [1024, 512, 512, 256, 256, 64]
    fcn_dims = [512, 256, 256, 64] # 70w
    # fcn_dims = [256, 128, 128, 64] # 44w
    # fcn_dims = [128, 64, 64, 64] # 33w
    # fcn_dims = [64, 32, 32, 32] # 26w
    
    dg = DataGenerator(x_train, y_train)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test, batch_size=args.batch_size)
    
    if args.model_name == "mlp":
        model = Mlp_2_Layer(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
    elif args.model_name == "star":
        assert task_num == 1, 'star only supports single-task'
        model = STAR(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=fcn_dims)
    elif args.model_name == "mmoe":
        # fcn_dims_mmoe = [512, 256, 256, 64]
        fcn_dims_mmoe = [256, 64]
        model = MMOE(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims_mmoe, expert_num=args.expert_num)
    elif args.model_name == 'ple':
        model = PLE(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num)
    elif args.model_name == "mtmd":
        model = MTMD(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num) 
    elif args.model_name == "mtmd_dt":
        model = MTMD_DT_Expert(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num) 
    elif args.model_name == "mtmd_adv":
        model = MTMD_ADV(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num)    
    elif args.model_name == "mtmd_domain":
        model = MTMD_Domain_Expert(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num)       
    elif args.model_name == "mtmd_domain_add":
        model = MTMD_Domain_Expert_Add(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num)    
    elif args.model_name == "mtmd_domain_wo_shared":
        model = MTMD_Domain_Expert_wo_Shared(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num)
    elif args.model_name == "mtmd_task":
        model = MTMD_Task_Expert(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num)
    elif args.model_name == "mtmd_task_wo_shared":
        model = MTMD_Task_Expert_wo_Shared(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num)
    elif args.model_name == "mtmd_domain_task":
        model = MTMD_Domain_Task_Expert(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num)
    elif args.model_name == "mtmd_domain_task_wo_shared":
        model = MTMD_Domain_Task_Expert_wo_shared(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num)
    elif args.model_name == "mtmd_domain_task_gate_shared":
        model = MTMD_Domain_Task_Expert_Gate_Shared(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num)
    elif args.model_name == "mtmd_domain_task_gate_shared_skip":
        model = MTMD_Domain_Task_Expert_Gate_Shared_Skip(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num, skip_weight=args.skip_weight)
    elif args.model_name == "mtmd_domain_task_gate_shared_concat":
        model = MTMD_Domain_Task_Expert_Gate_Shared_Concat(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num)
    elif args.model_name == "mtmd_domain_task_gate_indi":
        model = MTMD_Domain_Task_Expert_Individual_Gate(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num)
    elif args.model_name == "mtmd_simple_star":
        model = MTMD_Simple_Star(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num)
    elif args.model_name == "mtmd_skip":
        model = MTMD_Skip(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num, skip_weight=args.skip_weight)
    elif args.model_name == "mtmd_skip_lora":
        model = MTMD_Skip_Lora(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num, skip_weight=args.skip_weight)
    elif args.model_name == "norm":
        fcn_dims = [256, 128, 128, 64]
        model = MTMD_Domain_Task_Expert_MI_Base(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num)
    elif args.model_name == "sdst":
        fcn_dims = [256, 128, 128, 64]
        model = MTMD_Domain_Task_Expert_MI_Close_SD_ST(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num)
    elif args.model_name == "push_t":
        fcn_dims = [256, 128, 128, 64]
        model = MTMD_Domain_Task_Expert_MI_Push_T(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num)
    elif args.model_name == "sdst_push_t":
        fcn_dims = [256, 128, 128, 64]
        model = MTMD_Domain_Task_Expert_MI_Close_SDST_Push_T(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num)
    elif args.model_name == "norm_concat":
        fcn_dims = [256, 128, 128, 64]
        model = MTMD_Domain_Task_Expert_Concat_MI_Base(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num, sigma_w=args.sigma_w)
    elif args.model_name == "norm_concat_star":
        fcn_dims = [256, 128, 128, 64]
        model = MTMD_Domain_Task_Expert_Concat_MI_STAR_MI_Base(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num, sigma_w=args.sigma_w)
    elif args.model_name == "norm_concat_wo_gate":
        fcn_dims = [256, 128, 128, 64]
        model = MTMD_Domain_Task_Expert_Wo_Gate_Base(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num, sigma_w=args.sigma_w)
    elif args.model_name == "norm_concat_neg":
        fcn_dims = [256, 128, 128, 64]
        model = MTMD_Domain_Task_Expert_Concat_MI_Base_Negative(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num)
    elif args.model_name == "sdst_concat":
        fcn_dims = [256, 128, 128, 64]
        model = MTMD_Domain_Task_Expert_Concat_MI_Close_SD_ST(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num, sigma_w=args.sigma_w)
    elif args.model_name == "sd_concat":
        fcn_dims = [256, 128, 128, 64]
        model = MTMD_Domain_Task_Expert_Concat_MI_Close_SD(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num, sigma_w=args.sigma_w)
    elif args.model_name == "st_concat":
        fcn_dims = [256, 128, 128, 64]
        model = MTMD_Domain_Task_Expert_Concat_MI_Close_ST(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num, sigma_w=args.sigma_w)
    elif args.model_name == "push_stsd_concat":
        fcn_dims = [256, 128, 128, 64]
        model = MTMD_Domain_Task_Expert_Concat_MI_Pull_SD_ST(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num, sigma_w=args.sigma_w)
    elif args.model_name == "push_sd_concat":
        fcn_dims = [256, 128, 128, 64]
        model = MTMD_Domain_Task_Expert_Concat_MI_Pull_SD(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num, sigma_w=args.sigma_w)
    elif args.model_name == "push_st_concat":
        fcn_dims = [256, 128, 128, 64]
        model = MTMD_Domain_Task_Expert_Concat_MI_Pull_ST(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num, sigma_w=args.sigma_w)
    elif args.model_name == "push_t_concat":
        fcn_dims = [256, 128, 128, 64]
        model = MTMD_Domain_Task_Expert_Concat_MI_Push_T(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num, sigma_w=args.sigma_w)
    elif args.model_name == "push_d_concat":
        fcn_dims = [256, 128, 128, 64]
        model = MTMD_Domain_Task_Expert_Concat_MI_Push_Domain(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num, sigma_w=args.sigma_w)
    elif args.model_name == "sdst_push_t_concat":
        fcn_dims = [256, 128, 128, 64]
        model = MTMD_Domain_Task_Expert_Concat_MI_Close_SDST_Push_T(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=fcn_dims, expert_num=args.expert_num, sigma_w=args.sigma_w)
    elif args.model_name == "mlp_7":
        model = Mlp_7_Layer(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=[512, 256, 256, 64])
    elif args.model_name == "mlp_adp":
        model = MLP_adap_2_layer_1_adp(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128],
                                       hyper_dims=[64], k=35)
    elif args.model_name == "dcn_md":
        model = DCN_MD(features=dense_feas + sparse_feas,num_domains=domain_num ,n_cross_layers=2, mlp_params={"dims": [256, 128]})
    elif args.model_name == "dcn_mmoe":
        fcn_dims = [128, 64]
        model = DCN_MMOE(features=dense_feas + sparse_feas, num_domains=domain_num, task_num=task_num, fcn_dims=fcn_dims, n_cross_layers=2, expert_num=args.expert_num, mlp_params={"dims": [256, 128]})
    elif args.model_name == "dcn_mtmd":
        fcn_dims = [128, 64]
        model = DCN_MTMD(features=dense_feas + sparse_feas, num_domains=domain_num, task_num=task_num, fcn_dims=fcn_dims, n_cross_layers=2, expert_num=args.expert_num, mlp_params={"dims": [256, 128]})
    elif args.model_name == "dcn_md_adp":
        model = DCN_MD_adp(features=dense_feas + sparse_feas,num_domains=domain_num, n_cross_layers=2, k = 30, mlp_params={"dims": [256, 128]}, hyper_dims=[128])
    elif args.model_name == "wd_md":
        model = WideDeep_MD(wide_features=dense_feas,num_domains= domain_num, deep_features=sparse_feas, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
    elif args.model_name == "wd_md_adp":
        model = WideDeep_MD_adp(wide_features=dense_feas,num_domains= domain_num, deep_features=sparse_feas,  k= 45,mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"}, hyper_dims=[128])
    else:
        assert 0, f'no such model: {args.model_name}'
    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable params: {trainable_params}')
    scheduler_fn=torch.optim.lr_scheduler.StepLR
    scheduler_params={"step_size": 4,"gamma": 0.8}
    # scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR
    # scheduler_params={"T_max": 150, "eta_min": 0}
    
    ctr_trainer = CTRTrainer(model, optimizer_params={"lr": args.learning_rate, "weight_decay": args.weight_decay}, n_epoch=args.epoch, earlystop_patience=10, device=args.device, model_path=args.save_dir,scheduler_fn=scheduler_fn,scheduler_params=scheduler_params)
    #scheduler_fn=torch.optim.lr_scheduler.StepLR,scheduler_params={"step_size": 2,"gamma": 0.8},
    
    ctr_trainer.fit(train_dataloader, warmup=args.warmup, val_dataloader=val_dataloader, n_w=args.n_w, sd_w=args.sd_w, st_w=args.st_w, t_w=args.t_w, orth_d_w=args.orth_d_w, orth_t_w=args.orth_t_w, w11=args.w11, w12=args.w12, w21=args.w21, w22=args.w22, w31=args.w31, w32=args.w32, domain_num=domain_num, task_num=task_num)
    logloss_list, auc_list, total_logloss, total_auc = ctr_trainer.evaluate_multi_domain(ctr_trainer.model, test_dataloader, domain_num, task_num=task_num)
    
    print(f'test auc: {total_auc} | test logloss: {total_logloss}')
    assert len(logloss_list) == domain_num*task_num
    col_name_ = list()
    auc_row_, log_row_ = list(), list()
    for d in range(domain_num):
        for t in range(task_num):
            auc_ = auc_list[d+t*domain_num]
            log_ = logloss_list[d+t*domain_num]
            col_name_.append(f'd{d}t{t}')
            auc_row_.append(auc_)
            log_row_.append(log_)
    col_name_ = [f'auc_{i}' for i in col_name_] + [f'log_{i}' for i in col_name_] + ['auc', 'log', 'seed', 'n_w', 'sd_w', 'st_w', 't_w', 'sigma_w', 'orth_d_w']
    # for t in range(task_num):
    #     for d in range(domain_num):
    #         print(f'domain: {d}, task: {t}, auc: {auc_list[d+t*domain_num]} | test logloss: {logloss_list[d+t*domain_num]}')
    #         col_name_.append(f'd{d}t{t}')
    # col_name_ = [f'auc_{i}' for i in col_name_] + [f'log_{i}' for i in col_name_] + ['auc', 'log']
            
    # save csv file
    import csv
    file_name_ = f'output/{args.model_name}_ml1m_domain_{args.domain}_task_{args.task}_exp_{args.expert_num}.csv'
    print_round_ = 4
    file_exists_ = os.path.isfile(file_name_)
    mode_ = 'a' if file_exists_ else 'w'
    with open(file_name_, mode_) as f:
        writer = csv.writer(f)
        if not file_exists_:
            writer.writerow(col_name_)
        writer.writerow([round(auc_, print_round_) for auc_ in auc_row_] + 
                        [round(log_, print_round_) for log_ in log_row_] + 
                        [round(total_auc, print_round_), round(total_logloss, print_round_), 
                         args.seed, args.n_w, args.sd_w, args.st_w, args.t_w, args.sigma_w, args.orth_d_w]
                       )
        
    print(f'wrote to file: {file_name_}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="../../data/ml-1m")
    parser.add_argument('--model_name', default='mlp_7')
    parser.add_argument('--epoch', type=int, default=50)  #100
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
    parser.add_argument('--decay_epoch', type=int, default=10, help='Decay learning rate after this epoch.')
    parser.add_argument('--batch_size', type=int, default=4096*10)  #4096
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0')  #cuda:0
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--expert_num', type=int, default=4)
    parser.add_argument('--task', type=str, default='all', choices=['click', 'like', 'all'])
    parser.add_argument('--domain', type=str, default='all', choices=['0', '1', '2', 'all'])
    parser.add_argument('--skip_weight', type=float, default=1)
    parser.add_argument('--sigma_w', type=float, default=1)
    parser.add_argument('--n_w', type=float, default=0.01)
    parser.add_argument('--sd_w', type=float, default=0.01)
    parser.add_argument('--st_w', type=float, default=1)
    parser.add_argument('--t_w', type=float, default=-1)
    parser.add_argument('--orth_d_w', type=float, default=-1, help='loss weight of orthogonal loss inside each domain expert')
    parser.add_argument('--orth_t_w', type=float, default=-1, help='loss weight of orthogonal loss between task experts')
    parser.add_argument('--w11', type=int, default=1, choices=[1, 2, 5])
    parser.add_argument('--w12', type=int, default=1, choices=[1, 2, 5])
    parser.add_argument('--w21', type=int, default=1, choices=[1, 2, 5])
    parser.add_argument('--w22', type=int, default=1, choices=[1, 2, 5])
    parser.add_argument('--w31', type=int, default=1, choices=[1, 2, 5])
    parser.add_argument('--w32', type=int, default=1, choices=[1, 2, 5])

    args = parser.parse_args()
    print(args)
    main(args)
    
