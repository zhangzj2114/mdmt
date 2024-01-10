import sys

sys.path.append("..")
import os 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import pandas as pd
import torch
from HAMUR.trainers import CTRTrainer
from HAMUR.basic.features import DenseFeature, SparseFeature
from HAMUR.utils.data import DataGenerator
from HAMUR.models.multi_domain import Mlp_7_Layer, MLP_adap_7_layer_2_adp, DCN_MD, DCN_MD_adp, WideDeep_MD, WideDeep_MD_adp


def get_ali_ccp_data_dict(task, domain, data_path='./data/ali-ccp'):
    # df_train = pd.read_csv(data_path + '/ali_ccp_train_sample.csv')
    # df_val = pd.read_csv(data_path + '/ali_ccp_val_sample.csv')
    # df_test = pd.read_csv(data_path + '/ali_ccp_test_sample.csv')

    _usecols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853'] + ['click', 'purchase'] + ['101', '121', '122', '301']
    # # 使用AliCCP全部数据集
    # df_train = pd.read_csv(data_path + '/ali_ccp_train.csv', usecols=_usecols)
    # df_val = pd.read_csv(data_path + '/ali_ccp_val.csv', usecols=_usecols)
    # df_test = pd.read_csv(data_path + '/ali_ccp_test.csv', usecols=_usecols)
    # print("train : val : test = %d %d %d" % (len(df_train), len(df_val), len(df_test)))
    # train_idx, val_idx = df_train.shape[0], df_train.shape[0] + df_val.shape[0]
    
    # # 选择域：1, 2, 3, all
    # if domain != 'all':
    #     data = pd.concat([df_train[df_train['301']==int(domain)], df_val[df_val['301']==int(domain)], df_test[df_test['301']==int(domain)]], axis=0)
    # else:
    #     data = pd.concat([df_train, df_val, df_test], axis=0)

    # 使用AliCCP训练集作为全部数据集
    # df_train = pd.read_csv(data_path + '/ali_ccp_val.csv', usecols=_usecols)
    # df_train = pd.read_csv(data_path + '/val_20w.csv', usecols=_usecols)
    # df_train = pd.read_csv(data_path + '/val_200w.csv', usecols=_usecols)
    df_train = pd.read_csv(data_path + '/val_1000w.csv', usecols=_usecols)

    df_train = df_train[df_train['301'] != 2]
    # 选择域：1, 2, 3, all
    if domain != 'all':
        data = df_train[df_train['301']==int(domain)]
    else:
        data = df_train
    print(f"data: {len(data)}")
    train_idx, val_idx = int(len(data)*0.6), int(len(data)*0.8)
    # domain_map = {1: 0, 2: 1, 3: 2}
    # domain_num = 3
    domain_map = {1: 0, 3: 1}
    domain_num = 2
    data["domain_indicator"] = data["301"].apply(lambda x : domain_map[x])
    data['id'] = list(range(train_idx)) + list(range(val_idx-train_idx)) + list(range(len(data)-val_idx))
    print(len(data['id']))
    data = data.set_index('id', drop=True)
    # data = data.reset_index()

    col_names = data.columns.values.tolist()
    dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
    sparse_cols = [col for col in col_names if col not in dense_cols and col not in ['click', 'purchase']]
    print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))

    dense_feas = [DenseFeature(col) for col in dense_cols]
    sparse_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in sparse_cols]

    # 选择任务：click, purchase, all
    if task == 'all':
        y = data[['click', 'purchase']]
        del data['click']
        del data['purchase']
        # del data[['click', 'purchase']]
    else:
        y = data[task]
        del data[task]
    x = data
    x_train, y_train = x[:train_idx], y[:train_idx]
    x_val, y_val = x[train_idx:val_idx], y[train_idx:val_idx]
    x_test, y_test = x[val_idx:], y[val_idx:]
    
    x_train, y_train = x_train.reset_index(), y_train.reset_index()
    x_val, y_val = x_val.reset_index(), y_val.reset_index()
    x_test, y_test = x_test.reset_index(), y_test.reset_index()
    
    print(f'x_train: {len(x_train)}')
    print(f'x_val: {len(x_val)}')
    print(f'x_test: {len(x_test)}')
    return dense_feas, sparse_feas, x_train, y_train, x_val, y_val, x_test, y_test, domain_num

def main(dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed, task, domain, w11, w12, w21, w22):
    torch.manual_seed(seed)
    dense_feas, sparse_feas, x_train, y_train, x_val, y_val, x_test, y_test,domain_num = get_ali_ccp_data_dict(task, domain, dataset_path)
    dg = DataGenerator(x_train, y_train)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test, batch_size=batch_size)
    if task == 'all':
        task_num = 2
    else:
        assert 'only support 2 tasks now.'

    if model_name == "mlp":
        model = Mlp_7_Layer(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num, fcn_dims=[1024, 512, 512, 256, 256, 64, 64])
    elif model_name == "mlp_adp":
        model = MLP_adap_7_layer_2_adp(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[1024, 512, 512, 256, 256, 64, 64],
                            hyper_dims=[64],  k=65)
    elif model_name == "dcn_md":
        model = DCN_MD(features=dense_feas + sparse_feas,num_domains=domain_num ,n_cross_layers=7, mlp_params={"dims": [512, 512, 256, 256, 64, 64]})
    elif model_name == "dcn_md_adp":
        model = DCN_MD_adp(features=dense_feas + sparse_feas, num_domains=domain_num, n_cross_layers=7, k = 25, mlp_params={"dims": [512, 512, 256, 256, 64, 64]}, hyper_dims=[64])
    elif model_name == "wd_md":
        model = WideDeep_MD(wide_features=dense_feas, deep_features=sparse_feas, num_domains=domain_num,mlp_params={"dims": [512, 512, 256, 256, 64, 64], "dropout": 0.2, "activation": "relu"})
    elif model_name == "wd_md_adp":
        model = WideDeep_MD_adp(wide_features=dense_feas, deep_features=sparse_feas, num_domains = domain_num, k= 25,mlp_params={"dims": [512, 512, 256, 256, 64, 64], "dropout": 0.2, "activation": "relu"}, hyper_dims=[64])

    print(model)
    ctr_trainer = CTRTrainer(model, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, earlystop_patience=5, scheduler_params={"step_size": 5,"gamma": 0.85},device=device, model_path=save_dir)
    # scheduler_fn=torch.optim.lr_scheduler.StepLR,scheduler_params={"step_size": 2,"gamma": 0.8},
    # auc1,auc2,auc3,auc4,auc5,auc6,auc = ctr_trainer.evaluate_multi_domain_auc(ctr_trainer.model, test_dataloader)
    # log1,log2,log3,log4,log5,log6,log = ctr_trainer.evaluate_multi_domain_logloss(ctr_trainer.model, test_dataloader)
    # auc1,auc2,auc4,auc5,auc = ctr_trainer.evaluate_multi_domain_auc(ctr_trainer.model, test_dataloader)
    # log1,log2,log4,log5,log = ctr_trainer.evaluate_multi_domain_logloss(ctr_trainer.model, test_dataloader)
    # print(f'test auc: {auc} | test logloss: {log}')
    ctr_trainer.fit(train_dataloader, val_dataloader=val_dataloader, w11=w11, w12=w12, w21=w21, w22=w22)
    auc1,auc2,auc4,auc5,auc = ctr_trainer.evaluate_multi_domain_auc(ctr_trainer.model, test_dataloader)
    log1,log2,log4,log5,log = ctr_trainer.evaluate_multi_domain_logloss(ctr_trainer.model, test_dataloader)
    print(f'test auc: {auc} | test logloss: {log}')
    print(f'domain 1 task 1 test auc: {auc1} | test logloss: {log1}')
    print(f'domain 2 task 1 test auc: {auc2} | test logloss: {log2}')
    # print(f'domain 3 task 1 test auc: {auc3} | test logloss: {log3}')
    print(f'domain 1 task 2 test auc: {auc4} | test logloss: {log4}')
    print(f'domain 2 task 2 test auc: {auc5} | test logloss: {log5}')
    # print(f'domain 3 task 2 test auc: {auc6} | test logloss: {log6}')

    # save csv file
    # import csv
    # with open(model_name+"_"+str(seed)+'.csv', "w", newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['model', 'seed', 'auc', 'log', 'auc1', 'log1', 'auc2', 'log2', 'auc3', 'log3'])
    #     writer.writerow([model_name, str(seed), auc, log, auc1,log1,auc2,log2,auc3,log3])


if __name__ == '__main__':
    import argparse
    import warnings
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="data/ali-ccp") #/home/xiaopli2/dataset/ali_ccp
    parser.add_argument('--model_name', default='mlp')
    parser.add_argument('--epoch', type=int, default=200)  #200
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4096*10)  #4096*10
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0')  #cuda:0
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--task', type=str, default='all', choices=['all'])
    parser.add_argument('--domain', type=str, default='all', choices=['all'])
    parser.add_argument('--w11', type=int, default=1, choices=[1, 2])
    parser.add_argument('--w12', type=int, default=1, choices=[1, 2])
    parser.add_argument('--w21', type=int, default=1, choices=[1, 2])
    parser.add_argument('--w22', type=int, default=1, choices=[1, 2])

    args = parser.parse_args()
    print(args)
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed, args.task, args.domain, args.w11, args.w12, args.w21, args.w22)
"""
python run_ali_ccp_ctr_ranking_multi_domain.py --model_name widedeep
python run_ali_ccp_ctr_ranking_multi_domain.py --model_name deepfm
python run_ali_ccp_ctr_ranking_multi_domain.py --model_name dcn
"""
