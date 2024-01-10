################################################################################
2024.1.8
Adversarial loss, generate user-wise loss weight (B*T values for T tasks and B samples), updated by 1batch validate data.
Modified based on MTMD. Dir is under baselines/MDMT_ADV/examples
cmd:
python run_movielens_rank_multi_domain.py --model_name mtmd_adv --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_adv --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_adv --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_adv --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_adv --device cuda:1 --seed 2026 

################################################################################
2023.12.26
Tune loss weight n_w (kl(expert||normal dist)), sd_w (kl(shared||domain)), st_w (kl(shared||task))
weight from "-1" "-0.1" "-0.03" "-0.01" "0" "0.01" "0.03" "0.1" "1"

Tune n_w
bash tune_norm_concat.sh
cmd inside:
python run_movielens_rank_multi_domain.py --model_name norm_concat --seed 2022 --sigma_w 0.01 --n_w $n_w

Tune sd_w
bash tune_close_sd.sh

Tune st_w
bash tune_close_st.sh

Tune sd_w and st_w
bash tune_close_sdst.sh

################################################################################
2023.12.26
please refer the mutual information part in 'old_files/readme.txt'

################################################################################
2023.12.19
Haven't implemented individual update yet. Modified mutual information 

################################################################################
2023.12.11
Change all BatchNorm to LayerNorm. Next update by domain individually; add min mutual information regularizer.
the dir is MDMT_Indi_Domain

################################################################################
2023.12.7
Tuned many domain/task specific expert and gate. Next build a branch to test automl-optimized loss weights.
Done, the dir is Auto-MTMD, where I modified ctr_trainer.py and adapter.py. 
The steps of training a model with automatically updated loss weights:
1. add parameters in init: , tau=1, tau_step=0.00005, softmax_type=0
2. add parameter loss_weight: self.loss_weight = Weights(domain_num*task_num, tau, tau_step, softmax_type)
3. identify the model and run cmd as in MDMT.


################################################################################ 
2023.11.30
Tuned all the preliminary research

Next:
Test several modifications:
1. fewer MLP structures to reach the trade-off between parameter volume and data complexity: 
v1. [1024, 512, 512, 256, 256, 64], v2. 70w [512, 256, 256, 64], v3. 40w [256, 128, 128, 64], v4: 33w [128, 64, 64, 64], v5: 26w [64, 32, 32, 32]


cmd:
Domain_specific expert
python run_movielens_rank_multi_domain.py --model_name mtmd_domain --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain --device cuda:1 --seed 2026

Domain_specific expert wo shared expert
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_wo_shared --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_wo_shared --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_wo_shared --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_wo_shared --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_wo_shared --device cuda:1 --seed 2026

Task_specific expert
python run_movielens_rank_multi_domain.py --model_name mtmd_task --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_task --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_task --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_task --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_task --device cuda:1 --seed 2026

Task_specific expert wo shared expert
python run_movielens_rank_multi_domain.py --model_name mtmd_task_wo_shared --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_task_wo_shared --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_task_wo_shared --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_task_wo_shared --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_task_wo_shared --device cuda:1 --seed 2026


Task_Domain specific expert
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --device cuda:1 --seed 2026

Task_Domain specific expert gate(emb)+d+t
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared --device cuda:1 --seed 2026

Task_Domain specific expert gate(emb) concat d concat t
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared_concat --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared_concat --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared_concat --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared_concat --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared_concat --device cuda:1 --seed 2026

Task_Domain specific expert gate(emb)||gate(d|)|gate(t)
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_indi --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_indi --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_indi --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_indi --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_indi --device cuda:1 --seed 2026

Compared with 7 shared experts
python run_movielens_rank_multi_domain.py --model_name mtmd --expert_num 7 --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --expert_num 7 --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --expert_num 7 --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --expert_num 7 --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --expert_num 7 --device cuda:1 --seed 2026
Compared with 9 shared experts
python run_movielens_rank_multi_domain.py --model_name mtmd --expert_num 9 --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --expert_num 9 --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --expert_num 9 --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --expert_num 9 --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --expert_num 9 --device cuda:1 --seed 2026

Task_Domain specific wo shared expert
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_wo_shared --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_wo_shared --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_wo_shared --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_wo_shared --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_wo_shared --device cuda:1 --seed 2026


Domain specific skip connection for each tower
python run_movielens_rank_multi_domain.py --model_name mtmd_skip --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip --device cuda:1 --seed 2026

python run_movielens_rank_multi_domain.py --model_name mtmd_skip --device cuda:1 --skip_weight 0.5 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip --device cuda:1 --skip_weight 0.5 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip --device cuda:1 --skip_weight 0.5 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip --device cuda:1 --skip_weight 0.5 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip --device cuda:1 --skip_weight 0.5 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip --device cuda:1 --skip_weight 2 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip --device cuda:1 --skip_weight 2 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip --device cuda:1 --skip_weight 2 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip --device cuda:1 --skip_weight 2 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip --device cuda:1 --skip_weight 2 --seed 2026

Domain/task-specific skip connection for each tower, with gated shared 
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared_skip --device cuda:1 --skip_weight 0.5 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared_skip --device cuda:1 --skip_weight 0.5 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared_skip --device cuda:1 --skip_weight 0.5 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared_skip --device cuda:1 --skip_weight 0.5 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared_skip --device cuda:1 --skip_weight 0.5 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared_skip --device cuda:1 --skip_weight 2 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared_skip --device cuda:1 --skip_weight 2 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared_skip --device cuda:1 --skip_weight 2 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared_skip --device cuda:1 --skip_weight 2 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task_gate_shared_skip --device cuda:1 --skip_weight 2 --seed 2026

Domain specific skip connection for each tower as lora
python run_movielens_rank_multi_domain.py --model_name mtmd_skip_lora --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip_lora --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip_lora --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip_lora --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip_lora --device cuda:1 --seed 2026

python run_movielens_rank_multi_domain.py --model_name mtmd_skip_lora --device cuda:1 --skip_weight 0.5 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip_lora --device cuda:1 --skip_weight 0.5 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip_lora --device cuda:1 --skip_weight 0.5 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip_lora --device cuda:1 --skip_weight 0.5 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip_lora --device cuda:1 --skip_weight 0.5 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip_lora --device cuda:1 --skip_weight 2 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip_lora --device cuda:1 --skip_weight 2 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip_lora --device cuda:1 --skip_weight 2 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip_lora --device cuda:1 --skip_weight 2 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_skip_lora --device cuda:1 --skip_weight 2 --seed 2026

Tune expert_num in task_Domain specific expert 
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --expert_num 1 --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --expert_num 1 --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --expert_num 1 --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --expert_num 1 --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --expert_num 1 --device cuda:1 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --expert_num 2 --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --expert_num 2 --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --expert_num 2 --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --expert_num 2 --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --expert_num 2 --device cuda:1 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --expert_num 3 --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --expert_num 3 --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --expert_num 3 --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --expert_num 3 --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd_domain_task --expert_num 3 --device cuda:1 --seed 2026


################################################################################
2023.11.20
Add a star layer before expert process as a new class MTMD

cmd:
Tune loss weight multi-domain multi-task star
python run_movielens_rank_multi_domain.py --model_name mtmd --w11 2 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w11 2 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w11 2 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w11 2 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w11 2 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w12 2 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w12 2 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w12 2 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w12 2 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w12 2 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w21 2 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w21 2 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w21 2 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w21 2 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w21 2 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w22 2 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w22 2 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w22 2 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w22 2 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w22 2 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w31 2 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w31 2 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w31 2 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w31 2 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w31 2 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w32 2 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w32 2 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w32 2 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w32 2 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w32 2 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w11 5 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w11 5 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w11 5 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w11 5 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w11 5 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w12 5 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w12 5 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w12 5 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w12 5 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w12 5 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w21 5 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w21 5 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w21 5 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w21 5 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w21 5 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w22 5 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w22 5 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w22 5 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w22 5 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w22 5 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w31 5 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w31 5 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w31 5 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w31 5 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w31 5 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w32 5 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w32 5 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w32 5 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w32 5 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --w32 5 --seed 2026


Single-domain multi-task mmoe
python run_movielens_rank_multi_domain.py --model_name mmoe --domain 0 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mmoe --domain 0 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mmoe --domain 0 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mmoe --domain 0 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mmoe --domain 0 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mmoe --domain 1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mmoe --domain 1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mmoe --domain 1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mmoe --domain 1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mmoe --domain 1 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mmoe --domain 2 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mmoe --domain 2 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mmoe --domain 2 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mmoe --domain 2 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mmoe --domain 2 --seed 2026 

Multi-domain single-task star
python run_movielens_rank_multi_domain.py --model_name star --task click --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name star --task click --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name star --task click --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name star --task click --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name star --task click --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name star --task like --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name star --task like --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name star --task like --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name star --task like --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name star --task like --seed 2026

Multi-domain multi-task mtmd 
python run_movielens_rank_multi_domain.py --model_name mtmd --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --seed 2027 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --seed 2028 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --seed 2029 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --seed 2020 &&
python run_movielens_rank_multi_domain.py --model_name mtmd --seed 2021

Sigle-domain single-task mlp_7
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 0 --task click --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 0 --task click --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 0 --task click --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 0 --task click --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 0 --task click --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 1 --task click --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 1 --task click --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 1 --task click --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 1 --task click --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 1 --task click --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 2 --task click --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 2 --task click --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 2 --task click --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 2 --task click --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 2 --task click --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 0 --task like --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 0 --task like --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 0 --task like --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 0 --task like --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 0 --task like --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 1 --task like --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 1 --task like --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 1 --task like --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 1 --task like --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 1 --task like --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 2 --task like --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 2 --task like --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 2 --task like --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 2 --task like --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mlp_7 --domain 2 --task like --seed 2026


Baselines:
DCN_MD:
python run_movielens_rank_multi_domain.py --model_name dcn_md --task click --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name dcn_md --task click --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name dcn_md --task click --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name dcn_md --task click --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name dcn_md --task click --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name dcn_md --task like --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name dcn_md --task like --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name dcn_md --task like --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name dcn_md --task like --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name dcn_md --task like --seed 2026
HAMUR:
python run_movielens_rank_multi_domain.py --model_name mlp_adp --task click --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mlp_adp --task click --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mlp_adp --task click --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mlp_adp --task click --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mlp_adp --task click --device cuda:1 --seed 2026 &&
python run_movielens_rank_multi_domain.py --model_name mlp_adp --task like --device cuda:1 --seed 2022 &&
python run_movielens_rank_multi_domain.py --model_name mlp_adp --task like --device cuda:1 --seed 2023 &&
python run_movielens_rank_multi_domain.py --model_name mlp_adp --task like --device cuda:1 --seed 2024 &&
python run_movielens_rank_multi_domain.py --model_name mlp_adp --task like --device cuda:1 --seed 2025 &&
python run_movielens_rank_multi_domain.py --model_name mlp_adp --task like --device cuda:1 --seed 2026


################################################################################
2023.11.17
Multi-task multi-domain mmoe 
cmd:

python run_movielens_rank_multi_domain.py --model_name mmoe --wt1 2 && 
python run_movielens_rank_multi_domain.py --model_name mmoe --wt1 2 && 
python run_movielens_rank_multi_domain.py --model_name mmoe --wt1 2 && 
python run_movielens_rank_multi_domain.py --model_name mmoe --wt1 2 && 
python run_movielens_rank_multi_domain.py --model_name mmoe --wt1 2  

python run_movielens_rank_multi_domain.py --model_name mmoe --wt2 2 && 
python run_movielens_rank_multi_domain.py --model_name mmoe --wt2 2 && 
python run_movielens_rank_multi_domain.py --model_name mmoe --wt2 2 && 
python run_movielens_rank_multi_domain.py --model_name mmoe --wt2 2 && 
python run_movielens_rank_multi_domain.py --model_name mmoe --wt2 2 

Next:
Implement Star as multi-domain fusion

