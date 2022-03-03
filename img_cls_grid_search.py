import os

lrs = [0.0005, 0.001, 0.005, 0.008, 0.01, 0.05, 0.08, 0.1, 0.5]
betas = [0.8, 0.85, 0.9, 0.95, 0.99, 0.999]
models = ["LeNet", "ResNet18"]
optimizers = [
  "Adam",
  "AMSGrad",
  "AdamW", 
  "AggMo", 
  "QHM",
  "QHAdam",
  'SGDM',
  'DemonSGD', 
  'DemonAdam'
]

optimizer_param = {}
for optimizer in optimizers:
    optimizer_param[optimizer] = {}
    
    optimizer_param[optimizer]["lr"] = lrs
    if optimizer == "AggMo":
        optimizer_param[optimizer]["aggmo_num_betas"] = list(range(2,7))
    elif optimizer == "Adam" or optimizer == "DemonAdam":
        optimizer_param[optimizer]["adam_beta1"] = betas
    elif  optimizer == "AdamW":
        optimizer_param[optimizer]["adamw_weight_decay"] = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]
    elif  optimizer == "QHM":
        optimizer_param[optimizer]["qhm_beta"] = betas
    elif  optimizer == "QHAdam":
        optimizer_param[optimizer]["qhadam_beta1"] = betas
    elif  optimizer == 'SGDM' or optimizer == 'DemonSGD':
        optimizer_param[optimizer]["sgdm_momentum"] = betas

cmds = []

for model in models:
    for optimizer in optimizer_param:
        param_name1, param_name2 = optimizer_param[optimizer].keys()
        for param_value1 in optimizer_param[optimizer][param_name1]:
            for param_value2 in optimizer_param[optimizer][param_name2]:
                cmd = "python3 main.py --task img_cls --dataset cifar10 --epochs 100 --batch_size 128 --optim {} --model_name {} --{} {} --{} {} --device cuda".format(optimizer, model, param_name1, param_value1, param_name2, param_value2)
                cmds.append(cmd)

print("Total run exp:", len(cmds))
for cmd in cmds:
    print("Running:", cmd)
    os.system(cmd)

print("==Finish=="*100)