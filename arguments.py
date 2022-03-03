import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Advanced DL')
    # Arguments
    parser.add_argument('--seed', type=int, default=21, 
                        help='initial seed')
    parser.add_argument('--task', default="img_cls", type=str,
                        choices= ["img_cls", "text_cls", "img_gen"],
                        help='task')
    parser.add_argument('--dataset', default="cifar10", type=str,
                        help='dataset')
    parser.add_argument('--model_name', default='LeNet', type=str, 
                        choices= ["LeNet", 
                                  "ResNet18",
                                  "LSTM",
                                  "BERT",
                                  "DistilBERT",
                                  'VAE',
                                  'NCSN'],
                        help='model type')
    parser.add_argument('--optim', default='AggMo', type=str, required=True,
                        choices= ["Adam",
                                  "AMSGrad",
                                  "AdamW", 
                                  "AggMo", 
                                  "QHM", 
                                  "QHAdam",
                                  'SGDM',
                                  'DemonSGD', 
                                  'DemonAdam'],
                        help='Optimizer')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate')
    parser.add_argument('--device', default="cuda", type=str, 
                        help='device to train')
    parser.add_argument('--valid_rate', default=0.1, type=float,
                        help='Valid split rate')
    # Text Classification
    parser.add_argument('--max_length', default=64, type=int, 
                        help='Max sequence length')

    
    # SGDM + Demon SGD parameter
    parser.add_argument('--sgdm_momentum', default=0.9, type=float,
                        help='sgdm momentum')

    # Adam + Demon Adam parameter
    parser.add_argument('--adam_beta1', default=0.9, type=float,
                        help='adam beta1')
    parser.add_argument('--adam_beta2', default=0.999, type=float,
                        help='adam beta2')

    # AMSGrad parameter
    parser.add_argument('--amsgrad_beta1', default=0.9, type=float,
                        help='amsgrad beta1')
    parser.add_argument('--amsgrad_beta2', default=0.999, type=float,
                        help='amsgrad beta2')

    # AdamW parameter
    parser.add_argument('--adamw_beta1', default=0.9, type=float,
                        help='adamw beta1')
    parser.add_argument('--adamw_beta2', default=0.999, type=float,
                        help='adamw beta2')
    parser.add_argument('--adamw_weight_decay', default=0.01, type=float,
                        help='adamw weight decay')

    # AggMo parameters
    parser.add_argument('--aggmo_num_betas', default=3, type=int,
                        help='Number of betas')

    # QHM parameter
    parser.add_argument('--qhm_nu', default=0.7, type=float,
                        help='v in the paper')
    parser.add_argument('--qhm_beta', default=0.9, type=float,
                        help='v in the paper')
    # QHAdam parameter
    parser.add_argument('--qhadam_nu1', default=0.7, type=float,
                        help='qhadam v1')
    parser.add_argument('--qhadam_nu2', default=1.0, type=float,
                        help='qhadam v2')
    parser.add_argument('--qhadam_beta1', default=0.9, type=float,
                        help='qhadam beta1')
    parser.add_argument('--qhadam_beta2', default=0.999, type=float,
                        help='qhadam beta2')
    args = parser.parse_args()
    return args