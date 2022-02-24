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
                                  "DistilBERT"],
                        help='model type')
    parser.add_argument('--optim', default='AggMo', type=str, required=True,
                        choices= ["Adam", 
                                  "AggMo", 
                                  "QHM", 
                                  "QHAdam"],
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
    parser.add_argument('--betas', default=0.9, type=float,
                        help='betas')
    # Text Classification
    parser.add_argument('--max_length', default=64, type=int, 
                        help='Max sequence length')

    # AggMo parameters
    parser.add_argument('--num_betas', default=3, type=int,
                        help='Number of betas')

    # QHM parameter
    parser.add_argument('--nu', default=0.7, type=float,
                        help='v in the paper')
    args = parser.parse_args()
    return args