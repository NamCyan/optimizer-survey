from arguments import get_args
import math
import random
import numpy as np
import torch
import json
from time import time
import os, csv

args = get_args()
list_args = args._get_kwargs()

optim_parameter_path = "../results/optim_parameters.json"
if not os.path.isfile(optim_parameter_path):
    with open(optim_parameter_path, "w") as f:
        json.dump({}, f, indent= 4)

with open(optim_parameter_path, "r") as f:
    exp_ids = json.load(f)

log_dir = "../results/log/"
new_exp_id = len(os.listdir(log_dir))
log_file = "{}_{}_{}_{}.csv".format(args.optim, args.task, args.model_name, new_exp_id)
args.log_file = os.path.join(log_dir, log_file)

# save new experiment parameters
# assert log_file not in exp_ids
exp_ids[log_file] = {}
for param in list_args:
    exp_ids[log_file][param[0]] = param[1]

with open(optim_parameter_path, "w") as f:
    json.dump(exp_ids, f, indent= 4) 

if args.model_name == "BERT":
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case= True)
elif args.model_name == "DistilBERT":
    from transformers import DistilBertTokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased", do_lower_case= True)


if args.dataset == "cifar10":
    from loader import cifar10 as dataloader
    loader_train, loader_valid, loader_test, num_classes = dataloader.load(args.valid_rate, args.batch_size)
elif args.dataset == "mnist":
    from loader import mnist as dataloader
    loader_train, loader_valid, loader_test, num_classes = dataloader.load(args.valid_rate, args.batch_size)
elif args.dataset == "cola":
    from loader import cola as dataloader
    if "bert" in args.model_name.lower():
        loader_train, loader_valid, loader_test, num_classes = dataloader.loadBERT(args.valid_rate, args.batch_size, args.max_length, tokenizer, args.model_name)
    else:
        loader_train, loader_valid, loader_test, num_classes = dataloader.load(args.valid_rate, args.batch_size, args.max_length, args.model_name)
elif args.dataset == "imdb":
    from loader import imdb as dataloader
    if "bert" in args.model_name.lower():
        loader_train, loader_valid, loader_test, num_classes = dataloader.loadBERT(args.valid_rate, args.batch_size, args.max_length, tokenizer, args.model_name)
    else:
        loader_train, loader_valid, loader_test, num_classes = dataloader.load(args.valid_rate, args.batch_size, args.max_length, args.model_name)
else:
    raise Exception("Do not support {} dataset".format(args.dataset))


if args.model_name == "LeNet":
    from models.LeNet import Net
    model = Net(num_classes= num_classes)
elif args.model_name == "ResNet18":
    from torchvision.models import resnet18 as Net
    model = Net(num_classes= num_classes)
elif args.model_name == "LSTM":
    from models.LSTM import Net
    model = Net(num_classes= num_classes)
elif args.model_name == "BERT":
    from transformers import BertForSequenceClassification
    model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels= num_classes)
elif args.model_name == "DistilBERT":
    from transformers import DistilBertForSequenceClassification
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels= num_classes)
elif args.model_name == "VAE":
    from models.VAE import Net
    model = Net()
elif args.model_name == "NCSN":
    from models.NCSN import Net
    model = Net()

if args.task == "img_cls":
    from trainer.ImageCLS import Trainer
    trainer = Trainer(args, model, loader_train, loader_valid, loader_test)
elif args.task == "text_cls":
    from trainer.TextCLS import Trainer
    trainer = Trainer(args, model, loader_train, loader_valid, loader_test)
elif args.task == "img_gen":
    from trainer.ImageGen import Trainer
    trainer = Trainer(args, model, loader_train, loader_valid, loader_test)
else:
    raise Exception("Wrong task name!")

model.to(args.device)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

print("Model information:")
print(model)
print("="*100)

print("Arguments:")
for k,v in list_args:
    print("\t{} : {}".format(k,v))
print("="*100)

trainer.train()

if args.task != "img_gen":
    time1= time()
    test_loss, test_acc, test_f1, test_report = trainer.eval(trainer.model, loader_test)
    time2= time()
    with open(args.log_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([args.epochs,
                      None, None, None, 
                      None, None, None,
                      test_loss, test_acc, test_f1,
                      time2 - time1])
    print("="*100)
    print("Test: loss= {:.3f} || accuracy= {:.3f}% || F1= {:.3f}%".format(test_loss, test_acc, test_f1))
    print("Report \n", test_report)
else:
    pass

# model_save_path = args.log_file.replace(".csv", ".pt").replace("log", "models")
# torch.save(trainer.model.state_dict(), model_save_path)
