from arguments import get_args
import math
import random
import numpy as np
import torch

args = get_args()

if args.dataset == "cifar10":
    from loader import cifar10 as dataloader
    loader_train, loader_valid, loader_test, num_classes = dataloader.load(args.valid_rate, args.batch_size)
elif args.dataset == "cola":
    from loader import cola as dataloader
    loader_train, loader_valid, loader_test, num_classes = dataloader.load(args.valid_rate, args.batch_size, args.max_length)
elif args.dataset == "imdb":
    from loader import imdb as dataloader
    loader_train, loader_valid, loader_test, num_classes = dataloader.load(args.valid_rate, args.batch_size, args.max_length)
else:
    raise Exception("Do not support {} dataset".format(args.dataset))
    
if args.task == "img_cls":
    from trainer.ImageCLS import Trainer
elif args.task == "text_cls":
    from trainer.TextCLS import Trainer
else:
    raise Exception("Wrong task name!")

if args.model_name == "LeNet":
    from models.LeNet import Net
    model = Net(num_classes= num_classes)
elif args.model_name == "ResNet18":
    from torchvision.models import resnet18 as Net
    model = Net(num_classes= num_classes)
elif args.model_name == "LSTM":
    from models.LSTM import Net
    model = Net(num_classes= num_classes)

model.to(args.device)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

print("Model information:")
print(model)
print("="*100)

print("Arguments:")
list_args = args._get_kwargs()
for k,v in list_args:
    print("\t{} : {}".format(k,v))
print("="*100)

trainer = Trainer(args, model, loader_train, loader_valid, loader_test)
trainer.train()

test_loss, test_acc, test_f1, test_report = trainer.eval(trainer.model, loader_test)

print("="*100)
print("Test: loss= {:.3f} || accuracy= {:.3f}% || F1= {:.3f}%".format(test_loss, test_acc, test_f1))
print("Report \n", test_report)