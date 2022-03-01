import torch_optimizer as optim
import torch
import math

def get_optimizer(args, model):
    if args.optim == "Adam":
        betas = [args.adam_beta1, args.adam_beta2]
        return torch.optim.Adam(model.parameters(), lr=args.lr, betas= betas)
    elif args.optim == "AMSGrad":
        betas = [args.amsgrad_beta1, args.amsgrad_beta1]
        return torch.optim.Adam(model.parameters(), lr=args.lr, betas= betas, amsgrad=True)
    elif args.optim == "AdamW":
        betas = [args.adamw_beta1, args.adamw_beta2]
        return torch.optim.AdamW(model.parameters(), lr=args.lr, betas= betas, weight_decay= args.adamw_weight_decay)
    elif args.optim == "SGD":
        return torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == "AggMo":
      betas = []
      for i in range(args.aggmo_num_betas):
          beta = 1 - math.pow(0.1, i)
          betas.append(beta)
      optimizer = optim.AggMo(model.parameters(), lr=args.lr, betas= betas)
      return optimizer
    elif args.optim == "QHM":
        optimizer = optim.QHM(model.parameters(), lr=args.lr, momentum=args.qhm_beta, nu=args.qhm_nu)
        return optimizer
    elif args.optim == "QHAdam":
        betas = [args.qhadam_beta1, args.qhadam_beta2]
        nus = [args.qhadam_nu1, args.qhadam_nu2]
        optimizer = optim.QHAdam(model.parameters(), lr= args.lr, nus= nus, betas= betas)
        return optimizer
    else:
        raise Exception('Have not implement {} optimizer yet'.format(args.optim))