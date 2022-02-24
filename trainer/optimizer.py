import torch_optimizer as optim
import torch
import math

def get_optimizer(args, model):
    if args.optim == "Adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "AdamW":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
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
        betas = args.qhm_beta
        optimizer = optim.QHM(model.parameters(), lr=args.lr, momentum=betas, nu=args.qhm_nu)
        return optimizer
    elif args.optim == "QHAdam":
        betas = [args.qhadam_beta1, args.qhadam_beta2]
        nus = [args.qhadam_nu1, args.qhadam_nu2]
        optimizer = optim.QHAdam(model.parameters(), lr= args.lr, nus= nus, betas= betas)
        return optimizer
    else:
        raise Exception('Have not implement {} optimizer yet'.format(args.optim))