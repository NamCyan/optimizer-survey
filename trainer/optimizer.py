import torch.optim as optim
import torch
import math
from torch import Tensor
from typing import List, Optional

# from torch import _functional as F
from torch.optim.optimizer import Optimizer
# import torch.optim.SGD  as SGD
# import torch.optim.Adam as Adam
from trainer.demon_utils import sgd, adam


def get_optimizer(args, model, T=None):
    if args.optim == "Adam":
        betas = [args.adam_beta1, args.adam_beta2]
        return torch.optim.Adam(model.parameters(), lr=args.lr, betas= betas)
    elif args.optim == "DemonAdam":
        betas = [args.adam_beta1, args.adam_beta2]
        return DemonAdam(T, model.parameters(), lr=args.lr, betas= betas)
    elif args.optim == "AMSGrad":
        betas = [args.amsgrad_beta1, args.amsgrad_beta1]
        return torch.optim.AMSGrad(model.parameters(), lr=args.lr, betas= betas, amsgrad=True)
    elif args.optim == "AdamW":
        betas = [args.adamw_beta1, args.adamw_beta2]
        return torch.optim.AdamW(model.parameters(), lr=args.lr, betas= betas, weight_decay= args.adamw_weight_decay)
    elif args.optim == "SGDM":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum = args.sgdm_momentum)
    elif args.optim == 'DemonSGD':
        return DemonSGD(T, model.parameters(), lr=args.lr, momentum = args.sgdm_momentum)
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
        
class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

class DemonSGD(Optimizer):
    r"""
    Demon - SGD - pytorch
    T: total number of iterations
    
    """

    def __init__(self, T, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(DemonSGD, self).__init__(params, defaults)
        self.T = T
        self._iterations = 1
        
    # see Algorithm 1 in the paper
    def _momentum_decay(self, beta_init):
        if self._iterations is not None:
            decay_rate = float(1.0 - self._iterations / self.T)
        else:
            decay_rate = 1.0
        beta_decay = beta_init * decay_rate
        beta = beta_decay / ((1.0 - beta_init) + beta_decay)
        self._iterations += 1
        return beta
    
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = self._momentum_decay(group['momentum'])
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            sgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  dampening=dampening,
                  nesterov=nesterov)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss
        
        
class DemonAdam(Optimizer):
    r"""
    DemonAdam
    T: total number of iterations
    """

    def __init__(self, T, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, *, foreach: Optional[bool] = None,
                 maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        maximize=maximize, foreach=foreach)
        super(DemonAdam, self).__init__(params, defaults)
        self.iterations = 1
        self.T = T
        
    # see Algorithm 2 in the paper
    def _momentum_decay(self, beta_init):
        if self.iterations is not None:
            decay_rate = float(1.0 - self.iterations / self.T)
        else:
            decay_rate = 1.0
        beta_decay = beta_init * decay_rate
        beta = beta_decay / ((1.0 - beta_init) + beta_decay)
        self.iterations += 1
        return beta
        
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            beta1 = self._momentum_decay(beta1)
            
            # print(group.keys())

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = torch.tensor(0.)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    state_steps.append(state['step'])

            adam(params_with_grad,
                 grads,
                 exp_avgs,
                 exp_avg_sqs,
                 max_exp_avg_sqs,
                 state_steps,
                 amsgrad=group['amsgrad'],
                 beta1=beta1,
                 beta2=beta2,
                 lr=group['lr'],
                 weight_decay=group['weight_decay'],
                 eps=group['eps'],
                 maximize=group['maximize'],
                 foreach=group['foreach'])

        return loss

