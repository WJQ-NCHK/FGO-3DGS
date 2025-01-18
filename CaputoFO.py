import torch
from torch.optim import Optimizer
from scipy.special import gamma


class CaputoFO(Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, beta=0.1, delta=1e-8,
                 lr_adjustment_factor=0.15, lr_upper_bound=1e-2, lr_lower_bound=1e-4,
                 ema_alpha=1e-4, abc_nu=0.2, ml_threshold=5.0,param_history_length=None,  # 默认较大 N_term
                  clip_value=1.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, beta=beta, delta=delta, abc_nu=abc_nu,param_history_length = param_history_length)
        super(CaputoFO, self).__init__(params, defaults)
        self.lr_adjustment_factor = lr_adjustment_factor
        self.lr_upper_bound = lr_upper_bound
        self.lr_lower_bound = lr_lower_bound
        self.ema_alpha = ema_alpha
        # self.N_term = N_term  # 保留 N 项
        # self.lambda_decay = lambda_decay
        self.clip_value = clip_value

    def adjust_learning_rate(self, exp_avg, exp_avg_sq, group, state):
        current_avg_magnitude = torch.mean(torch.abs(exp_avg))
        current_avg_variance = torch.mean(exp_avg_sq)

        if 'ema_avg_magnitude' not in state:
            state['ema_avg_magnitude'] = current_avg_magnitude
        else:
            state['ema_avg_magnitude'] = (self.ema_alpha * state['ema_avg_magnitude'] +
                                          (1 - self.ema_alpha) * current_avg_magnitude)

        if 'ema_avg_variance' not in state:
            state['ema_avg_variance'] = current_avg_variance
        else:
            state['ema_avg_variance'] = (self.ema_alpha * state['ema_avg_variance'] +
                                         (1 - self.ema_alpha) * current_avg_variance)

        avg_magnitude = state['ema_avg_magnitude']
        avg_variance = state['ema_avg_variance']

        lr = group['lr']

        if avg_magnitude > avg_variance:
            new_lr = min(lr * (1 + self.lr_adjustment_factor), self.lr_upper_bound)
        else:
            new_lr = max(lr * (1 - self.lr_adjustment_factor), self.lr_lower_bound)

        # 使用 EMA 更新学习率
        group['lr'] = (1 - self.ema_alpha) * lr + self.ema_alpha * new_lr

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            param_history_length = group['param_history_length']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('CaputoFO does not support sparse gradients, consider SparseAdam instead')

                state = self.state[p]
                param_name = group.get('name', '')
                N_term = param_history_length.get(param_name, 1)
                # 初始化状态
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['history_grads'] = []
                    state['history_params'] = []
                else:
                    # 检查参数形状变化
                    if len(state['history_params']) > 0 and state['history_params'][-1].shape != p.data.shape:
                        # 重置历史，因为形状发生了变化
                        state['history_grads'] = []
                        state['history_params'] = []

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                beta = group['beta']
                delta = group['delta']
                # lambda_decay = self.lambda_decay
                # clip_value = self.clip_value
                state['step'] += 1

                # 保存当前梯度和参数到历史中
                if len(state['history_grads']) >= N_term:
                    state['history_grads'].pop(0)
                state['history_grads'].append(grad.clone())

                if len(state['history_params']) >= N_term:
                    state['history_params'].pop(0)
                state['history_params'].append(p.data.clone())

                # 计算分数阶梯度累积
                frac_grad = torch.zeros_like(p.data)
                gamma_values = torch.tensor(
                    [gamma(i + 2 - beta) for i in range(len(state['history_grads']))],
                    device=p.device,
                    dtype=p.data.dtype
                )
                # lambda_powers = torch.tensor(
                #     [lambda_decay ** i for i in range(len(state['history_grads']))],
                #     device=p.device,
                #     dtype=p.data.dtype
                # )

                weights = []
                weight_sum = 0.0
                for i in range(len(state['history_grads'])):
                    if i == 0:
                        param_diff = torch.zeros_like(p.data)
                    else:
                        param_diff = torch.abs(state['history_params'][i] - state['history_params'][i - 1])

                    weight = (param_diff + delta).pow(i + 1 - beta)
                    # weight = weight * lambda_decay ** i  # 应用衰减因子
                    weight = weight / gamma_values[i]
                    weights.append(weight)
                    weight_sum += weight

                # 归一化权重
                for i, hist_grad in enumerate(state['history_grads']):
                    frac_grad += hist_grad * weights[i] / weight_sum

                # 正则化或裁剪 frac_grad
                # frac_grad = torch.clamp(frac_grad, min=-clip_value, max=clip_value)
                frac_grad = frac_grad / (frac_grad.norm() + group['eps'])  # 可选归一化

                # 更新一阶和二阶动量
                exp_avg.mul_(beta1).add_(frac_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(frac_grad, frac_grad, value=1 - beta2)

                # 调整学习率
                self.adjust_learning_rate(exp_avg, exp_avg_sq, group, state)

                denom = (exp_avg_sq.sqrt() / torch.sqrt(
                    torch.tensor(1 - beta2 ** state['step'], device=p.device))).add_(group['eps'])
                step_size = group['lr'] * torch.sqrt(torch.tensor(1 - beta1 ** state['step'], device=p.device))
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
