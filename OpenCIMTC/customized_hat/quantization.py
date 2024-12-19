import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 
def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad
# 
def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def floor_pass(x):
    y = x.floor()
    y_grad = x
    return (y - y_grad).detach() + y_grad

# Add noise to input data
def add_noise(weight, method = 'add', n_scale = 0.074, n_range = 'max'):
    # 
    if n_scale == 0:
        return weight
    std = weight.std()

    if n_range == 'max':
        factor = weight.abs().max()
    if n_range == 'std':
        factor = std
    if n_range == 'max_min':
        factor = weight.max() - weight.min()
    if n_range == 'maxabs_2':
        factor = 2 * torch.max(torch.abs(weight))

    if method == 'add':
        w_noise = torch.randn_like(weight, device=weight.device).clamp_(-3.0, 3.0) * factor * n_scale
        weight_noise = weight + w_noise
    if method == 'mul':
        w_noise = torch.randn_like(weight, device=weight.device).clamp_(-3.0,
                                                                        3.0) * n_scale + 1  ## whether clamp randn to (-3,3)
        weight_noise = weight * w_noise
    weight_noise = (weight_noise - weight).detach() + weight
    return weight_noise
# 
def data_quantization(data_float, symmetric = True, bit = 8, clamp_std = None,
                        th_point='max', th_scale=None, all_positive=False):
    # data_float -> Input data needs to be quantized
    # symmetric -> whether use symmetric quantized, int range: [-(2**(bit-1)-1), 2**(bit-1)-1]
    # bit -> quant bits
    # clamp_std -> Clamp data_float to [- std * clamp_std, std * clamp_std]
    # th_point -> clamp data_float mode
    # th_scale -> scale the clamp thred, used together with th_point
    # all_positive -> whether data_float is all positive, int range: [0, 2**bit-1]

    std = data_float.std()
    max_data = data_float.max()
    min_data = data_float.min()
    # if min_data.item() >= 0:
    #     all_positive = True

    if clamp_std != None and clamp_std != 0 and th_scale != None:
        raise ValueError("clamp_std and th_scale, only one clamp method can be used. ")
    if clamp_std != None and clamp_std != 0:
        data_float = torch.clamp(data_float, min = -clamp_std * std, max = clamp_std * std)
    else:
        if min_data.item() * max_data.item() < 0. and th_point == 'min':
            th = min(max_data.abs().item(), min_data.abs().item())
        else:
            th = max(max_data.abs().item(), min_data.abs().item())
        if th_scale != None:
            th *= th_scale
        data_float = torch.clamp(data_float, min = -th, max = th)

    if all_positive:
        if data_float.min().item() < 0:
            raise ValueError("all_positive uniform_quantizer's data_float is not all positive. ")
        data_range = data_float.max()
        quant_range = 2**bit-1
        zero_point = 0
    elif symmetric:
        data_range = 2*abs(data_float).max()
        quant_range = 2**bit - 2
        zero_point = 0
    else:
        data_range = data_float.max() - data_float.min()
        quant_range = 2**bit - 1
        zero_point = data_float.min() / data_range * quant_range

    if data_range == 0:
        return data_float

    scale = data_range / quant_range
    data_quantized = (round_pass(data_float / scale - zero_point) + zero_point) * scale

    return data_quantized

def data_quantization_int(data_float, symmetric = True, bit = 8, clamp_std = None,
                        th_point='max', th_scale=None, all_positive=False):
    # data_float -> Input data needs to be quantized
    # symmetric -> whether use symmetric quantized
    # bit -> quant bits
    # clamp_std -> Clamp data_float to [- std * clamp_std, std * clamp_std]
    # th_point -> clamp data_float mode
    # th_scale -> scale the clamp thred, used together with th_point
    # all_positive -> whether data_float is all positive

    std = data_float.std()
    max_data = data_float.max()
    min_data = data_float.min()
    # if min_data.item() >= 0:
    #     all_positive = True

    if clamp_std != None and clamp_std != 0 and th_scale != None:
        raise ValueError("clamp_std and th_scale, only one clamp method can be used. ")
    if clamp_std != None and clamp_std != 0:
        data_float = torch.clamp(data_float, min = -clamp_std * std, max = clamp_std * std)
    else:
        if min_data.item() * max_data.item() < 0. and th_point == 'min':
            th = min(max_data.abs().item(), min_data.abs().item())
        else:
            th = max(max_data.abs().item(), min_data.abs().item())
        if th_scale != None:
            th *= th_scale
        data_float = torch.clamp(data_float, min = -th, max = th)

    if all_positive:
        if data_float.min().item() < 0:
            raise ValueError("all_positive uniform_quantizer's data_float is not all positive. ")
        data_range = data_float.max()
        quant_range = 2**bit-1
        zero_point = 0
    elif symmetric:
        data_range = 2*abs(data_float).max()
        quant_range = 2**bit - 2
        zero_point = 0
    else:
        data_range = data_float.max() - data_float.min()
        quant_range = 2**bit - 1
        zero_point = data_float.min() / data_range * quant_range

    if data_range == 0:
        return data_float, 1

    scale = data_range / quant_range
    data_quantized = (round_pass(data_float / scale - zero_point) + zero_point) * scale

    if zero_point != 0:
        raise ValueError("asymmetric uniform quantizer can not be valid next step yet. ")

    return round_pass(data_float / scale - zero_point), scale.item()

class uniform_quantizer(nn.Module):
    def __init__(self, symmetric=False, bit=4, clamp_std=0, th_point='max', th_scale=None, all_positive=False, noise_scale=0,
                noise_method='add', noise_range='max', int_flag=False, *args, **kwargs):
        # symmetric -> whether use symmetric quantized
        # bit -> quant bits
        # clamp_std -> Clamp data_float to [- std * clamp_std, std * clamp_std]
        # th_point -> clamp data_float mode
        # th_scale -> scale the clamp thred, used together with th_point
        # all_positive -> whether data_float is all positive
        # noise_scale -> noise scale
        # noise_method -> noise method, {'add', 'mul'}
        # noise_range -> noise range, activated when noise_method is 'add'
        super(uniform_quantizer, self).__init__()
        self.symmetric = symmetric
        self.bit = bit
        self.clamp_std = clamp_std
        self.th_point = th_point
        self.th_scale = th_scale
        self.all_positive = all_positive
        self.noise_scale = noise_scale
        self.noise_method = noise_method
        self.noise_range = noise_range
        self.int_flag = int_flag
        self.scale = 0
    
    def forward(self, weight):
        weight_int, scale = data_quantization_int(weight, self.symmetric, self.bit, self.clamp_std,
                                        self.th_point, self.th_scale, self.all_positive)
        self.scale = scale
        if self.noise_scale != 0:
            weight_int = add_noise(weight_int, self.noise_method, self.noise_scale, self.noise_range)
        if self.int_flag:
            return weight_int, torch.tensor(scale).cuda()
        else:
            return weight_int * scale
    
    def get_scale(self):
        return self.scale
    
    def get_int(self, weight): # without noise
        return data_quantization_int(weight, self.symmetric, self.bit, self.clamp_std,
                                        self.th_point, self.th_scale, self.all_positive)
    
    def get_quant_params(self):
        members= {}
        members['quant_name'] = 'uniform'
        members['bit'] = self.bit
        members['symmetric'] = self.symmetric
        members['clamp_std'] = self.clamp_std
        members['th_point'] = self.th_point
        members['th_scale'] = self.th_scale
        members['all_positive'] = self.all_positive
        members['noise_scale'] = self.noise_scale
        members['noise_method'] = self.noise_method
        members['noise_range'] = self.noise_range
        return members


class LSQ_weight_quantizer(nn.Module):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=False, noise_scale=0,
                    noise_method='add', noise_range='max', s_init=2,
                    init_mode='origin', init_percent=0.95, int_flag=False, s_grad_alpha=None, *args, **kwargs):
        # bit -> quant bits
        # all_positive -> set int_quant range to [0, 2**bit-1]
        # symmetric -> True: set int_quant range to [-(2**(bit-1)-1), 2**(bit-1)-1], False: set int_quant range to [-2**(bit-1), 2**(bit-1)-1]
        # per_channel -> channel wise quantizer or tensor wise
        # init_mode -> choice of {'origin', 'percent'}
        super(LSQ_weight_quantizer, self).__init__()

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric: # not full_range
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                if bit == 4:
                    self.thd_neg = - 2 ** (bit - 1) + 2
                    self.thd_pos = 2 ** (bit - 1) - 2
                else:
                    self.thd_neg = - 2 ** (bit - 1) + 1
                    self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1
        self.bit = bit
        self.per_channel = per_channel
        self.s = nn.Parameter(torch.tensor(1.0))
        self.noise_scale = noise_scale
        self.noise_method = noise_method
        self.noise_range = noise_range
        self.init_mode = init_mode
        self.s_init = s_init
        self.init_percent = init_percent
        self.int_flag = int_flag
        self.min_s = torch.tensor(1e-7)  # s > 0
        self.s_grad_alpha = s_grad_alpha
    
    def init_scale(self, x, *args, **kwargs):
        if self.init_mode == 'origin':
            if self.per_channel:
                self.s = nn.Parameter(
                    x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * self.s_init / (
                            self.thd_pos ** 0.5))
            else:
                self.s = nn.Parameter(x.detach().abs().mean() * self.s_init / (self.thd_pos ** 0.5))
        elif self.init_mode == 'percent':
            if self.per_channel:
                raise ValueError('per_channel weight quant not supported yet. ')
            else:
                val, ind = torch.sort(x.detach().view(-1).abs())
                self.s = nn.Parameter(val[math.ceil(len(val)*self.init_percent) - 1] / self.thd_pos)
        else:
            raise ValueError('Unknown s init_mode {}. '.format(self.init_mode))

    def get_thred(self):
        return self.s.data.detach() * self.thd_pos

    def get_scale(self):
        return self.s.data.detach()
    
    def compute_s_grad_scale(self, x):
        if self.s_grad_alpha is None:
            return 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            return 1.0 / (self.s_grad_alpha * ((self.thd_pos * x.numel()) ** 0.5))
        
    def forward(self, x):
        
        if self.s < self.min_s:
            self.s.data = self.min_s.to(self.s.device)
        # s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_grad_scale = self.compute_s_grad_scale(x)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x_int = round_pass(x)
        if self.noise_scale != 0:
            x_int = add_noise(x_int, self.noise_method, self.noise_scale, self.noise_range)
        # if self.int_flag:
        #     return x_int, s_scale
        # else:
        #     return x_int * s_scale
        return x_int, s_scale

    def get_quant_params(self):
        members = {}
        members['quant_name'] = 'lsq'
        members['bit'] = self.bit
        members['thd_neg'] = self.thd_neg
        members['thd_pos'] = self.thd_pos
        members['s'] = self.s.data.cpu().numpy()
        members['noise_scale'] = self.noise_scale
        members['noise_method'] = self.noise_method
        members['noise_range'] = self.noise_range
        return members
    
    def get_int(self, x):
        x = x / self.s
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        return x, self.s.item()

class LSQ_act_quantizer(nn.Module):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=False,
                 noise_scale=0, noise_method='add', noise_range='max', s_init=2,
                 init_mode='origin', init_percent=0.95, int_flag=False, s_grad_alpha=None, *args, **kwargs):
        super(LSQ_act_quantizer, self).__init__()

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.bit = bit
        self.per_channel = per_channel
        self.s = nn.Parameter(torch.tensor(1.0))
        self.init_batch_mode = False
        self.init_batch_num = 0
        self.noise_scale = noise_scale
        self.noise_method = noise_method
        self.noise_range = noise_range
        self.init_mode = init_mode
        self.s_init = s_init
        self.init_percent = init_percent
        self.int_flag = int_flag
        self.min_s = torch.tensor(1e-7)
        self.s_grad_alpha = s_grad_alpha

    def init_scale(self, x):
        if self.init_mode == 'origin':
            if self.per_channel:
                self.s = nn.Parameter(
                    x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * self.s_init / (
                            self.thd_pos ** 0.5))
            else:
                self.s = nn.Parameter(x.detach().abs().mean() * self.s_init / (self.thd_pos ** 0.5))
        elif self.init_mode == 'percent':
            if self.per_channel:
                raise ValueError('per_channel weight quant not supported yet. ')
            else:
                val, ind = torch.sort(x.detach().view(-1).abs())
                self.s = nn.Parameter(val[math.ceil(len(val)*self.init_percent) - 1] / self.thd_pos)
        else:
            raise ValueError('Unknown s init_mode {}. '.format(self.init_mode))
        
    def get_scale(self):
        return self.s.data.detach()

    def get_thred(self):
        return self.s.data.detach() * self.thd_pos
        
    def compute_s_grad_scale(self, x):
        if self.s_grad_alpha is None:
            return 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            return 1.0 / (self.s_grad_alpha * ((self.thd_pos * x.numel()) ** 0.5))
        
    def forward(self, x):
        if self.init_batch_mode:
            self.init_batch_num += 1
            self.init_scale(x.clone())
        if self.s < self.min_s:
            self.s.data = self.min_s.to(self.s.device)
        # s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_grad_scale = self.compute_s_grad_scale(x)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x_int = round_pass(x)
        if self.noise_scale != 0:
            x_int = add_noise(x_int, self.noise_method, self.noise_scale, self.noise_range)
        # if self.int_flag:
        #     return x_int, s_scale
        # else:
        #     return x_int * s_scale
        return x_int, s_scale
    
    def get_quant_params(self):
        members = {}
        members['quant_name'] = 'lsq'
        members['bit'] = self.bit
        members['thd_neg'] = self.thd_neg
        members['thd_pos'] = self.thd_pos
        members['s'] = self.s.data.cpu().numpy()
        members['noise_scale'] = self.noise_scale
        members['noise_method'] = self.noise_method
        members['noise_range'] = self.noise_range
        return members