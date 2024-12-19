import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def input_multi_bits_pulse_expansion(input_matrix, pulse_half_level=1, device='cpu'):
    input_matrix = torch.round(input_matrix)
    assert torch.max(input_matrix) <= 127
    assert torch.min(input_matrix) >= -128

    if torch.equal(input_matrix, torch.zeros_like(input_matrix)):
        return input_matrix, []

    if len(input_matrix.shape) == 1:
        input_matrix = input_matrix.view(-1, 1).to(torch.int8)
    else:
        input_matrix = input_matrix.to(torch.int8)
    
    max_expansion_times = math.ceil(torch.max(torch.abs(input_matrix)).item() / pulse_half_level)
    batches, rows = input_matrix.shape

    input_matrix = input_matrix.reshape(-1, 1)

    input_expanded = torch.zeros((rows * batches, max_expansion_times), dtype=torch.int8).to(device)

    for i in range(max_expansion_times - 1):
        pulse_cur = (input_matrix >= pulse_half_level).to(torch.int8) * pulse_half_level \
                    + (input_matrix <= -pulse_half_level).to(torch.int8) * -pulse_half_level
        input_expanded[:, i:i + 1] = pulse_cur
        input_matrix -= pulse_cur

    input_expanded[:, -1] = input_matrix.view(-1)

    input_expanded = bitwise_shift(input_expanded,device=device)

    input_expanded = input_expanded.view(batches, rows, max_expansion_times)
    
    return input_expanded

def bitwise_shift(input_expanded, device='cpu'):
    bitlen = input_expanded.shape[1]
    if bitlen <= 1:
        return input_expanded
    input_shift_count = input_expanded.clone()

    roll = torch.abs(input_shift_count).sum(dim=1)

    roll = (torch.cumsum(roll, dim=0) - roll) % bitlen

    rows = torch.arange(input_expanded.shape[0]).unsqueeze(1).to(device)
    column_indices = torch.arange(input_expanded.shape[1]).unsqueeze(0).to(device)

    roll = roll.unsqueeze(1)
    column_indices = (column_indices - roll) % bitlen

    input_expanded_shifted = input_expanded[rows, column_indices]
    
    return input_expanded_shifted


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
    data_quantized = ((data_float / scale - zero_point).round() + zero_point) * scale

    if zero_point != 0:
        raise ValueError("asymmetric uniform quantizer can not be valid next step yet. ")

    return (data_float / scale - zero_point).round(), scale.item()

def img2col(input, kernel_size, stride=1, padding=0):
    # input dim: B,C,H,W
    assert len(input.shape) == 4
    if padding > 0:
        input = F.pad(input, (padding,padding,padding,padding))
    unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)
    input_vector = unfold(input)
    #
    input_vector = input_vector.permute(0, 2, 1)
    # output dim: Batchsize*OH*OW, (Kernel*kernel*IC)
    input_vector = input_vector.reshape(-1,input_vector.shape[-1])
    return input_vector