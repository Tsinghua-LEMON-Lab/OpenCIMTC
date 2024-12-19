import json
from .utils import *
from ..customized_hat.pdt_utils import get_bit_value
import os

current_file_path = os.path.abspath(__file__)
project_directory = os.path.dirname(current_file_path)
with open(project_directory + r'/current_range.json', 'r') as f:
    current_range = json.load(f)
    
def Macro_MVM_1bIN_4bOUT_SIM(input_data, weight_data, *, DAC_noise = 0, conductance_noise = 0.0, ADC_noise = 0, ADC_offset = 0,
                            integration_time = 0, max_conductance = 19.16, max_voltage = 0.1, device='cpu',):
    
    # According to the input and weight, it is converted into the corresponding conductivity value and voltage value
    input_voltage = input_data * max_voltage
    weight_conductance = weight_data / 7 * max_conductance
    
    # add noise to input voltage
    if DAC_noise != 0:
        input_voltage = input_voltage + DAC_noise * torch.randn_like(input_voltage).to(device)
    
    # add noise to weight conductance
    if conductance_noise != 0:
        # weight noise std
        conductance_noise = max_conductance * conductance_noise
        # weight conductance + noise
        weight_conductance = weight_conductance + conductance_noise * torch.randn_like(weight_conductance).clamp_(-3.0, 3.0).to(device)
    
    # calculate output current
    output_current = F.linear(input_voltage, weight_conductance)
    
    # add noise to output current
    if ADC_noise != 0:
        # current output_current max * noise
        ADC_noise = torch.abs(output_current).max() * ADC_noise
        # output current + noise
        output_current = output_current + ADC_noise * torch.randn_like(output_current).clamp_(-3.0, 3.0).to(device)
    
    # output current quantization
    output_quant = output_current / current_range[f'{integration_time*100}'] * 7  # 4-bit quantization
    output_quant = torch.round(output_quant).to(torch.int32)
    
    # quantization offset and noise
    if ADC_offset != 0:
        ADC_offset = torch.round( torch.randn_like(output_current).clamp_(-3.0, 3.0) * ADC_offset).to(device)
        output_quant = output_quant + ADC_offset
    
    # output clamp
    output_quant = torch.clamp(output_quant, min=-8, max=7) # 4-bit quantization
    
    return output_quant

def Macro_Conv2d_SIM(inputs, weights, *, kernel = 1, stride = 1, padding = 0, input_quant_scale = 1.0, output_dequant_scale = 1.0,
                    activation_bits = 4, input_expansion_method = 0, integration_time = 0, weight_row_copy = 1, PE_weight_noise=0.0, 
                    ADC_offset = 0, device='cpu'):
    # Input quantization, input quant scale is learned during training
    inputs_int = (inputs / input_quant_scale).round()
    thd_value = 2 ** (activation_bits - 1) - 1
    inputs_int = torch.clamp(inputs_int, -thd_value, thd_value)
    
    # record the dimensions of input and weight
    batch_size = inputs.shape[0]
    in_h = inputs.shape[2]
    in_w = inputs.shape[3]
    oc = weights.shape[0]
    
    # input and weight repeat
    if weight_row_copy > 1:
        assert isinstance(weight_row_copy, int)
        inputs_int = inputs_int.repeat(1, weight_row_copy, 1, 1)
        weights = weights.repeat(1, weight_row_copy)
    
    # image2col
    inputs_int = img2col(inputs_int, (kernel,kernel), stride=stride, padding=padding)
    
    # Expand according to the number of DAC bits (the default DAC is 1 bit)
    # input expansion method = 0 expands equallly;
    # input expansion method = 1 expands according to the number of bits
    if input_expansion_method == 0:
        inputs_int = input_multi_bits_pulse_expansion(inputs_int, pulse_half_level=1)
        inputs_int = inputs_int.to(device)
        len_ = inputs_int.shape[-1]
        outputs = 0
        for i in range(len_):
            partial_sum = Macro_MVM_1bIN_4bOUT_SIM(inputs_int[:,:,i], weights, conductance_noise=PE_weight_noise, ADC_offset = ADC_offset, integration_time=integration_time, device=device)
            outputs = outputs + partial_sum
    
    elif input_expansion_method == 1:
        
        bit_length = activation_bits - 1
        outputs = 0
        for bit_index in range(bit_length):
            input_bit_value = get_bit_value(inputs_int, bit_index)
            output_bit_value = Macro_MVM_1bIN_4bOUT_SIM(input_bit_value, weights, conductance_noise=PE_weight_noise, ADC_offset = ADC_offset, 
                                                       integration_time=integration_time, device=device)
            outputs = outputs + output_bit_value * 2**(bit_index)
        
    else:
        raise ValueError(f'Not supported input expansion method: {input_expansion_method} !!!')
    
    # reshape output shape
    out_h = (in_h + 2 * padding - kernel) // stride + 1
    out_w = (in_w + 2 * padding - kernel) // stride + 1
    outputs = outputs.reshape(batch_size, out_h, out_w, oc).permute(0, 3, 1, 2).contiguous()
    
    # rescale output
    outputs = outputs * output_dequant_scale
       
    return outputs

def Macro_FC_SIM(inputs, weights, *, input_quant_scale = 1, output_dequant_scale = 1.0, activation_bits = 4, 
                input_expansion_method = 0, integration_time = 0, weight_row_copy = 1, PE_weight_noise=0.0,
                ADC_offset = 0, device='cpu'):
    # input quantization, input quant scale is learned during training
    inputs_int = (inputs / input_quant_scale).round()
    thd_value = 2 ** (activation_bits - 1) - 1
    inputs_int = torch.clamp(inputs_int, -thd_value, thd_value)
    
    if weight_row_copy > 1:
        assert isinstance(weight_row_copy, int)
        inputs_int = inputs_int.repeat(1, weight_row_copy)
        weights = weights.repeat(1, weight_row_copy)
    
    # Expand according to the number of DAC bits (the default DAC is 1 bit)
    # input expansion method = 0 expands equallly;
    # input expansion method = 1 expands according to the number of bits
    if input_expansion_method == 0:
        inputs_int = input_multi_bits_pulse_expansion(inputs_int)
        inputs_int = inputs_int.to(device)
        len_ = inputs_int.shape[-1]
        outputs = 0
        for i in range(len_):
            partial_sum = Macro_MVM_1bIN_4bOUT_SIM(inputs_int[:,:,i], weights, integration_time=integration_time, ADC_offset = ADC_offset, device=device)
            outputs = outputs + partial_sum
            
    elif input_expansion_method == 1:
        # 
        bit_length = activation_bits - 1
        outputs = 0
        for bit_index in range(bit_length):
            input_bit_value = get_bit_value(inputs_int, bit_index)
            output_bit_value = Macro_MVM_1bIN_4bOUT_SIM(input_bit_value, weights, conductance_noise=PE_weight_noise, ADC_offset = ADC_offset, 
                                                       integration_time=integration_time, device=device)
            outputs = outputs + output_bit_value * 2**(bit_index)
             
    else:
        raise ValueError(f'Not supported input expansion method: {input_expansion_method} !!!')
    
    # rescale output
    outputs = outputs * output_dequant_scale
    
    return outputs


def Macro_Add(inputs):
    output = 0
    # sum
    for i in inputs:
        output += i
    return output

def Macro_ReLU(input): 
    return torch.clamp(input, min=0)

def Macro_Concat(inputs, axis=1):
    output = torch.cat(inputs, axis=axis)
    return output

def Macro_Split(input, split, dim=0):
    output = torch.split(input, split, dim=dim)
    return output

def Macro_Maxpool2d(input_data, kernel_size=1, stride=0, padding=0):
    input_data = input_data.to(torch.float32)
    output_data = torch.max_pool2d(input_data, kernel_size, stride, padding)
    return output_data

def Macro_Avgpool2d(input_data, kernel_size=1, stride=0, padding=0):
    output_data = F.avg_pool2d(input_data, kernel_size, stride, padding)
    return output_data

def Macro_GlobalAvgpool2d(input_data, out_size=1):
    output_data = F.adaptive_avg_pool2d(input_data, output_size=out_size)
    return output_data

def Macro_Resize(input_data, size=None, scale_factor=[1, 1]):
    input_data = input_data.to(torch.float32)
    output_data = F.interpolate(input_data, size=size, scale_factor=scale_factor)
    return output_data

def Macro_Pad(inputs, pad, value):
    output = F.pad(inputs, pad=pad, value=value)
    return output 

def Macro_Relu(inputs):
    output = torch.clamp(inputs, min=0)
    return output

def Macro_BatchNorm2d(inputs, *, epsilon=None, weights=None, bias=None, mean=None, var=None):
    weights = weights.view(1, -1, 1, 1)
    bias = bias.view(1, -1, 1, 1)
    mean = mean.view(1, -1, 1, 1)
    var = var.view(1, -1, 1, 1) 
    return ((inputs - mean) / torch.sqrt(var + epsilon)) * weights + bias

def Macro_SiLU(inputs):
    return F.silu(inputs)

def Macro_Flatten(inputs, start_dim=0):
    return torch.flatten(inputs, start_dim=start_dim)

def Macro_Softmax(inputs, dim=1):
    return F.softmax(inputs, dim=dim)