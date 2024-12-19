import torch
from torch.autograd.function import Function

def get_bit_value(tensor, bit_index):
    tensor = tensor.to(torch.int8)
    
    # calculate sign bit
    sign_bit = tensor < 0
    
    # calculate absolute value and specific bit value
    abs_tensor = tensor.abs()
    bit_value = (abs_tensor >> bit_index) & 1

    # if sign bit is 1 and specific bit value is 1, then return -1
    # if sign bit is 0 and specific bit value is 0, then return 0
    result = torch.where(sign_bit & (bit_value == 1), torch.tensor(-1, dtype=torch.int8).to(tensor.device), bit_value.to(tensor.device))

    return result.float()

def train_multi_bit(input, scale, bit_index, bit_length):
    class train_multi_bit_operater(Function):
        @staticmethod
        def forward(ctx, input, scale, bit_index, bit_length):
            
            input_int = (input / scale).round()
            input_bit = get_bit_value(input_int, bit_index)
            ctx.save_for_backward(input_bit, torch.tensor([bit_length]).to(input_int.device), 
                                  torch.tensor([bit_index]).to(input_int.device))
            input_bit = input_bit * scale
            return input_bit.float()
        
        @staticmethod
        def backward(ctx, grad_output):
            input, bit_length, bit_index = ctx.saved_tensors
            if torch.sum(input) == 0.0:
                grad_input = None
            else:
                grad_input = grad_output / 2**(bit_index) / bit_length
            return grad_input, None, None, None
    
    return train_multi_bit_operater.apply(input, scale, bit_index, bit_length)
