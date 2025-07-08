from .quantization import *
from .pdt_utils import *

def get_weight_quantizer(cfg):
    cfg = cfg.copy()
    if cfg['quant_name'] == 'uniform':
        q = uniform_quantizer
    elif cfg['quant_name'] == 'lsq':
        q = LSQ_weight_quantizer
    else:
        raise ValueError('Cannot find quantizer `%s`', cfg['quant_name'])
    cfg.pop('quant_name')
    return q(**cfg)

def get_act_quantizer(cfg):
    cfg = cfg.copy()
    if cfg['quant_name'] == 'uniform':
        q = uniform_quantizer
    elif cfg['quant_name'] == 'lsq':
        q = LSQ_act_quantizer
    else:
        raise ValueError('Cannot find quantizer `%s`', cfg['quant_name'])
    cfg.pop('quant_name')
    return q(**cfg)

class Conv2dPDT(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, layer_name, quantization_config ):
        super(Conv2dPDT, self).__init__(in_channels, out_channels, kernel_size, stride, 
                                        padding = padding, dilation=dilation, groups=groups, bias=bias)
        assert layer_name is not None 
        assert quantization_config is not None
        self.layer_name = layer_name
        self.quantization_config = quantization_config
        self.init_quantizer()
    
    def init_quantizer(self):
        input_quant_config = self.quantization_config['input']
        weight_quant_config = self.quantization_config['weight']
        output_quant_config = self.quantization_config['output'] 
        # init input quantizer
        if self.layer_name in input_quant_config.keys():
            current_in_quant_config = input_quant_config[self.layer_name]
        else:
            current_in_quant_config = input_quant_config['default']
        self.input_quantizer = get_act_quantizer(current_in_quant_config)
        # init weight quantizer
        if self.layer_name in weight_quant_config.keys():
            current_w_quant_config = weight_quant_config[self.layer_name]
        else:
            current_w_quant_config = weight_quant_config['default']
        self.weight_quantizer = get_weight_quantizer(current_w_quant_config)
        # init output quantizer
        if self.layer_name in output_quant_config.keys():
            current_out_quant_config = output_quant_config[self.layer_name]
        else:
            current_out_quant_config = output_quant_config['default']
        self.output_quantizer = get_act_quantizer(current_out_quant_config)
            
    def get_int_weight(self):
        weight_int, scale = self.weight_quantizer.get_int(self.weight)
        return weight_int, scale

    def init_weight_scale(self):
        # init weight scale
        if isinstance(self.weight_quantizer, LSQ_weight_quantizer):
            self.weight_quantizer.init_scale(self.weight)
        
    
    def get_params(self):
        d = {}
        d['weight_int'] = self.get_int_weight()[0].cpu()
        input_s = self.input_quantizer.get_scale()
        weight_s = self.weight_quantizer.get_scale()
        output_s  = self.output_quantizer.get_scale()
        d['soft_total_scale'] = (input_s * weight_s / output_s).detach().cpu()
        d['input_scale'] = input_s
        d['weight_scale'] = weight_s    
        d['output_scale'] = output_s
        d['input_bit'] = self.input_quantizer.bit
        d['weight_bit'] = self.weight_quantizer.bit
        d['output_bit'] = self.output_quantizer.bit
        d['bias'] = self.bias
        return d
        
    def forward(self, input):
        # print(self.input_quantizer.s)
        # print(self.input_quantizer.bit)
        # input()
        # quantize inputs
        input_int, input_s = self.input_quantizer(input)
        # return input_int
        # quantize weights
        weight_int, weight_s = self.weight_quantizer(self.weight)
        # calculat  output
        out_int = 0
        out_s = 0
        out_value_list = []
        input_bit = self.input_quantizer.bit
        bit_length = input_bit - 1
        for bit_index in range(bit_length):
            input_bit_value = train_multi_bit(input_int, 1, bit_index, bit_length)
            out_bit_value = self._conv_forward(input_bit_value * input_s, weight_int * weight_s, bias=None)
            out_bit_int, out_bit_s = self.output_quantizer(out_bit_value)
            # 
            out_value_list.append(out_bit_value)
            # sum out_s, out_int
            out_int += out_bit_int * 2**(bit_index)
            out_s += out_bit_s
        # average scale
        out_s = out_s / bit_length
        # 
        out = out_int * out_s
        if isinstance(self.bias, nn.Parameter):
            out += self.bias.view(1, -1, 1, 1)
        return out

class ConvTranspose2dPDT(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, groups, bias, dilation, layer_name, quantization_config ):
        super(ConvTranspose2dPDT, self).__init__(in_channels, out_channels, kernel_size, stride, 
                                        padding = padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
        assert layer_name is not None 
        assert quantization_config is not None
        self.layer_name = layer_name
        self.quantization_config = quantization_config
        self.init_quantizer()
    
    def init_quantizer(self):
        input_quant_config = self.quantization_config['input']
        weight_quant_config = self.quantization_config['weight']
        output_quant_config = self.quantization_config['output'] 
        # init input quantizer
        if self.layer_name in input_quant_config.keys():
            current_in_quant_config = input_quant_config[self.layer_name]
        else:
            current_in_quant_config = input_quant_config['default']
        self.input_quantizer = get_act_quantizer(current_in_quant_config)
        # init weight quantizer
        if self.layer_name in weight_quant_config.keys():
            current_w_quant_config = weight_quant_config[self.layer_name]
        else:
            current_w_quant_config = weight_quant_config['default']
        self.weight_quantizer = get_weight_quantizer(current_w_quant_config)
        
        # init output quantizer
        if self.layer_name in output_quant_config.keys():
            current_out_quant_config = output_quant_config[self.layer_name]
        else:
            current_out_quant_config = output_quant_config['default']
        self.output_quantizer = get_act_quantizer(current_out_quant_config)
            
    def get_int_weight(self):
        weight_int, scale = self.weight_quantizer.get_int(self.weight)
        return weight_int, scale

    def init_weight_scale(self):
        # init weight scale
        if isinstance(self.weight_quantizer, LSQ_weight_quantizer):
            self.weight_quantizer.init_scale(self.weight)
    
    def get_params(self):
        d = {}
        d['weight_int'] = self.get_int_weight()[0].cpu()
        input_s = self.input_quantizer.get_scale()
        weight_s = self.weight_quantizer.get_scale()
        output_s  = self.output_quantizer.get_scale()
        d['soft_total_scale'] = (input_s * weight_s / output_s).detach().cpu()
        d['input_scale'] = input_s
        d['weight_scale'] = weight_s    
        d['output_scale'] = output_s
        d['input_bit'] = self.input_quantizer.bit
        d['weight_bit'] = self.weight_quantizer.bit
        d['output_bit'] = self.output_quantizer.bit
        d['bias'] = self.bias
        return d
        
    def forward(self, input):
        # quantize inputs
        input_int, input_s = self.input_quantizer(input)
        # quantize weights
        weight_int, weight_s = self.weight_quantizer(self.weight)
        # calculat  output
        out_int = 0
        out_s = 0
        out_value_list = []
        input_bit = self.input_quantizer.bit
        bit_length = input_bit - 1
        for bit_index in range(bit_length):
            input_bit_value = train_multi_bit(input_int, 1, bit_index, bit_length)
            out_bit_value = self._conv_forward(input_bit_value * input_s, weight_int * weight_s)
            out_bit_int, out_bit_s = self.output_quantizer(out_bit_value)
            # 
            out_value_list.append(out_bit_value)
            # sum out_s, out_int
            out_int += out_bit_int * 2**(bit_index)
            out_s += out_bit_s
        # average scale
        out_s = out_s / bit_length
        out = out_int * out_s
        if isinstance(self.bias, nn.Parameter):
            out += self.bias.view(1, -1, 1, 1)
        return out

            
class LinearPDT(nn.Linear):
    def __init__(self, in_features, out_features, bias, layer_name, quantization_config ):
        super(LinearPDT, self).__init__(in_features, out_features, bias=bias)
        assert layer_name is not None 
        assert quantization_config is not None
        self.layer_name = layer_name
        self.quantization_config = quantization_config
        self.init_quantizer()
    
    def init_quantizer(self):
        input_quant_config = self.quantization_config['input']
        weight_quant_config = self.quantization_config['weight']
        output_quant_config = self.quantization_config['output'] 
        # init input quantizer
        if self.layer_name in input_quant_config.keys():
            current_in_quant_config = input_quant_config[self.layer_name]
        else:
            current_in_quant_config = input_quant_config['default']
        self.input_quantizer = get_act_quantizer(current_in_quant_config)
        # init weight quantizer
        if self.layer_name in weight_quant_config.keys():
            current_w_quant_config = weight_quant_config[self.layer_name]
        else:
            current_w_quant_config = weight_quant_config['default']
        self.weight_quantizer = get_weight_quantizer(current_w_quant_config)
        # init output quantizer
        if self.layer_name in output_quant_config.keys():
            current_out_quant_config = output_quant_config[self.layer_name]
        else:
            current_out_quant_config = output_quant_config['default']
        self.output_quantizer = get_act_quantizer(current_out_quant_config)
    
    def init_weight_scale(self):
        # init weight scale
        if isinstance(self.weight_quantizer, LSQ_weight_quantizer):
            self.weight_quantizer.init_scale(self.weight)
    
    def get_int_weight(self):
        weight_int, scale = self.weight_quantizer.get_int(self.weight)
        return weight_int, scale

    def get_params(self):
        d = {}
        d['weight_int'] = self.get_int_weight()[0].cpu()
        input_s = self.input_quantizer.get_scale()
        weight_s = self.weight_quantizer.get_scale()
        output_s  = self.output_quantizer.get_scale()
        d['soft_total_scale'] = (input_s * weight_s / output_s).detach().cpu()
        d['input_scale'] = input_s
        d['weight_scale'] = weight_s    
        d['output_scale'] = output_s
        d['input_bit'] = self.input_quantizer.bit
        d['weight_bit'] = self.weight_quantizer.bit
        d['output_bit'] = self.output_quantizer.bit
        d['bias'] = self.bias
        return d
    
    def forward(self, input):
        
       # quantize inputs
        input_int, input_s = self.input_quantizer(input)
        # quantize weights
        weight_int, weight_s = self.weight_quantizer(self.weight)
        # calculat  output
        out_int = 0
        out_s = 0
        out_value_list = []
        input_bit = self.input_quantizer.bit
        bit_length = input_bit - 1
        for bit_index in range(bit_length):
            input_bit_value = train_multi_bit(input_int, 1, bit_index, bit_length)
            out_bit_value = F.linear(input_bit_value * input_s, weight_int * weight_s)
            out_bit_int, out_bit_s = self.output_quantizer(out_bit_value)
            # 
            out_value_list.append(out_bit_value)
            # sum out_s, out_int
            out_int += out_bit_int * 2**(bit_index)
            out_s += out_bit_s
        # average scale
        out_s = out_s / bit_length
        # 
        out = out_int * out_s
        if isinstance(self.bias, nn.Parameter):
            out += self.bias.view(1, -1)
        return out

