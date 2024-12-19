import math
import numpy as np
import copy
from matplotlib import pyplot as plt
import warnings

from ..hw_paras_def.macro import *
from ..frontend.onnx2ir.converter import ConvertONNX
from ..irtool.core.layer import make_layer, make_op, DataDef


def get_max_time_layer(layer_time):
    '''
    input: 
        layer_time: {'layer_name':layer_time}
    return: 
        dict: The layer with the longest time.
        If there are multiple layers with the same time, the first layer traversed is returned.
    '''
    
    a1 = sorted(layer_time.items(),key = lambda x:x[1],reverse = True)
    layer_name = a1[0][0]
    max_ = a1[0][1]
    return {layer_name : max_}

def split_node(node_shape,split_num):
    '''
    input: 
        node_shape: {'node_name':[w,h]}
        split_num: {'node_name':[para_diff_array,para_same_array,w_num,h_num]}
    return:
        {'node_name_new':[w_split,h_split]},'node_name_new' = 'node_name' + '.repeat_index' + '.h_index' + '.w_index'
    '''
    node_shape_split = {}
    for node_name in node_shape.keys():
        h = []
        w = []
        [W,H] = node_shape[node_name]
        [pda,psa,w_split,h_split] = split_num[node_name]
        h_i = h_split 
        w_i = w_split 
        _h = math.floor(H/h_i)
        _w = math.floor(W/w_i)
        for i in range(h_i):    
            if H - _h > 0:
                h.append(_h)
                H = H - _h
            else:
                h.append(H)
        for j in range(w_i):
            if W - _w > 0:
                w.append(_w)
                W = W - _w
            else:
                w.append(W)
        for k in range(pda):
            for i in range(len(h)):
                for j in range(len(w)):
                    node_shape_split[node_name+'.'+str(k)+'.'+str(i)+'.'+str(j)] = [w[j],h[i]]
    return node_shape_split

def split_node_window_duplicate(node_info,xb_size,split_num):
    '''
    input: 
        node_shape: {'node_name':[w,h]}
        split_num: {'node_name':[parallel_num,copy_num,w_num,h_num]}
    return:
        {'node_name_new':[w_split,h_split]},'node_name_new' = 'node_name' + '.parallel_index' + '.h_index' + '.w_index'
    '''
    node_shape_split = {}
    for node_name in node_info.keys():
        
        h = []
        w = []
        if node_info[node_name]['op_type'] in ['conv2d', 'fused_conv2d', 'conv_transpose2d']:
            [para_num,repeat_num,w_split,h_split] = split_num[node_name]
            kz = node_info[node_name]['kernel_size']
            stride = node_info[node_name]['stride']
            in_channel = node_info[node_name]['in_channel']
            out_channel = node_info[node_name]['out_channel']
            # cc = node_info[node_name]['copy_constraint']
            W = out_channel * repeat_num
            H = ( kz  + (repeat_num - 1) * stride ) * in_channel * kz
            h_i =  math.ceil(H /  xb_size[1])
            w_i =  math.ceil(W /  xb_size[0])
            
            _h = math.floor(H/h_i)
            _w = math.floor(W/w_i)
            split_num[node_name] = [para_num,repeat_num,w_i,h_i]
            
        elif node_info[node_name]['op_type'] in ['matmul','fc','linear', 'fused_fc']:
            [para_num, w_split, h_split] = split_num[node_name]
            _h = h_split
            _w = w_split

        for i in range(h_i):    
            if H - _h > 0:
                h.append(_h)
                H = H - _h
            else:
                h.append(H)
        for j in range(w_i):
            if W - _w > 0:
                w.append(_w)
                W = W - _w
            else:
                w.append(W)
        
        for k in range(para_num):
            for i in range(len(h)):
                for j in range(len(w)):
                    node_shape_split[node_name+'.'+str(k)+'.'+str(i)+'.'+str(j)+'_wd'] = [w[j],h[i]]
    
    return node_shape_split,split_num

def split_node_HWC(node_weight, node_info, para_num, XB_size, device='rram-144k'):
    '''
    input: 
        node_shape: {'node_name':[w,h]}
        node_info: {'node_name':node_info}
        para_num: {'node_name':para_num}
        XB_Size: [w,h]
    return:
        {'node_name_new':[w_split,h_split]},'node_name_new' = 'node_name' + '.repeat_index' + '.h_index' + '.w_index'
    '''
    
    node_shape_split = {}
    node_split_num = {}

    for node_name in node_weight.keys():
        
        h = []
        w = []
        [W, H] = node_weight[node_name]
        array_size = XB_size    
                
        if node_info[node_name]['op_type'] in ['conv2d', 'fused_conv2d', 'conv_transpose2d']:
            kernel_size = node_info[node_name]['kernel_size']
            in_channel = node_info[node_name]['in_channel']
            out_channel = node_info[node_name]['out_channel']
            assert (H % (kernel_size**(2) * in_channel) == 0)
            row_repeat_avg = H / (kernel_size**(2) * in_channel)
            if H <= array_size[1]:
                h.append(H)
            else:
                h_temp = H
                # t = 1
                split_ic = []
                row_split_num = math.ceil(h_temp / array_size[1])
                while in_channel % row_split_num != 0:
                    row_split_num += 1
                max_split_channel_num, split_ic = get_max_channel_split_num(in_channel,row_split_num)
                if np.array(split_ic).mean() != max_split_channel_num:
                    warnings.warn(f'Current layer {node_name}, the input channel split is non-uniform!!! \
                                  The number of channels after splitting is: {split_ic} !!!')
                
                assert (split_ic != [])
                for ic_ in split_ic:
                    h.append(ic_ * kernel_size * kernel_size * row_repeat_avg )
        else:
            
            h_split = math.ceil(H /  array_size[1])
            h_i = h_split  
            _h = math.floor(H/h_i)
            for i in range(h_i):    
                if H - _h > 0:
                    h.append(_h)
                    H = H - _h
                else:
                    h.append(H)
                    
        w_split = math.ceil(W /  array_size[0]) 
        w_i = w_split
        _w = math.floor(W/w_i)
        for j in range(w_i):
            if W - _w > 0:
                w.append(_w)
                W = W - _w
            else:
                w.append(W)
        # repeat，diff_array
        repeat = 1
        if para_num != None:
            repeat = para_num[node_name][0]
        diff_array_repeat = repeat
        same_array_repeat = 1
        node_split_num[node_name] = [diff_array_repeat, same_array_repeat, len(w), len(h)]
        for k in range(repeat):
            for i in range(len(h)):
                for j in range(len(w)):
                    node_shape_split[node_name+'.'+str(k)+'.'+str(i)+'.'+str(j)] = [w[j],h[i]]
    
    return node_shape_split, node_split_num

def get_max_channel_split_num(ic,split_num):
    '''
    input:
        ic: int number
        split_num: int value, which means the number of splits
    return:
        max_num: the maximum number after splitting
    '''
    t = math.ceil(ic / split_num)
    w = []
    rest = ic
    for i in range(split_num):
        temp = rest - t
        if temp > 0: 
            w.append(t)
            rest = temp
        else:
            w.append(rest)

    return np.array(w).max(), w

def get_layer_ref(inputs, layer, ref):
    
    # MAX_count = 10
    for i in inputs:
        ref_name = i.ref
        if ':' in ref_name:
            ref_name = ref_name.split(':')[0]
        if 'graph_input' in ref_name:
            ref.append(ref_name)
        elif layer[ref_name].type == 'reuse':
            ref.append(ref_name)
        elif layer[ref_name].op.op_id in ['conv2d', 'fused_conv2d', 'conv_transpose2d', 'linear','matmul', 'fc', 'fused_fc']:
            ref.append(ref_name)
        elif layer[ref_name].op.op_id in ['constant', 'split', 'add', 'fused_add', 'fused_concat', 'concat']:
            ref.append(ref_name)
        else:
            get_layer_ref(layer[ref_name].inputs, layer, ref)

def get_conv_shape(op_info):
    kernel_size = op_info.kernel
    in_channel = op_info.in_channel
    out_channel = op_info.out_channel
    bias = False
    if bias:
        unroll_shape_h = kernel_size * kernel_size * in_channel + 1
    else:
        unroll_shape_h = kernel_size * kernel_size * in_channel
    unroll_shape_w = out_channel
    
    return [unroll_shape_w,unroll_shape_h]

def get_linear_shape(op_info):

    in_channel = op_info.in_channel
    out_channel = op_info.out_channel
    bias = False
    if bias:
        unroll_shape_h = in_channel + 1
    else:
        unroll_shape_h = in_channel
    unroll_shape_w = out_channel
    
    return [unroll_shape_w,unroll_shape_h]

def get_conv_info(layer):
    '''
    input:
        layer: layer object
    return:
        {'in_channel':INT,'out_channel':INT,'kernel_size':INT,'calc_num':INT,'stride':INT,
                    'in_precision':INT,'out_precision':INT,'copy_constraint':INT}
    '''
    
    if layer.inputs[0].dtype != None:
        intype = layer.inputs[0].dtype
    else:
        intype = 8
        
    if layer.outputs[0].dtype != None:
        outtype = layer.outputs[0].dtype
    else:
        outtype = 8
        
    kz = layer.op.kernel
    stride = layer.op.stride
    padding  = layer.op.padding
    
    out_height = layer.outputs[0].height
    out_width = layer.outputs[0].width
    
    in_channel = layer.op.in_channel
    out_channel = layer.op.out_channel
    
    copy_const = out_height
    calc_num = out_height * out_width
    op_type = layer.op.op_id
    
     
    in_data_len = (layer.inputs[0].height + 2 * padding) * (layer.inputs[0].width + 2 * padding) * layer.inputs[0].channel
    out_data_len = out_height * out_width * out_channel
    
    input_shape = [layer.inputs[0].height, layer.inputs[0].width]
    
    return dict(op_type=op_type,in_channel=in_channel,out_channel=out_channel,kernel_size=kz,
                stride=stride,calc_num=calc_num,in_precision=intype,
                out_precision=outtype,copy_constraint=copy_const, in_data_len=in_data_len,
                out_data_len = out_data_len, input_shape=input_shape)

def get_linear_info(layer):
    '''
    input:
        layer: layer object
    return:
        {'in_channel':INT,'out_channel':INT,'kernel_size':-1,'calc_num':INT,'stride':-1,
                    'in_precision':INT,'out_precision':INT,'copy_constraint':INT}
    '''
    if layer.inputs[0].dtype != None:
        intype = layer.inputs[0].dtype
    else:
        intype = 8
        
    if layer.outputs[0].dtype != None:
        outtype = layer.outputs[0].dtype
    else:
        outtype = 8
    in_channel = layer.op.in_channel
    out_channel = layer.op.out_channel
    
    calc_num = 1
    kz = 1
    stride = 1
    op_type = layer.op.op_id
    copy_const = 1
    
    return dict(op_type=op_type, in_channel=in_channel, out_channel=out_channel, kernel_size=kz,
                stride=stride, calc_num=calc_num, in_precision=intype, out_precision=outtype,
                copy_constraint=copy_const, in_data_len = in_channel, out_data_len = out_channel)

def get_split_concat_info(layer):
    '''
    input:
        layer: layer object
    return:
        {'in_channel': LSIT[INT], 'out_channel':LSIT[INT], 'axis': INT,}
    '''
    in_channel = []
    input_shape = []
    for in_ in layer.inputs:
        in_channel.append(in_.channel)
        input_shape.append([in_.height, in_.width])
    
    out_channel = []
    if layer.outputs != None:
        for out_ in layer.outputs:
            out_channel.append(out_.channel)
        
    op_type = layer.op.op_id
    axis = layer.op.axis
    
    return dict(op_type=op_type, in_channel=in_channel, out_channel=out_channel, axis = axis, input_shape=input_shape)

def get_add_info(layer):
    '''
    input:
        layer: layer object
    return:
        {'in_channel': LSIT[INT], 'out_channel':LSIT[INT], 'axis': INT,}
    '''
    in_channel = []
    input_shape = []
    for in_ in layer.inputs:
        in_channel.append(in_.channel)
        input_shape.append([in_.height, in_.width])
    
    out_channel = []
    for out_ in layer.outputs:
        out_channel.append(out_.channel)
        
    op_type = layer.op.op_id
    
    return dict(op_type=op_type, in_channel=in_channel, out_channel=out_channel, input_shape=input_shape)

def list_reverse(list_):
    len_ = len(list_)
    reverse_list = []
    for i in range(len_-1,-1,-1):
        reverse_list.append(list_[i])
    return reverse_list

def make_mapped_ir(ir, split_info, place_info, copy_info=None, cpu_layer=None,
                   calc_info=None, device='rram-144k', runtime = 'simulation', **kwargs):
    '''
    add mapping info into IR
    input:
        ir: ir object 
        split_info: {'node_name':[r,w,h],...}
        place_info: {node_name: [Mapping info object],...}
    return:
        ir object with mapping info
    '''
    for name, layer in ir.iter_layers():
        if layer.type == 'op' :
            if cpu_layer != None and name in cpu_layer:
                continue
            if layer.op.op_id in ['conv2d', 'fused_conv2d', 'conv_transpose2d'] or layer.op.op_id in ['linear', 'matmul', 'fc', 'fused_fc']:
                if 'rram-144k' in device:
                    if name not in split_info.keys():
                        continue
                    split_num = split_info[name]
                    if copy_info != None and name in copy_info.keys():
                        col_repeat_num = copy_info[name][1]
                        row_repeat_num = copy_info[name][0]
                    else:
                        col_repeat_num = 1
                        row_repeat_num = 1
                    
                    assert(len(split_num) == 4)
                    if isinstance(runtime,dict):
                        if name in runtime.keys():
                            runtime_ = runtime[name]
                        else:
                            runtime_ = 'simulation'
                    elif isinstance(runtime,str):
                        runtime_ = runtime
                    else:
                        raise ValueError(f'Not support runtime type: {type(runtime)} !!!')
                    layer.macro_mapping_info = MacroMappingInfo(col_split_num=split_num[2],row_split_num=split_num[3],
                                                    col_repeat_num=col_repeat_num,row_repeat_num=row_repeat_num,
                                                    para_same_array=split_num[1],para_diff_array=split_num[0],
                                                    runtime=runtime_, mappings=place_info[name],)
                    if calc_info == None:
                        layer.macro_calc_info = MacroCalcInfo().clone()
                    elif isinstance(calc_info,dict):
                        if name not in calc_info.keys():
                            if layer.macro_calc_info == None:
                                Warning(f"layer {name} does not set calc_info, use default calc_info !!!")
                                layer.macro_calc_info = MacroCalcInfo().clone()
                            
                        else:
                            layer.macro_calc_info = calc_info[name]
                    else:
                        layer.macro_calc_info = calc_info.clone()
                          
                else:
                    raise ValueError(f'Not support device : {device} !!!')
                
    return ir

def make_device_ir(ir,device=None):
    '''
    add device info into IR
    '''
    if ir.devices != None:
        raise ValueError(f'IR already exits devices info : { ir.devices.keys() } !!!')

    if device != None:
        if isinstance(device,list):
            for dev_ in device:
                dev_copy = copy.deepcopy(dev_)
                if 'num' in dev_.keys():
                    dev_copy.pop('name')
                    dev_copy.pop('kind')
                    dev_copy.pop('num')
                    ir.add_device(dev_['name'], dev_['kind'], number=dev_['num'], **dev_copy)
                else:
                    dev_copy.pop('name')
                    dev_copy.pop('kind')
                    ir.add_device(dev_['name'], dev_['kind'], **dev_copy)
        elif isinstance(device,dict):
            dev_copy = copy.deepcopy(device)
            dev_copy.pop('name')
            dev_copy.pop('kind')
            dev_copy.pop('num')
            ir.add_device(device['name'], device['kind'], number=device['num'], **dev_copy)
        else:
            raise TypeError(f'device type {type(device)} error!!!')
        return ir
    else:
        raise ValueError('No device info !!!')

def make_onnx_ir(onnx_file,return_weight=False):
    '''
    convert onnx into ir object
    input:
        onnx_file: onnx model name
        return_weight: boolean ; 
    return:
        case1 : ir object and weight value when return_weight is True
        case2 : ir object
    '''
    t = ConvertONNX(onnx_file)
    if return_weight:
        return t.ir,t.model_parser.weight_numpy
    else:
        return t.ir

def make_node_id(split_nodes):
    node_id = {}
    for i in range(len(split_nodes)):
        if len(split_nodes[i]) != 1:
            raise ValueError(f"Multiple layers in the same XB are not supported yet !!! {split_nodes}")
        else:
            node_name = list(split_nodes[i][0].keys())[0]
            node_id[node_name] = i
    return node_id

def get_device_ip(device):
    device_ip = {}
    if isinstance(device, list):
        for dev_ in device:
            assert isinstance(dev_, dict)
            if 'ip' in dev_.keys():
                device_ip[dev_['name']] = dev_['ip']
    elif isinstance(device, dict):
        if 'ip' in dev_.keys():
            device_ip[dev_['name']] = dev_['ip']
    else:
        raise ValueError(f'Not support device type: {type(device)}!!!')
    
    return device_ip

def gen_macro_mapping_info(placed_nodes, hardware_name, window_copy=False):
    node_mapping_info = {}
    for index in range(len(placed_nodes)):            
        device_ref = hardware_name[index]
        for node_addr in placed_nodes[index]:
            key = list(node_addr.keys())[0]
            value = list(node_addr.values())[0]
            name_ = key.split('.')
            node_name = name_[0]
            if window_copy:
                index_ = [int(name_[1]),int(name_[2]),int(name_[3].split('_')[0])]
            else:
                index_ = [int(name_[1]),int(name_[2]),int(name_[3])]
            if node_name not in node_mapping_info.keys():
                node_mapping_info[node_name] = []
            mapping_info = MacroDeviceMappingInfo(index = index_, device=device_ref, address=value)
            node_mapping_info[node_name].append(mapping_info)


def get_pre_layer(layers):
    ''''
    input: {layer_name:layer_object}
    return: {current_layer_name: pre_layer_name}
    '''
    prefix_layer = {}
    for name, layer in layers.items():
        if layer.type not in ['input']:
            prefix_layer[name] =  []
            if layer.type == 'op' and layer.op.op_id in ['constant']:
                continue
            for i in layer.inputs:
                if 'graph_input' not in i.ref:
                    ref = i.ref
                    if ':' in ref:
                        ref = ref.split(':')[0]
                    pre_layer = layers[ref]
                    if pre_layer.type == 'op' and pre_layer.op.op_id in ['flatten','reshape']:
                        for j in pre_layer.inputs:
                            prefix_layer[name].append(j.ref)
                    else:
                        prefix_layer[name].append(ref)
                else:        
                    prefix_layer[name].append(i.ref)
    
    return prefix_layer   


def get_next_layer(layers):
    '''
    input: {layer_name:layer_object}
    return: {current_layer_name: next_layer_name}
    '''
    next_layer = {}
    pre_layer = get_pre_layer(layers)  
    
    for k,v in pre_layer.items():
        if layers[k].type == 'op' and layers[k].op.op_id in ['flatten']:
            continue
    
        for name in v:
            if name not in next_layer.keys():
                next_layer[name] = []
            next_layer[name].append(k)
                
    return next_layer