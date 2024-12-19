import numpy as np
import math
import warnings

def hardware_adaptive_split_weight(onnx_weight, array_size=[576, 128], split_method='uniform'):
    
    array_data = {}
    split_layer = []
    for k,v in onnx_weight.items():
        
        layer_name = k.split('.')[0] 
        data_shape = v.shape
        IsNeedSplit = False
        
        # constant
        if 'Constant' in k:
            array_data[k] = v
            continue
        
        # layernorm
        if 'LayerNormalization' in k:
            array_data[k] = v
            continue
        
        # batchnorm
        if 'BatchNormalization' in k:
            array_data[k] = v
            continue
        
        if 'weight' in k:
            in_row = 0
            if len(data_shape) == 4:
                [oc, ic, h1, h2] = data_shape
                in_row = ic * h1 * h2
                if in_row > array_size[0] or oc > array_size[1]:
                    IsNeedSplit = True
            elif len(data_shape) == 2:
                [oc, ic] = data_shape
                in_row = ic
                if ic > array_size[0] or oc > array_size[1]:
                    IsNeedSplit = True
            else:
                raise ValueError(f'Not support weight shape: {data_shape} !!!')
            
            if IsNeedSplit:
                if split_method == 'uniform':
                    row_split_num = math.ceil(in_row / array_size[0])
                    col_split_num = math.ceil(oc / array_size[1])
                    _, row_value = get_split_num(ic, row_split_num)
                    _, col_value = get_split_num(oc, col_split_num)
                    
                    for rn in range(row_split_num):
                        for cn in range(col_split_num):
                            start_row = int(np.sum(np.array(row_value[:rn])))
                            end_row =  int(start_row + row_value[rn])
                            start_col =  int(np.sum(np.array(col_value[:cn])))
                            end_col =  int(start_col + col_value[cn])
                            
                            if len(data_shape) == 4:
                                array_data[f'{layer_name}_{rn}_{cn}.weight'] = onnx_weight[k][start_col:end_col,start_row:end_row,:,:]
                            elif len(data_shape) == 2:
                                array_data[f'{layer_name}_{rn}_{cn}.weight'] = onnx_weight[k][start_col:end_col,start_row:end_row]
                            
                            if rn == row_split_num - 1:    
                                if f'{layer_name}.bias' in onnx_weight.keys():
                                    array_data[f'{layer_name}_{rn}_{cn}.bias'] = onnx_weight[f'{layer_name}.bias'][start_col:end_col]
                            else:
                                if f'{layer_name}.bias' in onnx_weight.keys():
                                    array_data[f'{layer_name}_{rn}_{cn}.bias'] = np.zeros(col_value[cn])
                             
                            # 
                            if f'{layer_name}_bn.weight' in onnx_weight.keys():
                                for ln in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                                    if f'{layer_name}_bn.{ln}' not in onnx_weight.keys():
                                        warnings.warn(f'lack the parameter: {layer_name}_bn.{ln}')
                                        continue
                                    if ln == 'num_batches_tracked':
                                        array_data[f'{layer_name}_{rn}_{cn}_bn.{ln}'] = onnx_weight[f'{layer_name}_bn.{ln}']
                                    elif ln in ['bias', 'running_mean']:
                                        array_data[f'{layer_name}_{rn}_{cn}_bn.{ln}'] = onnx_weight[f'{layer_name}_bn.{ln}'][start_col:end_col] / row_split_num
                                    else:
                                        array_data[f'{layer_name}_{rn}_{cn}_bn.{ln}'] = onnx_weight[f'{layer_name}_bn.{ln}'][start_col:end_col]
                            
                else:
                    raise ValueError(f'Not support split method: {split_method} !!!')
                split_layer.append(layer_name)
            else:
                array_data[k] = v
                # batchnorm
                if f'{layer_name}_bn.weight' in onnx_weight.keys():
                    for ln in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                        if f'{layer_name}_bn.{ln}' not in onnx_weight.keys():
                            warnings.warn(f'lack the parameter: {layer_name}_bn.{ln} !!!')
                            continue
                        array_data[f'{layer_name}_bn.{ln}'] = onnx_weight[f'{layer_name}_bn.{ln}']
        elif 'bias' in k:
            if layer_name not in split_layer:
                array_data[k] = v
        else:
            raise ValueError(f'Not support weight key:{k}!!!')
        
    return array_data

def get_split_num(ic, split_num):
    t = int(math.ceil(ic / split_num))
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