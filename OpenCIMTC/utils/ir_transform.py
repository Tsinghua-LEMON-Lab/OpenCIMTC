import os
import torch
import numpy as np
import copy
from pathlib import Path
#
from ..compiler import * 
from ..simulator.torch_utils import current_range
from .gen_weights import hardware_adaptive_split_weight

def gen_ir(onnx_model, save_onnx_weight_to_torch=True):
    # device define
    devices = [{ 'name':'macro-0', 'kind':'rram-144k-cluster', 'num':8}]  
    # # 
    # parse file path
    model_path = os.path.dirname(onnx_model)
    onnx_name = os.path.basename(onnx_model).split('.')[0]
    #  onnx_model
    # onnx to ir
    onnx_obj = frontend.ConvertONNX(onnx_model, fix_layer_name=True)
    onnx_ir = onnx_obj.ir
    
    # placement
    map = mapper.MacroPlacement(ir=onnx_ir, device = devices, weight_format = 'HWC', place_strategy = 'LLA')
    map.run()
    mapped_ir = map.ir
    
    # 
    if save_onnx_weight_to_torch:
        # save float weight
        onnx_weight = onnx_obj.model_parser.weight_numpy
        onnx_weight = hardware_adaptive_split_weight(onnx_weight)
        torch_float_weight = {}
        for k,v in onnx_weight.items():
            torch_float_weight[k] = torch.from_numpy(v.copy()).float()
        torch.save(torch_float_weight, model_path + f'/{onnx_name}_float_weight.pth.tar')

    # dump mapped ir
    if not os.path.exists(model_path + '/ir'):
        os.makedirs(model_path + '/ir')
    mapped_ir.dump_json(file= model_path + f'/ir/{onnx_name}_mapped_ir.yaml')

def hardware_paras_identification(software_scale):
    # hardware scale
    hardware_in_scale = 0.1
    hardware_w_scale = 19.16 / 7
    # get w_int
    total_scale = software_scale
        
    diff = []
    for it_time, adc_range_value in current_range.items():
        hardware_out_scale = 7 / adc_range_value 
        # real chip weight
        w_int = round(total_scale / (hardware_in_scale * hardware_w_scale * hardware_out_scale))
        if w_int != 1:
            w_int = 1
        hard_scale = w_int * (hardware_in_scale * hardware_w_scale * hardware_out_scale)
        diff.append(abs(hard_scale / total_scale - 1))
        
    # get index with the minimal diff        
    min_diff_index = np.argmin(np.array(diff))
    adc_range = int(min_diff_index) + 1
    scale_diff = diff[min_diff_index] + 1
    return adc_range, scale_diff

def modify_ir_with_pdt_weight(mapped_ir, pdt_weight, save_quant_weight=True):
    # load ir
    ir = irtool.core.load_ir(file=mapped_ir)
    ir = copy.deepcopy(ir)
    # get scale and trained int weight
    trained_paras = torch.load(pdt_weight, map_location='cpu')
    quantized_weight = {}
    # 
    for k,v in ir.layers.items():
        if v.type == 'op' :
            if v.op.op_id in ['conv2d', 'matmul', 'fc', 'conv_transpose2d']:
                if k in trained_paras['quantization_info'].keys():
                    w = trained_paras['quantization_info'][k]['weight_int']
                    if len(w.shape) == 4:
                        w = w.reshape(w.shape[0], -1)
                    quantized_weight[f'{k}.weight'] = w
                    # 
                    software_scale = trained_paras['quantization_info'][k]['soft_total_scale'].detach().numpy()
                    adc_range, scale_diff = hardware_paras_identification(software_scale)
                    v.macro_calc_info.it_time = adc_range
                    v.macro_calc_info.input_quant_scale = float(trained_paras['quantization_info'][k]['input_scale'].detach().numpy())
                    v.macro_calc_info.assigned_output_quant_scale = float(trained_paras['quantization_info'][k]['output_scale'].detach().numpy())
                    v.macro_calc_info.activation_bits = trained_paras['quantization_info'][k]['input_bit']
                    
            elif v.op.op_id in ['batch_norm2d'] and f'{k}.weight' in trained_paras['state_dict'].keys():
                # 
                quantized_weight[f'{k}.weight'] = trained_paras['state_dict'][f'{k}.weight']
                quantized_weight[f'{k}.bias'] = trained_paras['state_dict'][f'{k}.bias']
                quantized_weight[f'{k}.running_mean'] = trained_paras['state_dict'][f'{k}.running_mean']
                quantized_weight[f'{k}.running_var'] = trained_paras['state_dict'][f'{k}.running_var']
                # 
                v.op.scale = trained_paras['state_dict'][f'{k}.weight'].numpy().tolist()
                v.op.bias = trained_paras['state_dict'][f'{k}.bias'].numpy().tolist()
                v.op.input_mean = trained_paras['state_dict'][f'{k}.running_mean'].numpy().tolist()
                v.op.input_var = trained_paras['state_dict'][f'{k}.running_var'].numpy().tolist()
                v.op.epsilon = 10**(-5)
                
    # saved file path
    ir_path = os.path.dirname(mapped_ir)
    ir_name = os.path.basename(mapped_ir).split('.')[0]
    
    # 
    ir.dump_json(file= ir_path + f'/{ir_name}_with_pdt_weight.yaml')
    ir_file_path = ir_path + f'/{ir_name}_with_pdt_weight.yaml'
    if save_quant_weight:
        weight_path = str(Path(ir_path).parent) + '/inference_weight/'
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)
        torch.save(quantized_weight, weight_path + f'/{ir_name}_pdt_weight.pth.tar')
    
    return ir_file_path