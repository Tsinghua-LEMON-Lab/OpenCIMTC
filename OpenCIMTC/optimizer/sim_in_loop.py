import torch.nn.functional as F
import time
import math
from sko.GA import GA
import copy
import logging
import os

# tool chain
from ..compiler.hw_paras_def.macro import * #noqa

class SimInLoopOptimizer:
    
    def __init__(self, ir, infer_module, test_input, test_target, weights=None, pe_weight_noise=0.1,
                 training_samples_num=256, optimized_params_category=None, beta = 0.1, loss_thr = 1,
                 log_dir=None, device='cuda', ):
        # 
        self.ir = ir
        self.optimized_params_category = optimized_params_category
        self.infer_module = infer_module
        self.weights = weights
        self.training_samples_num = training_samples_num
        self.test_input = test_input
        self.test_target = test_target
        self.loss_thr = loss_thr
        self.beta = beta
        self.log_dir = log_dir
        self.device = device
        self.pe_weight_noise = pe_weight_noise
        
        # 
        self.get_category()
        self.get_base_params()
        self.get_params_bound()

        # 
        self.best_x = None
        self.best_y = None
        
        self.init_logger()
        
    def init_logger(self):
        time_str = time.strftime("%Y%m%d_%H%M%S")
        if self.log_dir == None:
            log_dir = os.getcwd()
            if not os.path.exists(log_dir + "/" + f'{time_str}'):
                os.mkdir(log_dir + "/" + f'{time_str}')
        else:
            log_dir = self.log_dir
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
        log_file = log_dir + "/" + f'sim_in_loop_optimize_{time_str}.log'
        logging.basicConfig(level=logging.DEBUG,
                        filename=log_file,
                        filemode='a',
                        format= '%(asctime)s - %(levelname)s: %(message)s'
                        )
        # cmd output
        console_handler = logging.StreamHandler()
        self.logger = logging.getLogger()
        self.logger.addHandler(console_handler)
        self.logger.info('Log file: ' + str(log_file))
    
    def get_category(self):
        if self.optimized_params_category == None:
            self.optimized_params_category = ['it_time', 'input_expansion_mode', 'weight_copy_num']
        
        self.is_it_optimize = False
        self.is_iem_optimize = False
        self.is_wcn_optimize = False
        
        if 'it_time' in self.optimized_params_category:
            self.is_it_optimize = True
            self.it_time_lb = [] # lower bound
            self.it_time_ub = [] # upper bound
        if 'input_expansion_mode' in self.optimized_params_category:
            self.is_iem_optimize = True
            self.in_expand_mode_lb = []
            self.in_expand_mode_ub = []
        if 'weight_copy_num' in self.optimized_params_category:
            self.is_wcn_optimize = True
            self.weight_copy_num_lb = []
            self.weight_copy_num_ub = []
            
    def get_base_params(self):
        
        self.calc_times = []
        self.weight_shape = []
        self.activations_bits = []
        self.input_expansion_mode = []
        self.weight_copy_num = []
        
        # 
        layers = self.ir.layers
        for name, layer in layers.items():
            if layer.type == 'op' and layer.op.op_id in ['conv2d', 'matmul', 'fused_conv2d', 'fused_fc', 'conv_transpose2d']:
                if layer.macro_calc_info != None:
                    # 
                    calc_times = layer.outputs[0].width * layer.outputs[0].height
                    self.calc_times.append(calc_times)
                    
                    activations_bits = layer.macro_calc_info.activation_bits
                    self.activations_bits.append(activations_bits)
                    
                    if layer.macro_calc_info.shift_expansion_mode == 'bit_shift':
                        self.input_expansion_mode.append(1)
                    elif layer.macro_calc_info.shift_expansion_mode == 'bit_pulse':
                        self.input_expansion_mode.append(0)
                    else:
                        raise ValueError(f'Not support input expansion mode: {layer.macro_calc_info.shift_expansion_mode} !!!')
                    
                    self.weight_copy_num.append(layer.macro_mapping_info.row_repeat_num) 
                    
                    if layer.op.op_id in ['fused_fc', 'matmul']:
                        weight_size = layer.op.in_channel * layer.op.out_channel
                    else:
                        weight_size = layer.op.in_channel * layer.op.out_channel * layer.op.kernel * layer.op.kernel
                    self.weight_shape.append(weight_size)
                    
    def get_params_bound(self):
        self.layer_index = {}
        c = 0
        # 
        layers = self.ir.layers
        for name, layer in layers.items():
            if layer.type == 'op' and layer.op.op_id in ['conv2d', 'matmul', 'fused_conv2d', 'fused_fc', 'conv_transpose2d']:
                if layer.macro_calc_info != None:
                    if self.is_iem_optimize:
                        self.in_expand_mode_lb.append(0)
                        self.in_expand_mode_ub.append(1) 
                    if self.is_wcn_optimize:
                        self.weight_copy_num_lb.append(1)
                        # weight shape
                        if layer.op.op_id in ['fused_fc', 'matmul']:
                            weight_height = layer.op.in_channel
                        else:
                            weight_height = layer.op.in_channel * layer.op.kernel * layer.op.kernel

                        if 576 // weight_height < 1:
                            self.weight_copy_num_ub.append(1)
                        else:
                            self.weight_copy_num_ub.append(int(576 // weight_height))
                    # 
                    if self.is_it_optimize:
                        current_it_time = layer.macro_calc_info.it_time
                        #
                        if self.is_wcn_optimize and self.is_iem_optimize:
                            max_wcn = self.weight_copy_num_ub[-1]
                            lb = max(int(current_it_time // max_wcn) - 20, 1)
                            ub = min(current_it_time + 20, 63)
                        elif self.is_wcn_optimize:
                            max_wcn = self.weight_copy_num_ub[-1]
                            lb = max(int(current_it_time // max_wcn) - 5, 1)
                            ub = min(current_it_time + 5, 63)
                        else:
                            lb = max(current_it_time - 20, 1)
                            ub = min(current_it_time + 20, 63)
                        self.it_time_lb.append(lb)
                        self.it_time_ub.append(ub)
                
                    self.layer_index[c] = name
                    c += 1
                    
        # bound
        self.lb = []
        self.ub = []
        if self.is_it_optimize:
            self.lb.extend(self.it_time_lb)
            self.ub.extend(self.it_time_ub)
        if self.is_iem_optimize:
            self.lb.extend(self.in_expand_mode_lb)
            self.ub.extend(self.in_expand_mode_ub)
        if self.is_wcn_optimize:
            self.lb.extend(self.weight_copy_num_lb)
            self.ub.extend(self.weight_copy_num_ub)
        
    def get_weight_factor(self, x):
        assert len(x) == len(self.weight_shape)
        # base line
        sum_base = 0
        for v in self.weight_shape:
            sum_base += v
            
        sum_cur = 0
        for k in range(len(x)):
            sum_cur += x[k] * self.weight_shape[k]
            
        overflow = False
        
        if sum_cur > 8 * 576 * 128:
            overflow = True
        
        return sum_base / sum_cur, overflow
        
    def get_time_factor(self, **kwargs):
        
        # 
        for k,v in kwargs.items():
            assert len(self.calc_times) == len(v)
        
        # baseline
        sum_base = 0
        for v in range(len(self.calc_times)):
            t_ = self.calc_times[v]
            if 'it_time' in kwargs.keys():
                t_ *= kwargs['it_time'][v]
            t_ *= (self.activations_bits[v] - 1)
            sum_base += t_
            
        sum_cur = 0
        for v in range(len(self.calc_times)):
            t_ = self.calc_times[v]
            if 'it_time' in kwargs.keys():
                t_ *= kwargs['it_time'][v]
            if 'input_expansion_mode' in kwargs.keys():
                if kwargs['input_expansion_mode'][v] == 1:
                    t_ *= (self.activations_bits[v] - 1)
                elif kwargs['input_expansion_mode'][v] == 0:
                    t_ *= 2**(self.activations_bits[v] - 1) - 1
                else:
                    raise ValueError(f"Not support input expansion mode: {kwargs['input_expansion_mode'][v]} !!!")
            else:
                t_ *= (self.activations_bits[v] - 1)
                
            sum_cur += t_
        
        return sum_base / sum_cur
    
    def get_loss_factor(self, **kwargs):
        # 
        kwargs['device'] = self.device
        kwargs['PE_weight_noise'] = self.pe_weight_noise
        batch_size = self.training_samples_num
        output_sim = self.infer_module(self.test_input[0:batch_size,:,:,:], self.weights, **kwargs)[0]
        loss = F.mse_loss(output_sim, self.test_target)
        if self.device == 'cuda':
            loss = loss.cpu()
        self.logger.info(f'current loss: {loss} !!!')
        return loss
    
    def parse_var(self, x):
        # parse x
        params = {}
        s = 0
        if self.is_it_optimize:
            len_ = len(self.it_time_lb)
            params['it_time'] = []
            for j in x[s:s+len_].tolist():
                if not math.isnan(j):
                    params['it_time'].append(int(j))
                else:
                    params['it_time'].append(1)
            s = s + len_
        else:
            raise ValueError(f'it_time need to be optimized !!! ')
            
        if self.is_iem_optimize:
            len_ = len(self.in_expand_mode_lb)
            params['input_expansion_mode'] = []
            for j in x[s:s+len_].tolist():
                if not math.isnan(j):
                    params['input_expansion_mode'].append(int(j))
                else:
                    params['input_expansion_mode'].append(1)
            s = s + len_
        else:
            params['input_expansion_mode'] = self.input_expansion_mode
            
        if self.is_wcn_optimize:
            len_ = len(self.weight_copy_num_lb)
            params['weight_copy_num'] = []
            for j in x[s:s+len_].tolist():
                if not math.isnan(j):
                    params['weight_copy_num'].append(int(j))
                else:
                    params['weight_copy_num'].append(1)
            s = s + len_
        else:
            params['weight_copy_num'] = self.weight_copy_num
        
        return params
    
    def fom(self, x):
        # 
        params = self.parse_var(x)
        
        # get loss
        loss = self.get_loss_factor(**params)
        
        # get weight factor
        w_factor, overflow = self.get_weight_factor(params['weight_copy_num'])
        if overflow:
            self.logger.info('oveflow !!!')
            return 10 ** (6)
            
        # get time factor
        time_factor = self.get_time_factor(**params)
        
        # get fom
        loss_factor = loss
        fom = 0
        if loss_factor >= self.loss_thr:
            fom = loss_factor 
        else:
            fom = loss_factor * (1 - (self.beta * w_factor + (1-self.beta) * time_factor) )
            
        # logger
        x_ = []
        for j in x:
            if not math.isnan(j):
                x_.append(j)
            else:
                x_.append(1)
        
        return fom
        
    def run(self, size_pop=100, max_iter=100):
        self.logger.info(f'lower bound: {self.lb}')
        self.logger.info(f'upper bound: {self.ub}')
        n_dim = len(self.lb)
        self.logger.info(f'variable number: {n_dim}')
        # input()
        ga = GA(func=self.fom, size_pop=size_pop, n_dim=n_dim, max_iter=max_iter, lb=self.lb, ub= self.ub, precision=1)
        self.best_x, self.best_y = ga.run()
        self.logger.info('best_x:', self.best_x, '\n', 'best_y:', self.best_y)
        x_ = []
        for j in self.best_x:
            if not math.isnan(j):
                x_.append(j)
            else:
                x_.append(1)
        self.logger.info(f'best_x:{ str(x_)}')
        self.logger.info(f'best_y:{ str(self.best_y)}')
    
    def get_optimized_ir(self, to_simulator=True):
        
        assert (self.best_x != None).any()
        
        params = self.parse_var(self.best_x)
        
        it_time = params['it_time']
        weight_copy_num = params['weight_copy_num']
        input_expansion_mode = params['input_expansion_mode']
        
        copy_para = {}
        for i in range(len(weight_copy_num)):
            copy_para[self.layer_index[i]] = [int(weight_copy_num[i]), 1]
        
        # 
        calc_info = {}
        for i in range(len(it_time)):
            layer_name = self.layer_index[i]
            new_calc_info = copy.deepcopy(self.ir.layers[layer_name].macro_calc_info)
            # 
            new_calc_info.it_time = int(it_time[i])
            if input_expansion_mode[i] == 1:
                new_calc_info.shift_expansion_mode = 'bit_shift'    
            else:
                new_calc_info.shift_expansion_mode = 'bit_pulse'
            calc_info[layer_name] = new_calc_info
        
        # mapping
        from ..compiler.hw_paras_def.macro import MacroMappingInfo, MacroCalcInfo #noqa
        from ..compiler.mapper.place_strategy import LLA
        from ..compiler.mapper import MacroPlacement
         
        if to_simulator:
            runtime = 'simulation'
        else:
            runtime = 'macro'
        
        map = MacroPlacement(ir=self.ir,
                    calc_info=calc_info,
                    place_strategy=LLA,
                    average_copy=copy_para,
                    runtime=runtime,)
        map.run()
        optimized_ir = map.mapped_ir
        
        return optimized_ir 
