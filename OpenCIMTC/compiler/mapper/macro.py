from .base import BasePlacement
from .helper import *
from .place_strategy import *

class MacroPlacement(BasePlacement):
    
    def __init__(self, * , ir=None, device=None, cpu_layer=None, weight_format='CHW',
                 average_copy=None, specify_para_num=None, specify_split_num = None,
                 place_strategy='LLA', window_copy=False,
                 calc_info=None, runtime = 'simulation', specify_device_id_list = None,
                 masked_device_id_list = None, adaptive_split_ir = False,  target_device = '144k',
                 ):
        
        ir_ = copy.deepcopy(ir)
        
        if ir_.devices == None:  
            self.device_ir = make_device_ir(ir_, device)
        else:
            self.device_ir = ir_
        super().__init__(self.device_ir, cpu_layer, specify_device_id_list, masked_device_id_list)
        self.device = device
        self.weight_format = weight_format
        self.average_copy = average_copy
        self.specify_split_num = specify_split_num
        self.specify_para_num = specify_para_num
        self.place_strategy = place_strategy
        self.window_copy = window_copy
        self.calc_info = calc_info
        self.runtime = runtime
        self.adaptive_split_ir = adaptive_split_ir
        self.target_device = target_device 
    
    def get_hardware_info(self):
        '''
        get hardware information
        '''
        self.XB_num = self.hardware_config['xb_number']
        self.XB_size = self.hardware_config['xb_shape']
        self.hd_name = self.hardware_config['name']
        self.dac_num = self.hardware_config['dac_num']
        self.adc_num = self.hardware_config['adc_num']
        self.dac_precision = self.hardware_config['dac_precision']
        
        # device type rram-144k
        self.device_field = self.hd_name[0]
        
        
    def split_average(self):
        
        if self.average_copy != None:
            for i in self.average_copy.keys():
                if i in self.node_weight.keys():
                    w, h = self.node_weight[i]
                    w_ = w * self.average_copy[i][1]
                    h_ = h * self.average_copy[i][0]
                    self.node_weight[i] = [w_, h_]
                else:
                    warnings.warn(f'The layers do not need to be mapped to the device : {i} !!!')
        
        if self.weight_format == 'HWC':
            XB_size = self.XB_size
            self.split_node_weight, self.split_num = split_node_HWC(self.node_weight,self.node_info,self.specify_para_num,
                                                                    XB_size, device=self.device_field)
            
        elif self.weight_format == 'CHW':
            
            self.split_num = {}
            for i in self.node_weight.keys():
                
                if self.specify_para_num != None and i in self.specify_para_num.keys():
                    p_diff_array,p_same_array = self.specify_para_num[i]
                else:
                    p_diff_array,p_same_array = 1, 1
                
                if self.window_copy and self.node_info[i]['op_type'] in ['conv2d', 'conv_transpose2d']:
                    # assert (i in self.specify_split_num.keys())
                    if self.specify_split_num != None and i in self.specify_split_num.keys():
                        _h = self.specify_split_num[i][0] 
                        _w = self.specify_split_num[i][1] 
                        self.split_num[i] = [p_diff_array, p_same_array, _w, _h] #
                    else:
                        self.split_num[i] = [p_diff_array, p_same_array, 1, 1] #
                else:
                    
                    self.node_weight[i][1] = self.node_weight[i][1] * p_same_array
                    self.node_weight[i][0] = self.node_weight[i][0] * p_same_array
                    
                    if self.specify_split_num != None and i in self.specify_split_num.keys():
                        _h = self.specify_split_num[i][0] 
                        _w = self.specify_split_num[i][1] 
                    else:
                        _h = math.ceil(self.node_weight[i][1] /  self.XB_size[1])
                        _w = math.ceil(self.node_weight[i][0] /  self.XB_size[0])
                    self.split_num[i] = [p_diff_array,p_same_array, _w, _h]
                    
            if self.window_copy:
                self.split_node_weight, self.split_num = split_node_window_duplicate(self.node_info,self.XB_size,self.split_num)
            else:
                self.split_node_weight = split_node(self.node_weight,self.split_num)
        
        else:
            raise ValueError(f"Not support the weight format : {self.weight_format}")
        
    def run(self):
        # get the hardware information
        self.get_hardware_info()
        
        # adaptive split the weight: {node_name : [r,w,h]}
        self.split_average()

        # Record the correspondence between the last layer after splitting and the previous layer names 
        self.split_weight_layer_dict = {}
        
        # Adaptively convert IR and link layers according to segmentation weights
        if self.adaptive_split_ir:
            
            # current laye information
            layers_info = self.ir.layers
            
            next_layer_dict = get_next_layer(self.ir.layers)
            split_layer_name = []
            new_split_num = {}
           
            for k, v in self.split_num.items():
                
                # Determine whether the row and column directions need to be split. 
                # v[2] indicates the number of splits in the column direction,
                # and v[3] indicates the number of splits in the row direction.
                if v[2] * v[3] != 1:
                    current_layer = layers_info[k]
                    split_layer_name.append(k)
                    if v[3] > 1:
                        # Determine whether the row direction is split. If so, insert the split operator
                        insert_split_node_name = f'{k}_Split'
                        assert current_layer.op.in_channel % v[3] == 0, f'{k}, {v}'
                        axis = 1
                        split = []
                        split_output = []
                        for i in range(v[3]):
                            # By default, the input channels are uniformly split.
                            split.append(current_layer.op.in_channel // v[3])
                            split_output.append({'channel': int(current_layer.op.in_channel // v[3]),
                                                 'width': current_layer.inputs[0].width,
                                                 'height': current_layer.inputs[0].height})
                        op_ = make_op('split', axis=axis, split=split)
                        split_input = current_layer.inputs
                        self.ir.add_layer(insert_split_node_name, op=op_, inputs=split_input, outputs=split_output)
                        
                    split_in_channel = current_layer.op.in_channel // v[3]
                    
                    if current_layer.op.out_channel % v[2] != 0:
                        warnings.warn(f"currrent layer {k}, output channel is {current_layer.op.out_channel}, split number: {v[2]}, split results: {math.ceil(current_layer.op.out_channel // v[2])} !!!")
                        current_layer.op.out_channel += 1
                    split_out_channel = int(math.ceil(current_layer.op.out_channel // v[2]))
                    
                    in_width = current_layer.inputs[0].width
                    in_height = current_layer.inputs[0].height
                    
                    out_width = current_layer.outputs[0].width
                    out_height = current_layer.outputs[0].height
                    
                
                    # First generate the add operator in the row direction,
                    # then generate the concat operator in the column direction
                    for w_ in range(v[2]):
                        for h_ in range(v[3]):
                            
                            new_insert_layer = current_layer.clone()
                            if v[3] > 1:
                                new_insert_layer.inputs[0].ref = insert_split_node_name + f':{h_}'
                            
                            new_node_name = k + f'_{h_}_{w_}' 
                            
                            new_insert_layer.inputs[0].channel = split_in_channel
                            new_insert_layer.outputs[0].channel = split_out_channel
                            
                            # print
                            original_weight_shape = current_layer.weights['weight'].shape
                            if len(original_weight_shape) == 4:
                                new_insert_layer.weights['weight'].shape = (split_out_channel, split_in_channel, original_weight_shape[2], original_weight_shape[3])
                            elif len(original_weight_shape) == 2:
                                new_insert_layer.weights['weight'].shape = (split_out_channel, split_in_channel)
                            else:
                                raise ValueError(f'Not support weight dimension : {original_weight_shape} !!!')
                            new_insert_layer.op.in_channel = split_in_channel
                            new_insert_layer.op.out_channel = split_out_channel
                            
                            if 'bias' in new_insert_layer.weights.keys():
                                new_insert_layer.weights['bias'].shape = (split_out_channel)
                            
                            self.ir.layers[new_node_name] = new_insert_layer

                            # update the split num
                            new_split_num[new_node_name] = [self.split_num[k][0], self.split_num[k][1], 1, 1]

                        # If the row direction is split, the add operator needs to be inserted at the output.
                        if v[3] > 1:
                            insert_add_node_name = f'{k}_Add_{w_}'
                            op_ = make_op('add')
                            add_input = []
                            for h_ in range(v[3]):
                                ref_name = k + f'_{h_}_{w_}' 
                                add_input.append(dict(ref=ref_name, channel=split_out_channel, width=out_width, height=out_height))
                            add_output = [dict(channel=split_out_channel, width=out_width, height=out_height)]
                            self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)
                            
                            
                    # If the column direction is split, insert the concat operator
                    if v[2] > 1:
                        insert_concat_node_name = f'{k}_Concat'
                        # By default, concat is performed in the channel dimension
                        op_ = make_op('concat', axis=1)
                        concat_input = []
                        for w_ in range(v[2]):
                            ref_name = k + f'_0_{w_}'
                            if v[3] > 1:
                                ref_name = f'{k}_Add_{w_}'
                            concat_input.append(dict(ref=ref_name, channel=split_out_channel, width=out_width, height=out_height))
                        concat_output = [dict(channel=current_layer.op.out_channel, width=out_width, height=out_height)]
                        self.ir.add_layer(insert_concat_node_name, op=op_, inputs=concat_input, outputs=concat_output)
                      
                    if v[2] > 1:
                        self.split_weight_layer_dict[k] = insert_concat_node_name
                    else:
                        self.split_weight_layer_dict[k] = insert_add_node_name
                    
                    next_layer_list = next_layer_dict[k]
                    replace_ref_name = k
                    
                    # update the ref name of the next layers
                    for nl in next_layer_list:
                        c = 0
                        if v[2] > 1:
                            for i in self.ir.layers[nl].inputs:
                                if i.ref == replace_ref_name:
                                    self.ir.layers[nl].inputs[c].ref = insert_concat_node_name
                                c += 1
                        elif v[3] > 1:
                            for i in self.ir.layers[nl].inputs:
                                if i.ref == replace_ref_name:
                                    self.ir.layers[nl].inputs[c].ref = insert_add_node_name
                                c += 1
                    
                    # remove the original layer
                    self.ir.layers.pop(k)
                        
                else:
                    new_split_num[k] = v
            
            # sort layers          
            self.ir.layers = dict(self.ir.iter_layers(deep=False, sorted=True))
            
            # update split_node_weight
            new_split_node_weight = {}
            for k,v in self.split_node_weight.items():
                k_ = k.split('.')
                if k_[0] in split_layer_name:
                    new_split_node_weight[f'{k_[0]}_{k_[2]}_{k_[3]}.0.0.0'] = v
                else:
                    new_split_node_weight[k] = v
            self.split_node_weight = new_split_node_weight
            
            # update node info
            ir_parser = BasePlacement(ir = self.ir)
            self.node_info = ir_parser.node_info
            
            # update split num
            self.split_num = new_split_num
                
        # place weight
        if self.place_strategy == 'LLA':
            self.place_strategy = LLA
        elif self.place_strategy == 'OoO':
            self.place_strategy = OoO
        else:
            raise ValueError(f'Not support the place strategy : {self.place_strategy} !!!')
        
        self.placed_nodes = self.place_strategy(self.split_node_weight, self.XB_size).run()
        sum_ = 0
        if 'rram-144k' in self.hd_name[0]:
            sum_ = len(self.placed_nodes)
        else:
            raise ValueError(f'Not support the device : {self.hd_name[0]} !!!')
        rest_xb = self.XB_num - sum_
        print(f'Current need: {sum_} array !!!')
        if rest_xb < 0 :
            raise ValueError(f'Current place strategy: {self.place_strategy.__name__}, At least {sum_} XBs are needed!!! Currently have {self.XB_num} XBs !!!')
        # 
        self.ref_to_device()
    
    def ref_to_device(self):
        self.node_mapping_info = {}
        assert len(self.placed_nodes) <= len(self.hd_name)
        # 144k placement
        if 'rram-144k' in self.hd_name[0]:
            for index in range(len(self.placed_nodes)):            
                device_ref = self.hd_name[index]
                for node_addr in self.placed_nodes[index]:
                    key = list(node_addr.keys())[0]
                    value = list(node_addr.values())[0]
                    for i in range(len(value)):
                        value[i] = int(value[i])
                    name_ = key.split('.')
                    node_name = name_[0]
                    if self.window_copy:
                        index_ = [int(name_[1]),int(name_[2]),int(name_[3].split('_')[0])]
                    else:
                        index_ = [int(name_[1]),int(name_[2]),int(name_[3])]
                    if node_name not in self.node_mapping_info.keys():
                        self.node_mapping_info[node_name] = []
                    mapping_info = MacroDeviceMappingInfo(index = index_, device=device_ref, address=value)
                    self.node_mapping_info[node_name].append(mapping_info)
        else:
            raise ValueError(f'Not support the device : {self.hd_name[0]} !!!')
        # 
        self.mapped_ir = make_mapped_ir(self.device_ir, self.split_num, self.node_mapping_info,
                                        self.average_copy, cpu_layer=self.cpu_layer,calc_info=self.calc_info,
                                        runtime=self.runtime)