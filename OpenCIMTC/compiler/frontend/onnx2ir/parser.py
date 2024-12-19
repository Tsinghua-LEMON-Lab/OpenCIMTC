from collections import defaultdict
import numpy as np
from .helper import *

class OnnxParser(object):
    
    def __init__(self, onnx_model,weight_half_level=None,
                 weight_scale=None, data_clamp_std = 0, data_range_specify = None):
        self.model = onnx_model
        # self.cpu_layer = cpu_layer
        self.weight_scale = weight_scale
        
        self.value_infos = {}
        self.parameters = {}
        self.nodes = {}
        # inputs
        self.inputs = []
        self.get_inputs()
        
        #get weight data
        self.weight_numpy = {}
        self.weight_numpy_quant = {}
        self.weight_half_level = weight_half_level
        self.weight_quant_scale = {}
        #get constant data
        self.constant = {}

        self.predecessors = defaultdict(list)
        self.successors = defaultdict(list)
        
        # quant data para
        self.data_clamp_std = data_clamp_std
        self.data_range_specify = data_range_specify
        
        # node name --> weight name
        self.node_weight_name = {}
        
        # input edge name --> node_name
        self.edge_node_name = {}
        
        # passes
        self.parse_names()
        
    def parse_names(self):
        self._map_value_infos()
        self._map_nodes()
        self._map_initializer_parameters()
        self._map_orders()
        self._save_constant_parameters()
        self._map_node_weight_name()
        self._map_edge_node_name()
        
        
    @property
    def graph(self):
        
        return self.model.graph
    
    # @property
    def get_inputs(self):
        for i in self.graph_input:
            self.inputs.append(i)
        # return [i.name]
    
    @property
    def graph_input(self):
        return [i.name for i in self.graph.input]
    
    @property
    def graph_output(self):
        return [o.name for o in self.graph.output]

    def _map_node_weight_name(self):
        """Mapping node name -> Conv / matmul weight name"""
        for node in self.graph.node:
            if node.op_type in ["Conv", "MatMul"]:
                self.node_weight_name[node.name] = node.input[1]
        
    def _map_value_infos(self):
        """Mapping name -> ValueInfoProto"""
        value_infos = self.graph.value_info
        
        inputs = self.graph.input
        outputs = self.graph.output
        value_infos.extend(inputs)
        value_infos.extend(outputs)
        
        for info in value_infos:
            self.value_infos[info.name] = info
        
        
    def _map_initializer_parameters(self):
        """Mapping name -> TensorProto for initializer parameters"""
        
        for tensor in self.graph.initializer:
            if tensor.name not in self.graph_input:
                self.parameters[tensor.name] = tensor
                
        for c in self.graph.node:
            if c.op_type == "Constant":
                for attr in c.attribute:
                    self.parameters[c.output[0]] = numpy_helper.to_array(attr.t)
        for node in self.nodes.keys():
            layer_name = None
            index = 0
            for weight_name in self.nodes[node].input:
                weight_trans = False
                node_name = None
                
                if weight_name in self.parameters.keys():
                    layer_name = self.nodes[node].name
                    
                    if self.nodes[node].op_type == 'MatMul' and weight_name == self.nodes[node].input[1]:
                        weight_trans = True
                        node_name = layer_name + '.weight'
                        
                    elif self.nodes[node].op_type == 'Gemm' and weight_name == self.nodes[node].input[1]:
                        node_name = layer_name + '.weight'
                        
                    elif self.nodes[node].op_type in ["Conv", "ConvTranspose"] and weight_name == self.nodes[node].input[1]:
                        node_name = layer_name + '.weight'
                        
                            
                    elif self.nodes[node].op_type == 'LSTM':
                        weight_trans = True
                        if weight_name == self.nodes[node].input[1]:  
                            node_name = layer_name + '_lstm_x' + '.weight'
                        if weight_name == self.nodes[node].input[2]:
                            node_name = layer_name + '_lstm_h' + '.weight'
                        if weight_name == self.nodes[node].input[3]:
                            node_name = layer_name + '_lstm_bias' + '.bias'
                    
                    elif self.nodes[node].op_type in ["BatchNormalization"]: 
                        if weight_name == self.nodes[node].input[1]:
                            node_name = layer_name + '.weight'
                        elif weight_name == self.nodes[node].input[2]:
                            node_name = layer_name + '.bias'
                        elif weight_name == self.nodes[node].input[3]:
                            node_name = layer_name + '.running_mean'
                        elif weight_name == self.nodes[node].input[4]:
                            node_name = layer_name + '.running_var'

                    elif self.nodes[node].op_type in ["LayerNormalization"]: 
                        if weight_name == self.nodes[node].input[1]:
                            node_name = layer_name + '.weight'
                        elif weight_name == self.nodes[node].input[2]:
                            node_name = layer_name + '.bias'
                    
                    elif self.nodes[node].op_type in ["Add", "Mul", 'Div', 'Concat']: 
                        if len(numpy_helper.to_array(self.parameters[weight_name]).shape) >= 1:
                            
                            weight_name_str = 'Constant_'+ self.nodes[node].name + f'_{str(index)}'
                            index += 1
                            node_name = f'x_{weight_name_str}'
                            
                    
                    elif len(self.nodes[node].input) == 3:
                        if self.nodes[node].op_type in ["Conv",'Gemm','ConvTranspose']:
                            node_name = layer_name + '.bias'
                    
                    if node_name != None:   
                        if weight_trans:
                            if '_lstm_x' in  node_name:
                                data = numpy_helper.to_array(self.parameters[weight_name])
                                data = data.squeeze()
                                data = data.transpose()
                                assert data.shape[1] % 4 == 0, f'Not supportlstm weight shape: {data.shape} !!!'
                                len_ = data.shape[1] // 4
                                self.weight_numpy[f'{layer_name}-input_gate_x-0.weight'] = data[:,0:len_]
                                self.weight_numpy[f'{layer_name}-output_gate_x-0.weight'] = data[:,len_:2*len_]
                                self.weight_numpy[f'{layer_name}-forget_gate_x-0.weight'] = data[:,2*len_:3*len_]
                                self.weight_numpy[f'{layer_name}-cell_state_x-0.weight'] = data[:,3*len_:4*len_]
                            elif '_lstm_h' in node_name:
                                data = numpy_helper.to_array(self.parameters[weight_name])
                                data = data.squeeze()
                                data = data.transpose()
                                assert data.shape[1] % 4 == 0, f'Not support lstm weigt shape: {data.shape}  !!!'
                                len_ = data.shape[1] // 4
                                self.weight_numpy[f'{layer_name}-input_gate_h-0.weight'] = data[:,0:len_]
                                self.weight_numpy[f'{layer_name}-output_gate_h-0.weight'] = data[:,len_:2*len_]
                                self.weight_numpy[f'{layer_name}-forget_gate_h-0.weight'] = data[:,2*len_:3*len_]
                                self.weight_numpy[f'{layer_name}-cell_state_h-0.weight'] = data[:,3*len_:4*len_]
                            elif '_lstm_bias' in node_name:
                                data = numpy_helper.to_array(self.parameters[weight_name])
                                data = data.squeeze()
                                assert data.shape[0] % 8 == 0, f'Not support lstm bias shape: {data.shape} !!!'
                                len_ = data.shape[0] // 8
                                self.weight_numpy[f'{layer_name}-input_gate_x-0.bias'] = data[0:len_]
                                self.weight_numpy[f'{layer_name}-output_gate_x-0.bias'] = data[len_:2*len_]
                                self.weight_numpy[f'{layer_name}-forget_gate_x-0.bias'] = data[2*len_:3*len_]
                                self.weight_numpy[f'{layer_name}-cell_state_x-0.bias'] = data[3*len_:4*len_]
                                self.weight_numpy[f'{layer_name}-input_gate_h-0.bias'] = data[4*len_:5*len_]
                                self.weight_numpy[f'{layer_name}-output_gate_h-0.bias'] = data[5*len_:6*len_]
                                self.weight_numpy[f'{layer_name}-forget_gate_h-0.bias'] = data[6*len_:7*len_]
                                self.weight_numpy[f'{layer_name}-cell_state_h-0.bias'] = data[7*len_:8*len_]
                            else:
                                self.weight_numpy[node_name] = np.transpose(numpy_helper.to_array(self.parameters[weight_name]))
                        else:
                            self.weight_numpy[node_name] = numpy_helper.to_array(self.parameters[weight_name])
                        
                        if self.weight_scale != None and weight_name in self.weight_scale.keys():
                            self.weight_quant_scale[node_name] = self.weight_scale[weight_name]
                            if self.weight_half_level != None:
                                raise ValueError("Only support with weight_half_level or weight_scale, not both !!!")
                        
                        if self.weight_half_level != None :
                            # if self.cpu_layer != None and layer_name in self.cpu_layer:
                            #     self.weight_numpy_quant[node_name] = self.weight_numpy[node_name]
                            #     continue
                            # bias 
                            if 'bias' in node_name:
                                if '_lstm_bias' in node_name:
                                    for n in ['-input_gate_x-0.bias', '-output_gate_x-0.bias', '-forget_gate_x-0.bias', '-cell_state_x-0.bias',
                                              '-input_gate_h-0.bias', '-output_gate_h-0.bias', '-forget_gate_h-0.bias', '-cell_state_h-0.bias']:
                                        self.weight_numpy_quant[f'{layer_name}{n}'] = self.weight_numpy[f'{layer_name}{n}']
                                else:
                                    self.weight_numpy_quant[node_name] = self.weight_numpy[node_name]
                            else:
                                # print(self.weight_half_level)
                                assert isinstance(self.weight_half_level,int)
                                assert (self.weight_half_level > 0)
                                if '_lstm_x' in node_name:
                                    for n in ['-input_gate_x-0.weight', '-output_gate_x-0.weight', '-forget_gate_x-0.weight', '-cell_state_x-0.weight']:
                                        tensor,scale = data_quantization_sym(self.weight_numpy[f'{layer_name}{n}'],
                                                                        half_level=self.weight_half_level,
                                                                        data_range = self.data_range_specify,
                                                                        clamp_std =  self.data_clamp_std,
                                                                        isint=1)
                                        self.weight_numpy_quant[f'{layer_name}{n}'] = tensor
                                        self.weight_quant_scale[f'{layer_name}{n}'] = scale
                                elif '_lstm_h' in node_name:
                                    for n in ['-input_gate_h-0.weight', '-output_gate_h-0.weight', '-forget_gate_h-0.weight', '-cell_state_h-0.weight']:
                                        tensor,scale = data_quantization_sym(self.weight_numpy[f'{layer_name}{n}'],
                                                                        half_level=self.weight_half_level,
                                                                        data_range = self.data_range_specify,
                                                                        clamp_std =  self.data_clamp_std,
                                                                        isint=1)
                                        self.weight_numpy_quant[f'{layer_name}{n}'] = tensor
                                        self.weight_quant_scale[f'{layer_name}{n}'] = scale
                                else:
                                    tensor,scale = data_quantization_sym(self.weight_numpy[node_name],
                                                                        half_level=self.weight_half_level,
                                                                        data_range = self.data_range_specify,
                                                                        clamp_std =  self.data_clamp_std,
                                                                        isint=1)
                                    self.weight_numpy_quant[node_name] = tensor
                                    self.weight_quant_scale[node_name] = scale
                
                
    def _save_constant_parameters(self):
        """save consatant parameters -> constant"""
        self.constant = get_model_constant_value(self.model)
        
        
    def _map_nodes(self):
        """Mapping name -> NodeProto"""
        
        for node in self.graph.node:
            self.nodes[node.name] = node
        
    def _map_orders(self):
        """Mapping predecessor and successor nodes for value infos"""
        for node in self.graph.node:
            for i in node.input:
                if i in self.graph_input:
                    self.predecessors[i].append(self.value_infos[i])
                self.successors[i].append(node)
            for o in node.output:
                if o in self.graph_output:
                    self.successors[o].append(self.value_infos[o])
                self.predecessors[o].append(node)
            
    def _map_edge_node_name(self):
        
        """Mapping edge name -> nodes name"""
        for node in self.graph.node:
            for i in node.input:
                if i not in self.edge_node_name.keys():
                    self.edge_node_name[i] = node.name
                    
            
                

    

        