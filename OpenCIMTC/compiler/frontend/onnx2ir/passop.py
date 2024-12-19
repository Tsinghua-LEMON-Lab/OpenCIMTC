
from ...irtool.core import make_op, make_layer, make_ir #noqa
from ...irtool.tools.loop import *
from .helper import *
import numpy

class MakeIROp:
    
    def Conv(self,ir,model_parser,node_name):
        node = model_parser.nodes[node_name]
        bias = False
        bias_shape = None
        if len(node.input) == 3:
            bias_name = node_name + '.bias'
            bias_value = model_parser.weight_numpy[bias_name]
            mean_ = numpy.mean(bias_value)
            std_ = numpy.std(bias_value)
            if mean_ != 0 or std_ != 0:
                bias = True 
                bias_shape = bias_value.shape
        input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        weight = node.input[1]
        weight_info = model_parser.value_infos[weight]
        weight_shape = dim_to_list(weight_info.type.tensor_type.shape.dim)
        stride,pad,kernel = get_conv_node_attr(node.attribute)
        dilation = get_conv_node_dilation(node.attribute)
        op_ = make_op('conv2d',in_channel=weight_shape[1],out_channel=weight_shape[0],kernel=kernel[0],
                    stride=stride[0], padding=pad[0], bias=bias, dilation = dilation[0])
        ref_name = model_parser.predecessors[node.input[0]][0].name
        # By default, a convolutional layer will only have one output value, which can be assigned to different layers.
        out_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        if ref_name in model_parser.inputs:
            index_ = model_parser.inputs.index(ref_name)
            ref_name = f'graph_input:{index_}'
        elif node.input[0] in model_parser.inputs:
            index_ = model_parser.inputs.index(node.input[0])
            ref_name = f'graph_input:{index_}'
        
        # Determine whether the previous layer of the convolution is a split layer. 
        # If so, you need to add the serial number to the ref name
        elif model_parser.nodes[ref_name].op_type == 'Split':
            node_in_name = node.input[0]
            index_ = list(model_parser.nodes[ref_name].output).index(node_in_name)
            ref_name = f'{ref_name}:{index_}'
        
        inputs_ = [get_input_info(input_shape,ref_name)]
        output_ = get_output_info(out_shape)
        weights_ = get_weight_info(weight_shape,bias_shape)
        ir.add_layer(node_name,op=op_,inputs=inputs_,outputs=output_,weights=weights_)
        
    def Add(self,ir,model_parser,node_name):
        op_ = make_op('add')
        input_ = []
        node = model_parser.nodes[node_name]
        c = 0
        for i in node.input:
            input_shape = dim_to_list(model_parser.value_infos[i].type.tensor_type.shape.dim)
            if input_shape == []:
                input_shape = [1]
            if i in model_parser.parameters.keys():
                # insert a constant op
                node_name_ = 'Constant_' + node.name + f'_{str(c)}'
                c += 1
                value = numpy_helper.to_array(model_parser.parameters[i])
                
                if value.shape == ():
                    out_shape = [1]
                else:
                    out_shape = value.shape
                op_1 = make_op('constant',value = value.tolist())
                output_ = get_output_info(out_shape)
                ir.add_layer(node_name_,op=op_1,outputs=output_)
                #
                t = get_input_info(out_shape,node_name_)
                input_.append(t)
            
            else:
                ref_name = model_parser.predecessors[i][0].name
                if ref_name in model_parser.inputs:
                    index_ = model_parser.inputs.index(ref_name)
                    ref_name = f'graph_input:{index_}'
                elif model_parser.nodes[ref_name].op_type == 'Split':
                    node_in_name = i
                    index_ = list(model_parser.nodes[ref_name].output).index(node_in_name)
                    ref_name = f'{ref_name}:{index_}'
                t = get_input_info(input_shape,ref_name)
                input_.append(t)
            
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(output_shape)
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_)
    
    def Constant(self,ir,model_parser,node_name):
        node = model_parser.nodes[node_name]
        value = model_parser.constant[node_name].tolist()
        out_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        if out_shape == [] or out_shape == [0]:
            out_shape = [1]
        if value == []:
            value = 0
        op_ = make_op('constant',value = value)
        output_ = get_output_info(out_shape)
        ir.add_layer(node_name,op=op_,outputs=output_)

    def Pad(self,ir,model_parser,node_name):
        node = model_parser.nodes[node_name]
        if len(node.input) > 1:
            print(model_parser.constant.keys())
            assert node.input[1] in model_parser.value_infos.keys(), f'{node.input[1]}'
            pads = model_parser.constant[node.input[1]].tolist()
        else:
            pads = get_node_pads(node)
        mode = get_pad_mode(node)
        value = get_pad_value(node)
        if b'constant' not in mode:
            raise ValueError(f'Not support padding mode: {mode} !!!')
        out_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        op_ = make_op('pad', pads=pads, value = value)
        
        input_ = []
        input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        ref_name = model_parser.predecessors[node.input[0]][0].name
        t = get_input_info(input_shape,ref_name)
        input_.append(t)
        output_ = get_output_info(out_shape)
        ir.add_layer(node_name,op=op_, inputs=input_, outputs=output_)
    
    def Transpose(self,ir,model_parser,node_name):
        
        node = model_parser.nodes[node_name]
        perm = get_perm(node)
        op_ = make_op('transpose', perm=perm)
        input_ = []
        for i in node.input:
            input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
            ref_name = model_parser.predecessors[i][0].name
            if ref_name in model_parser.inputs:
                index_ = model_parser.inputs.index(ref_name)
                ref_name = f'graph_input:{index_}'
            t = get_input_info(input_shape,ref_name)
            input_.append(t)
        out_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(out_shape)
        ir.add_layer(node_name, op=op_, inputs=input_, outputs=output_)
    
    def MatMul(self,ir,model_parser,node_name):
        # By default, this layer has only 1/2 outputs, which can be referenced by other layers.
        node = model_parser.nodes[node_name]
        weight = node.input[1]
        # input dimension
        input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        ref_name = model_parser.predecessors[node.input[0]][0].name
        out_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        if ref_name in model_parser.inputs:
            index_ = model_parser.inputs.index(ref_name)
            ref_name = f'graph_input:{index_}'
        elif node.input[0] in model_parser.inputs:
            index_ = model_parser.inputs.index(node.input[0])
            ref_name = f'graph_input:{index_}'
        # Determine whether the previous layer of the convolution is a split layer. 
        # If so, you need to add the serial number to the ref name
        elif model_parser.nodes[ref_name].op_type == 'Split':
            node_in_name = node.input[0]
            index_ = list(model_parser.nodes[ref_name].output).index(node_in_name)
            ref_name = f'{ref_name}:{index_}'
            
        # Determine whether weight is a constant
        if weight in model_parser.parameters.keys():
            weight_info = model_parser.value_infos[weight]
            weight_shape = dim_to_list(weight_info.type.tensor_type.shape.dim)
            inputs_ = [get_input_info(input_shape,ref_name)]
            # Swap the horizontal and vertical axes to keep the first dimension consistent with the output dimension
            temp = weight_shape[1]
            weight_shape[1] = weight_shape[0]
            weight_shape[0] = temp
            weights_ = get_weight_info(weight_shape)
            op_ = make_op('matmul',in_channel=weight_shape[1],out_channel=weight_shape[0],bias=False)
            outputs_ = get_output_info(out_shape)
            ir.add_layer(node_name,op=op_,inputs=inputs_,outputs=outputs_,weights=weights_)
        # Dynamic matrix multiplication (attention)   
        else:
            weight_shape = dim_to_list(model_parser.value_infos[node.input[1]].type.tensor_type.shape.dim)
            weight_ref_name = model_parser.predecessors[node.input[1]][0].name
            inputs_ = [get_input_info(input_shape,ref_name), get_input_info(weight_shape,weight_ref_name)]
            op_ = make_op('matmul',in_channel=weight_shape[0],out_channel=weight_shape[1],bias=False)
            outputs_ = get_output_info(out_shape)
            ir.add_layer(node_name,op=op_,inputs=inputs_,outputs=outputs_)
        
    def Gemm(self,ir,model_parser,node_name):
        node = model_parser.nodes[node_name]
        bias = False
        bias_shape = None
        if len(node.input) == 3:
            bias_name = node_name + '.bias'
            bias_value = model_parser.weight_numpy[bias_name]
            mean_ = numpy.mean(bias_value)
            std_ = numpy.std(bias_value)
            if mean_ != 0 or std_ != 0:
                bias = True 
                bias_shape = bias_value.shape
        weight = node.input[1]
        weight_info = model_parser.value_infos[weight]
        weight_shape = dim_to_list(weight_info.type.tensor_type.shape.dim)
        input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        out_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        op_ = make_op('matmul',in_channel=weight_shape[1],out_channel=weight_shape[0],bias=bias)
        ref_name = model_parser.predecessors[node.input[0]][0].name
        if ref_name in model_parser.inputs:
            index_ = index_ = model_parser.inputs.index(ref_name)
            ref_name = f'graph_input:{index_}'
        inputs_ = [get_input_info(input_shape,ref_name)]
        outputs_ = get_output_info(out_shape)
        weights_ = get_weight_info(weight_shape, bias_shape)
        ir.add_layer(node_name,op=op_,inputs=inputs_,outputs=outputs_,weights=weights_)
    
    def Mul(self,ir,model_parser,node_name):
        op_ = make_op('mul')
        input_ = []
        node = model_parser.nodes[node_name]
        c = 0
        for i in node.input:
            input_shape = dim_to_list(model_parser.value_infos[i].type.tensor_type.shape.dim)
            if input_shape == []:
                input_shape = [1]
            if i in model_parser.parameters.keys():
                # insert a constant op
                node_name_ = 'Constant_' + node.name + f'_{str(c)}'
                c += 1
                value = numpy_helper.to_array(model_parser.parameters[i])
                
                if value.shape == ():
                    out_shape = [1]
                else:
                    out_shape = value.shape
                op_1 = make_op('constant',value = value.tolist())
                output_ = get_output_info(out_shape)
                ir.add_layer(node_name_,op=op_1,outputs=output_)
                #
                t = get_input_info(out_shape,node_name_)
                input_.append(t)
            else:
                ref_name = model_parser.predecessors[i][0].name
                t = get_input_info(input_shape,ref_name)
                input_.append(t)
            # ref_name = model_parser.predecessors[i][0].name
            # t = get_input_info(input_shape,ref_name)
            # input_.append(t)
        out_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(out_shape)
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_)
        
    def Reshape(self,ir,model_parser,node_name):
        input_ = []
        node = model_parser.nodes[node_name]
        # if node.input[1] in model_parser.parameters.keys():
        #     shape = model_parser.parameters[node.input[1]]
        #     shape = numpy_helper.to_array(shape).tolist()
        # else:
        #     pre_node = model_parser.predecessors[node.input[1]][0]
        #     shape = model_parser.constant[pre_node.name].tolist()
        
        shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        # batch size default to -1
        shape[0] = -1
        # shape = [1, -1]
        op_ = make_op('reshape',shape=shape)
        for i in node.input:
            if i in model_parser.parameters.keys():
                continue
            input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
            ref_name = model_parser.predecessors[i][0].name
            if ref_name in model_parser.inputs:
                index_ = model_parser.inputs.index(ref_name)
                ref_name = f'graph_input:{index_}'
            elif model_parser.nodes[ref_name].op_type == 'Split':
                node_in_name = i
                index_ = list(model_parser.nodes[ref_name].output).index(node_in_name)
                ref_name = f'{ref_name}:{index_}'
            t = get_input_info(input_shape,ref_name)
            input_.append(t)
        out_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(out_shape)
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_)
    
    def Resize(self,ir,model_parser,node_name):
        input_ = []
        node = model_parser.nodes[node_name]
        
        if len(node.input) == 3:
            if node.input[2] in model_parser.parameters.keys():
                scale = numpy_helper.to_array(model_parser.parameters[node.input[2]]).astype(np.int32).tolist()
            else:
                pre_node = model_parser.predecessors[node.input[2]][0]
                scale = model_parser.constant[pre_node.name].tolist()
        else:
            if node.input[1] != '':
                if node.input[1] in model_parser.parameters.keys():
                    if isinstance(model_parser.parameters[node.input[1]], numpy.ndarray):
                        scale = model_parser.parameters[node.input[1]].astype(np.int32).tolist()
                    else:
                        scale = numpy_helper.to_array(model_parser.parameters[node.input[1]]).astype(np.int32).tolist()
                else:
                    pre_node = model_parser.predecessors[node.input[1]][0]
                    scale = model_parser.constant[pre_node.name].tolist()
        
        input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        out_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        if scale == [] or np.array(scale).mean() == 0:
            scale = []
            for i in range(len(input_shape)):
                scale.append(out_shape[i] // input_shape[i])
        
        mode = get_resize_mode(node)
        op_ = make_op('resize', scale=scale, mode = mode)
        
        ref_name = model_parser.predecessors[node.input[0]][0].name
        if ref_name in model_parser.inputs:
            index_ = model_parser.inputs.index(ref_name)
            ref_name = f'graph_input:{index_}'
        t = get_input_info(input_shape,ref_name)
        input_.append(t)
        
        output_ = get_output_info(out_shape)
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_)
    
    def Flatten(self,ir,model_parser,node_name):
        
        # input_ = []
        node = model_parser.nodes[node_name]
        axis = get_axis(node)
        op_ = make_op('flatten', start_dim=axis)
        # default this layer has only one input
        input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        ref_name = model_parser.predecessors[node.input[0]][0].name
        # default this layer has only one output
        out_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        if ref_name in model_parser.inputs:
            index_ = model_parser.inputs.index(ref_name)
            ref_name = f'graph_input:{index_}'
        inputs_ = [get_input_info(input_shape,ref_name)]
        out_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(out_shape)
        ir.add_layer(node_name,op=op_,inputs=inputs_,outputs=output_)
    
    def Relu(self,ir,model_parser,node_name):
        op_ = make_op('relu')
        input_ = []
        node = model_parser.nodes[node_name]
        for i in node.input:
            input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
            ref_name = model_parser.predecessors[i][0].name
            if ref_name in model_parser.inputs:
                index_ = model_parser.inputs.index(ref_name)
                ref_name = f'graph_input:{index_}'
            if i in model_parser.inputs:
                index_ = model_parser.inputs.index(i)
                ref_name = f'graph_input:{index_}'
            t = get_input_info(input_shape,ref_name)
            input_.append(t)
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(output_shape)
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_)
    
    def LeakyRelu(self,ir,model_parser,node_name):
        
        node = model_parser.nodes[node_name]
        alpha = get_alpha(node)
        op_ = make_op('leaky_relu', alpha = alpha)
        input_ = []
        for i in node.input:
            input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
            ref_name = model_parser.predecessors[i][0].name
            t = get_input_info(input_shape,ref_name)
            input_.append(t)
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(output_shape)
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_)
    
    def Concat(self,ir,model_parser,node_name):
        input_ = []
        node = model_parser.nodes[node_name]
        axis = get_axis(node)
        index = 0
        c =  0
        for i in node.input:
            if i in model_parser.parameters.keys():
                value = numpy_helper.to_array(model_parser.parameters[i])
                if i not in model_parser.constant.keys():
                    # node_name_ = 'Constant_' + node.name
                    node_name_ = 'Constant_' + node.name + f'_{str(c)}'
                    c += 1
                    if value.shape == ():
                        out_shape = [1]
                    else:
                        out_shape = value.shape
                    op_1 = make_op('constant',value = value.tolist())
                    output_ = get_output_info(out_shape)
                    ir.add_layer(node_name_,op=op_1,outputs=output_)
                    # 
                    input_shape = out_shape
                    ref_name = node_name_
                else:
                    input_shape = value.shape
                    ref_name = i
                t = get_input_info(input_shape,ref_name)
                
            else:
                input_shape = dim_to_list(model_parser.value_infos[node.input[index]].type.tensor_type.shape.dim)
                ref_name = model_parser.predecessors[i][0].name
                if ref_name in model_parser.inputs:
                    index_ = model_parser.inputs.index(ref_name)
                    ref_name = f'graph_input:{index_}'
                elif model_parser.nodes[ref_name].op_type == 'Split':
                    node_in_name = i
                    index_ = list(model_parser.nodes[ref_name].output).index(node_in_name)
                    ref_name = f'{ref_name}:{index_}'
                t = get_input_info(input_shape,ref_name)
            input_.append(t)
            index += 1
        op_ = make_op('concat',axis=axis)
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(output_shape)    
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_) 
    
    def ConvTranspose(self,ir,model_parser,node_name):
        node = model_parser.nodes[node_name]
        # Bias
        bias = False
        bias_shape = None
        if len(node.input) == 3:
            bias_name = node_name + '.bias'
            bias_value = model_parser.weight_numpy[bias_name]
            mean_ = numpy.mean(bias_value)
            std_ = numpy.std(bias_value)
            if mean_ != 0 or std_ != 0:
                bias = True 
                bias_shape = bias_value.shape
        input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        weight = node.input[1]
        weight_info = model_parser.value_infos[weight]
        weight_shape = dim_to_list(weight_info.type.tensor_type.shape.dim)
        stride,pad,kernel = get_conv_node_attr(node.attribute)
        # Transposed convolution, the first dimension of the weight is the input channel,
        # the second dimension is the output channel
        op_ = make_op('conv_transpose2d',in_channel=weight_shape[0],out_channel=weight_shape[1],kernel=kernel[0],
                    stride=stride[0],padding=pad[0],bias=bias)
        ref_name = model_parser.predecessors[node.input[0]][0].name
        # By default, this layer will only have one output value, which can be assigned to different layers.
        out_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        if ref_name in model_parser.inputs:
            index_ = model_parser.inputs.index(ref_name)
            ref_name = f'graph_input:{index_}'
        inputs_ = [get_input_info(input_shape,ref_name)]
        output_ = get_output_info(out_shape)
        weights_ = get_weight_info(weight_shape,bias_shape)
        ir.add_layer(node_name,op=op_,inputs=inputs_,outputs=output_,weights=weights_)
    
    def Div(self,ir,model_parser,node_name):
        op_ = make_op('div')
        input_ = []
        node = model_parser.nodes[node_name]
        c = 0
        for i in node.input:
            if i in model_parser.parameters.keys():
                # node_name_ = 'Constant_' + node.name
                node_name_ = 'Constant_' + node.name + f'_{str(c)}'
                c += 1
                value = numpy_helper.to_array(model_parser.parameters[i])
                
                if value.shape == ():
                    out_shape = [1]
                else:
                    out_shape = value.shape
                op_1 = make_op('constant',value = value.tolist())
                output_ = get_output_info(out_shape)
                ir.add_layer(node_name_,op=op_1,outputs=output_)
                #
                t = get_input_info(out_shape,node_name_)
                input_.append(t)
            else:
                input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
                ref_name = model_parser.predecessors[i][0].name
                if ref_name in model_parser.inputs:
                    index_ = model_parser.inputs.index(ref_name)
                    ref_name = f'graph_input:{index_}'
                t = get_input_info(input_shape,ref_name)
                input_.append(t)
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(output_shape)    
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_) 
    
    def Softmax(self,ir,model_parser,node_name):
        axis = get_axis(model_parser.nodes[node_name])
        op_ = make_op('softmax',axis=axis)
        input_ = []
        node = model_parser.nodes[node_name]
        for i in node.input:
            input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
            ref_name = model_parser.predecessors[i][0].name
            t = get_input_info(input_shape,ref_name)
            input_.append(t)
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(output_shape)    
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_) 
    
    def Sigmoid(self,ir,model_parser,node_name):
        op_ = make_op('sigmoid')
        input_ = []
        node = model_parser.nodes[node_name]
        for i in node.input:
            input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
            ref_name = model_parser.predecessors[i][0].name
            t = get_input_info(input_shape,ref_name)
            input_.append(t)
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(output_shape)    
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_) 
    
    def Silu(self,ir,model_parser,node_name):
        # y= x * sigmoid(x)
        op_ = make_op('silu')
        input_ = []
        node = model_parser.nodes[node_name]
        for i in node.input:
            input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
            ref_name = model_parser.predecessors[i][0].name
            t = get_input_info(input_shape,ref_name)
            input_.append(t)
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(output_shape)    
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_) 
    
    def Tanh(self,ir,model_parser,node_name):
        op_ = make_op('tanh')
        input_ = []
        node = model_parser.nodes[node_name]
        for i in node.input:
            input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
            ref_name = model_parser.predecessors[i][0].name
            t = get_input_info(input_shape,ref_name)
            input_.append(t)
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(output_shape)    
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_)
    
    def LogSoftmax(self,ir,model_parser,node_name):
        op_ = make_op('logsoftmax')
        input_ = []
        node = model_parser.nodes[node_name]
        for i in node.input:
            input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
            ref_name = model_parser.predecessors[i][0].name
            t = get_input_info(input_shape,ref_name)
            input_.append(t)
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(output_shape)    
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_)

    def MaxPool(self,ir,model_parser,node_name):
        node = model_parser.nodes[node_name]
        stride,pad,kernel = get_conv_node_attr(node.attribute)
        op_ = make_op('maxpool2d',kernel=kernel[0],stride=stride[0],padding=pad[0])
        input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        ref_name = model_parser.predecessors[node.input[0]][0].name
        t = get_input_info(input_shape,ref_name)
        inputs_ = [t]
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        outputs_ = get_output_info(output_shape)
        ir.add_layer(node_name,op=op_,inputs=inputs_,outputs=outputs_)

    def AveragePool(self,ir,model_parser,node_name):
        node = model_parser.nodes[node_name]
        stride,pad,kernel = get_conv_node_attr(node.attribute)
        op_ = make_op('avgpool2d',kernel=kernel[0],stride=stride[0],padding=pad[0])
        input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        ref_name = model_parser.predecessors[node.input[0]][0].name
        t = get_input_info(input_shape,ref_name)
        inputs_ = [t]
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        outputs_ = get_output_info(output_shape)
        ir.add_layer(node_name,op=op_,inputs=inputs_,outputs=outputs_)
    
    def GlobalAveragePool(self,ir,model_parser,node_name):
        node = model_parser.nodes[node_name]
        input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        op_ = make_op('global_avg_pool2d', kernel=input_shape[2],stride=1,padding=0)
        ref_name = model_parser.predecessors[node.input[0]][0].name
        if ref_name in model_parser.inputs:
            index_ = model_parser.inputs.index(ref_name)
            ref_name = f'graph_input:{index_}'
        elif node.input[0] in model_parser.inputs:
            index_ = model_parser.inputs.index(node.input[0])
            ref_name = f'graph_input:{index_}'
        
        t = get_input_info(input_shape,ref_name)
        inputs_ = [t]
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        outputs_ = get_output_info(output_shape)
        ir.add_layer(node_name,op=op_,inputs=inputs_,outputs=outputs_)
        
    def Split(self, ir, model_parser, node_name):
        node = model_parser.nodes[node_name]
        # get input info
        input_ = []
        # for i in node.input:
        input_ref = node.input[0]
        input_shape = dim_to_list(model_parser.value_infos[input_ref].type.tensor_type.shape.dim)
        ref_name = model_parser.predecessors[input_ref][0].name
        if ref_name in model_parser.inputs: 
            index_ = model_parser.inputs.index(ref_name)
            ref_name = f'graph_input:{index_}'
        elif input_ref in model_parser.inputs:
            index_ = model_parser.inputs.index(input_ref)
            ref_name = f'graph_input:{index_}'
        t = get_input_info(input_shape,ref_name)
        input_.append(t)
            
        
        axis = get_axis(node)
        if axis == -1:
            axis = len(input_shape) - 1
        split = get_split(node)
        if split == None:
            # split = dim_to_list(model_parser.value_infos[node.input[1]].type.tensor_type.shape.dim)
            assert node.input[1] in model_parser.parameters.keys()
            split = numpy_helper.to_array(model_parser.parameters[node.input[1]]).tolist()
        op_ = make_op('split',axis=axis, split=split)
        # output info
        output_ = []
        for j in node.output:
            output_shape = dim_to_list(model_parser.value_infos[j].type.tensor_type.shape.dim)
            t = get_output_info(output_shape)[0] 
            output_.append(t)
           
        ir.add_layer(node_name, op=op_, inputs=input_, outputs=output_)
    
    def Pow(self,ir,model_parser,node_name):
        op_ = make_op('pow')
        input_ = []
        node = model_parser.nodes[node_name]
        for i in node.input:
            input_shape = dim_to_list(model_parser.value_infos[i].type.tensor_type.shape.dim)
            if i in model_parser.parameters.keys():
                node_name_ = 'Constant_' + node.name
                value = numpy_helper.to_array(model_parser.parameters[i])
                if value.shape == ():
                    out_shape = [1]
                else:
                    out_shape = value.shape
                op_1 = make_op('constant',value = value.tolist())
                output_ = get_output_info(out_shape)
                ir.add_layer(node_name_,op=op_1,outputs=output_)
                # 
                t = get_input_info(out_shape,node_name_)
                input_.append(t)
            else:
                ref_name = model_parser.predecessors[i][0].name
                if ref_name in model_parser.inputs:
                    index_ = model_parser.inputs.index(ref_name)
                    ref_name = f'graph_input:{index_}'
                t = get_input_info(input_shape,ref_name)
                input_.append(t)
        out_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(out_shape)
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_)
    
    
    def ReduceMean(self,ir,model_parser,node_name):
        
        input_ = []
        node = model_parser.nodes[node_name]
        axes = get_axes(node)[0]
        keepdims = get_keepdims(node)
        op_ = make_op('reducemean',axes=axes, keepdims=keepdims)
        for i in node.input:
            input_shape = dim_to_list(model_parser.value_infos[i].type.tensor_type.shape.dim)
            ref_name = model_parser.predecessors[i][0].name
            t = get_input_info(input_shape,ref_name)
            input_.append(t)
            
        out_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(out_shape)
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_)
        
        # ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_)
    
    def Sqrt(self,ir,model_parser,node_name):
        op_ = make_op('sqrt')
        input_ = []
        node = model_parser.nodes[node_name]
        for i in node.input:
            input_shape = dim_to_list(model_parser.value_infos[i].type.tensor_type.shape.dim)
            ref_name = model_parser.predecessors[i][0].name
            t = get_input_info(input_shape,ref_name)
            input_.append(t)
        out_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(out_shape)
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_)
        
    def BatchNormalization(self,ir,model_parser,node_name):
        input_ = []
        node = model_parser.nodes[node_name]
        # input
        input_ = []
        node = model_parser.nodes[node_name]
        input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        ref_name = model_parser.predecessors[node.input[0]][0].name
        if ref_name in model_parser.inputs:
            index_ = model_parser.inputs.index(ref_name)
            ref_name = f'graph_input:{index_}'
        t = get_input_info(input_shape,ref_name)
        input_.append(t)
        # epsilon
        epsilon = get_node_epsilon(node)
        
        # scale, bias, input_mean, input_var
        scale = numpy_helper.to_array(model_parser.parameters[node.input[1]]).tolist()
        bias = numpy_helper.to_array(model_parser.parameters[node.input[2]]).tolist()
        input_mean = numpy_helper.to_array(model_parser.parameters[node.input[3]]).tolist()
        input_var = numpy_helper.to_array(model_parser.parameters[node.input[4]]).tolist()
        
        op_ = make_op('batch_norm2d', channel=input_shape[1], epsilon = float(epsilon), scale=scale,
                    bias= bias, input_mean=input_mean, input_var=input_var)
        
        out_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(out_shape)
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_)
    
    
    def LSTM(self, ir, model_parser, node_name):
        
        node = model_parser.nodes[node_name]
        hidden_size = get_hidden_size(node)
        word_size = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)[-1]
        repeat = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)[0]
        ref_name = model_parser.predecessors[node.input[0]][0].name
        
        #  constant 
        h0 = make_layer(type='op', op=make_op('constant', value=[0] * hidden_size))
        c0 = make_layer(type='op', op=make_op('constant', value=[0] * hidden_size))

        fc_x = make_op('matmul', in_channel= word_size, out_channel = hidden_size, bias=True)
        fc_h = make_op('matmul', in_channel= hidden_size, out_channel = hidden_size, bias=True)
        sigmoid = make_op('sigmoid')
        mul = make_op('mul')
        tanh = make_op('tanh')
        add = make_op('add')
        relu = make_op('relu')

        # lstm_layer: loop layer
        lstm_layer = make_layer(type='loop', repeat=repeat, inputs=[ref_name, 'h0', 'c0'])

        # loop layer
        lstm_in_layer_in_data = LoopDataDef(loop=dict(source='split', axis=0))
        lstm_in_layer_h = LoopDataDef(loop=dict(source='output', index=0))
        lstm_in_layer_c = LoopDataDef(loop=dict(source='output', index=1))

        # input
        lstm_indata_layer = make_layer(type='input',datadef=LoopDataDef, inputs=[lstm_in_layer_in_data.clone(), lstm_in_layer_h.clone(), lstm_in_layer_c.clone()])
        lstm_layer.add_layer('lstm_in', lstm_indata_layer)

        lstm_layer.add_layer('lstm_in_x_relu', op=relu.clone(), 
                            inputs=[dict(ref='lstm_in:0', channel=100, width=1, height=1)],
                            outputs=[dict(channel=100, width=1, height=1)]
                            )
        lstm_layer.add_layer('lstm_in_h_relu', op=relu.clone(), 
                            inputs=[dict(ref='lstm_in:1', channel=100, width=1, height=1)],
                            outputs=[dict(channel=100, width=1, height=1)]
                            )

        # input gate
        lstm_layer.add_layer('input_gate_x', op=fc_x.clone(), 
                            inputs=[dict(ref='lstm_in_x_relu', channel=word_size, width=1, height=1)],
                            weights={'weight':{'shape':[hidden_size, word_size]}, 'bias':{'shape':[word_size]}}, 
                            outputs=[dict(channel=hidden_size, width=1, height=1)]
                            )
        lstm_layer.add_layer('input_gate_h', op=fc_h.clone(), 
                            inputs=[dict(ref='lstm_in_h_relu', channel=hidden_size, width=1, height=1)],
                            weights={'weight':{'shape':[hidden_size, hidden_size]}}, 
                            outputs=[dict(channel=hidden_size, width=1, height=1)]
                            )
        lstm_layer.add_layer('input_gate', op=add.clone(), 
                            inputs=[dict(ref='input_gate_x', channel=hidden_size, width=1, height=1),
                                    dict(ref='input_gate_h', channel=hidden_size, width=1, height=1)], 
                            outputs=[dict(channel=hidden_size, width=1, height=1)]
                            )
        lstm_layer.add_layer('activated_input_gate', op=sigmoid.clone(), 
                            inputs=[dict(ref='input_gate', channel=hidden_size, width=1, height=1)],
                            outputs=[dict(channel=hidden_size, width=1, height=1)])

        # forget gate
        lstm_layer.add_layer('forget_gate_x', op=fc_x.clone(), 
                            inputs=[dict(ref='lstm_in_x_relu', channel=word_size, width=1, height=1)],
                            weights={'weight':{'shape':[hidden_size, word_size]}, 'bias':{'shape':[hidden_size]}}, 
                            outputs=[dict(channel=hidden_size, width=1, height=1)]
                            )
        lstm_layer.add_layer('forget_gate_h', op=fc_h.clone(), 
                            inputs=[dict(ref='lstm_in_h_relu', channel=hidden_size, width=1, height=1)],
                            weights={'weight':{'shape':[hidden_size, hidden_size]}},
                            outputs=[dict(channel=hidden_size, width=1, height=1)]
                            )
        lstm_layer.add_layer('forget_gate', op=add.clone(), 
                            inputs=[dict(ref='forget_gate_x', channel=hidden_size, width=1, height=1),
                                    dict(ref='forget_gate_h', channel=hidden_size, width=1, height=1)], 
                            outputs=[dict(channel=hidden_size, width=1, height=1)]
                            )
        lstm_layer.add_layer('activated_forget_gate', op=sigmoid.clone(), 
                            inputs=[dict(ref='forget_gate',channel=hidden_size, width=1, height=1)],
                            outputs=[dict(channel=hidden_size, width=1, height=1)])
        # output gate
        lstm_layer.add_layer('output_gate_x', op=fc_x.clone(), 
                            inputs=[dict(ref='lstm_in_x_relu', channel=word_size, width=1, height=1)],
                            weights={'weight':{'shape':[hidden_size, word_size]}, 'bias':{'shape':[hidden_size]}}, 
                            outputs=[dict(channel=hidden_size, width=1, height=1)]
                            )
        lstm_layer.add_layer('output_gate_h', op=fc_h.clone(), 
                            inputs=[dict(ref='lstm_in_h_relu', channel=hidden_size, width=1, height=1)],
                            weights={'weight':{'shape':[hidden_size, hidden_size]}},
                            outputs=[dict(channel=hidden_size, width=1, height=1)]
                            )
        lstm_layer.add_layer('output_gate', op=add.clone(), 
                            inputs=[dict(ref='output_gate_x', channel=hidden_size, width=1, height=1),
                                    dict(ref='output_gate_h', channel=hidden_size, width=1, height=1)], 
                            outputs=[dict(channel=hidden_size, width=1, height=1)]
                            )
        lstm_layer.add_layer('activated_output_gate', op=sigmoid.clone(),
                            inputs=[dict(ref='output_gate', channel=hidden_size, width=1, height=1)],
                            outputs=[dict(channel=hidden_size, width=1, height=1)])
        # cell state
        lstm_layer.add_layer('cell_state_x', op=fc_x.clone(), 
                            inputs=[dict(ref='lstm_in_x_relu', channel=word_size, width=1, height=1)],
                            weights={'weight':{'shape':[hidden_size, word_size]}, 'bias':{'shape':[hidden_size]}}, 
                            outputs=[dict(channel=hidden_size, width=1, height=1)]
                            )
        lstm_layer.add_layer('cell_state_h', op=fc_h.clone(), 
                            inputs=[dict(ref='lstm_in_h_relu', channel=hidden_size, width=1, height=1)],
                            weights={'weight':{'shape':[hidden_size, hidden_size]}},
                            outputs=[dict(channel=hidden_size, width=1, height=1)]
                            )
        lstm_layer.add_layer('cell_state', op=add.clone(), 
                            inputs=[dict(ref='cell_state_x', channel=hidden_size, width=1, height=1),
                                    dict(ref='cell_state_h', channel=hidden_size, width=1, height=1)], 
                            outputs=[dict(channel=hidden_size, width=1, height=1)]
                            )
        lstm_layer.add_layer('activated_cell_state', op=tanh.clone(), 
                            inputs=[dict(ref='cell_state',channel=hidden_size, width=1, height=1)],
                            outputs=[dict(channel=hidden_size, width=1, height=1)])
        # update c
        lstm_layer.add_layer('update_cell_state_mul_1', op=mul.clone(), 
                            inputs=[dict(ref='activated_forget_gate',channel=hidden_size, width=1, height=1),
                                    dict(ref='lstm_in:2',channel=hidden_size, width=1, height=1)],
                            outputs=[dict(channel=hidden_size, width=1, height=1)]
                            )
        lstm_layer.add_layer('update_cell_state_mul_2', op=mul.clone(), 
                            inputs=[dict(ref='activated_input_gate',channel=hidden_size, width=1, height=1),
                                    dict(ref='activated_cell_state',channel=hidden_size, width=1, height=1)],
                            outputs=[dict(channel=hidden_size, width=1, height=1)])
        lstm_layer.add_layer('updated_cell_state', op=add.clone(), 
                            inputs=[dict(ref='update_cell_state_mul_1',channel=hidden_size, width=1, height=1),
                                    dict(ref='update_cell_state_mul_2',channel=hidden_size, width=1, height=1)],
                            outputs=[dict(channel=hidden_size, width=1, height=1)])
        # update h
        lstm_layer.add_layer('update_h_tanh_1', op=tanh.clone(), 
                            inputs=[dict(ref='updated_cell_state',channel=hidden_size, width=1, height=1)],
                            outputs=[dict(channel=hidden_size, width=1, height=1)])
        lstm_layer.add_layer('updated_h', op=mul.clone(), 
                            inputs=[dict(ref='activated_output_gate',channel=hidden_size, width=1, height=1),
                                    dict(ref='update_h_tanh_1',channel=hidden_size, width=1, height=1)],
                            outputs=[dict(channel=hidden_size, width=1, height=1)])
        # output layer
        lstm_layer.add_layer('lstm_out',type='output', 
                            inputs=[dict(ref='updated_h',channel=hidden_size, width=1, height=1),
                                    dict(ref='updated_cell_state',channel=hidden_size, width=1, height=1)]
                            )
        ir.add_layer('h0', h0)
        ir.add_layer('c0', c0)
        ir.add_layer(node_name, lstm_layer)
        
    def Squeeze(self,ir,model_parser,node_name):
        node = model_parser.nodes[node_name]
        axes = get_axes(node)
        if axes == []:
            assert len(node.input) > 1
            axes = numpy_helper.to_array(model_parser.parameters[node.input[1]]).tolist()            
        op_ = make_op('squeeze', axes=axes)
        input_ = []
        node = model_parser.nodes[node_name]
        # input
        input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        ref_name = model_parser.predecessors[node.input[0]][0].name
        t = get_input_info(input_shape,ref_name)
        input_.append(t)
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(output_shape)
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_)
        
    def Gather(self,ir,model_parser,node_name):
        node = model_parser.nodes[node_name]
        axis = get_axis(node)
        if node.input[1] in model_parser.predecessors.keys():
            pre_node = model_parser.predecessors[node.input[1]][0]
            indices = model_parser.constant[pre_node.name].tolist()
        elif node.input[1] in model_parser.parameters.keys():
            indices = numpy_helper.to_array(model_parser.parameters[node.input[1]]).tolist()
        else:
            raise ValueError(f'Gather op indices attribute is not found!!!')
        op_ = make_op('gather', axis = axis, indices = indices)
        # inputs
        input_ = []
        input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        ref_name = model_parser.predecessors[node.input[0]][0].name
        t = get_input_info(input_shape,ref_name)
        input_.append(t)
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(output_shape)
        ir.add_layer(node_name, op=op_, inputs=input_, outputs=output_)
    
    
    def Unsqueeze(self,ir,model_parser,node_name):
        node = model_parser.nodes[node_name]
        axes = get_axes(node)
        if axes == []:
            assert len(node.input) > 1
            axes = numpy_helper.to_array(model_parser.parameters[node.input[1]]).tolist()
        op_ = make_op('unsqueeze', axes = axes)
        # inputs
        input_ = []
        input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        ref_name = model_parser.predecessors[node.input[0]][0].name
        t = get_input_info(input_shape,ref_name)
        input_.append(t)
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(output_shape)
        ir.add_layer(node_name, op=op_, inputs=input_, outputs=output_)
    
    def LayerNormalization(self,ir,model_parser,node_name):
        input_ = []
        node = model_parser.nodes[node_name]
        # input
        input_ = []
        node = model_parser.nodes[node_name]
        input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        ref_name = model_parser.predecessors[node.input[0]][0].name
        if ref_name in model_parser.inputs:
            index_ = model_parser.inputs.index(ref_name)
            ref_name = f'graph_input:{index_}'
        t = get_input_info(input_shape,ref_name)
        input_.append(t)
        # epsilon
        epsilon = get_node_epsilon(node)
        axis = get_axis(node)
        
        # scale, bias
        scale = numpy_helper.to_array(model_parser.parameters[node.input[1]]).tolist()
        bias = numpy_helper.to_array(model_parser.parameters[node.input[2]]).tolist()
        
        op_ = make_op('layer_norm', axis=axis, epsilon = float(epsilon), scale=scale, bias= bias)
        
        out_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(out_shape)
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_)
    
    def Erf(self,ir,model_parser,node_name):
        op_ = make_op('erf')
        input_ = []
        node = model_parser.nodes[node_name]
        # input
        input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        ref_name = model_parser.predecessors[node.input[0]][0].name
        if ref_name in model_parser.inputs:
            index_ = model_parser.inputs.index(ref_name)
            ref_name = f'graph_input:{index_}'
        if node.input[0] in model_parser.inputs:
            index_ = model_parser.inputs.index(node.input[0])
            ref_name = f'graph_input:{index_}'
        t = get_input_info(input_shape,ref_name)
        input_.append(t)
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(output_shape)
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_)
    
    
    def Slice(self, ir, model_parser, node_name):
        node = model_parser.nodes[node_name]
        starts = numpy_helper.to_array(model_parser.parameters[node.input[1]]).tolist()
        ends = numpy_helper.to_array(model_parser.parameters[node.input[2]]).tolist()
        axes = numpy_helper.to_array(model_parser.parameters[node.input[3]]).tolist()       
        steps = numpy_helper.to_array(model_parser.parameters[node.input[4]]).tolist() 
        op_ = make_op('slice', starts=starts, ends=ends, axes=axes, steps=steps)
        input_ = []
        input_shape = dim_to_list(model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
        ref_name = model_parser.predecessors[node.input[0]][0].name
        t = get_input_info(input_shape,ref_name)
        input_.append(t)
        output_shape = dim_to_list(model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
        output_ = get_output_info(output_shape)
        ir.add_layer(node_name,op=op_,inputs=input_,outputs=output_)