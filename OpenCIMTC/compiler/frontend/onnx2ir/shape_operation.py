
from numpy import random
from .helper import *
import onnxruntime as ort
import numpy as np 
from onnx import helper, shape_inference
from onnx import TensorProto
import json
from .shape_inferece import ShapeInference
import os


def is_input_ready(value_info,input_list,inferenced_output_name_list,initializer_dict,constant_value_dict):
    isready = True
    for i in input_list:
        if value_info_has_shape(value_info[i]) or i in inferenced_output_name_list or i in initializer_dict.keys() or i in constant_value_dict.keys():
            continue
        else:
            isready = False
            break
    return isready

def shape_inference_(model):
    outputs = [o.name for o in model.graph.output]
    value_info = {}
    for i in model.graph.input:
        value_info[i.name] = i
    for n in model.graph.value_info:
        value_info[n.name] = n
    initializer_dict = get_model_initializer_value(model)
    constant_value_dict = get_model_constant_value(model)
    constant_output_node_dict = get_model_constant_value(model, OutputNode=True)
    inferenced_output_name_list = []
    node_list = list(model.graph.node)
    while node_list != []:
        i = node_list[0]
        if i.output[0] not in outputs :
            if i.output[0] not in value_info.keys() or not value_info_has_shape(value_info[i.output[0]]):
                if is_input_ready(value_info,i.input,inferenced_output_name_list,initializer_dict,constant_value_dict):
                    node_output_shape = []
                    node_output_type = value_info[i.output[0]].type.tensor_type.elem_type
                    ShapeInference_ = ShapeInference(i, value_info, initializer_dict, constant_value_dict, constant_output_node_dict)
                    node_output_shape = getattr(ShapeInference_,i.op_type)()
                    node_output_value_info = helper.make_tensor_value_info(
                        i.output[0],
                        node_output_type,
                        node_output_shape,
                    )
                    value_info[i.output[0]] = node_output_value_info
                    inferenced_output_name_list.append(i.output[0])
                    node_list.pop(0)
                else:
                    random.shuffle(node_list)
                    continue
            else:
                node_list.pop(0)
        else:
            node_list.pop(0)

    graph = helper.make_graph(
        model.graph.node,
        model.graph.name,
        model.graph.input,
        model.graph.output,
        model.graph.initializer,
        doc_string = None,
        value_info = value_info.values()
    )  
    model = helper.make_model(graph)
    return model

def load_onnx_model(model, label_file=None, output_npu_model_filename = None, store_intermediate_model=False, fix_layer_name=False):

    file_path = os.getcwd()
    # model = onnx.load(model_path)
    model = shape_inference.infer_shapes(model)
    #print(len(list(model.graph.node)))
    constant_dict = get_model_constant_node(model)
    initializer_dict = get_model_initializer_value(model)   
    node_list,node_output = find_shape_nodes(model, constant_dict, initializer_dict)
    if node_output != []:
        value_info = get_model_value_info(model)
        ort_inputs = {}
        for i in range(len(model.graph.input)):
            input_name = model.graph.input[i].name
            model_input = model.graph.input[i].type.tensor_type.shape.dim
            input_dim = dim_to_list(model_input)
            if len(input_dim) == 4:
                n,c,h,w = input_dim
                ort_inputs[input_name] = np.random.rand(n,c,h,w).astype(np.float32)
            elif len(input_dim) == 2:
                h,w = input_dim
                ort_inputs[input_name] = np.random.rand(h,w).astype(np.float32)
            elif len(input_dim) == 3:
                c,h,w = input_dim
                ort_inputs[input_name] = np.random.rand(c,h,w).astype(np.float32)
            else:
                raise ValueError(f"Not support input dimention:{input_dim}")
        has_no_shape_node_name = []
        # get the length of original model output, apply the value after that
        original_model_output_length = len(model.graph.output)
        for name in node_output:
            model.graph.output.append(value_info[name])
        # save_onnx_model(model,r'data\c100_144k\transformer\new_model11.onnx')
        # x = np.random.rand(n,c,h,w).astype(np.float32)
        # outputs_val= backend.run(model,x,'CPU')
        
        save_onnx_model(model,os.path.join(file_path,'intermediate_convert_model.onnx'))
        ort_session =  ort.InferenceSession(os.path.join(file_path,'intermediate_convert_model.onnx'))
        # ort_inputs = {input_name:np.random.rand(n,c,h,w).astype(np.float32) }
        outputs_val = ort_session.run(None,ort_inputs)
        # print(outputs_val)
        # input()
        initializer_dict_ = initializer_dict
        new_initializer_dict = []
        
        for n in range(len(node_output)):
            new_initializer_dict.append(make_tensor(node_output[n],outputs_val[n+original_model_output_length]))
        for name in initializer_dict_.keys():
            new_initializer_dict.append(initializer_dict_[name])
        
        # remove node_output
        for name in node_output:
            model.graph.output.remove(value_info[name]) 
        npu_compatible_graph = helper.make_graph(
            node_list,
            model.graph.name,
            model.graph.input,
            model.graph.output,
            new_initializer_dict,
            doc_string = None,
            value_info = model.graph.value_info
        )  
        npu_model = helper.make_model(npu_compatible_graph)
        
        # save_onnx_model(npu_model,os.path.join(file_path , r'model1.onnx'))
        if not store_intermediate_model:
            os.remove(os.path.join(file_path,'intermediate_convert_model.onnx')) 
    else:          
        npu_model = model

    if store_intermediate_model:
        # onnx opset version
        npu_model.ir_version = 5
        npu_model.opset_import[0].version = 10
        save_onnx_model(npu_model,os.path.join(file_path,'intermediate_convert_model.onnx'))
         
    #input()
    
    include_initializer_model = add_value_info_for_constants(npu_model)

    npu_model = shape_inference_(include_initializer_model)

    npu_model = MeaninglessOpPass(npu_model, constant_dict, initializer_dict)
    
    updated_name_dict = None    
    if fix_layer_name:
        npu_model, updated_name_dict = fix_node_name(npu_model)
        # specify the onnx opset version
        npu_model.ir_version = 5
        npu_model.opset_import[0].version = 10
        # save_onnx_model(npu_model,os.path.join(file_path , f'model_with_fixed_name.onnx'))
    
    
    if label_file :
        model_config = json.load(open(label_file))
        if model_config['input'] and model_config['output']:
            npu_model = get_npu_compatible_model(npu_model,model_config,output_npu_model_filename)
    
    return npu_model, updated_name_dict

def make_tensor(value_name,value):
    d_ = str(value.dtype).upper()
    if d_ == 'FLOAT32':
        d_ = 'FLOAT'
    dtype= TensorProto.DataType.Value(d_)
    tensor = helper.make_tensor(
        name = value_name,
        data_type= dtype,
        dims= value.shape,
        vals= value.reshape(-1,),
    )
    return tensor

def find_shape_nodes(model,constant_dict,initializer_dict):
    
    new_model_nodes = []
    valid_node_output_name = []
    calc_input_name =[]
    model_input_name = []
    for i in model.graph.input:
        model_input_name.append(i.name)
    
    for i in model.graph.node:
        
        if i.op_type != 'Constant':
            
            if i.input[0] in model_input_name and i.op_type != 'Shape':
                new_model_nodes.append(i)
                if i.op_type == 'Split':
                    for n in i.output:
                        valid_node_output_name.append(n)
                else:
                    valid_node_output_name.append(i.output[0])
                if i.op_type == 'Reshape' and i.input[1] not in initializer_dict.keys():
                    calc_input_name.append(i.input[1])
            elif i.op_type == 'Shape':
                continue
            else:
                valid = False
                for node_input in i.input:
                    if node_input in valid_node_output_name:
                        valid = True
                        continue
                    elif node_input in constant_dict.keys() or node_input in initializer_dict.keys():
                        continue
                if valid:
                    for node_input in i.input:
                        
                        if node_input in constant_dict.keys():
                            new_model_nodes.append(constant_dict[node_input])
                            continue
                        elif node_input in initializer_dict.keys() or node_input in valid_node_output_name:
                            continue
                        else:
                            if i.op_type == 'Clip' and node_input == '':
                                continue
                            if node_input == '':
                                continue
                            calc_input_name.append(node_input)
                    for node_output in i.output:
                        # valid_node_output_name.append(i.output[0])
                        valid_node_output_name.append(node_output)
                    new_model_nodes.append(i)
    
    return new_model_nodes, calc_input_name        

def createInputNode(model):
    InputNode = {}
    for node in model.graph.node:
        if node.op_type != 'Constant':    
            InputNode[node.input[0]] = node
    return InputNode

def createOutputNode(model):
    OutputNode = {}
    for node in model.graph.node:
        if node.op_type != 'Constant':
            OutputNode[node.output[0]] = node
    return OutputNode

def createValueinfo(model):
    valueinfo = {}
    for v in model.graph.value_info:
        valueinfo[v.name] = v
    return valueinfo

def MeaninglessOpPass(model, constant_dict, initializer_dict):
    inputnode = createInputNode(model)
    outputnode = createOutputNode(model)
    valueinfo = createValueinfo(model)
    index = 0
    
    for node in model.graph.node:
        
        if node.op_type == 'Pad':
            
            if len(node.input) == 2:
                # 
                assert node.input[1] in constant_dict.keys()
                c = constant_dict[node.input[1]]
                for attr in c.attribute:
                    # constant_dict[c.output[0]] = numpy_helper.to_array(attr.t)
                    pad_value = numpy_helper.to_array(attr.t)
            else:
                pad_value = get_node_pads(node)
            
            if np.mean(pad_value) == 0 and np.std(pad_value) == 0:
                input_name = node.input[0]
                output_name = node.output[0]
                outputNode = inputnode[output_name]
                inputs = []
                inputs.append(input_name)
                for i in range(1, len(outputNode.input)):
                    inputs.append(outputNode.input[i])
                
                attr = get_model_addr(outputNode.attribute)
                
                outputNode_new =  make_node(outputNode.op_type,
                                            inputs,
                                            outputNode.output,
                                            outputNode.name,
                                            **attr)
                model.graph.node.remove(node)
                if len(node.input) == 2:
                    assert node.input[1] in constant_dict.keys()
                    c = constant_dict[node.input[1]]
                    model.graph.node.remove(c)
                #    print(outputNode)
                model.graph.node.remove(outputNode)
                model.graph.node.insert(index,outputNode_new)
        
        elif node.op_type == 'Add':
            # When performing an add operation, if the second operand is all 0, remove the node
            add_value = None
            if node.input[1] in constant_dict.keys():
                # the second input is a constant node
                c = constant_dict[node.input[1]]
                for attr in c.attribute:
                    add_value = numpy_helper.to_array(attr.t)
            elif node.input[1] in initializer_dict.keys():
                add_value = numpy_helper.to_array(initializer_dict[node.input[1]])
            else:
                continue
            
            # remove add node
            if np.mean(add_value) == 0 and np.std(add_value) == 0:
                input_name = node.input[0]
                output_name = node.output[0]
                outputNode = inputnode[output_name]
                # Replace the first input of outputNode with the original input, and keep the rest of the input unchanged
                inputs = []
                inputs.append(input_name)
                for i in range(1, len(outputNode.input)):
                    inputs.append(outputNode.input[i])
                
                attr = get_model_addr(outputNode.attribute)
                outputNode_new =  make_node(outputNode.op_type,
                                            inputs,
                                            outputNode.output,
                                            outputNode.name,
                                            **attr)
                model.graph.node.remove(node)
                if node.input[1] in constant_dict.keys():
                    c = constant_dict[node.input[1]]
                    model.graph.node.remove(c)
                model.graph.node.remove(outputNode)
                model.graph.node.insert(index,outputNode_new)
            
        index += 1
    return model

def get_model_addr(model_attr):
    attr = {}
    for a in model_attr:
        # if i.name == 'ceil_mode':
        #     attr[i.name] = i.i
        # else:
        #     attr[i.name] = i.ints
        if a.name in ['group','ceil_mode','axis']:
            attr[a.name] = a.i
        elif a.type == 7:
            attr[a.name] = a.ints
        elif a.type == 4:
            attr[a.name] = a.t
        elif a.type == 1:
            attr[a.name] = a.f
        elif a.type == 3:
            attr[a.name] = a.s
        else:
            raise ValueError(f'Not support attribute type: {a} !!!')
        
    return attr

def make_node(op_type, inputs, outputs, name=None, domain=None, doc_string=None, **kwargs):
    if name is None:
        name = op_type
    
    node = helper.make_node(
        op_type,
        [i for i in inputs if i is not None],
        [o for o in outputs if o is not None],
        name,
        doc_string,
        domain,
        **kwargs
    )
    
    return node