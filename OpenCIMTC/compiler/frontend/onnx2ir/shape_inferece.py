from .helper import *

class ShapeInference(object):

    def __init__(self, node, value_info, initializer_dict, constant_dict, constant_output_node_dict):
        self.node = node
        self.value_info = value_info
        self.initializer_dict = initializer_dict
        self.constant_dict = constant_dict
        self.constant_output_node_dict = constant_output_node_dict

    def Abs(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1
    
    def Acos(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Acosh(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Add(self):
        '''
        support broadcast add 
        '''
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        input_shape2 = dim_to_list(self.value_info[self.node.input[1]].type.tensor_type.shape.dim)
        x = np.ones(input_shape1)
        y = np.ones(input_shape2)
        z = x + y
        node_output_shape = z.shape
        return node_output_shape
    
    def And(self):
        '''
        logical and element-wise
        '''
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        input_shape2 = dim_to_list(self.value_info[self.node.input[1]].type.tensor_type.shape.dim)
        x = np.ones(input_shape1)
        y = np.ones(input_shape2)
        z = np.logical_and(x,y)
        node_output_shape = z.shape
        return node_output_shape

    def ArgMax(self):      
        '''
        do not test
        '''  
        select_last_index = get_select_last_index(self.node)
        axis = get_axis(self.node)
        keepdims = get_keepdims(self.node)
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        x = np.ones(input_shape1)
        if select_last_index == 0:
            y = np.argmax(x,axis=axis)
            if keepdims == 1:
                y = np.expand_dims(y,axis=axis)
        else:
            x = np.flip(x,axis=axis)
            y = np.argmax(x,axis=axis)
            y = x[axis] - y - 1
            if keepdims == 1:
                y = np.expand_dims(y,axis=axis)
        node_output_shape = y.shape   
        return node_output_shape

    def ArgMin(self):        
        '''
        do not test
        '''  
        select_last_index = get_select_last_index(self.node)
        axis = get_axis(self.node)
        keepdims = get_keepdims(self.node)
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        x = np.ones(input_shape1)
        if select_last_index == 0:
            y = np.argmin(x,axis=axis)
            if keepdims == 1:
                y = np.expand_dims(y,axis=axis)
        else:
            x = np.flip(x,axis=axis)
            y = np.argmin(x,axis=axis)
            y = x[axis] - y - 1
            if keepdims == 1:
                y = np.expand_dims(y,axis=axis)
        node_output_shape = y.shape
        return node_output_shape
    
    def Asin(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1
    
    def Asinh(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Atan(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Atanh(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def AveragePool(self):
        node_input_shape = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        node_output_shape = []
        node_output_shape.append(node_input_shape[0])
        node_output_shape.append(node_input_shape[1])
        node_output_shape.append(int(node_input_shape[2]/2))
        node_output_shape.append(int(node_input_shape[3]/2))
        return node_output_shape

    def BatchNormalization(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1
    
    def BitShift(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Cast(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Ceil(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Clip(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Compress(self):
        '''
        TODO
        '''
        pass

    def Concat(self):
        axis = get_axis(self.node)
        array = []
        for i in range(len(self.node.input)):
            input_shape1 = dim_to_list(self.value_info[self.node.input[i]].type.tensor_type.shape.dim)
            x = np.ones(input_shape1)
            array.append(x)
        y = np.concatenate(array,axis=axis)
        node_output_shape = y.shape
        return node_output_shape

    def ConcatFromSequence(self):
        '''
        do not test
        '''
        axis = get_axis(self.node)
        new_axis = get_new_axis(self.node)
        if new_axis == 0:
            for i in range(len(self.node.input)):
                if i == 0:
                    input_shape1 = dim_to_list(self.value_info[self.node.input[i]].type.tensor_type.shape.dim)
                    x = np.ones(input_shape1)
                else:
                    input_shape = dim_to_list(self.value_info[self.node.input[i]].type.tensor_type.shape.dim)
                    y = np.ones(input_shape)
                    x = np.concatenate((x,y),axis=axis)
        else:
            array = []
            for i in range(len(self.node.input)):
                input_shape1 = dim_to_list(self.value_info[self.node.input[i]].type.tensor_type.shape.dim)
                y = np.ones(input_shape1)
                array.append(y)
            x = np.stack(array,axis=new_axis)
        node_output_shape = x.shape
        return node_output_shape

    def Constant(self):
        '''
        TODO 
        '''
        pass

    def ConstantOfShape(self):
        '''
        TODO 
        '''
        pass

    def Conv(self):
        node_weight_shape = dim_to_list(self.value_info[self.node.input[1]].type.tensor_type.shape.dim)
        stride,pad,kernel = get_conv_node_attr(self.node.attribute)
        node_input_shape = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        output_shape_h = int((node_input_shape[2]-node_weight_shape[2]+2*pad[0])/stride[0] + 1) 
        output_shape_w = int((node_input_shape[3]-node_weight_shape[3]+2*pad[0])/stride[0] + 1) 
        node_output_shape = []
        node_output_shape.append(node_input_shape[0])
        node_output_shape.append(node_weight_shape[0])
        node_output_shape.append(output_shape_h)
        node_output_shape.append(output_shape_w)
        return node_output_shape

    def ConvInteger(self):
        '''
        TODO 
        '''
        pass

    def ConvTranspose(self):
        node_weight_shape = dim_to_list(self.value_info[self.node.input[1]].type.tensor_type.shape.dim)
        stride,pad,kernel = get_conv_node_attr(self.node.attribute)
        node_input_shape = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        new_input_shape_h = node_input_shape[2] + (node_input_shape[2] -1)*(stride[0]-1)
        new_input_shape_w = node_input_shape[3] + (node_input_shape[3] -1)*(stride[0]-1)
        new_stride = 1
        new_pad = kernel[0] - pad[0] - 1
        output_shape_h = int((new_input_shape_h-node_weight_shape[2]+2*new_pad)/new_stride + 1) 
        output_shape_w = int((new_input_shape_w-node_weight_shape[3]+2*new_pad)/new_stride + 1) 
        node_output_shape =[]
        node_output_shape.append(node_input_shape[0])
        node_output_shape.append(node_weight_shape[1])
        node_output_shape.append(output_shape_h)
        node_output_shape.append(output_shape_w)
        return node_output_shape

    def Cos(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Cosh(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def CumSum(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def DepthToSpace(self):
        '''
        TODO 
        '''
        pass
    
    def DequantizaLinear(self):
        '''
        TODO 
        '''
        pass    

    def Det(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        x = np.ones(input_shape1)
        y = np.linalg.det(x)
        node_output_shape = y.shape
        return node_output_shape

    def Div(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Dropout(self):
        '''
        TODO 
        '''
        pass

    def Einsum(self):
        '''
        TODO 
        '''
        pass

    def Elu(self):
        alpha = get_alpha(self.node)
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        x = np.ones(input_shape1)
        y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * alpha
        node_output_shape = y.shape
        return node_output_shape

    def Equal(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        input_shape2 = dim_to_list(self.value_info[self.node.input[1]].type.tensor_type.shape.dim)
        x = np.ones(input_shape1)
        y = np.ones(input_shape2)
        z = np.equal(x,y)
        node_output_shape = z.shape
        return node_output_shape

    def Erf(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Exp(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Expand(self):
        axes = get_axes(self.node)
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        x = np.ones(input_shape1)
        y = np.expand_dims(x, axis=axes)
        node_output_shape = y.shape
        return node_output_shape

    def Eyelike(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Flatten(self):
        axis = get_axis(self.node)
        node_output_shape = []
        node_input_shape = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        if axis == 0 :
            node_output_shape.append(1)
            t = 1
            for k in node_input_shape[0:]:
                t = k * t
            node_output_shape.append(t)
        elif axis >= 1:
            node_output_shape = [int(np.prod(node_input_shape[0:axis])), int(np.prod(node_input_shape[axis:]))]
        elif axis <= -1:
            node_output_shape = [int(np.prod(node_input_shape[0:len(node_input_shape)+axis])), int(np.prod(node_input_shape[len(node_input_shape)+axis:]))]
        return node_output_shape
    
    def Floor(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def GRU(self):
        '''
        TODO 
        '''
        pass

    def Gather(self):
        axis = get_axis(self.node)
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        indices = self.constant_output_node_dict[self.node.input[1]]
        x = np.ones(input_shape1)
        z = np.take(x,indices=indices,axis=axis)
        node_output_shape = z.shape
        return node_output_shape

    def GatherElements(self):
        '''
        TODO 
        '''
        pass

    def GatherND(self):
        '''
        TODO 
        '''
        pass

    def Gemm(self):
        node_weight_shape = dim_to_list(self.value_info[self.node.input[1]].type.tensor_type.shape.dim)
        node_input_shape =  dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        node_output_shape = []
        node_output_shape.append(node_input_shape[0])
        node_output_shape.append(node_weight_shape[0])
        return node_output_shape

    def GlobalAveragePool(self):
        '''
        can be modified
        '''
        node_input_shape = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        node_output_shape = []
        node_output_shape.append(node_input_shape[0])
        node_output_shape.append(node_input_shape[1])
        node_output_shape.append(1)
        node_output_shape.append(1)
        return node_output_shape

    def GlobalLpPool(self):
        '''
        TODO 
        '''
        pass

    def GlobalMaxPool(self):
        '''
        can be modified
        '''
        node_input_shape = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        node_output_shape = []
        node_output_shape.append(node_input_shape[0])
        node_output_shape.append(node_input_shape[1])
        node_output_shape.append(1)
        node_output_shape.append(1)
        return node_output_shape

    def Greater(self):
        '''
        TODO 
        '''
        pass

    def HardSigmoid(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Hardmax(self):
        '''
        Realizing with the help of onnx test hardmax api.
        '''
        axis = get_axis(self.node)
        input_shape = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        x = np.random.random(input_shape)
        y = hardmax(x,axis=axis)
        node_output_shape = y.shape
        return node_output_shape

    def Identity(self):
        '''
        Support multi inputs and multi outputs;and outputs are the same as inputs.
        '''
        input_len = len(self.node.input)
        if input_len == 1:
            input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
            return input_shape1
        else:
            node_output_shape = []
            for i in range(input_len):
                input_shape1 = dim_to_list(self.value_info[self.node.input[i]].type.tensor_type.shape.dim)
                node_output_shape.append(i)
            return node_output_shape

    def If(self):
        '''
        TODO 
        '''
        pass

    def InstanceNormalization(self):
        '''
        TODO 
        '''
        pass

    def Isinf(self):
        '''
        TODO 
        '''
        pass

    def IsNaN(self):
        '''
        TODO 
        '''
        pass 

    def LRN(self):
        '''
        TODO 
        '''
        pass

    def LSTM(self):
        '''
        TODO 
        '''
        pass

    def LeakyRelu(self):
        '''
        TODO 
        '''
        pass

    def Less(self):
        '''
        TODO 
        '''
        pass

    def Log(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Loop(self):
        '''
        TODO 
        '''
        pass
    
    def LpNormalization(self):
        '''
        TODO 
        '''
        pass

    def LpPool(self):
        '''
        TODO 
        '''
        pass

    def MatMul(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        input_shape2 = dim_to_list(self.value_info[self.node.input[1]].type.tensor_type.shape.dim)
        x = np.ones(input_shape1)
        y = np.ones(input_shape2)
        z = np.matmul(x,y)
        node_output_shape = z.shape 
        return node_output_shape

    def MatMulInteger(self):
        '''
        Output tensor shape is same as MatMul op. But the behavior is different. 
        '''
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        input_shape2 = dim_to_list(self.value_info[self.node.input[1]].type.tensor_type.shape.dim)
        x = np.ones(input_shape1)
        y = np.ones(input_shape2)
        z = np.matmul(x,y)
        node_output_shape = z.shape 
        return node_output_shape

    def Max(self):
        input_list = []
        for i in self.node.input:
            node_input_shape = dim_to_list(self.value_info[i].type.tensor_type.shape.dim)
            input_ = np.random.random(node_input_shape)
            input_list.append(input_)
        
        if input_list != []:
            inter_results = 0
            for j in range(len(input_list)):
                if j == 0:
                    inter_results = input_list[j]
                else:
                    inter_results = np.maximum(inter_results,input_list[j])
            node_output_shape = inter_results.shape
            return node_output_shape
        else:
            raise ValueError('Input tensors do not be loaded completely!')
    
    def MaxPool(self):
        '''
        can be modified --> pool kernel
        '''
        node_input_shape = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        node_output_shape = []
        node_output_shape.append(node_input_shape[0])
        node_output_shape.append(node_input_shape[1])
        node_output_shape.append(int(node_input_shape[2]/2))
        node_output_shape.append(int(node_input_shape[3]/2))
        return node_output_shape

    def MaxRoiPool(self):
        '''
        TODO 
        '''
        pass

    def MaxUnpool(self):
        '''
        TODO 
        '''
        pass

    def Mean(self):
        input_list = []
        for i in self.node.input:
            node_input_shape = dim_to_list(self.value_info[i].type.tensor_type.shape.dim)
            input_ = np.random.random(node_input_shape)
            input_list.append(input_)
        
        if input_list != []:
            inter_results = 0
            for j in range(len(input_list)):
                if j == 0:
                    inter_results = input_list[j]
                else:
                    inter_results = np.add(inter_results,input_list[j])
            results = np.divide(inter_results ,len(input_list))
            node_output_shape = results.shape
            return node_output_shape
        else:
            raise ValueError('Input tensors do not be loaded completely!')

    def Min(self):
        input_list = []
        for i in self.node.input:
            node_input_shape = dim_to_list(self.value_info[i].type.tensor_type.shape.dim)
            input_ = np.random.random(node_input_shape)
            input_list.append(input_)
        
        if input_list != []:
            inter_results = 0
            for j in range(len(input_list)):
                if j == 0:
                    inter_results = input_list[j]
                else:
                    inter_results = np.minimum(inter_results,input_list[j])
            node_output_shape = inter_results.shape
            return node_output_shape
        else:
            raise ValueError('Input tensors do not be loaded completely!')

    def Mod(self):
        fmod = get_fmod(self.node)
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        input_shape2 = dim_to_list(self.value_info[self.node.input[1]].type.tensor_type.shape.dim)
        x = np.random.random(input_shape1)
        y = np.ones(input_shape2)
        if fmod == 1:
            z = np.fmod(x,y)
        else:
            z = np.mod(x,y)
        node_output_shape = z.shape
        return node_output_shape

    def Mul(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        input_shape2 = dim_to_list(self.value_info[self.node.input[1]].type.tensor_type.shape.dim)
        x = np.ones(input_shape1)
        y = np.ones(input_shape2)
        z = x * y
        node_output_shape = z.shape
        return node_output_shape

    def Multinomial(self):
        '''
        TODO 
        '''
        pass

    def Neg(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def NonMaxSuppression(self):
        '''
        TODO 
        '''
        pass

    def Nonzero(self):
        '''
        TODO 
        '''
        pass

    def Not(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def OneHot(self):
        '''
        TODO 
        '''
        pass

    def Optional(self):
        '''
        TODO 
        '''
        pass

    def OptionalGetElement(self):
        '''
        TODO 
        '''
        pass
    
    def OptionalHasElement(self):
        '''
        TODO 
        '''
        pass

    def Or(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        input_shape2 = dim_to_list(self.value_info[self.node.input[1]].type.tensor_type.shape.dim)
        x = np.ones(input_shape1)
        y = np.ones(input_shape2)
        z = np.logical_or(x,y)
        node_output_shape = z.shape
        return node_output_shape

    def PRelu(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Pad(self):
        '''
        TODO 
        '''
        pass

    def Pow(self):
        '''
        can be modified
        '''
        if self.node.input[0] not in self.constant_dict.keys():
            input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        elif self.node.input[1] not in self.constant_dict.keys():
            input_shape1 = dim_to_list(self.value_info[self.node.input[1]].type.tensor_type.shape.dim)
        else:
            input_shape1 = None
        return input_shape1
    
    def QlinearConv(self):
        '''
        TODO 
        '''
        pass

    def QlinearMatMul(self):
        '''
        TODO 
        '''
        pass

    def QuantizaLinear(self):
        '''
        TODO 
        '''
        pass

    def RNN(self):
        '''
        TODO 
        '''
        pass

    def RandomNormal(self):
        '''
        TODO 
        '''
        pass

    def RandomNormalLike(self):
        '''
        TODO 
        '''
        pass

    def RandomUniform(self):
        '''
        TODO 
        '''
        pass

    def RandomUniformLike(self):
        '''
        TODO 
        '''
        pass

    def Reciprocal(self):
        '''
        TODO 
        '''
        pass

    def ReduceL1(self):
        '''
        this op can not be calculated the output shape without the input data
        so can not inference output shape 
        except when keepdims == 1 , axes = None
        '''
        keepdims = get_keepdims(self.node)
        axes = get_axes(self.node)
        if keepdims == 1 and axes == []:
            input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
            node_output_shape = [1]*len(input_shape1)
            return node_output_shape  

    def ReduceL2(self):
        '''
        this op can not be calculated the output shape without the input data
        so can not inference output shape
        except when keepdims == 1 , axes = None
        '''
        keepdims = get_keepdims(self.node)
        axes = get_axes(self.node)
        if keepdims == 1 and axes == []:
            input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
            node_output_shape = [1]*len(input_shape1)
            return node_output_shape 

    def ReduceLogSum(self):
        '''
        this op can not be calculated the output shape without the input data
        so can not inference output shape
        except when keepdims == 1 ,  axes = None
        '''
        keepdims = get_keepdims(self.node)
        axes = get_axes(self.node)
        if keepdims == 1 and axes == []:
            input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
            node_output_shape = [1]*len(input_shape1)
            return node_output_shape
    
    def ReduceLogSumExp(self):
        '''
        this op can not be calculated the output shape without the input data
        so can not inference output shape
        except when keepdims == 1, axes == []
        '''
        keepdims = get_keepdims(self.node)
        axes = get_axes(self.node)
        if keepdims == 1 and axes == []:
            input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
            node_output_shape = [1]*len(input_shape1)
            return node_output_shape

    def ReduceMax(self):
        '''
        this op can not be calculated the output shape without the input data
        so can not inference output shape
        except when keepdims == 1,axes = []
        '''
        keepdims = get_keepdims(self.node)
        axes = get_axes(self.node)
        if keepdims == 1 and axes == []:
            input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
            node_output_shape = [1]*len(input_shape1)
            return node_output_shape

    def ReduceMean(self):
        '''
        this op can not be calculated the output shape without the input data
        so can not inference output shape
        except when keepdims == 1,axes == []
        '''
        keepdims = get_keepdims(self.node)
        axes = get_axes(self.node)
        if keepdims == 1 and axes == []:
            input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
            node_output_shape = [1]*len(input_shape1)
            return node_output_shape

    def ReduceMin(self):
        '''
        this op can not be calculated the output shape without the input data
        so can not inference output shape
        except when keepdims == 1, axes == []
        '''
        keepdims = get_keepdims(self.node)
        axes = get_axes(self.node)
        if keepdims == 1 and axes == []:
            input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
            node_output_shape = [1]*len(input_shape1)
            return node_output_shape
    
    def ReduceProd(self):
        '''
        this op can not be calculated the output shape without the input data
        so can not inference output shape
        except when keepdims == 1 , axes == []
        '''
        keepdims = get_keepdims(self.node)
        axes = get_axes(self.node)
        if keepdims == 1 and axes == []:
            input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
            node_output_shape = [1]*len(input_shape1)
            return node_output_shape

    def ReduceSum(self):
        '''
        this op can not be calculated the output shape without the input data
        so can not inference output shape
        except when keepdims == 1 , axes == []
        '''
        keepdims = get_keepdims(self.node)
        axes = get_axes(self.node)
        if keepdims == 1 and axes == []:
            input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
            node_output_shape = [1]*len(input_shape1)
            return node_output_shape

    def ReduceSumSquare(self):
        '''
        TODO 
        '''
        pass

    def Relu(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Reshape(self):
        allowzero = get_allowzero(self.node)
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        input_shape2 = self.initializer_dict[self.node.input[1]].int64_data
        x = np.ones(input_shape1)
        y = np.copy(input_shape2)
        if allowzero == 0:
            zeros_index = np.where(input_shape2 == 0)
            y[zeros_index] = np.array(x.shape)[zeros_index]
        z = np.reshape(x, y)
        
        node_output_shape = z.shape
        return node_output_shape
        # '''
        # TODO 
        # '''
        # pass
        
    def Resize(self):
        '''
        TODO 
        '''
        pass

    def ReverseSequence(self):
        '''
        TODO 
        '''
        pass

    def RoiAlign(self):
        '''
        TODO 
        '''
        pass

    def Round(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Scan(self):
        '''
        TODO 
        '''
        pass

    def ScatterElements(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def ScatterND(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Selu(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def SequenceAt(self):
        '''
        TODO 
        '''
        pass

    def SequenceConstruct(self):
        '''
        TODO 
        '''
        pass

    def SequenceEmpty(self):
        '''
        TODO 
        '''
        pass

    def SequenceErase(self):
        '''
        TODO 
        '''
        pass

    def SequenceInsert(self):
        '''
        TODO 
        '''
        pass

    def SequenceLength(self):
        '''
        TODO 
        '''
        pass

    def Shape(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        x = np.ones(input_shape1)
        node_output_shape = x.shape
        return node_output_shape

    def Shrink(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Sigmoid(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Sign(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Sin(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Sinh(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Size(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        x = np.ones(input_shape1)
        y = x.size
        node_output_shape = y.shape
        return node_output_shape

    def Slice(self):

        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        starts = self.constant_dict[self.node.input[1]]
        ends = self.constant_dict[self.node.input[2]]
        axes = self.constant_dict[self.node.input[3]]
        steps = self.constant_dict[self.node.input[4]]
        axes_list = axes.tolist()
        for i in range(len(axes_list)):
            input_shape_axes_i = input_shape1[axes_list[i]] 
            if starts[i] < 0 :
                starts[i] = input_shape_axes_i + starts[i]
            if ends[i] < 0 :
                ends[i] = input_shape_axes_i + ends[i]
            n = 0
            for j in range(starts[i],ends[i],steps[i]):
                n = n + 1
            input_shape1[axes_list[i]] = n
        return input_shape1
        
    def Softplus(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Softsign(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def SpaceToDepth(self):
        '''
        TODO 
        '''
        pass

    def Split(self):
        '''
        return output shape list because it has more than one output Probabilistically
        '''
        axis = get_axis(self.node)
        split = get_split(self.node)
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        output_shape = []
        if axis >= 0:
            for i in range(len(split)):
                node_output_shape = input_shape1
                node_output_shape[axis] = split[i]
                output_shape.append(node_output_shape.copy())  
        else:
            axis_ = len(input_shape1) + axis
            for i in range(len(split)):
                node_output_shape = input_shape1
                node_output_shape[axis_] = split[i]
                output_shape.append(node_output_shape.copy())
        return output_shape

    def SplitToSequence(self):
        '''
        TODO 
        '''
        pass

    def Sqrt(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Squeeze(self):
        axes = get_axes(self.node)
        axes = sort_index(axes)
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        x = np.ones(input_shape1)
        for i in range(len(axes)):
            if axes[i] >= 0:
                if i == 0:
                    y = np.squeeze(x,axis=axes[i])
                else:
                    y = np.squeeze(y,axis=axes[i]-1)
            else:
                if i == 0:
                    y = np.squeeze(x,axis=len(input_shape1) + axes[i])
                else:
                    y = np.squeeze(y,axis=len(input_shape1) + axes[i]-1)
        node_output_shape = y.shape
        return node_output_shape

    def StringNormalizer(self):
        '''
        TODO 
        '''
        pass

    def Sub(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Tan(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Tanh(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def TfldfVectorizer(self):
        '''
        TODO 
        '''
        pass

    def ThresholdedRelu(self):
        '''
        TODO 
        '''
        pass

    def Tile(self):
        '''
        TODO 
        '''
        pass

    def TopK(self):
        '''
        TODO 
        '''
        pass

    def Transpose(self):
        perm = get_perm(self.node)
        node_input_shape = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        node_output_shape = []
        for j in perm:
            node_output_shape.append(node_input_shape[j])
        return node_output_shape

    def Trilu(self):
        '''
        TODO 
        '''
        pass

    def Unique(self):
        '''
        TODO 
        '''
        pass

    def Unsqueeze(self):
        axes = get_axes(self.node)
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        x = np.ones(input_shape1)
        for ax in range(len(axes)):
            if ax == 0:
                y = np.expand_dims(x,axis=axes[ax])
            else:
                y = np.expand_dims(y,axis=axes[ax])
        node_output_shape = y.shape
        return node_output_shape

    def Upsample(self):
        '''
        TODO 
        '''
        pass

    def Where(self):
        '''
        TODO 
        '''
        pass

    def Xor(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def Bernoulli(self):
        '''
        TODO 
        '''
        pass

    def CastLike(self):
        '''
        TODO 
        '''
        pass

    def Celu(self):
        '''
        TODO 
        '''
        pass

    def DynamicQuantizeLinear(self):
        '''
        TODO 
        '''
        pass

    def GreaterOrEqual(self):
        '''
        TODO 
        '''
        pass

    def HardSwish(self):
        '''
        TODO 
        '''
        pass
    
    def LessOrEqual(self):
        '''
        TODO 
        '''
        pass

    def LogSoftmax(self):
        '''
        Realizing with the help of onnx test hardmax api.
        '''
        axis = get_axis(self.node)
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        x = np.random.random(input_shape1)
        y = logsoftmax(x,axis=axis)
        node_output_shape = y.shape
        return node_output_shape

    def MeanVarianceNormalization(self):
        '''
        TODO 
        '''
        pass

    def NegativeLogLikelihoodLoss(self):
        '''
        TODO 
        '''
        pass

    def Range(self):
        start = self.initializer_dict[self.node.input[0]]
        limit = self.initializer_dict[self.node.input[1]]
        delta = self.initializer_dict[self.node.input[2]]
        number_of_elements = max(np.ceil( (limit - start) / delta ), 0 )
        return number_of_elements

    def Softmax(self):
        input_shape1 = dim_to_list(self.value_info[self.node.input[0]].type.tensor_type.shape.dim)
        return input_shape1

    def SoftmaxCrossEntropyloss(self):
        '''
        TODO 
        '''
        pass


