
import time
import numpy as np
import torch
import argparse
import importlib.util

def convert_module_to_object(path, obj_name):
    # Specify the absolute path of the module
    module_path = path

    # Define a name for the module (optional)
    module_name = 'module'

    # Create a spec for the module
    spec = importlib.util.spec_from_file_location(module_name, module_path)

    # Load the module using the spec
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # get the object in module
    obj = getattr(module, obj_name)
    return obj

if __name__ == "__main__":

    p = argparse.ArgumentParser(description='training, inferencing and optimizing with Open CIM Toolchain')
    p.add_argument('-id', '--input_data', default=None, help='path of the input data')
    p.add_argument('-il', '--input_label', default=None, help='path of the input label')
    p.add_argument('-w', '--pdt_trained_weights', default=None, help='the trained weights and parameters with PDT method')
    p.add_argument('-n', '--module_name', help='the name of the generated module')
    p.add_argument('-m', '--module_path', help='the path of the generated module')
    p.add_argument('-bs', '--batch_size', default=1000, help='the batch size of test data')
    p.add_argument('-bn', '--batch_num', default=1, help='the batch number of test data')
    args = p.parse_args()
    
    # load input
    feature_map_in = torch.load(args.input_data)
    # label
    d_label = torch.load(args.input_label)
    #
    label = np.array(d_label)
    # torch params
    weights_torch = torch.load(args.pdt_trained_weights)
    
    # load module
    func = convert_module_to_object(args.module_path, args.module_name)
    
    # cuda
    device = 'cpu'
    if torch.cuda.is_available():
        print(f'Using CUDA infernce === >')
        feature_map_in_cuda = feature_map_in.to('cuda') 
        # cuda params
        weights_cuda = {}
        for k,v in weights_torch.items():
            if torch.is_tensor(v):
                weights_cuda[k] = v.to('cuda')
        device = 'cuda'
        weights = weights_cuda
    else:
        print(f'Using CPU infernce === >')
        weights = weights_torch 
    # 
    torch.cuda.init()
    batch_size = args.batch_size
    batch_num = args.batch_num
    # 
    time1 = time.time()
    predict = []
    for i in range(batch_num):
        output_torch_cuda = func(feature_map_in_cuda[i*batch_size:(i+1)*batch_size,:,:,:], weights, device=device)
        output_torch_cuda = torch.squeeze(output_torch_cuda[0])
        re = torch.argmax(output_torch_cuda, axis=1).cpu().detach().numpy()
        # 
        predict.append(re)
    predict = np.concatenate(predict, axis=0)
    
    # acc
    sum_ = np.sum(predict == label[0:batch_num * batch_size])
    acc_ = sum_ / (batch_num * batch_size)
    
    time2 = time.time()
    print(f'CUDA simulation {batch_num * batch_size} images time: {time2 - time1} s')
    print(f'Current Accuracy: {acc_} !!!')
        