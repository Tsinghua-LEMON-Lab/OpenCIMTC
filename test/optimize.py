import  torch
import argparse
import importlib.util
import os
from pathlib import Path
# 
from OpenCIMTC.optimizer.sim_in_loop import SimInLoopOptimizer
from OpenCIMTC.compiler.irtool.core import load_ir

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
    p.add_argument('-i', '--ir', default=None, help='path of the ir')
    p.add_argument('-id', '--input_data', default=None, help='path of the input data')
    p.add_argument('-il', '--input_label', default=None, help='path of the input label')
    p.add_argument('-w', '--pdt_trained_weights', default=None, help='the trained weights and parameters with PDT method')
    p.add_argument('-n', '--module_name', help='name of the generated module')
    p.add_argument('-m', '--module_path', help='path of the generated module')
    p.add_argument('-lt', '--loss_thr', default=1, help='the threshold of the loss for optimization')
    p.add_argument('-b', '--beta', default=0.5, help='the value of beta')
    p.add_argument('-opc', '--optimized_params_category', default=['it_time', 'weight_copy_num'], help='the category of optimized parameters')
    p.add_argument('-sp', '--size_pop', default=100, help='the size of population')
    p.add_argument('-mi', '--max_iter', default=200, help='the max iteration of ga')
    args = p.parse_args()
    
    # 
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    # training samples    
    ir = load_ir(file=args.ir)
    # load input
    feature_map_in = torch.load(args.id).to(device)
    # torch params
    weights_torch = torch.load(args.pdt_trained_weights)
    # label
    test_label = torch.load(args.input_label).to(device)
    #
    func = convert_module_to_object(args.module_path, args.module_name)
    # 
    if device == 'cuda':
        # cuda params
        weights_cuda = {}
        for k,v in weights_torch.items():
            if torch.is_tensor(v):
                weights_cuda[k] = v.to(device)
        weights_torch = weights_cuda
        
    # optimize
    loss_thr = args.loss_thr
    beta = args.beta
    # optimization parameters category
    optimized_params_category=args.optimized_params_category
    # log directory
    ir_path = os.path.dirname(args.ir)
    ir_name = os.path.basename(args.ir).split('.')[0]
    ir_path = str(Path(ir_path).parent)
    log_dir = ir_path + f'/sil_opt_log/'
    # 
    optimizer = SimInLoopOptimizer(ir, func, feature_map_in, test_label,
                                weights_torch, loss_thr=loss_thr, beta=beta, optimized_params_category=optimized_params_category,
                                log_dir=log_dir, device=device)
    optimizer.run(size_pop=args.size_pop, max_iter=args.max_iter)
    # 
    optimized_ir = optimizer.get_optimized_ir()
    # 
    optimized_ir.dump_json(file = ir_path + f'/{ir_name}_sil_opt.yaml')