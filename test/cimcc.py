import argparse
# 
from OpenCIMTC.utils import gen_trainable_codes, gen_inference_codes, gen_ir, modify_ir_with_pdt_weight
    
if __name__ == "__main__":
    
    # 
    p = argparse.ArgumentParser(description='training, inferencing and optimizing with Open CIM Toolchain')
    p.add_argument('-i', '--ir_path', default=None, help='path of the onnx model or ir file')
    p.add_argument('-o', '--onnx_model', default=None, help='path of the onnx model or ir file')
    p.add_argument('-w', '--pdt_trained_weights', default=None, help='the trained weights and parameters with PDT method')
    p.add_argument('--modify_ir', action='store_true', help='modify the ir file with trained weights and parameters')
    p.add_argument('--save_onnx_weight', action='store_true', default=True, help='save the onnx weight to torch format')
    p.add_argument('--to_ir', action='store_true', help='convert onnx model to ir file')
    p.add_argument('--to_train', action='store_true', help='convert ir file to trainable codes')
    p.add_argument('--to_sim', action='store_true', help='convert ir file to simulation codes')
    p.add_argument('--to_optim', action='store_true', help='convert ir file to chip inference codes')
    args = p.parse_args()
    
    # 
    ir_path = args.ir_path
    if args.modify_ir:
        assert ir_path is not None, 'Please provide the ir file path.'
        assert args.pdt_trained_weights is not None, 'Please provide the trained weights and parameters with PDT method.'
        modified_ir_path = modify_ir_with_pdt_weight(ir_path, args.pdt_trained_weights)
        ir_path = modified_ir_path
    #     
    if args.to_ir:
        assert args.onnx_model is not None, 'Please provide the onnx model path.'
        gen_ir(args.onnx_model, save_onnx_weight_to_torch=args.save_onnx_weight)
    elif args.to_train:
        gen_trainable_codes(ir_path)
    elif args.to_sim:
        gen_inference_codes(ir_path, to_simulator=True)
    elif args.to_optim:
        gen_inference_codes(ir_path, to_optimization=True)
    elif not args.modify_ir:
        raise ValueError('Invalid option. Please choose one of the options: [to ir,to_train, to_sim, to_optim].')