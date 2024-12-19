import os
from ..compiler import * 
from pathlib import Path

def gen_trainable_codes(mapped_ir_path):
    '''
    code generation with spliting information and customized trainable module
    '''
    self_define_layer = {'conv2d': 'Conv2dPDT', 'matmul':'LinearPDT', 'conv_transpose2d':'ConvTranspose2dPDT'}
    ir_path = os.path.dirname(mapped_ir_path)
    ir_path = str(Path(ir_path).parent)
    ir_name = os.path.basename(mapped_ir_path).split('.')[0]
    if not os.path.exists(ir_path + '/scripts'):
        os.makedirs(ir_path + '/scripts')
    scripts_path = ir_path + f'/scripts/'
    mapped_ir = irtool.core.load_ir(file=mapped_ir_path)
    code = backend.TrainingCodeGen(mapped_ir, module_name=f'{ir_name}_PDT', self_define_layer=self_define_layer)
    code.to_code(generator=code.gen_layers(), file=scripts_path + f'{ir_name}_PDT.py')
# 
def gen_inference_codes(mapped_ir_path, to_simulator=False, to_optimization= False, to_chip=False):
    '''
    code generation for inference
    '''
    ir_path = os.path.dirname(mapped_ir_path)
    scripts_path = str(Path(ir_path).parent) + f'/scripts/'
    ir_name = os.path.basename(mapped_ir_path).split('.')[0]
    # 
    mapped_ir = irtool.core.load_ir(file=mapped_ir_path)
    if to_simulator:
        code = backend.InferenceCodeGen(mapped_ir, module_name=f'{ir_name}_SIM', simulation=True)
        code.to_code(generator=code.gen_layers(), file=scripts_path + f'{ir_name}_SIM.py')
    elif to_optimization:
        code = backend.InferenceCodeGen(mapped_ir, module_name=f'{ir_name}_SIM_OPT', simulation=True, is4search=True)
        code.to_code(generator=code.gen_layers(), file=scripts_path + f'{ir_name}_SIM_OPT.py')
    elif to_chip:
        code = backend.InferenceCodeGen(mapped_ir, module_name=f'{ir_name}_CHIP')
        code.to_code(generator=code.gen_layers(), file=scripts_path + f'{ir_name}_CHIP.py')
    
    