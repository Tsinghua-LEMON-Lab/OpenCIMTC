
from ..irtool.core.ir import load_ir,BaseIR
from .helper import *
from ..hw_paras_def.macro import *
import re

class BasePlacement(object):
    
    def __init__(self,ir=None, cpu_layer=None, specify_device_id_list=None, masked_device_id_list=None):
        
        if isinstance(ir, BaseIR):
            self.ir = ir
        elif isinstance(ir, str):
            self.ir = load_ir(ir)
        else:
            raise ValueError(f'lack of a valid ir : {ir} !!! ')
            
        self.node_weight = {}
        self.node_info = {}        
        self.hardware_config = {}
        self.cpu_layer = cpu_layer
        self.specify_device_id_list = specify_device_id_list
        self.masked_device_id_list = masked_device_id_list
        self.parser()
        
    def parser(self):
        '''
        parsing the ir object 
        '''
        for i in self.ir.layers.keys():
            layer = self.ir.layers[i]
            if layer.type == 'op' :
                # print(i)
                if self.cpu_layer != None and i in self.cpu_layer:
                    continue
                op_type = layer.op.op_id
                if op_type in ['conv2d', 'fused_conv2d', 'conv_transpose2d']:
                    self.node_weight[i] = get_conv_shape(layer.op)
                    temp1 = get_conv_info(layer)
                    temp1.update(dict(weight_shape=get_conv_shape(layer.op)))
                    ref = []
                    get_layer_ref(layer.inputs, self.ir.layers, ref)
                    temp1.update(dict(ref=ref))
                    self.node_info[i] = temp1
                    
                elif op_type in ['linear','matmul', 'fc', 'fused_fc']:
                    if len(layer.inputs) == 2:
                        continue
                    self.node_weight[i] = get_linear_shape(layer.op)
                    temp1 = get_linear_info(layer)
                    temp1.update(dict(weight_shape=get_linear_shape(layer.op)))
                    ref = []
                    get_layer_ref(layer.inputs, self.ir.layers, ref)
                    temp1.update(dict(ref=ref))
                    self.node_info[i] = temp1
                
                elif op_type in ['split', 'concat', 'fused_concat']:
                    # self.node_weight[i] = get_linear_shape(layer.op)
                    temp1 = get_split_concat_info(layer)
                    # temp1.update(dict(weight_shape=get_linear_shape(layer.op)))
                    ref = []
                    get_layer_ref(layer.inputs, self.ir.layers, ref)
                    temp1.update(dict(ref=ref))
                    self.node_info[i] = temp1
                
                elif op_type in ['add', 'fused_add']:
                    # self.node_weight[i] = get_linear_shape(layer.op)
                    temp1 = get_add_info(layer)
                    # temp1.update(dict(weight_shape=get_linear_shape(layer.op)))
                    ref = []
                    get_layer_ref(layer.inputs, self.ir.layers, ref)
                    temp1.update(dict(ref=ref))
                    self.node_info[i] = temp1
                
                
        # parsing the hardware config    
        device = DeviceParser(self.ir.devices, self.specify_device_id_list, self.masked_device_id_list)
        self.hardware_config = device.info
    
    
class DeviceParser(object):
    
    def __init__(self, device, specify_device_id_list, masked_device_id_list):
        '''
        device: device object
        '''
        self.device = device
        self.specify_device_id_list = specify_device_id_list
        self.masked_device_id_list = masked_device_id_list
        self.info = {}
        self.parser()
    
    def parser(self):
        
        profile = []
        self.get_device_profile(self.device, profile)

        full_name = []
        full_name = self.get_device_full_name(full_name,None,self.device)
        
        # specify the mapping device id
        specified_device_id_list = []
        if self.specify_device_id_list != None:
            assert isinstance(self.specify_device_id_list, list)
            for tile_id in self.specify_device_id_list:
                if 'rram-144k' in full_name[0]:
                    device_kind = full_name[0].split('.')[1].split(':')[0]
                    specified_device_id_list.append(f'{device_kind}:{tile_id}')
        
        # specify the mask device id
        masked_device_id_list = []
        if self.masked_device_id_list != None:
            for device_id in self.masked_device_id_list:
                if 'rram-144k' in full_name[0]:
                    device_kind = full_name[0].split('.')[1].split(':')[0]
                    masked_device_id_list.append(f'{device_kind}:{tile_id}') 

                else:
                    raise ValueError(f'Not support device: {full_name[0]} !!!')
        
        # Get the number of all devices with xb and 'rram'
        num = 0
        all_xb_name = []
        for name in full_name:
            if 'xb' in name or 'rram' in name or 'dmac' in name:
                if specified_device_id_list != []:
                    if 'rram-144k' in name:
                        tile_name = name.split('.')[1]
                        if tile_name in specified_device_id_list:
                            num += 1 
                            all_xb_name.append(name)
                    else:
                        raise ValueError(f'Not support device: {name} !!!')
                elif masked_device_id_list != []:
                    if 'rram-144k' in name :
                        tile_name = name.split('.')[1]
                        if tile_name in masked_device_id_list:
                            continue
                        else:
                            num += 1 
                            all_xb_name.append(name)
                    else:
                        raise ValueError(f'Not support device: {name} !!!')
                else:
                    num += 1 
                    all_xb_name.append(name)
         
        # Get the number of output ADCs and input DACs of the hardware
        rram_profile = profile[0]
        dac_num = rram_profile.in_channel
        if 'dac_num' in rram_profile.__dict__.keys():
            dac_num = rram_profile.dac_num
            
        adc_num = rram_profile.out_channel    
        if 'adc_num' in rram_profile.__dict__.keys():
            adc_num = rram_profile.adc_num
        
        dac_precision = 4
        if 'dac_precision' in rram_profile.__dict__.keys():
            dac_precision =  rram_profile.dac_precision
            
        self.info = {'name':all_xb_name,
                     'xb_number':num,
                     'xb_shape':[rram_profile.out_channel,rram_profile.in_channel],
                     'adc_num':adc_num,
                     'dac_num':dac_num,
                     'dac_precision':dac_precision
                    }
        
    def get_device_name(self,name,device):
        '''
        input:
            device object
        return:
            all the kind of device
        '''
        # name = []
        t = []
        for i in device.keys():
            t.append(device[i].kind)
            if device[i].devices != None:
                self.get_device_name(name,device[i].devices)
        name.append(t)
        return name
    
    def get_device_profile(self, device, profile):
        '''
        input:
            device: object
            profile: list
        '''
        for i in device.keys():
            
            if 'profile' in device[i].__dict__.keys():
                profile.append(device[i].profile)
            
            if device[i].devices != None:
                self.get_device_profile(device[i].devices, profile)
        

    def get_device_number(self,number,device):
        '''
        input:
            device object
        return:
            all the device number
        '''
        t = []
        for i in device.keys():
            if 'number' in device[i].__dict__.keys():
                t.append(device[i].number)
            else:
                t.append(1)
            if device[i].devices != None:
                self.get_device_number(number,device[i].devices)
        number.append(t)    
        return number
    
    def get_device_full_name(self,name,prefix,device):
        '''
        input:
            name: deive name list
            prefix: The prefix of the current device
            device: IR device obeject
        return:
            the full name of allthe device
        '''
        count = 0 
        for i in device.keys():
            num = 1
            if 'number' in device[i].__dict__.keys():
                num = device[i].number
            
            for j in range(num):
                if prefix != None:
                    suffix = prefix + '.' + device[i].kind + f':{j}'
                else:
                    # The first level of hardware needs to be named
                    suffix = i + '.' + device[i].kind + f':{j}'
                if device[i].devices != None:
                    self.get_device_full_name(name,suffix,device[i].devices)
                else:
                    name.append(suffix)  
        return name