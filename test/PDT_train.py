import argparse
import torch
import torch.nn as nn
import yaml
import importlib.util
# OpenCIMTC 
from OpenCIMTC.customized_hat.train_utils import load_data, init_logger, load_model_checkpoint, save_checkpoint, eval, train, \
                                                 ProgressMonitor, TensorBoardMonitor, Lr_schedule, PerformanceScoreboard
                                                 
from OpenCIMTC.customized_hat.quantization import LSQ_act_quantizer
from OpenCIMTC.customized_hat.pdt import Conv2dPDT, LinearPDT

def convert_module_to_object(path, obj_name, *args, **kwargs):
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
    obj = getattr(module, obj_name)(*args, **kwargs)
    return obj

def main():
    p = argparse.ArgumentParser(description='Post-deployment Training (PDT) For CIM')
    p.add_argument('-C', '--config', help='path of the training configuration file')
    p.add_argument('-M', '--module', help='a trainable pytorch module')
    arg = p.parse_args()
    
    # parsing training configuration
    with open(arg.config, 'r') as f:
        cfg = yaml.safe_load(f)
    # init output dir
    output_dir = cfg['output_dir']
    logger, log_dir = init_logger(cfg['model']['name'], output_dir)
    logger.info(f'Training Config file: {arg.config}')
    logger.info(f'Training Module file: {arg.module}')
    # perparing training data 
    train_loader, val_loader, test_loader = load_data(cfg['dataloader'])
    logger.info('Dataset `%s` size:' % cfg['dataloader']['dataset'] +
                '\n          Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
                '\n        Validation Set = %d (%d)' % (len(val_loader.sampler), len(val_loader)) +
                '\n              Test Set = %d (%d)' % (len(test_loader.sampler), len(test_loader)))
    # perparing model
    if 'quantization' in cfg.keys():
        model = convert_module_to_object(arg.module, cfg['model']['name'], cfg['quantization'])
    else:
        model = convert_module_to_object(arg.module, cfg['model']['name'])
    model.to(cfg['device']['type'])
    
    if cfg['device']['type'] == 'cpu' or not torch.cuda.is_available() or cfg['device']['gpu'] == []:
        cfg['device']['gpu'] = []
    else:
        available_gpu = torch.cuda.device_count()
        for dev_id in cfg['device']['gpu']:
            if dev_id >= available_gpu:
                logger.error('GPU device ID {0} requested, but only {1} devices available'
                             .format(dev_id, available_gpu))
                exit(1)
        # Set default device in case the first one on the list
        torch.cuda.set_device(cfg['device']['gpu'][0])
    
    # training
    pymonitor = ProgressMonitor(logger)
    tbmonitor = TensorBoardMonitor(logger, log_dir)
    monitors = [pymonitor, tbmonitor]
    # 
    criterion = nn.CrossEntropyLoss().to(cfg['device']['type'])
    
    # init the model accuracy
    v_top1_0 = 0.
    v_top5_0 = 0.
    
    logger.info("***model***\n"+str(model))

    if cfg['device']['gpu'] and not cfg['dataloader']['serialized']:
        model = torch.nn.DataParallel(model, device_ids=cfg['device']['gpu'])

    # training optimizer and lr scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg['optimizer']['learning_rate'],
                                momentum=cfg['optimizer']['learning_rate'],
                                weight_decay=cfg['optimizer']['weight_decay'])
    lr_scheduler = Lr_schedule(optimizer,  ##
                                batch_size=train_loader.batch_size,
                                num_samples=len(train_loader.sampler),
                                **cfg['lr_scheduler'])
    
    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % lr_scheduler)
    perf_scoreboard = PerformanceScoreboard(cfg['log']['num_best_scores'])
    
    # ************************ train & eval ****************************

    if cfg['model']['checkpoint']is not None:
        load_model_checkpoint(model, checkpoint=cfg['model']['checkpoint'],
                              device=cfg['device']['type'], strict=cfg['model']['strict'])
        # init the input, weight, and output scale for quantization
        if 'quantization' in cfg.keys() and 'init_scale' in cfg['quantization'].keys() and cfg['quantization']['init_scale']:
            for _, module in model.named_modules():
                if isinstance(module, LSQ_act_quantizer):
                    module.init_batch_mode = True
                    # logger.info("Quantizer {} set init_batch_mode True. ".format(name))
                elif isinstance(module, Conv2dPDT) or isinstance(module, LinearPDT):
                    # init weight scale
                    module.init_weight_scale()
            for batch_idx,(inputs, targets)in enumerate(train_loader):
                if batch_idx >= cfg['quantization']['init_batch_num']:
                    break
                _ = model(inputs)
            for _, module in model.named_modules():
                if isinstance(module, LSQ_act_quantizer):
                    module.init_batch_mode = False
                    # logger.info("Quantizer {} set init_batch_mode False. ".format(name))
        #  
        v_top1_0, v_top5_0, _ = eval(model, test_loader, criterion, -1, monitors, cfg)
    
    if cfg['eval']:
        eval(model, test_loader, criterion, -1, monitors, cfg)
    else:
        if cfg['resume']['path'] or cfg['model']['pre_trained']:
            logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation val_loader)')
            top1, top5, _ = eval(model, val_loader, criterion, start_epoch - 1, monitors, cfg)
            perf_scoreboard.update(top1, top5, start_epoch - 1)
        # 
        start_epoch = 0
        for epoch in range(start_epoch, cfg['epochs']):
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            t_top1, t_top5, t_loss = train(model, train_loader, criterion, optimizer,
                                                   lr_scheduler, epoch, monitors, cfg)
            v_top1, v_top5, v_loss = eval(model, val_loader, criterion, epoch, monitors, cfg)

            tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

            perf_scoreboard.update(v_top1, v_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)
            save_checkpoint(epoch, cfg['model']['name'], model.module, optimizer, {'top1': v_top1, 'top5': v_top5}, is_best, log_dir)
        
        logger.info('>>>>>>>> checkpoint model evalution')
        logger.info('==> Top1: %.3f    Top5: %.3f \n', v_top1_0, v_top5_0)
        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        eval(model, test_loader, criterion, -1, monitors, cfg)

    tbmonitor.writer.close()  # close the TensorBoard
    logger.info('Program completed successfully ... exiting ...')

if __name__ == "__main__":
    main()
