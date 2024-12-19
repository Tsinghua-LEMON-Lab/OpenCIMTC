import os
import numpy as np
import torch as t
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import time
import logging
import logging.config
import math
import operator
# 
from .quantization import data_quantization
from .pdt import Conv2dPDT, ConvTranspose2dPDT, LinearPDT

logger = logging.getLogger()

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, fmt='%.6f'):
        self.fmt = fmt
        self.val = self.avg = self.sum = self.count = 0

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        s = self.fmt % self.avg
        return s

def __balance_val_split(dataset, val_split=0.):
    targets = np.array(dataset.targets)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=val_split,
        stratify=targets
    )
    train_dataset = t.utils.data.Subset(dataset, indices=train_indices)
    val_dataset = t.utils.data.Subset(dataset, indices=val_indices)
    return train_dataset, val_dataset

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with t.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def __deterministic_worker_init_fn(worker_id, seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)


def load_data(cfg):
    if cfg['val_split'] < 0 or cfg['val_split'] >= 1:
        raise ValueError('val_split should be in the range of [0, 1) but got %.3f' % cfg['val_split'])

    tv_normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
    if cfg['dataset'] == 'imagenet':
        train_transform = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv_normalize
        ])
        val_transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv_normalize
        ])

        train_set = tv.datasets.ImageFolder(
            root=os.path.join(cfg['path'], 'train'), transform=train_transform)
        test_set = tv.datasets.ImageFolder(
            root=os.path.join(cfg['path'], 'val'), transform=val_transform)

    elif cfg['dataset'] == 'cifar10':
        tv_normalize = tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transform = tv.transforms.Compose([
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(32, 4),
            tv.transforms.ToTensor(),
            tv_normalize
        ])
        val_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv_normalize
        ])
        train_set = tv.datasets.CIFAR10(cfg['path'], train=True, transform=train_transform, download=False)
        test_set = tv.datasets.CIFAR10(cfg['path'], train=False, transform=val_transform, download=False)
         
    elif cfg['dataset'] == 'cifar100':
        tv_normalize = tv.transforms.Normalize((0.50705882, 0.48666667, 0.44078431), (0.26745098, 0.25568627, 0.27607843))
        train_transform = tv.transforms.Compose([
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(32, 4),
            tv.transforms.ToTensor(),
            tv_normalize
        ])
        val_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv_normalize
        ])

        train_set = tv.datasets.CIFAR100(cfg['path'], train=True, transform=train_transform, download=True)
        test_set = tv.datasets.CIFAR100(cfg['path'], train=False, transform=val_transform, download=True)
    
    elif cfg['dataset'] == 'FashionMNIST':
        train_transform = tv.transforms.Compose([
            tv.transforms.ToTensor()
        ])
        train_set = tv.datasets.FashionMNIST(root = cfg['path'], train = True,
                                            transform = tv.transforms.Compose([tv.transforms.ToTensor()]),
                                            download = False)
        test_set = tv.datasets.FashionMNIST(root = cfg['path'], train = False,
                                            transform = tv.transforms.Compose([tv.transforms.ToTensor()]), download = False)
    elif cfg['dataset'] == 'mnist':
        train_set = tv.datasets.MNIST(root=cfg['path'], train=True, download=False,
                        transform=tv.transforms.Compose([
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize((0.1307,), (0.3081,))
                        ]))
        test_set = tv.datasets.MNIST(root=cfg['path'], train=False, transform=tv.transforms.Compose([
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize((0.1307,), (0.3081,))
                        ]))

    else:
        raise ValueError('load_data does not support dataset %s' % cfg.dataset)

    if cfg['val_split'] != 0:
        train_set, val_set = __balance_val_split(train_set, cfg['val_split'])
    else:
        val_set = test_set

    worker_init_fn = None
    if cfg['deterministic']:
        worker_init_fn = __deterministic_worker_init_fn

    train_loader = t.utils.data.DataLoader(
        train_set, cfg['batch_size'], shuffle=True, num_workers=cfg['workers'], pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = t.utils.data.DataLoader(
        val_set, cfg['batch_size'], num_workers=cfg['workers'], pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = t.utils.data.DataLoader(
        test_set, cfg['batch_size'], num_workers=cfg['workers'], pin_memory=True, worker_init_fn=worker_init_fn)

    return train_loader, val_loader, test_loader

def init_logger(experiment_name, output_dir):
    time_str = time.strftime("%Y%m%d_%H%M%S")
    # time_str = 'F'
    exp_full_name = time_str if experiment_name is None else experiment_name + '_' + time_str
    log_dir = output_dir + "/" + exp_full_name
    # log_dir.mkdir(exist_ok=True)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/" + 'model_training.log'
    logging.basicConfig(level=logging.DEBUG,
                    filename=log_file,
                    filemode='a',
                    format= '%(asctime)s - %(levelname)s: %(message)s'
                    )
    # cmd output
    console_handler = logging.StreamHandler()
    # logging.config.fileConfig(cfg_file, defaults={'logfilename': log_file})
    logger = logging.getLogger()
    logger.addHandler(console_handler)
    logger.info('Log file for this run: ' + str(log_file))
    return logger, log_dir

def load_model_checkpoint(model, checkpoint=None, device='cuda', strict=True):
    checkpoint = t.load(checkpoint, map_location=device)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    check1 = {}
    for k in checkpoint:
        if k.split(".")[0] == "module":
            check1[('.'.join(k.split('.')[1:]))] = checkpoint[k]
        else: 
            check1[k] = checkpoint[k]
    if isinstance(model, t.nn.DataParallel):
        model = model.module
    model.load_state_dict(check1, strict=strict)

def save_checkpoint(epoch, name, model, optimizer, extras=None, is_best=None, output_dir='.'):
    """Save a pyTorch training checkpoint
    Args:
        epoch: current epoch number
        name: name of the network architecture/topology
        model: a pyTorch model
        extras: optional dict with additional user-defined data to be saved in the checkpoint.
            Will be saved under the key 'extras'
        is_best: If true, will save a copy of the checkpoint with the suffix 'best'
        output_dir: directory in which to save the checkpoint
    """
    if not os.path.isdir(output_dir):
        raise IOError('Checkpoint directory does not exist at', os.path.abspath(dir))

    if extras is None:
        extras = {}
    if not isinstance(extras, dict):
        raise TypeError('extras must be either a dict or None')
    
    # 
    quantization_info = {}
    for n, module in model.named_modules():
        if isinstance(module, Conv2dPDT) or isinstance(module, ConvTranspose2dPDT) or isinstance(module, LinearPDT):
            quantization_info[n] = module.get_params()
    
    filename = 'checkpoint.pth.tar' if name is None else name + '_checkpoint.pth.tar'
    filepath = os.path.join(output_dir, filename)
    filename_best = 'best.pth.tar' if name is None else name + '_best.pth.tar'
    filepath_best = os.path.join(output_dir, filename_best)

    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'name': name,
        'extras': extras,
        'quantization_info': quantization_info
    }

    msg = 'Saving checkpoint to:\n'
    msg += '             Current: %s\n' % filepath
    t.save(checkpoint, filepath)
    if is_best:
        msg += '                Best: %s\n' % filepath_best
        t.save(checkpoint, filepath_best)
    logger.info(msg)

def train(model, train_loader, criterion, optimizer, lr_scheduler, epoch, monitors, cfg):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    logger.info('Training: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.train()
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(cfg['device']['type'])
        targets = targets.to(cfg['device']['type'])
        # convert to bit image
        if cfg['img_quant_bits'] != 0:
            inputs = data_quantization(inputs, bit=cfg['img_quant_bits'])
            
        outputs = model(inputs)
        
        # outputs = 1
        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        if lr_scheduler is not None:
            lr_scheduler.step(epoch=epoch, batch=batch_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (batch_idx + 1) % cfg['log']['print_freq'] == 0:
            for m in monitors:
                m.update(epoch, batch_idx + 1, steps_per_epoch, 'Training', {
                    'Loss': losses,
                    'Top1': top1,
                    'Top5': top5,
                    'BatchTime': batch_time,
                    'LR': optimizer.param_groups[0]['lr']
                })

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg

def eval(model, data_loader, criterion, epoch, monitors, cfg):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)

    logger.info('Validation: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.eval()
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        with t.no_grad():
            inputs = inputs.to(cfg['device']['type'])
            targets = targets.to(cfg['device']['type'])
            if cfg['img_quant_bits'] != 0:
                inputs = data_quantization(inputs, bit=cfg['img_quant_bits'])
            
            outputs= model(inputs)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (batch_idx + 1) % cfg['log']['print_freq'] == 0:
                for m in monitors:
                    m.update(epoch, batch_idx + 1, steps_per_epoch, 'Validation', {
                        'Loss': losses,
                        'Top1': top1,
                        'Top5': top5,
                        'BatchTime': batch_time
                    })

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n', top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg

class ProgressMonitor():
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def update(self, epoch, step_idx, step_num, prefix, meter_dict):
        msg = prefix
        if epoch > -1:
            msg += ' [%d][%5d/%5d]   ' % (epoch, step_idx, int(step_num))
        else:
            msg += ' [%5d/%5d]   ' % (step_idx, int(step_num))
        for k, v in meter_dict.items():
            msg += k + ' '
            if isinstance(v, AverageMeter):
                msg += str(v)
            else:
                msg += '%.6f' % v
            msg += '   '
        self.logger.info(msg)

class TensorBoardMonitor():
    def __init__(self, logger, log_dir):
        super().__init__()
        self.writer = SummaryWriter(log_dir + '/' + 'tb_runs')
        logger.info('TensorBoard data directory: %s/tb_runs' % log_dir)

    def update(self, epoch, step_idx, step_num, prefix, meter_dict):
        current_step = epoch * step_num + step_idx
        for k, v in meter_dict.items():
            val = v.val if isinstance(v, AverageMeter) else v
            self.writer.add_scalar(prefix + '/' + k, val, current_step)

class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                        idx + 1, score['epoch'], score['top1'], score['top5'])

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch

class LrScheduler:
    def __init__(self, optimizer, batch_size, num_samples, update_per_batch):
        self.optimizer = optimizer
        self.current_lr = self.get_lr()
        self.base_lr = self.get_lr()
        self.num_groups = len(self.base_lr)

        self.batch_size = batch_size
        self.num_samples = num_samples
        self.update_per_batch = update_per_batch

    def get_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]

    def set_lr(self, lr):
        for i in range(self.num_groups):
            self.current_lr[i] = lr[i]
            self.optimizer.param_groups[i]['lr'] = lr[i]

    def step(self, epoch, batch):
        raise NotImplementedError

    def __str__(self):
        s = '`%s`' % self.__class__.__name__
        s += '\n    Update per batch: %s' % self.update_per_batch
        for i in range(self.num_groups):
            s += '\n             Group %d: %g' % (i, self.current_lr[i])
        return s


class FixedLr(LrScheduler):
    def step(self, epoch, batch):
        pass

class MultiStepLr(LrScheduler):
    def __init__(self, milestones=[30, ], gamma=0.1, **kwargs):
        super(MultiStepLr, self).__init__(**kwargs)
        self.milestones = milestones
        self.gamma = gamma

    def step(self, epoch, batch):
        n = sum([1 for m in self.milestones if m <= epoch])
        scale = self.gamma ** n
        for i in range(self.num_groups):
            self.current_lr[i] = self.base_lr[i] * scale
        self.set_lr(self.current_lr)

def Lr_schedule(optimizer, mode, batch_size=None, num_samples=None, update_per_batch=False, **kwargs):
    # variables batch_size & num_samples are only used when the learning rate updated every epoch
    if update_per_batch:
        assert isinstance(batch_size, int) and isinstance(num_samples, int)
    if mode == 'fixed':
        scheduler = FixedLr
    elif mode == 'multi_step':
        scheduler = MultiStepLr
    else:
        raise ValueError('LR scheduler `%s` is not supported', mode)

    return scheduler(optimizer=optimizer, batch_size=batch_size, num_samples=num_samples,
                     update_per_batch=update_per_batch, **kwargs)