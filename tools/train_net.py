# ------------------------------------------------------------------------------
# Training code.
# Example command:
# python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --cfg PATH_TO_CONFIG_FILE
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import argparse
import logging
import os
import pprint
import time
from contextlib import suppress
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
from fvcore.common.file_io import PathManager
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import _init_paths
from segmentation.config import config, update_config
from segmentation.data import build_train_loader_from_cfg, build_test_loader_from_cfg
from segmentation.model import build_segmentation_model_from_cfg
from segmentation.solver import build_optimizer, build_lr_scheduler
from segmentation.solver import get_lr_group_id
from segmentation.utils import AverageMeter
from segmentation.utils import comm
from segmentation.utils import save_debug_images
from segmentation.utils.logger import setup_logger
from segmentation.utils.utils import get_loss_info_str, to_cuda, get_module, get_loss_info_dict

torch.backends.cudnn.benchmark = True
logger = logging.getLogger('segmentation')


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts', help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--amp', type=int, default=1, help='use Native AMP for mixed precision training')
    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory layout')
    parser.add_argument('--opt-model', default='', type=str, metavar='MODEL',
                        help='Path to optimize model (default: "")')

    # WANDB Parameters
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='log training and validation metrics to wandb')
    parser.add_argument('--experiment', default='panoptic-deeplabv3', type=str, metavar='NAME',
                        help='name of train experiment, name of sub-folder for output')
    parser.add_argument('--group', default='', type=str, metavar='NAME', help='name of experiment version')

    args = parser.parse_args()
    update_config(config, args)

    return args


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def main():
    args = parse_args()

    has_wandb = False
    if args.log_wandb:
        try:
            import wandb
            if not args.group:
                raise ValueError('Please provide experiment group < --group "0.0.1" > ')
            wandb.init(project=args.experiment, group=args.group, config=args)
            has_wandb = True
        except ImportError:
            logger.warning("You've requested to log metrics to wandb but package not found. "
                           "Metrics not being logged to wandb, try `pip install wandb`")

    output_dir = os.path.join(config.OUTPUT_DIR, "train_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger(output=output_dir, distributed_rank=args.local_rank)
    logger.info(pprint.pformat(args))
    logger.info(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                    % (args.rank, args.world_size))
    else:
        logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # build model
    if args.opt_model:
        model = torch.load(args.opt_model, map_location='cpu')
        logger.info(f"Load: {args.opt_model}")
    else:
        model = build_segmentation_model_from_cfg(config)
        # torch.save(model, 'models/~panoptic_deeplab_R50_os32_cityscapes.pth')

    # logger.debug("Model:\n{}".format(model))
    logger.info("Rank of current process: {}. World size: {}".format(comm.get_rank(), comm.get_world_size()))

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if comm.get_world_size() > 1:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    data_loader = build_train_loader_from_cfg(config)
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_lr_scheduler(config, optimizer)

    data_loader_iter = iter(data_loader)

    start_iter = 0
    max_iter = config.TRAIN.MAX_ITER
    best_param_group_id = get_lr_group_id(optimizer)

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            logger.info('AMP not enabled. Training in float32.')

    # initialize model
    if os.path.isfile(config.MODEL.WEIGHTS):
        model_weights = torch.load(config.MODEL.WEIGHTS)
        get_module(model, args.distributed).load_state_dict(model_weights, strict=False)
        logger.info('Pre-trained model from {}'.format(config.MODEL.WEIGHTS))
    elif not config.MODEL.BACKBONE.PRETRAINED:
        if os.path.isfile(config.MODEL.BACKBONE.WEIGHTS):
            pretrained_weights = torch.load(config.MODEL.BACKBONE.WEIGHTS)
            get_module(model, args.distributed).backbone.load_state_dict(pretrained_weights, strict=False)
            logger.info('Pre-trained backbone from {}'.format(config.MODEL.BACKBONE.WEIGHTS))
        else:
            logger.info('No pre-trained weights for backbone, training from scratch.')
    # load model
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            start_iter = checkpoint['start_iter']
            get_module(model, args.distributed).load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info('Loaded checkpoint (starting from iter {})'.format(checkpoint['start_iter']))

    data_time = AverageMeter()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    # Debug output.
    if config.DEBUG.DEBUG:
        debug_out_dir = os.path.join(output_dir, 'debug_train')
        PathManager.mkdirs(debug_out_dir)

    if config.CKPT_FREQ > 0:
        model_out_dir = os.path.join(output_dir, 'models')
        PathManager.mkdirs(model_out_dir)

    # Train loop.
    model.train()
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    try:
        for i in range(start_iter, max_iter):
            # data
            start_time = time.time()
            data = next(data_loader_iter)
            if not args.distributed:
                data = to_cuda(data, args.device)
            data_time.update(time.time() - start_time)

            image = data.pop('image')
            if args.channels_last:
                image = image.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                out_dict = model(image, data)
                loss = out_dict['loss']

            optimizer.zero_grad()
            if loss_scaler is not None:
                loss_scaler(loss, optimizer, parameters=model.parameters(), create_graph=second_order)
            else:
                loss.backward(create_graph=second_order)
                optimizer.step()

            # Get lr.
            lr = optimizer.param_groups[best_param_group_id]["lr"]
            lr_scheduler.step()

            batch_time.update(time.time() - start_time)
            loss_meter.update(loss.detach().cpu().item(), image.size(0))

            if i == 0 or (i + 1) % config.PRINT_FREQ == 0:
                msg = '[{0}/{1}] LR: {2:.7f}\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'.format(
                    i + 1, max_iter, lr, batch_time=batch_time, data_time=data_time)
                msg += get_loss_info_str(get_module(model, args.distributed).loss_meter_dict)
                logger.info(msg)

                if comm.is_main_process() and has_wandb:
                    rowd = get_loss_info_dict(get_module(model, args.distributed).loss_meter_dict)
                    rowd['iter'] = i + 1
                    rowd['lr'] = lr
                    rowd['batch_time'] = batch_time.avg
                    wandb.log(rowd)

            if i == 0 or (i + 1) % config.DEBUG.DEBUG_FREQ == 0:
                if comm.is_main_process() and config.DEBUG.DEBUG:
                    save_debug_images(
                        dataset=data_loader.dataset,
                        batch_images=image,
                        batch_targets=data,
                        batch_outputs=out_dict,
                        out_dir=debug_out_dir,
                        iteration=i,
                        target_keys=config.DEBUG.TARGET_KEYS,
                        output_keys=config.DEBUG.OUTPUT_KEYS,
                        iteration_to_remove=i - config.DEBUG.KEEP_INTERVAL
                    )
            if i == 0 or (i + 1) % config.CKPT_FREQ == 0:
                if comm.is_main_process():
                    torch.save({
                        'start_iter': i + 1,
                        'state_dict': get_module(model, args.distributed).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                    }, os.path.join(model_out_dir, f'checkpoint_{i+1}.pth.tar'))
    except Exception:
        logger.exception("Exception during training:")
        raise
    finally:
        if comm.is_main_process():
            torch.save(get_module(model, args.distributed).state_dict(), os.path.join(output_dir, 'final_state.pth'))
        logger.info("Training finished.")


if __name__ == '__main__':
    main()
