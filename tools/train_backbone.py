# ------------------------------------------------------------------------------
# Training code.
# Example command:
# python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --cfg PATH_TO_CONFIG_FILE
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import os
import random
import sys
import time
import pprint
import logging
import argparse
from datetime import datetime
from contextlib import suppress
import numpy

import onnx
import onnxsim
import torch
import torch.backends.cudnn as cudnn
import torchvision.models.resnet
from PIL import Image

from torch import nn
from torch.nn.parallel import DistributedDataParallel

from fvcore.common.file_io import PathManager

import _init_paths
from segmentation.config import config, update_config
from segmentation.data import build_train_loader_from_cfg
from segmentation.data import build_test_loader_from_cfg
from segmentation.data import build_analyze_loader_from_cfg
from segmentation.model import build_segmentation_model_from_cfg
from segmentation.solver import build_optimizer, build_lr_scheduler
from segmentation.solver import get_lr_group_id
from segmentation.utils import AverageMeter
from segmentation.utils import comm
from segmentation.utils import save_debug_images
from segmentation.utils.logger import setup_logger
from segmentation.utils.utils import get_loss_info_str, to_cuda, get_module, get_loss_info_dict

from torchvision.models import resnet50, ResNet50_Weights

logger = logging.getLogger('segmentation')

logger.warning('TO DO: Check if we can move it main')
torch.backends.cudnn.benchmark = True


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

    if torch.has_mps:
        logger.warning("TO DO: Ugly hack, config.DATALOADER.NUM_WORKERS is not MPS but on python version and platform. Not working on MacOS ")
        config.defrost()
        config.DATALOADER.NUM_WORKERS = 0
        config.freeze()


    output_dir = os.path.join(config.OUTPUT_DIR, "train_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    output_dir = os.path.join(config.OUTPUT_DIR, "train")
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger(output=output_dir, distributed_rank=args.local_rank)
    logger.info(pprint.pformat(args))
    logger.info(config)
    data_loader = build_analyze_loader_from_cfg(config)
    data_loader_iter = iter(data_loader)

    data_time = AverageMeter()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    model = torchvision.models.resnet.resnet50(torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4)
    model.eval()
    # try:
    total = 0
    classes = {}
    fraction = {}
    image_count = 0
    for data in data_loader_iter:
        # data
        #data = next(data_loader_iter)
        image = data['image']
        label = data['semantic']
        # n = label.flatten().numpy()
        # x = numpy.unique(n, return_counts=True)
        # total += x[1].sum()
        # for i in range(len(x[0])):
        #     classes[f'{x[0][i]}'] = (classes[f'{x[0][i]}'] + x[1][i]) if f'{x[0][i]}' in classes.keys() else x[1][i]
        #     fraction[f'{x[0][i]}'] = classes[f'{x[0][i]}'] / total
        # print(fraction)
        size = 223
        # for oy in range(0, image.shape[-2] - size):
        #     for ox in range(0, image.shape[-1] - size):
        for k in range(1000):
                oy = random.randint(0, image.shape[-2] - size)
                ox = random.randint(0, image.shape[-1] - size)
                # if 255 in label[:, oy:oy + size, ox:ox + size].flatten().numpy():
                #     continue
                i = image[:,:,oy:oy+size,ox:ox+size]
                l = label[:,oy:oy+size,ox:ox+size]
                c = l[:, size // 2, size // 2].numpy()[0]
                if c == 255:
                    continue
                x = model(i)
                n = numpy.unique(l.flatten().numpy(), return_counts=True)
                print(x.shape, c, n)
                path = 'triplet'
                if not os.path.exists(path):
                    os.mkdir(path)
                path = os.path.join(path, 'test' if random.randint(0, 9) == 0 else 'train' )
                if not os.path.exists(path):
                    os.mkdir(path)
                path = os.path.join(path,f'{c}')
                if not os.path.exists(path):
                    os.mkdir(path)
                print(i.shape)
                i = torch.permute(i, dims = (0, 2, 3, 1))
                print(i.shape)
                img = Image.fromarray((i * 255).to(torch.uint8).numpy()[0])
                img.save(os.path.join(path,f'{image_count}_{oy}_{ox}.jpg'))
                # numpy.save(os.path.join(path,f'{image_count}_{oy}_{ox}.npy'), i.numpy())
        image_count = image_count + 1
    # except Exception:
    #     logger.exception("Exception during training:")
    #     raise
    # finally:
    #     logger.info("Generation finished.")


if __name__ == '__main__':
    main()
