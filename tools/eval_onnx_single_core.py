# ------------------------------------------------------------------------------
# Testing code.
# Example command:
# python tools/test_net_single_core.py --cfg PATH_TO_CONFIG_FILE
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import argparse
import sys

import cv2
import logging
import numpy as np
import onnxruntime as ort
import os
import pprint
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from datetime import datetime
from fvcore.common.file_io import PathManager

from segmentation.config import config, update_config
from segmentation.data import build_test_loader_from_cfg
from segmentation.evaluation import (
    SemanticEvaluator, CityscapesInstanceEvaluator, CityscapesPanopticEvaluator,
    COCOInstanceEvaluator, COCOPanopticEvaluator)
from segmentation.model.post_processing import get_cityscapes_instance_format
from segmentation.model.post_processing import get_semantic_segmentation, get_panoptic_segmentation
from segmentation.utils import AverageMeter
from segmentation.utils import save_annotation, save_instance_annotation, save_panoptic_annotation
from segmentation.utils import save_debug_images
from segmentation.utils.logger import setup_logger

logger = logging.getLogger('segmentation')


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network with single process')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--opt-model', default='', type=str, metavar='MODEL', required=True,
                        help='Path to optimize model (default: "")')
    parser.add_argument('--opt-weights', default='', type=str, metavar='MODEL',
                        help='Path to optimize weights (default: "")')

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    output_dir = os.path.join(config.OUTPUT_DIR, "test_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger(output=output_dir)
    logger.info(pprint.pformat(args))
    logger.info(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.TEST.GPUS)
    if len(gpus) > 1:
        raise ValueError('Test only supports single core.')
    device = 'mps' if torch.has_mps else ( 'cuda' if torch.cuda.is_available() else 'cpu' )

    # create inference session
    providers = ['CUDAExecutionProvider', ] if device == 'cuda' else ['CPUExecutionProvider', ]
    if device == 'mps' and 'CoreMLExecutionProvider' in ort.get_available_providers():
        providers = ['CoreMLExecutionProvider', ]
    # device = 'cpu'
    # providers = ['CPUExecutionProvider', ]

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    onnx_session = ort.InferenceSession(args.opt_model, sess_options=options, providers=providers)

    # build data_loader
    data_loader = build_test_loader_from_cfg(config)

    data_time = AverageMeter()
    net_time = AverageMeter()
    post_time = AverageMeter()
    timing_warmup_iter = 10

    semantic_metric = SemanticEvaluator(
        num_classes=data_loader.dataset.num_classes,
        ignore_label=data_loader.dataset.ignore_label,
        output_dir=None,  # os.path.join(output_dir, config.TEST.SEMANTIC_FOLDER),
        train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id()
    )

    instance_metric = None
    panoptic_metric = None

    if config.TEST.EVAL_INSTANCE:
        if 'cityscapes' in config.DATASET.DATASET:
            instance_metric = CityscapesInstanceEvaluator(
                output_dir=os.path.join(output_dir, config.TEST.INSTANCE_FOLDER),
                train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(),
                gt_dir=os.path.join(config.DATASET.ROOT, 'gtFine', config.DATASET.TEST_SPLIT)
            )
        elif 'coco' in config.DATASET.DATASET:
            instance_metric = COCOInstanceEvaluator(
                output_dir=os.path.join(output_dir, config.TEST.INSTANCE_FOLDER),
                train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(),
                gt_dir=os.path.join(config.DATASET.ROOT, 'annotations',
                                    'instances_{}.json'.format(config.DATASET.TEST_SPLIT))
            )
        else:
            raise ValueError('Undefined evaluator for dataset {}'.format(config.DATASET.DATASET))

    if config.TEST.EVAL_PANOPTIC:
        if 'cityscapes' in config.DATASET.DATASET:
            panoptic_metric = CityscapesPanopticEvaluator(
                output_dir=os.path.join(output_dir, config.TEST.PANOPTIC_FOLDER),
                train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(),
                label_divisor=data_loader.dataset.label_divisor,
                void_label=data_loader.dataset.label_divisor * data_loader.dataset.ignore_label,
                gt_dir=config.DATASET.ROOT,
                split=config.DATASET.TEST_SPLIT,
                num_classes=data_loader.dataset.num_classes
            )
        elif 'coco' in config.DATASET.DATASET:
            panoptic_metric = COCOPanopticEvaluator(
                output_dir=os.path.join(output_dir, config.TEST.PANOPTIC_FOLDER),
                train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(),
                label_divisor=data_loader.dataset.label_divisor,
                void_label=data_loader.dataset.label_divisor * data_loader.dataset.ignore_label,
                gt_dir=config.DATASET.ROOT,
                split=config.DATASET.TEST_SPLIT,
                num_classes=data_loader.dataset.num_classes
            )
        else:
            raise ValueError('Undefined evaluator for dataset {}'.format(config.DATASET.DATASET))

    foreground_metric = None
    if config.TEST.EVAL_FOREGROUND:
        foreground_metric = SemanticEvaluator(
            num_classes=2,
            ignore_label=data_loader.dataset.ignore_label,
            output_dir=os.path.join(output_dir, config.TEST.FOREGROUND_FOLDER)
        )

    image_filename_list = [
        os.path.splitext(os.path.basename(ann))[0] for ann in data_loader.dataset.ann_list]

    # Debug output.
    if config.TEST.DEBUG:
        debug_out_dir = os.path.join(output_dir, 'debug_test')
        PathManager.mkdirs(debug_out_dir)

    if not config.TEST.TEST_TIME_AUGMENTATION:
        if config.TEST.FLIP_TEST or len(config.TEST.SCALE_LIST) > 1:
            config.TEST.TEST_TIME_AUGMENTATION = True
            logger.warning(
                "Override TEST.TEST_TIME_AUGMENTATION to True because test time augmentation detected."
                "Please check your config file if you think it is a mistake.")

    # Train loop.
    try:
        for i, data in enumerate(data_loader):
            # if i == 1:
            #     break
            if i == timing_warmup_iter:
                data_time.reset()
                net_time.reset()
                post_time.reset()

            # data
            start_time = time.time()
            for key in data.keys():
                try:
                    data[key] = data[key].to(device)
                except:
                    pass

            image = data.pop('image')
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize(device)
                except:
                    pass
            data_time.update(time.time() - start_time)

            start_time = time.time()
            #image = image * 256 / 127.5 - 1
            #BMW_base_768_1536_sim.onnx {'mIoU': 75.48262947484066, 'fwIoU': 92.23071932792664, 'mACC': 83.37469100952148, 'pACC': 95.8065927028656}
            #PanDeepLab_Model1_sim.onnx {'mIoU': 75.49928865934673, 'fwIoU': 91.99857711791992, 'mACC': 83.4284079702277, 'pACC': 95.6729531288147}
            #PanDeepLab_Model2_sim.onnx {'mIoU': 75.1311653538754, 'fwIoU': 91.78708791732788, 'mACC': 84.01618254812139, 'pACC': 95.5343246459961}

            permute = True
            # run onnx session
            if permute:
                image = image * 256 / 127.5 - 1  # The best
                onnx_inputs = {"input": torch.permute(image, [0, 2, 3, 1]).cpu().numpy()}
            else:
                onnx_inputs = {"input": image.cpu().numpy()}

            out_list = onnx_session.run(None, onnx_inputs)
            semantic = -1
            offset = -1
            center = -1
            for j in range(3):
                out_list[j] = torch.from_numpy(out_list[j])
                if permute:
                    out_list[j] = torch.permute(out_list[j], [0, 3, 1, 2])
                if out_list[j].shape[1] == 1:
                    center = j
                if out_list[j].shape[1] == 2:
                    offset = j
                if out_list[j].shape[1] == 19:
                    semantic = j

            out_dict = {"semantic": out_list[semantic], "center": out_list[center], "offset": out_list[offset]}
            out_dict["semantic"] = F.interpolate(out_dict["semantic"],
                                                 size=image.shape[-2:],
                                                 mode='bilinear',
                                                 align_corners=True)
            out_dict["offset"] = F.interpolate(out_dict["offset"],
                                                 size=image.shape[-2:],
                                                 mode='bilinear',
                                                 align_corners=True)
            out_dict["center"] = F.interpolate(out_dict["center"],
                                                 size=image.shape[-2:],
                                                 mode='bilinear',
                                                 align_corners=True)
            try:
                torch.cuda.synchronize(device)
            except:
                pass
            net_time.update(time.time() - start_time)

            start_time = time.time()
            semantic_pred = get_semantic_segmentation(out_dict['semantic'])
            if 'foreground' in out_dict:
                foreground_pred = get_semantic_segmentation(out_dict['foreground'])
            else:
                foreground_pred = None

            # Oracle experiment
            if config.TEST.ORACLE_SEMANTIC:
                # Use predicted semantic for foreground
                foreground_pred = torch.zeros_like(semantic_pred)
                for thing_class in data_loader.dataset.thing_list:
                    foreground_pred[semantic_pred == thing_class] = 1
                # Use gt semantic
                semantic_pred = data['semantic']
                # Set it to a stuff label
                stuff_label = 0
                while stuff_label in data_loader.dataset.thing_list:
                    stuff_label += 1
                semantic_pred[semantic_pred == data_loader.dataset.ignore_label] = stuff_label
            if config.TEST.ORACLE_FOREGROUND:
                foreground_pred = data['foreground']
            if config.TEST.ORACLE_CENTER:
                out_dict['center'] = data['center']
            if config.TEST.ORACLE_OFFSET:
                out_dict['offset'] = data['offset']

            if config.TEST.EVAL_INSTANCE or config.TEST.EVAL_PANOPTIC:
                panoptic_pred, center_pred = get_panoptic_segmentation(
                    semantic_pred,
                    out_dict['center'],
                    out_dict['offset'],
                    thing_list=data_loader.dataset.thing_list,
                    label_divisor=data_loader.dataset.label_divisor,
                    stuff_area=config.POST_PROCESSING.STUFF_AREA,
                    void_label=(
                            data_loader.dataset.label_divisor *
                            data_loader.dataset.ignore_label),
                    threshold=config.POST_PROCESSING.CENTER_THRESHOLD,
                    nms_kernel=config.POST_PROCESSING.NMS_KERNEL,
                    top_k=config.POST_PROCESSING.TOP_K_INSTANCE,
                    foreground_mask=foreground_pred)
            else:
                panoptic_pred = None

            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize(device)
                except:
                    pass
            post_time.update(time.time() - start_time)
            logger.info('[{}/{}]\t'
                        'Data Time: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                        'Network Time: {net_time.val:.3f}s ({net_time.avg:.3f}s)\t'
                        'Post-processing Time: {post_time.val:.3f}s ({post_time.avg:.3f}s)\t'.format(
                i, len(data_loader), data_time=data_time, net_time=net_time, post_time=post_time))

            semantic_pred = semantic_pred.squeeze(0).cpu().numpy()
            if panoptic_pred is not None:
                panoptic_pred = panoptic_pred.squeeze(0).cpu().numpy()
            if foreground_pred is not None:
                foreground_pred = foreground_pred.squeeze(0).cpu().numpy()

            # Crop padded regions.
            image_size = data['size'].squeeze(0).cpu().numpy()
            semantic_pred = semantic_pred[:image_size[0], :image_size[1]]
            if panoptic_pred is not None:
                panoptic_pred = panoptic_pred[:image_size[0], :image_size[1]]
            if foreground_pred is not None:
                foreground_pred = foreground_pred[:image_size[0], :image_size[1]]

            # Resize back to the raw image size.
            raw_image_size = data['raw_size'].squeeze(0).cpu().numpy()
            if raw_image_size[0] != image_size[0] or raw_image_size[1] != image_size[1]:
                semantic_pred = cv2.resize(semantic_pred.astype(float), (raw_image_size[1], raw_image_size[0]),
                                           interpolation=cv2.INTER_NEAREST).astype(np.int32)
                if panoptic_pred is not None:
                    panoptic_pred = cv2.resize(panoptic_pred.astype(float),
                                               (raw_image_size[1], raw_image_size[0]),
                                               interpolation=cv2.INTER_NEAREST).astype(np.int32)
                if foreground_pred is not None:
                    foreground_pred = cv2.resize(foreground_pred.astype(float),
                                                 (raw_image_size[1], raw_image_size[0]),
                                                 interpolation=cv2.INTER_NEAREST).astype(np.int32)

            # Evaluates semantic segmentation.
            semantic_metric.update(semantic_pred,
                                   data['raw_label'].squeeze(0).cpu().numpy(),
                                   image_filename_list[i])

            # Optional: evaluates instance segmentation.
            if instance_metric is not None:
                raw_semantic = F.softmax(out_dict['semantic'][:, :, :image_size[0], :image_size[1]], dim=1)
                center_hmp = out_dict['center'][:, :, :image_size[0], :image_size[1]]
                if raw_image_size[0] != image_size[0] or raw_image_size[1] != image_size[1]:
                    raw_semantic = F.interpolate(raw_semantic,
                                                 size=(raw_image_size[0], raw_image_size[1]),
                                                 mode='bilinear',
                                                 align_corners=False)  # Consistent with OpenCV.
                    center_hmp = F.interpolate(center_hmp,
                                               size=(raw_image_size[0], raw_image_size[1]),
                                               mode='bilinear',
                                               align_corners=False)  # Consistent with OpenCV.

                raw_semantic = raw_semantic.squeeze(0).cpu().numpy()
                center_hmp = center_hmp.squeeze(1).squeeze(0).cpu().numpy()

                instances = get_cityscapes_instance_format(panoptic_pred,
                                                           raw_semantic,
                                                           center_hmp,
                                                           label_divisor=data_loader.dataset.label_divisor,
                                                           score_type=config.TEST.INSTANCE_SCORE_TYPE)
                instance_metric.update(instances, image_filename_list[i])

            # Optional: evaluates panoptic segmentation.
            if panoptic_metric is not None:
                image_id = '_'.join(image_filename_list[i].split('_')[:3])
                panoptic_metric.update(panoptic_pred,
                                       image_filename=image_filename_list[i],
                                       image_id=image_id)

            # Optional: evaluates foreground segmentation.
            if foreground_metric is not None:
                semantic_label = data['raw_label'].squeeze(0).cpu().numpy()
                foreground_label = np.zeros_like(semantic_label)
                for sem_lab in np.unique(semantic_label):
                    # Both `stuff` and `ignore` are background.
                    if sem_lab in data_loader.dataset.thing_list:
                        foreground_label[semantic_label == sem_lab] = 1

                # Use semantic segmentation as foreground segmentation.
                if foreground_pred is None:
                    foreground_pred = np.zeros_like(semantic_pred)
                    for sem_lab in np.unique(semantic_pred):
                        if sem_lab in data_loader.dataset.thing_list:
                            foreground_pred[semantic_pred == sem_lab] = 1

                foreground_metric.update(foreground_pred,
                                         foreground_label,
                                         image_filename_list[i])

            if config.TEST.DEBUG:
                # Raw outputs
                save_debug_images(
                    dataset=data_loader.dataset,
                    batch_images=image,
                    batch_targets=data,
                    batch_outputs=out_dict,
                    out_dir=debug_out_dir,
                    iteration=i,
                    target_keys=config.DEBUG.TARGET_KEYS,
                    output_keys=config.DEBUG.OUTPUT_KEYS,
                    is_train=False,
                )
                if panoptic_pred is not None:
                    # Processed outputs
                    save_annotation(semantic_pred, debug_out_dir, 'semantic_pred_%d' % i,
                                    add_colormap=True, colormap=data_loader.dataset.create_label_colormap())
                    pan_to_sem = panoptic_pred // data_loader.dataset.label_divisor
                    save_annotation(pan_to_sem, debug_out_dir, 'pan_to_sem_pred_%d' % i,
                                    add_colormap=True, colormap=data_loader.dataset.create_label_colormap())
                    ins_id = panoptic_pred % data_loader.dataset.label_divisor
                    pan_to_ins = panoptic_pred.copy()
                    pan_to_ins[ins_id == 0] = 0
                    save_instance_annotation(pan_to_ins, debug_out_dir, 'pan_to_ins_pred_%d' % i)

                    save_panoptic_annotation(panoptic_pred, debug_out_dir, 'panoptic_pred_%d' % i,
                                             label_divisor=data_loader.dataset.label_divisor,
                                             colormap=data_loader.dataset.create_label_colormap())

    except Exception:
        logger.exception("Exception during testing:")
        raise
    finally:
        logger.info("Inference finished.")
        semantic_results = semantic_metric.evaluate()
        logger.info(semantic_results)
        if instance_metric is not None:
            instance_results = instance_metric.evaluate()
            logger.info(instance_results)
        if panoptic_metric is not None:
            panoptic_results = panoptic_metric.evaluate()
            logger.info(panoptic_results)
        if foreground_metric is not None:
            foreground_results = foreground_metric.evaluate()
            logger.info(foreground_results)


if __name__ == '__main__':
    main()
    sys.exit(0)