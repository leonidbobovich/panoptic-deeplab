import os.path
import sys

import onnx
import onnx_tool
import onnxsim
import torch

import _init_paths
print(sys.path)

from segmentation.config import config, update_config
# from segmentation.data import build_test_loader_from_cfg
# from segmentation.evaluation import (
#     SemanticEvaluator, CityscapesInstanceEvaluator, CityscapesPanopticEvaluator,
#     COCOInstanceEvaluator, COCOPanopticEvaluator)
from segmentation.model import build_segmentation_model_from_cfg
# from segmentation.model.post_processing import get_cityscapes_instance_format
# from segmentation.model.post_processing import get_semantic_segmentation, get_panoptic_segmentation
# from segmentation.utils import AverageMeter
# from segmentation.utils import save_annotation, save_instance_annotation, save_panoptic_annotation
# from segmentation.utils import save_debug_images
# from segmentation.utils.logger import setup_logger
# from segmentation.utils.test_utils import multi_scale_inference
from torch_receptive_field import receptive_field

class Args:
    def __init__(self, cfg_name:str):
        self.cfg = cfg_name
        self.opts = ''

def main():
    args = Args('/Users/leonidbobovich/Work/ml/qualcomm-panoptic-deeplab/configs/panoptic_deeplab_vian.yaml')
    update_config(config, args)
    model = build_segmentation_model_from_cfg(config)
    receptive_field(model.backbone, (3, 768, 1536))
    checkpoint  = torch.load('/Users/leonidbobovich/Work/ml/qualcomm-panoptic-deeplab/output/train/models/checkpoint_99000.pth.tar', map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint['state_dict'], strict=False)
    # print(model)
    m = torch.nn.Sequential(model.backbone, model.decoder)
    output_name =  'test.onnx'
    torch.onnx.export(m, f=output_name, args=torch.rand(1,3,768,1536), input_names=['input'], output_names=['semantic', 'center', 'offset'])
    model = onnx.load('test.onnx')
    model = onnxsim.simplify(model=model, overwrite_input_shapes={model.graph.input[0].name:[1,3,768,1536]})
    onnx.save(model[0], output_name)
    onnx_tool.model_profile(model[0])

if __name__ == '__main__':
    main()
    sys.exit(0)