import sys

import onnx
import onnxsim
from onnx import version_converter

# outputname = '../evaluate.71.2.onnx'
# model = onnx.load('../aimet-ptq-export-71.2/Final.onnx')
#outputname = '../evaluate.16.75.80.72.20.onnx'
#model = onnx.load('../panoptic-deep-stem-16-75.8_quantized_to_72.2/Final.onnx')
outputname = '/Users/leonidbobovich/Work/ml/qualcomm-panoptic-deeplab/bmw_pytorch_deep_stem_baseline_768x1536_sim.onnx'
model = onnx.load('/Users/leonidbobovich/Work/ml/qualcomm-panoptic-deeplab/bmw_pytorch_deep_stem_baseline_768x1536.onnx')
model = onnxsim.simplify(model, overwrite_input_shapes={model.graph.input[0].name:[1,3,768,1536]})
onnx.save(model[0], outputname)
model = onnx.load(outputname)
print(model.graph.output)
model.graph.output.clear()

semantic_name = None
center_name = None
offset_name = None

for n in model.graph.node:
    if n.name == 'decoder.semantic_head.classifier.semantic.1' or n.name == '/decoder/semantic_head/semantic/semantic.1/Conv':
        semantic_name = n.output[0]
        output_value_info = onnx.helper.make_tensor_value_info(n.output[0], onnx.TensorProto.FLOAT, shape=[1, 19,192,384])
        output_value_info.name = 'semantic'
        n.output.clear()
        n.output.append(output_value_info.name)
        model.graph.output.append(output_value_info)
    if n.name == 'decoder.instance_head.classifier.center.1' or n.name == '/decoder/instance_head/center/center.1/Conv':
        center_name = n.output[0]
        output_value_info = onnx.helper.make_tensor_value_info(n.output[0], onnx.TensorProto.FLOAT, shape=[1, 1,192,384])
        output_value_info.name = 'center'
        n.output.clear()
        n.output.append(output_value_info.name)
        model.graph.output.append(output_value_info)
    if n.name == 'decoder.instance_head.classifier.offset.1' or n.name == '/decoder/instance_head/offset/offset.1/Conv':
        offset_name = n.output[0]
        output_value_info = onnx.helper.make_tensor_value_info(n.output[0], onnx.TensorProto.FLOAT, shape=[1, 2,192,384])
        output_value_info.name = 'offset'
        n.output.clear()
        n.output.append(output_value_info.name)
        model.graph.output.append(output_value_info)

clean_up_list = [offset_name, center_name, semantic_name]
stop = False
while not stop:
    stop = True
    for n in model.graph.node:
        print(n.input[0])
        if n.input[0] in clean_up_list:
            model.graph.node.remove(n)
            for o in n.output:
                clean_up_list.append(o)
            stop = False
            break

onnx.checker.check_model(model)
model = onnxsim.simplify(model, overwrite_input_shapes={model.graph.input[0].name:[1,3,768,1536]})
onnx.save(model[0], outputname)

model = onnx.load(outputname)
converted_model = version_converter.convert_version(model, 13)

sys.exit(0)
