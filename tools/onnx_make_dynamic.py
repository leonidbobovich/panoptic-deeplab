import sys
import onnx
import onnxsim

def make_dims(v):
    r = []
    for d in v.dim:
        r.append(d.dim_value)
    r[0] = 0
    return r
# outputname = '../evaluate.71.2.onnx'
# model = onnx.load('../aimet-ptq-export-71.2/Final.onnx')
outputname = '../dynamic.75.88.72.06.onnx'
model = onnx.load('../evaluate.75.88.72.06.onnx')
new_value_info = []
for n in model.graph.value_info:
    new_value_info.append(onnx.helper.make_tensor_value_info(name=n.name, elem_type=n.type.tensor_type.elem_type, shape=None))
model.graph.value_info.clear()
for n in new_value_info:
    model.graph.value_info.append(n)

new_value_info=[]
for n in model.graph.output:
    new_value_info.append(onnx.helper.make_tensor_value_info(name=n.name, elem_type=n.type.tensor_type.elem_type, shape=make_dims(n.type.tensor_type.shape)))
model.graph.output.clear()
for n in new_value_info:
    model.graph.output.append(n)

new_value_info=[]
for n in model.graph.input:
    new_value_info.append(onnx.helper.make_tensor_value_info(name=n.name, elem_type=n.type.tensor_type.elem_type, shape=make_dims(n.type.tensor_type.shape)))
model.graph.input.clear()
for n in new_value_info:
    model.graph.input.append(n)

onnx.checker.check_model(model)
#model = onnxsim.simplify(model, overwrite_input_shapes={model.graph.input[0].name:[1,3,768,1536]})
onnx.save(model, outputname)
sys.exit(0)
