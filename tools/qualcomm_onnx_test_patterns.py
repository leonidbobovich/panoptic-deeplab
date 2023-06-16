import onnx
import onnx_tool as ot

m = onnx.load('../panoptic-deep-stem-16-75.8_pruned_0.8.sim.onnx')
ot.model_profile(m=m)