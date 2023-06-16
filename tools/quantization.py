import sys

import onnx
from onnx import version_converter, helper
import numpy
import onnxruntime
import os
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, QuantFormat, CalibrationMethod
from onnxruntime.quantization.qdq_loss_debug import (
    collect_activations, compute_activation_error, compute_weight_error,
    create_activation_matching, create_weight_matching,
    modify_model_output_intermediate_tensors)

def _preprocess_images(images_folder: str, height: int, width: int, size_limit=0):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    # image_names = os.listdir(images_folder)
    # if size_limit > 0 and len(image_names) >= size_limit:
    #     batch_filenames = [image_names[i] for i in range(size_limit)]
    # else:
    #     batch_filenames = image_names
    unconcatenated_batch_data = []

    # for image_name in batch_filenames:
    for root, folder, files_in_dir in os.walk(images_folder):
        for image_name in files_in_dir:
            image_filepath = os.path.join(root, image_name)
            pillow_img = Image.new("RGB", (width, height))
            pillow_img.paste(Image.open(image_filepath).resize((width, height)))
            # input_data = numpy.float32(pillow_img) - numpy.array(
            #     [123.68, 116.78, 103.94], dtype=numpy.float32
            # )
            input_data = numpy.float32(pillow_img) # * 2 - 1
            nhwc_data = numpy.expand_dims(input_data, axis=0)
            nchw_data = nhwc_data.transpose((0, 3, 1, 2))  # ONNX Runtime standard
            unconcatenated_batch_data.append(nchw_data)
            if len( unconcatenated_batch_data ) >= size_limit:
                break
        if len(unconcatenated_batch_data) >= size_limit:
            break
    batch_data = numpy.concatenate(
        numpy.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    return batch_data

class DataReader1(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, sess_options=None,
                                               providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(
            calibration_image_folder, height, width, size_limit=10
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

class DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, sess_options=None,
                                               providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
        if session.get_inputs()[0].shape[1] > session.get_inputs()[0].shape[3]:
            self.transposed = True
            width = session.get_inputs()[0].shape[2]
            height = session.get_inputs()[0].shape[1]
        else:
            self.transposed = False
            width = session.get_inputs()[0].shape[2]
            height = session.get_inputs()[0].shape[3]
        self.height = height
        self.width = width
        # Convert image to input data
        self.nhwc_data_list = []
        i = 0
        for root, folder, files_in_dir in os.walk(calibration_image_folder):
            for image_name in files_in_dir:
                image_filepath = os.path.join(root, image_name)
                self.nhwc_data_list.append(image_filepath)
                i = i + 1
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)
        self.next_read = 0

    def get_next(self):
        if self.next_read == self.datasize:
            return None
        pillow_img = Image.new("RGB", (self.width, self.height))
        pillow_img.paste(Image.open(self.nhwc_data_list[self.next_read]).resize((self.width, self.height)))
        print("[{} {}]".format(self.next_read, self.nhwc_data_list[self.next_read]))
        self.next_read = self.next_read + 1
        # input_data = numpy.float32(pillow_img) - numpy.array(
        #     [123.68, 116.78, 103.94], dtype=numpy.float32
        # )
        input_data = numpy.float32(pillow_img) * 2 - 1
        nhwc_data = numpy.expand_dims(input_data, axis=0)
        return {self.input_name: nhwc_data}
        # nchw_data = nhwc_data.transpose((0, 3, 1, 2))  # ONNX Runtime standard
        # return {self.input_name: nchw_data}

    def rewind(self):
        self.next_read = 0

def _generate_aug_model_path(model_path: str) -> str:
    aug_model_path = (
        model_path[: -len(".onnx")] if model_path.endswith(".onnx") else model_path
    )
    return aug_model_path + ".save_tensors.onnx"

def convert_model_to_opset(model_path, converted_model_path, opset):
    model = onnx.load(model_path)
    while model.opset_import[0].version != opset:
        offset = 1 if model.opset_import[0].version < opset else -1
        model = version_converter.convert_version(model, model.opset_import[0].version + offset)
        # print(model.opset_import[0].version)
    onnx.save(model, converted_model_path)
    print(f"The model after conversion:\n{converted_model_path}")

def main():
    name='pd2_sim'
    model_fp32 = f'/Users/leonidbobovich/Work/ml/qualcomm-panoptic-deeplab/{name}.onnx'
    model_fp32_converted = f'/Users/leonidbobovich/Work/ml/qualcomm-panoptic-deeplab/{name}_converted.onnx'
    model_fp32_preprocessed = f'/Users/leonidbobovich/Work/ml/qualcomm-panoptic-deeplab/{name}_preprocessed.onnx'
    model_quant = f'/Users/leonidbobovich/Work/ml/qualcomm-panoptic-deeplab/{name}_quantized.onnx'
    data_reader = DataReader(
        '/Users/leonidbobovich/Work/ml/qualcomm-panoptic-deeplab/datasets/cityscapes/leftImg8bit/val', model_fp32)
    convert_model_to_opset(model_fp32, model_fp32_converted, 13)
    onnxruntime.quantization.quant_pre_process(input_model_path=model_fp32_converted, output_model_path=model_fp32_preprocessed)
    # Process input parameters and setup model input data reader
    # args = get_args()
    # float_model_path = args.float_model
    # qdq_model_path = args.qdq_model
    # calibration_dataset_path = args.calibrate_dataset

    float_model_path = model_fp32_preprocessed
    qdq_model_path = model_quant
    calibration_dataset_path = '/Users/leonidbobovich/Work/ml/qualcomm-panoptic-deeplab/datasets/cityscapes/leftImg8bit/val'

    # model_path = float_model_path
    # original_model = onnx.load(model_path)
    # print(f"The model before conversion:\n{original_model}")
    # converted_model = version_converter.convert_version(original_model, 10)
    # original_model = converted_model
    # converted_model = version_converter.convert_version(original_model, 11)
    # original_model = converted_model
    # onnx.save(original_model, model_fp32_11)
    # model_fp32 = model_fp32_11
    # print(f"The model after conversion:\n{converted_model}")

    quantized_model = quantize_static(float_model_path, model_quant, data_reader, quant_format=QuantFormat.QDQ,
          weight_type=QuantType.QUInt8,
          per_channel=False,
          activation_type=QuantType.QUInt8,
          optimize_model=True,
          calibrate_method=CalibrationMethod.MinMax, # CalibrationMethod.Entropy,
          # extra_options={'AddQDQPairToWeight': True}
          # extra_options={'QuantizeBias': False}
      )
    # quantized_model = quantize_dynamic(float_model_path, model_quant, weight_type=QuantType.QUInt8)

    print("------------------------------------------------\n")
    print("Comparing weights of float model vs qdq model.....")

    matched_weights = create_weight_matching(float_model_path, qdq_model_path)
    weights_error = compute_weight_error(matched_weights)
    for weight_name, err in weights_error.items():
        print(f"Cross model error of '{weight_name}': {err}")

    print("------------------------------------------------\n")
    print("Augmenting models to save intermediate activations......")

    aug_float_model = modify_model_output_intermediate_tensors(float_model_path)
    aug_float_model_path = _generate_aug_model_path(float_model_path)
    onnx.save(
        aug_float_model,
        aug_float_model_path,
        save_as_external_data=False,
    )
    del aug_float_model

    aug_qdq_model = modify_model_output_intermediate_tensors(qdq_model_path)
    aug_qdq_model_path = _generate_aug_model_path(qdq_model_path)
    onnx.save(
        aug_qdq_model,
        aug_qdq_model_path,
        save_as_external_data=False,
    )
    del aug_qdq_model

    print("------------------------------------------------\n")
    print("Running the augmented floating point model to collect activations......")
    input_data_reader = DataReader(
        calibration_dataset_path, float_model_path
    )
    float_activations = collect_activations(aug_float_model_path, input_data_reader)

    print("------------------------------------------------\n")
    print("Running the augmented qdq model to collect activations......")
    input_data_reader.rewind()
    qdq_activations = collect_activations(aug_qdq_model_path, input_data_reader)

    print("------------------------------------------------\n")
    print("Comparing activations of float model vs qdq model......")

    # act_matching = create_activation_matching(qdq_activations, float_activations)
    act_matching = create_activation_matching(qdq_activations, float_activations)
    act_error = compute_activation_error(act_matching)
    for act_name, err in act_error.items():
        print(f"Cross model error of '{act_name}': {err['xmodel_err']}")
        print(f"QDQ error of '{act_name}': {err['qdq_err']}")


if __name__ == "__main__":
    main()
    sys.exit()
#
#
# # float_model = onnx.load_model(model_fp32)
# quantized_model = quantize_dynamic(model_fp32, model_quant)
# matched_weights = create_weight_matching(model_fp32, model_quant)
# weights_error = compute_weight_error(matched_weights)
# for weight_name, err in weights_error.items():
#     print(f"Cross model error of '{weight_name}': {err}\n")