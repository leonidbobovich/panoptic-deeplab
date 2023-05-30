import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
import sys
import copy
import torch
import numpy
import numpy as np
import random
import importlib
from pathlib import Path
from PIL import Image
import onnx
import onnxruntime as ort
from onnxruntime.quantization import qdq_loss_debug

class DataReader:
    def __init__(self, calibration_image_folder: str, height: int, width: int, batch: int):
        self.enum_data = None
        self.height = height
        self.width = width
        self.batch = batch
        # Convert image to input data
        self.nhwc_data_list = []
        for root, folder, files_in_dir in os.walk(calibration_image_folder):
            for image_name in files_in_dir:
                image_filepath = os.path.join(root, image_name)
                self.nhwc_data_list.append(image_filepath)

        self.datasize = len(self.nhwc_data_list)
        self.next_read = 0

    def get_next(self):
        if self.next_read + self.batch >= self.datasize:
            return None
        nchw_ret = None
        for i in range(self.batch):
            pillow_img = Image.new("RGB", (self.width, self.height))
            pillow_img.paste(Image.open(self.nhwc_data_list[self.next_read]).resize((self.width, self.height)))
            #print("[{} {}]".format(self.next_read, self.nhwc_data_list[self.next_read]))
            self.next_read = self.next_read + 1
            # input_data = numpy.float32(pillow_img) - numpy.array( [123.68, 116.78, 103.94], dtype=numpy.float32 )
            input_data = numpy.float32(pillow_img) * 2 - 1
            nhwc_data = numpy.expand_dims(input_data, axis=0)
            nchw_data = nhwc_data.transpose((0, 3, 1, 2))  # ONNX Runtime standard
            nchw_ret = nchw_data if nchw_ret is None else numpy.concatenate((nchw_ret, nchw_data), axis=0)
        return torch.Tensor(nchw_ret)

    def rewind(self):
        self.next_read = 0
        random.shuffle(self.nhwc_data_list)


def check_onnx_export(model, x, tag):
    model.eval()
    print('\nEXPORTING', tag.upper(), 'TO ONNX')
    path = '{}.onnx'.format(tag)
    # torch_output = model(x) #.detach()
    # torch.onnx.export(model, x, path, verbose=True)
    torch.onnx.export(model=model, args=x, f=path, input_names=['input'], verbose=False)
    #torch.onnx.export(model, x, path, export_params=True, verbose=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    # print('CHECKING')
    # model = onnx.load(path)
    # onnx.checker.check_model(model)
    # ort_session = ort.InferenceSession(path, providers=ort.get_available_providers())
    # ort_outputs = ort_session.run(None, {'input': np.array(x).astype(np.float32)})
    # print(torch_output[0].shape, ort_outputs[0].shape)
    # np.testing.assert_allclose(np.array(torch_output[0]), ort_outputs[0], rtol=1e-03, atol=1e-05)
    print('FINISH')

filename = os.path.join(os.environ['HOME'],
                        'Work/ml/qualcomm-panoptic-deeplab/68c2c45f54765eafc5cfecd361b85997_test.py')
directory_path, file_name = os.path.split(filename)
sys.path.insert(0, directory_path)
print(Path(file_name).stem)
print(directory_path)
cd = os.curdir
os.chdir(directory_path)
network = importlib.import_module(Path(file_name).stem).network
opt_model = network()
ref_model = network()
os.chdir(cd)


device = 'mps' if torch.has_mps else 'cuda' if torch.cuda else 'cpu'
torch.backends.quantized.engine = "qnnpack"
#qconfig =  torch.quantization.default_qconfig
# qconfig = torch.quantization.get_default_qconfig(torch.backends.quantized.engine)
qconfig = torch.quantization.default_qat_qconfig_v2
#qconfig = torch.quantization.default_qat_qconfig
refer = network(False)
refer.eval()
refer = refer.to(device)

model = network(True)
model.qconfig = qconfig
model = torch.quantization.prepare(model)
model.train()
model = model.to(device)

n_epochs = 100

#optimizer = torch.optim.SGD(opt_model.parameters(), lr=1e0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-17)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, verbose=True)
# loss_fn = lambda out, tgt: torch.pow(tgt - out, 2).mean()
semantic_loss_fn = torch.nn.CrossEntropyLoss()
# loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.L1Loss()
data_reader = DataReader(
    os.path.join(os.environ['HOME'], 'Work/ml/qualcomm-panoptic-deeplab/datasets/cityscapes/leftImg8bit/train'),
    768, 1536, 2)
average_loss = 0
average_count =0
for epoch in range(n_epochs):
    data_reader.rewind()
    x = data_reader.get_next()

    while x is not None:
        x = x.to(device)
        ref = refer(x)
        out = model(x)
        loss = semantic_loss_fn(out[0], torch.argmax(ref[0],dim=1))  # + loss_fn(out[1], ref[1]) + loss_fn(out[2], ref[2])
        average_loss = average_loss + loss.detach().cpu().numpy().sum()
        average_count = average_count + 1
        print(optimizer.param_groups[0]['lr'], average_count, loss, average_loss/average_count,
              qdq_loss_debug.compute_signal_to_quantization_noice_ratio(ref[0].detach().cpu().numpy(), out[0].detach().cpu().numpy()),
              numpy.abs(ref[0].detach().cpu().numpy() - out[0].detach().cpu().numpy()).sum())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        x = data_reader.get_next()
    scheduler.step(average_loss / average_count)
    # x = torch.from_numpy(np.random.rand(1, 3, 768, 1536).astype(np.float32)).to(dtype=torch.float)
    # model_export = torch.quantization.convert(model.cpu(), remove_qconfig=False)
    # check_onnx_export(model_export, x.to('cpu'), f'opt_model_int8_{epoch}')
    # model = model.to(device)

model = model.cpu()
model.eval()
model = torch.quantization.convert(model)

x_numpy = np.random.rand(1, 3, 768, 1536).astype(np.float32)
x = torch.from_numpy(x_numpy).to(dtype=torch.float)
check_onnx_export(model, x, 'opt_model_int8_final')

sys.exit(0)



