import torch
import numpy
from models import MVCNN

svcnn = MVCNN.SVCNN("svcnn")
svcnn.load_state_dict(torch.load("model-00001.pth"))

dummy_input = torch.randn(1, 3, 244, 244).cpu()
onnx_model_path = 'model-00001.onnx'

torch.onnx.export(
    svcnn,
    dummy_input,
    onnx_model_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'}, 
        'output': {0: 'batch_size'}
    }
)