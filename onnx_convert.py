import torch
from models import MVCNN
import gc

gc.set_threshold(0)

svcnn = MVCNN.SVCNN("svcnn").to("cuda")
svcnn.load_state_dict(torch.load("model-00001.pth"))

dummy_input = torch.randn(1, 3, 244, 244, dtype=torch.float32).to("cuda")
onnx_model_path = 'model-00001.onnx'

torch.onnx.export(
    svcnn,
    dummy_input,
    onnx_model_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    verbose=True,
    input_names=['input'],
    output_names=['output'],
)
