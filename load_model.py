import sys

def load_model(weights_path):
    from torch import cuda as tcuda
    from torch import device as tdevice
    device = tdevice("cuda" if tcuda.is_available() else "cpu")
    sys.modules.pop('device')
    del tdevice
    
    from torch import load
    from models import MVCNN
    
    svcnn_model = MVCNN.SVCNN('svcnn').to(device)
    svcnn_weights = load(weights_path)
    
    sys.modules.pop('MVCNN')
    del MVCNN
    sys.modules.pop('load')
    del load
    
    
    
    svcnn_model.load_state_dict(svcnn_weights)
    
    return svcnn_model

if __name__ == "__main__":
    path = input("enter path: ").strip()
    svcnn_model = load_model(path)
    
    print(svcnn_model)