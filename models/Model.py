import torch.nn as nn
import os
import glob
import sys


class Model(nn.Module):

    def __init__(self, name):
        super(Model, self).__init__()
        self.name = name


    def save(self, path, epoch=0):
        from torch import save as tsave
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        tsave(self.state_dict(), 
                os.path.join(complete_path, 
                    "model-{}.pth".format(str(epoch).zfill(5))))

        sys.modules.pop('save')
        del save


    def save_results(self, path, data):
        raise NotImplementedError("Model subclass must implement this method.")
        

    def load(self, path, modelfile=None):
        from torch import load as tload
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            raise IOError("{} directory does not exist in {}".format(self.name, path))

        if modelfile is None:
            model_files = glob.glob(complete_path+"/*")
            mf = max(model_files)
        else:
            mf = os.path.join(complete_path, modelfile)

        self.load_state_dict(tload(mf))
        
        sys.modules.pop('load')
        del load