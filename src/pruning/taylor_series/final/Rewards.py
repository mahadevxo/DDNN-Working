import torch
from MVCNN_Trainer import MVCNN_Trainer
from LearnData import LearnData

class Reward:
    def __init__(self, min_acc, min_size):
        self.min_acc = min_acc
        self.min_size = min_size
        self.reward = 0
    def _get_accuracy_reward(self, accuracy_new):
        if accuracy_new > (self.min_acc - 10**-3) and accuracy_new < (self.min_acc + 10**-3):
            return 500.0
        elif accuracy_new > (self.min_acc + 10**-3):
            return (accuracy_new - self.min_acc)*200
        else:
            return (self.min_acc - accuracy_new)*-1000.0

    def _get_model_size_reward(self, model_size_new):
        delta_model_size = model_size_new - self.min_size
        if delta_model_size < 0:
            return 600*delta_model_size*-1
        else:
            return 1000*delta_model_size # Penalty

    def _get_comp_time_reward(self, comp_time_new,  comp_time_last):
        delta_comp_time = comp_time_new - comp_time_last    
        return 100*delta_comp_time if delta_comp_time < 0 else 300 * delta_comp_time

    def getReward(self, accuracy,  model_size,  comp_time,  comp_time_last,  x=0.33,  y=0.33,  z=0.33):
        
        if not (0.99 <= x + y + z <= 1.01):
            raise ValueError("x, y, and z must sum to 1 or 0")
        
        accuracy_reward = self._get_accuracy_reward(accuracy)
        model_size_reward = self._get_model_size_reward(model_size)
        comp_time_reward = self._get_comp_time_reward(comp_time, comp_time_last)
        return (x*accuracy_reward) + (y*model_size_reward) + (z*comp_time_reward)

class GetResults:
    def __init__(self):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.learndata = LearnData()
        self.learndata.load_state_dict(torch.load('learn_data.pth', weights_only=False, map_location=self.device))
        self.learndata.to(self.device)
        
    def get_results(self, model):
        model.to(self.device)
        model.eval()
        mvcnntrainer = MVCNN_Trainer(optimizer=torch.optim.Adam(model.parameters(), lr=0.001))
        acctime = mvcnntrainer.get_val_accuracy(model)
        acc, time = acctime[1], acctime[3]
        total_size = sum(
            param.nelement() * param.element_size() for param in model.parameters()
        )
        total_size = total_size / (1024 ** 2)
        
        return [acc, time, total_size]
    
    def get_approx_acc(self, pruning_amount, pre_pruning_acc, initial_acc):
        self.learndata.eval()

        with torch.no_grad():
            x = torch.tensor([pruning_amount, pre_pruning_acc, initial_acc])
            x = x.unsqueeze(0).to(self.device)
            return self.learndata(x)
        
