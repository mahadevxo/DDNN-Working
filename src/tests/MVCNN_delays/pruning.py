class DynamicPruning:
    def __init__(self, model, accuracy_threshold, param_target):
        self.model = model
        self.accuracy_threshold = accuracy_threshold
        self.param_target = param_target
        self.sparsity = 0
        self.lookup_table = {}
        
class CNNModel:
    def __init__(self):
        params = 0
        for layer in CNN.