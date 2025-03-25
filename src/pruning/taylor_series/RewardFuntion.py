class RewardFuntion:
    def __init__(self, INITIAL_ACC: float, ACC_IMP: float, 
                 COMP_IMP: float, MODEL_SIZE_IMP: float):
        self.X = ACC_IMP
        self.Y = COMP_IMP
        self.Z = MODEL_SIZE_IMP
        self.accuracy_min = INITIAL_ACC
        
        if self.X + self.Y + self.Z != 1:
            raise ValueError("Sum of all weights should be 1")
    
    def _accuracy_reward(self, accuracy_new: float) -> float:
        
        if accuracy_new > (self.accuracy_min - 10**-3) and accuracy_new < (self.accuracy_min + 10**-3):
            return 10000.0
        elif accuracy_new > (self.accuracy_min + 10**-3):
            return (accuracy_new - self.accuracy_min)*100
        else:
            return (self.accuracy_min - accuracy_new)*-10000.0
    
    def _model_size_reward(self, model_size_new: float, model_size_previous: float) -> float:
        
        delta_model_size = model_size_new - model_size_previous
        if delta_model_size < 0:
            return 1000 * delta_model_size
        else:
            return 3000 * delta_model_size
    
    def _comp_time_reward(self, comp_time_new: float, comp_time_previous: float) -> float:
        
        delta_comp_time = comp_time_new - comp_time_previous
        if delta_comp_time < 0:
            return 1000 * delta_comp_time
        else:
            return 3000 * delta_comp_time
    
    def get_reward(self, acc_new: float, model_size_new: float, model_size_previous: float, 
                   comp_time_new: float, comp_time_previous: float) -> float:
        
        """Calculate the reward based on accuracy, model size, and computation time.

        This function calculates a weighted reward based on the new accuracy, model size, and
        computation time compared to the previous model size and computation time. The weights
        for accuracy, model size, and computation time are determined by the `ACC_IMP`,
        `MODEL_SIZE_IMP`, and `COMP_IMP` parameters passed to the constructor, respectively.

        Args:
            acc_new: The new accuracy of the model.
            model_size_new: The new size of the model.
            model_size_previous: The previous size of the model.
            comp_time_new: The new computation time of the model.
            comp_time_previous: The previous computation time of the model.

        Returns:
            The calculated reward.
        """
        
        acc_reward = self._accuracy_reward(acc_new)
        model_size_reward = self._model_size_reward(model_size_new, model_size_previous)
        comp_time_reward = self._comp_time_reward(comp_time_new, comp_time_previous)
        
        return self.X*acc_reward + self.Y*model_size_reward + self.Z*comp_time_reward