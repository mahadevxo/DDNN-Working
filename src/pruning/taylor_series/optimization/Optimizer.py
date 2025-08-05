import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from ModelStatsRewards import ModelStats, Rewards
from tabulate import tabulate
from pymoo.mcdm.pseudo_weights import PseudoWeights

# Initialize stats & rewards
model_stats = ModelStats()
rewards     = Rewards()

# Feature importances (length 12)
I = model_stats.xgbmodel.feature_importances_  # noqa: E741

# Globals & hyperparameters
MAX_MODEL_SIZES     = [100]*12
GLOBAL_MIN_ACCURACY = 1.0
DEVICE_PERF         = np.ones(12)
POP_SIZE            = 400
X_TOL               = 1e-6
p_min               = np.zeros(12)
N_GENERATIONS       = 200

class ImportanceSampling(Sampling):
    def __init__(self, importances):
        super().__init__()
        imp = importances / np.max(importances)
        self.alpha = 1 + (1 - imp) * 9
        self.beta  = np.ones_like(self.alpha)

    def _do(self, problem, n_samples, **kwargs): # type: ignore
        X = np.zeros((n_samples, problem.n_var))
        for j in range(problem.n_var):
            X[:, j] = np.random.beta(self.alpha[j], self.beta[j], size=n_samples)
        return X

class MultiViewProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=12,
            n_obj=2,
            n_constr=2,
            xl=np.zeros(12),
            xu=np.ones(12)*0.95,
        )
    
    def _evaluate(self, X, out):
        # X shape: (pop_size, 12)
        n = X.shape[0]
        f1 = np.zeros(n)   # inference time
        f2 = np.zeros(n)   # accuracy 
        f3 = np.zeros(n)   # negative reward
        g1 = np.zeros(n)   # accuracy  constraint
        g2 = np.zeros(n)   # model size constraint
        
        for i in range(n):
            x = X[i]
            t   = model_stats.get_inf_time(x, DEVICE_PERF)
            acc = model_stats.get_model_accuracy(x)[0]
            size = model_stats.get_model_size(x)
            reward = rewards.get_reward(x, min_accuracy=GLOBAL_MIN_ACCURACY, max_model_size=MAX_MODEL_SIZES)
            
            f1[i] = np.max(t)    # Max inference time
            f2[i] = -acc*100         # Model accuracy
            f3[i] = -reward      # Negative reward
            
            g1[i] = GLOBAL_MIN_ACCURACY[0] - acc*100     # type: ignore # Accuracy constraint
            g2[i] = max(size/MAX_MODEL_SIZES-1.0)  # type: ignore # Model size constraint

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])

def init():
    global MAX_MODEL_SIZES
    global GLOBAL_MIN_ACCURACY
    global DEVICE_PERF
    global p_min
    global model_stats
    global rewards

    # Initialize model stats and rewards
    model_stats = ModelStats()
    rewards = Rewards()
    
    uniform_sample = np.random.uniform(1.5, 5, 12)
    
    max_sizes = [MAX_MODEL_SIZES[i] * uniform_sample[i] for i in range(12)]
    MAX_MODEL_SIZES = np.array(max_sizes)
    max_possible_acc = model_stats.get_model_accuracy(np.zeros(12))
    
    # More aggressive accuracy requirement (higher alpha in beta distribution)
    GLOBAL_MIN_ACCURACY = np.random.default_rng().beta(9.0, 0.8, size=1)*max_possible_acc[0]
    
    DEVICE_PERF = np.random.uniform(0.0, 0.99, size=12)  # Simulated device performance degradation
    
    
    problem = MultiViewProblem()
    sampling = ImportanceSampling(I)

    algorithm = NSGA2(
        pop_size=POP_SIZE, 
        sampling=sampling, # type: ignore
        )
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', N_GENERATIONS),
        verbose=False,
    )
    weights = [0.4, 0.6] 
    dm = PseudoWeights(weights)
    idx_arr = dm.do(res.F)
    idx = int(idx_arr)          # type: ignore
    P = res.X[idx] # type: ignore

    print(f"Pareto-optimal prune vectors (X): {P}") # type: ignore
    print(f"Accuracy: {model_stats.get_model_accuracy(P)[0]:.2f}") # type: ignore
    #tabulate results as a table: View, Pruning Amount, Size of Model, Max Size, Inference Time, Violated Accuracy
    t      = model_stats.get_inf_time(P, DEVICE_PERF)
    sizes  = model_stats.get_model_size(P)
    acc    = model_stats.get_model_accuracy(P)[0]
    violated = True if np.round(acc*100, 1) < np.round(GLOBAL_MIN_ACCURACY, 1) else False

    table_data = []
    for i in range(len(P)):
        table_data.append([
            i,
            f"{P[i]:.3f}",
            f"{sizes[i]:.2f}MB", # type: ignore
            f"{MAX_MODEL_SIZES[i]:.2f}MB",
            f"{t[i]:.4f}ms",
            "VIOLATED" if sizes[i] > MAX_MODEL_SIZES[i] else "No", # type: ignore
            f"{DEVICE_PERF[i]:.2f}"
        ])

    print("\nPer-view summary:")
    print(tabulate(
        table_data,
        headers=["View", "Pruning Amount", "Size of Model", "Max Size", "Inference Time", "Violated Model Size", "Device Performance"],
        tablefmt="github"
    ))
    print("\n")
    print(f"Required minimum accuracy: {GLOBAL_MIN_ACCURACY[0]*100:.2f}")
    print(f"Current accuracy: {acc*100:.2f}")
    if violated:
        print('Model does not meet minimum accuracy requirement.')
    else:
        print('Model meets minimum accuracy requirement.')
    inf_time = np.max(t)
    org_time = np.max(model_stats.get_inf_time(np.zeros(12), DEVICE_PERF))
    
    print(f"Original inference time: {org_time:.4f}ms")
    print(f"Optimized inference time: {inf_time:.4f}")

    print(f"{(org_time)/(inf_time):.2f}x speedup in inference time")


if __name__ == "__main__":
    init()