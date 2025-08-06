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

def calculate_min_pruning_for_size(target_size):
    """Calculate minimum pruning needed to meet target model size constraint."""
    # Coefficients from get_model_size
    coeffs = [507.9, -8.516, 0.04994, -7.665e-5]
    
    left, right = 0.0, 0.95  # Valid pruning range
    
    while right - left > 1e-6:
        mid = (left + right) / 2
        mid_percent = mid * 100
        size = sum(c * (mid_percent)**i for i, c in enumerate(coeffs))
        
        if size > target_size:
            left = mid
        else: 
            right = mid
    
    return right 

def distribute_pruning(min_pruning, feature_importances, device_perf, extra_budget=0.05):
    """
    Intelligently distribute pruning across models based on importance and device performance.
    """
    # Normalize importances (higher = more important to preserve)
    norm_importance = feature_importances / np.sum(feature_importances)
    
    # Combine importance and device performance
    # Less important models on slower devices should be pruned more
    prune_weights = (1 - norm_importance) * (1 + device_perf)
    prune_weights = prune_weights / np.sum(prune_weights)  # Normalize
    
    # Calculate extra pruning beyond minimum
    extra_pruning = extra_budget * np.sum(min_pruning)
    
    # Distribute extra pruning according to weights
    pruning_distribution = min_pruning.copy()
    for i in range(len(pruning_distribution)):
        additional = prune_weights[i] * extra_pruning
        pruning_distribution[i] = min(0.95, pruning_distribution[i] + additional)
    
    return pruning_distribution

class SmartSampling(Sampling):
    def __init__(self, importances, min_pruning, smart_pruning=None, device_perf=None):
        super().__init__()
        # Importance-based sampling parameters
        imp = importances / np.max(importances)
        self.alpha = 1 + (1 - imp) * 9
        self.beta = np.ones_like(self.alpha)
        
        # Store smart solutions
        self.min_pruning = min_pruning
        self.smart_pruning = smart_pruning if smart_pruning is not None else min_pruning
        self.device_perf = device_perf
        
    def _do(self, problem, n_samples, **kwargs): # type: ignore
        X = np.zeros((n_samples, problem.n_var))
        
        # First solution is smart pruning
        X[0] = self.smart_pruning
        
        # Second solution is minimum pruning
        if n_samples > 1:
            X[1] = self.min_pruning
        
        # Third solution is device-optimized pruning (more pruning for slower devices)
        if n_samples > 2 and self.device_perf is not None:
            device_weights = self.device_perf / np.sum(self.device_perf)
            X[2] = self.min_pruning * (1 + device_weights * 0.2)
            X[2] = np.clip(X[2], self.min_pruning, 0.95)
        if n_samples > 3:
            X[3] = np.clip(self.smart_pruning * 1.5, self.min_pruning, 0.95)
        
        # Rest are importance-based random samples, but respect minimum pruning
        for i in range(3, n_samples):
            for j in range(problem.n_var):
                raw_val = np.random.beta(self.alpha[j], self.beta[j])
                # Ensure it's at least min_pruning for this dimension
                X[i, j] = max(self.min_pruning[j], raw_val)
        
        return X

class MultiViewProblem(Problem):
    def __init__(self, min_pruning=None):
        # Set lower bounds to minimum pruning if provided
        xl = min_pruning if min_pruning is not None else np.zeros(12)
        super().__init__(
            n_var=12,
            n_obj=2,
            n_constr=2,
            xl=xl,
            xu=np.ones(12)*0.95,
        )
    
    def _evaluate(self, X, out):
        # X shape: (pop_size, 12)
        n = X.shape[0]
        f1 = np.zeros(n)   # inference time
        f2 = np.zeros(n)   # accuracy 
        g1 = np.zeros(n)   # accuracy constraint
        g2 = np.zeros(n)   # model size constraint
        
        for i in range(n):
            x = X[i]
            t = model_stats.get_inf_time(x, DEVICE_PERF)
            acc = model_stats.get_model_accuracy(x)[0]
            size = model_stats.get_model_size(x)
            
            f1[i] = np.max(t)    # Max inference time
            f2[i] = abs((acc*100)-(GLOBAL_MIN_ACCURACY[0]*1.01))*100 if acc>=GLOBAL_MIN_ACCURACY[0]*1.01 else abs((acc*100)-(GLOBAL_MIN_ACCURACY[0]*1.01))*150  # type: ignore

            g1[i] = GLOBAL_MIN_ACCURACY[0]*100 - acc*100  # type: ignore # Accuracy constraint
            g2[i] = max(size/MAX_MODEL_SIZES-1.0)         # type: ignore # Model size constraint

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])

def get_fallback_solution(min_pruning, feature_importances, device_perf):
    """Create fallback solutions with different strategies if optimization fails."""
    solutions = []
    
    # Strategy 1: Minimum pruning plus small increments on less important features
    s1 = distribute_pruning(min_pruning, feature_importances, device_perf, 0.02)
    solutions.append(s1)
    
    # Strategy 2: Conservative pruning based purely on feature importance
    s2 = min_pruning.copy()
    norm_imp = feature_importances / np.sum(feature_importances)
    for i in range(len(s2)):
        s2[i] = min_pruning[i] * (1 + (1-norm_imp[i])*0.1)  # Prune less important features more
    solutions.append(np.clip(s2, min_pruning, 0.95))
    
    # Strategy 3: Device performance focused
    s3 = min_pruning.copy()
    norm_perf = device_perf / np.sum(device_perf)
    for i in range(len(s3)):
        s3[i] = min_pruning[i] * (1 + norm_perf[i]*0.15)  # Prune slower devices more
    solutions.append(np.clip(s3, min_pruning, 0.95))
    
    # Find best solution by accuracy
    best_acc = -1  # noqa: F841
    best_sol = min_pruning
    target_accuracy = GLOBAL_MIN_ACCURACY[0] * 1.01 # type: ignore
    
    for sol in solutions:
        acc = model_stats.get_model_accuracy(sol)[0]
        sizes = model_stats.get_model_size(sol)
        
        # Check if size constraints are satisfied
        if all(sizes[i] <= MAX_MODEL_SIZES[i] for i in range(len(sizes))): # type: ignore
            # Check if accuracy is above minimum
            if acc >= GLOBAL_MIN_ACCURACY[0]: # type: ignore
                # Find solution closest to target accuracy
                proximity = abs(acc - target_accuracy)
                if proximity < best_proximity: # type: ignore  # noqa: F821
                    best_proximity = proximity  # noqa: F841
                    best_sol = sol
    
    return best_sol

def init():
    global MAX_MODEL_SIZES
    global GLOBAL_MIN_ACCURACY
    global DEVICE_PERF
    global p_min
    global model_stats
    global rewards
    
    MAX_MODEL_SIZES     = [100]*12
    GLOBAL_MIN_ACCURACY = 1.0
    DEVICE_PERF         = np.ones(12)
    p_min               = np.zeros(12)

    # Initialize model stats and rewards
    model_stats = ModelStats()
    rewards = Rewards()
    
    uniform_sample = np.random.uniform(1.5, 5, 12)
    
    max_sizes = [MAX_MODEL_SIZES[i] * uniform_sample[i] for i in range(12)]
    MAX_MODEL_SIZES = np.array(max_sizes)
    max_possible_acc = model_stats.get_model_accuracy(np.zeros(12))
    
    # More aggressive accuracy requirement (higher alpha in beta distribution)
    GLOBAL_MIN_ACCURACY = np.round(np.random.default_rng().beta(9.0, 0.8, size=1)*max_possible_acc[0], 2)

    DEVICE_PERF = np.random.uniform(0.0, 0.99, size=12)  # Simulated device performance degradation

    if max(max_possible_acc) < GLOBAL_MIN_ACCURACY[0]:
        raise ValueError(f"Maximum possible accuracy is lower than the global minimum accuracy requirement. Cannot optimize. Max possible acc: {max(max_possible_acc)*100:.2f}% Global min acc: {GLOBAL_MIN_ACCURACY[0]*100:.2f}%")
    
    # Calculate minimum pruning for each model to meet size constraints
    min_pruning = np.array([calculate_min_pruning_for_size(MAX_MODEL_SIZES[i]) for i in range(12)])
    
    # Check if minimum pruning is already feasible
    min_acc = model_stats.get_model_accuracy(min_pruning)[0]
    print(f"Minimum pruning for size constraints would yield accuracy: {min_acc*100:.2f}%")
    print(f"Required minimum accuracy: {GLOBAL_MIN_ACCURACY[0]*100:.2f}%")
    
    smart_pruning = distribute_pruning(min_pruning, I, DEVICE_PERF)
    
    # Use minimum pruning as lower bounds for optimization
    problem = MultiViewProblem(min_pruning)
    
    # Create specialized sampling with smart distributions
    sampling = SmartSampling(I, min_pruning, smart_pruning, DEVICE_PERF)

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
    
    # Check if we found feasible solutions
    feasible_mask = np.all(res.G <= 0, axis=1) # type: ignore
    if not np.any(feasible_mask):
        print("WARNING: No feasible solutions found. Using fallback solution.")
        P = get_fallback_solution(min_pruning, I, DEVICE_PERF)
    else:
        # Select solution from Pareto front
        feasible_indices = np.where(feasible_mask)[0]
        feasible_F = res.F[feasible_indices] # type: ignore
        feasible_X = res.X[feasible_indices] # type: ignore
        
        # Apply weights for decision making
        weights = [0.8, 0.2]  # Inference time vs accuracy
        dm = PseudoWeights(weights)
        
        # Apply decision making to feasible solutions only
        idx_in_feasible = dm.do(feasible_F)
        P = feasible_X[idx_in_feasible]

    print(f"Pareto-optimal prune vectors (X): {P}")
    print(f"Accuracy: {model_stats.get_model_accuracy(P)[0]:.2f}")
    
    # Tabulate results
    t = model_stats.get_inf_time(P, DEVICE_PERF)
    sizes = model_stats.get_model_size(P)
    acc = model_stats.get_model_accuracy(P)[0]
    accuracy_violated = True if np.round(acc*100, 1) < np.round(GLOBAL_MIN_ACCURACY[0]*100, 1) else False

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
        tablefmt="csv"
    ))
    print("\n")
    print(f"Required minimum accuracy: {GLOBAL_MIN_ACCURACY[0]*100:.2f}%")
    print(f"Current accuracy: {acc*100:.2f}%")
    
    if accuracy_violated:
        print('Model does not meet minimum accuracy requirement.')
    else:
        print('Model meets minimum accuracy requirement.')
        
    inf_time = np.max(model_stats.get_inf_time(min(P), min(DEVICE_PERF)))
    org_time = np.max(model_stats.get_inf_time(np.zeros(12), min(DEVICE_PERF)))

    print(f"Original inference time: {org_time:.4f}ms")
    print(f"Optimized inference time: {inf_time:.4f}ms")
    print(f"{(org_time)/(inf_time):.2f}x speedup in inference time")

    model_size_violated = any(np.round(sizes[i],0) > np.round(MAX_MODEL_SIZES[i],0) for i in range(len(sizes))) # type: ignore
    
    return (accuracy_violated, model_size_violated)

if __name__ == "__main__":
    iters = int(input("Enter number of iterations: "))
    failed = 0
    k=0
    i = 0
    while i < iters:
        print("-" * 50)
        print(f"\nIteration {i+1}/{iters}")
        try:
            a_v, s_v = init()
            if a_v or s_v:
                failed += 1
                print(f"Iteration {i+1} failed. Total failures: {failed}")
            else:
                print(f"Iteration {i+1} succeeded.")
                i+=1
        except Exception as e:
            k += 1
            print(f"Iteration {i+1} failed with error: {e}")
            print(f"Total error failures: {k}")
    print(f"\nTotal iterations: {iters}, Failed: {failed}, Errors: {k}")