import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from ModelStatsRewards import ModelStats, Rewards
from tabulate import tabulate
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.algorithms.soo.nonconvex.ga import GA

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

def calculate_min_pruning_newton_newton_newton(target_size):
    """
    Calculate minimum pruning using Newton's Method for faster convergence.
    """
    # Coefficients from get_model_size
    coeffs = [507.9, -8.516, 0.04994, -7.665e-5]

    # The function f(p) = size(p) - target_size
    def f(p):
        p_percent = p * 100
        size = sum(c * (p_percent)**i for i, c in enumerate(coeffs))
        return size - target_size

    # The derivative of the function, f'(p)
    def f_prime(p):
        p_percent = p * 100
        # Derivative: (c1 + 2*c2*x + 3*c3*x^2) * dx/dp, where x = 100p
        derivative = coeffs[1] + 2*coeffs[2]*p_percent + 3*coeffs[3]*(p_percent**2)
        return derivative * 100

    # Start with an initial guess (e.g., 0.5)
    p_current = 0.5 
    
    # Iterate a fixed number of times or until convergence
    for _ in range(10): # 10 iterations is usually more than enough
        fp = f(p_current)
        f_prime_p = f_prime(p_current)

        # Avoid division by zero
        if abs(f_prime_p) < 1e-9:
            break 
            
        # Newton's Method update step
        p_next = p_current - fp / f_prime_p
        
        # Stop if the change is negligible
        if abs(p_next - p_current) < 1e-7:
            p_current = p_next
            break
            
        p_current = p_next

    # Ensure the result is within the valid [0, 0.99] range
    return np.clip(p_current, 0.0, 0.99)

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
        pruning_distribution[i] = min(0.99, pruning_distribution[i] + additional)
    
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
        
        if n_samples > 3:
            X[3] = np.clip(self.smart_pruning * 1.5, self.min_pruning, 0.99)
            
        # Third solution is device-optimized pruning (more pruning for slower devices)
        if n_samples > 2 and self.device_perf is not None:
            device_weights = self.device_perf / np.sum(self.device_perf)
            X[2] = self.min_pruning * (1 + device_weights * 0.2)
            X[2] = np.clip(X[2], self.min_pruning, 0.99)
        
        
        # Rest are importance-based random samples, but respect minimum pruning
        for i in range(4, n_samples):
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
            n_obj=3,
            n_constr=2,
            xl=xl,
            xu=np.ones(12)*0.99,
        )
    
    def _evaluate(self, X, out):
        # X shape: (pop_size, 12)
        n = X.shape[0]
        f1 = np.zeros(n)   # inference time
        f2 = np.zeros(n)   # accuracy 
        f3 = np.zeros(n)   # reward
        g1 = np.zeros(n)   # accuracy constraint
        g2 = np.zeros(n)   # model size constraint
        
        for i in range(n):
            x = X[i]
            t = model_stats.get_inf_time(x, DEVICE_PERF)
            acc = model_stats.get_model_accuracy(x)[0]
            size = model_stats.get_model_size(x)
            
            f1[i] = np.max(t)    # Max inference time
            
            if acc >= GLOBAL_MIN_ACCURACY[0]+1: # type: ignore
                f2[i] = abs((acc*100)-(GLOBAL_MIN_ACCURACY[0]*100))*100 # type: ignore
            else:
                f2[i] = abs((acc*100)-(GLOBAL_MIN_ACCURACY[0]*100))*150 # type: ignore
                
            f3[i] = rewards.get_reward(x, GLOBAL_MIN_ACCURACY[0], MAX_MODEL_SIZES)# type: ignore

            g1[i] = (GLOBAL_MIN_ACCURACY[0] - acc)*100  # type: ignore # Accuracy constraint
            g2[i] = np.max(size / MAX_MODEL_SIZES) - 1.0 # type: ignore

        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1, g2])

class FallBackProblem(Problem):
    def __init__(self, min_pruning=None):
        super().__init__(
            n_var=12,
            n_obj=1, #handle only accuracy
            n_constr=0,
            xl=min_pruning if min_pruning is not None else np.zeros(12),
            xu=np.ones(12)*0.99,
        )
    def _evaluate(self, X, out):
        all_violations = []
        for x in X:
            acc = model_stats.get_model_accuracy(x)[0]
            sizes = model_stats.get_model_size(x)
            
            acc_violation = max(0, GLOBAL_MIN_ACCURACY[0]*100 - acc*100) * 1000  # type: ignore
            size_violation = sum(max(0, sizes[i] - MAX_MODEL_SIZES[i]) for i in range(len(sizes))) # type: ignore
            
            total_violation = acc_violation + size_violation
            all_violations.append(total_violation)
            
        out["F"] = np.array(all_violations).reshape(-1, 1)

def fallback_GA_Optimization(min_pruning, feature_importances, device_perf):
    print("No feasible solutions found by NSGA-II. Using GA solution.")
    fallback_problem = FallBackProblem(min_pruning=min_pruning)
    algorithm = GA(
        pop_size=POP_SIZE,
        sampling=SmartSampling(
            importances=feature_importances,
            min_pruning=min_pruning,
            smart_pruning=None,
            device_perf=device_perf,
            ), # type: ignore
        eliminate_duplicates=True,
    )
    
    res = minimize(
        fallback_problem,
        algorithm,
        termination=('n_gen', N_GENERATIONS),
        verbose=False,
        tol=X_TOL, # type: ignore
    )
    
    best_solution = res.X[0] # type: ignore
    min_violation_score = float(res.F[0]) # type: ignore

    best = None
    
    if min_violation_score < 1e-4:
        print("Fallback solution found with acceptable violations.")
        best = best_solution
    else:
        print("Fallback solution still violates constraints. Using minimum pruning.")
        best = min_pruning
    
    sizes = model_stats.get_model_size(best)
    headroom = (MAX_MODEL_SIZES - sizes)/MAX_MODEL_SIZES # type: ignore
    extra_budget = np.sum(np.clip(headroom, 0, 1)) * 0.05
    
    return distribute_pruning(best, feature_importances, device_perf, extra_budget=extra_budget) # type: ignore

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
    
    uniform_sample = np.random.uniform(1.0, 4.5, 12)
    
    max_sizes = [MAX_MODEL_SIZES[i] * uniform_sample[i] for i in range(12)]
    MAX_MODEL_SIZES = np.array(max_sizes)
    
    max_possible_acc = model_stats.get_model_accuracy(np.zeros(12))
    GLOBAL_MIN_ACCURACY = np.round(np.random.default_rng().beta(9.0, 2.0, size=1)*max_possible_acc[0], 1)

    DEVICE_PERF = np.random.uniform(0.0, 0.99, size=12)  # Simulated device performance degradation

    if max(max_possible_acc) < GLOBAL_MIN_ACCURACY[0]:
        raise ValueError(f"Maximum possible accuracy is lower than the global minimum accuracy requirement. Cannot optimize. Max possible acc: {max(max_possible_acc)*100:.2f}% Global min acc: {GLOBAL_MIN_ACCURACY[0]*100:.2f}%")
    
    # Calculate minimum pruning for each model to meet size constraints
    min_pruning = np.array([calculate_min_pruning_newton_newton_newton(MAX_MODEL_SIZES[i]) for i in range(12)])

    # Check if minimum pruning is already feasible
    min_acc = model_stats.get_model_accuracy(min_pruning)[0]
    print(f"Minimum pruning for size constraints would yield accuracy: {min_acc*100:.2f}%")
    print(f"Required minimum accuracy: {GLOBAL_MIN_ACCURACY[0]*100:.2f}%")
    print(f"Max Size vector: {MAX_MODEL_SIZES}")
    
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
        tol = X_TOL
    )
    
    # Check if we found feasible solutions
    if res.G is None or res.X is None or res.F is None:
        print("WARNING: Optimization failed to converge. Using fallback solution.")
        P = fallback_GA_Optimization(min_pruning, I, DEVICE_PERF)
    else:
        # Check if we found feasible solutions
        feasible_mask = np.all(res.G <= 0, axis=1)
        if not np.any(feasible_mask):
            print("WARNING: No feasible solutions found. Using fallback solution.")
            P = fallback_GA_Optimization(min_pruning, I, DEVICE_PERF)
        else:
            # Select solution from Pareto front
            feasible_indices = np.where(feasible_mask)[0]
            feasible_F = res.F[feasible_indices]
            feasible_X = res.X[feasible_indices]
            
            # Apply weights for decision making
            weights = [0.33, 0.33, 0.34]  # Inference time vs accuracy
            dm = PseudoWeights(weights)
            
            # Apply decision making to feasible solutions only
            idx_in_feasible = dm.do(feasible_F)
            P = feasible_X[idx_in_feasible]

    # print(f"Pareto-optimal prune vectors (X): {P}")    
    # Tabulate results
    t = model_stats.get_inf_time(P, DEVICE_PERF)
    sizes = model_stats.get_model_size(P)
    acc = model_stats.get_model_accuracy(P)[0]
    accuracy_violated = True if np.round(acc*100, 1) < np.round(GLOBAL_MIN_ACCURACY[0]*100, 1) else False

    table_data = []
    for i in range(len(P)): # type: ignore
        table_data.append([
            i,
            f",{P[i]:.3f}", # type: ignore
            f",{sizes[i]:.2f}MB", # type: ignore
            f",{MAX_MODEL_SIZES[i]:.2f}MB",
            f",{t[i]:.4f}ms",
            ",VIOLATED" if np.round(sizes[i], 0) > np.round(MAX_MODEL_SIZES[i], 0) else ",No", # type: ignore
            f",{DEVICE_PERF[i]:.2f}"
            f",{I[i]:.3f}" # type: ignore
        ])

    print("\nPer-view summary:")
    print(tabulate(
        table_data,
        headers=["View", "Pruning Amount", "Size of Model", "Max Size", "Inference Time", "Violated Model Size", "Device Performance", "Importance"],
        tablefmt="csv"
    ))
    print("\n")
    print(f"Required minimum accuracy: {GLOBAL_MIN_ACCURACY[0]*100:.2f}%")
    print(f"Current accuracy: {acc*100:.2f}%")
    
    if accuracy_violated:
        print('Model does not meet minimum accuracy requirement.')
    else:
        print('Model meets minimum accuracy requirement.')
        
    inf_time = np.max(model_stats.get_inf_time(P, DEVICE_PERF))
    org_time = np.max(model_stats.get_inf_time(np.zeros_like(P), DEVICE_PERF))

    print(f"Original inference time: {org_time:.4f}ms")
    print(f"Optimized inference time: {inf_time:.4f}ms")
    print(f"{(org_time)/(inf_time):.2f}x speedup in inference time")

    model_size_violated = any(np.round(sizes[i],0) > np.round(MAX_MODEL_SIZES[i],0) for i in range(len(sizes))) # type: ignore
    
    return (accuracy_violated, model_size_violated)