import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from ModelStatsRewards import ModelStats, Rewards
from tabulate import tabulate
from pymoo.algorithms.soo.nonconvex.ga import GA

model_stats = ModelStats()
rewards = Rewards()

I = model_stats.xgbmodel.feature_importances_  # noqa: E741

MAX_MODEL_SIZES = [100] * 12
GLOBAL_MIN_ACCURACY = 1.0
DEVICE_PERF = np.ones(12)
POP_SIZE = 400
X_TOL = 1e-6
p_min = np.zeros(12)
N_GENERATIONS = 200

def calculate_min_pruning_newton_newton_newton(target_size):
    coeffs = [507.9, -8.516, 0.04994, -7.665e-5]

    def f(p):
        p_percent = p * 100
        size = sum(c * (p_percent) ** i for i, c in enumerate(coeffs))
        return size - target_size

    def f_prime(p):
        p_percent = p * 100
        derivative = coeffs[1] + 2 * coeffs[2] * p_percent + 3 * coeffs[3] * (p_percent ** 2)
        return derivative * 100

    p_current = 0.5

    for _ in range(10):
        fp = f(p_current)
        f_prime_p = f_prime(p_current)

        if abs(f_prime_p) < 1e-9:
            break

        p_next = p_current - fp / f_prime_p

        if abs(p_next - p_current) < 1e-7:
            p_current = p_next
            break

        p_current = p_next

    return np.clip(p_current, 0.0, 0.99)

def distribute_pruning(min_pruning, feature_importances, device_perf, extra_budget=0.5):
    norm_importance = feature_importances / np.sum(feature_importances)
    prune_weights = (1 - norm_importance) * (1 + device_perf)
    prune_weights = prune_weights / np.sum(prune_weights)

    extra_pruning = extra_budget * np.sum(min_pruning)

    pruning_distribution = min_pruning.copy()
    for i in range(len(pruning_distribution)):
        additional = prune_weights[i] * extra_pruning
        pruning_distribution[i] = min(0.99, pruning_distribution[i] + additional)

    return pruning_distribution

class SmartSampling(Sampling):
    def __init__(self, importances, min_pruning, smart_pruning=None, device_perf=None):
        super().__init__()

        imp = importances / np.max(importances)
        self.alpha = 1 + (1 - imp) * 9
        self.beta = np.ones_like(self.alpha)

        self.min_pruning = min_pruning
        self.smart_pruning = smart_pruning if smart_pruning is not None else min_pruning
        self.device_perf = device_perf

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var))

        X[0] = self.smart_pruning

        if n_samples > 1:
            X[1] = self.min_pruning

        if n_samples > 3:
            X[3] = np.clip(self.smart_pruning * 1.5, self.min_pruning, 0.99)

        if n_samples > 2 and self.device_perf is not None:
            device_weights = self.device_perf / np.sum(self.device_perf)
            X[2] = self.min_pruning * (1 + device_weights * 0.2)
            X[2] = np.clip(X[2], self.min_pruning, 0.99)

        for i in range(4, n_samples):
            for j in range(problem.n_var):
                raw_val = np.random.beta(self.alpha[j], self.beta[j])
                X[i, j] = max(self.min_pruning[j], raw_val)

        return X

class MultiViewProblem(Problem):
    def __init__(self, min_pruning=None):
        xl = min_pruning if min_pruning is not None else np.zeros(12)
        super().__init__(
            n_var=12,
            n_obj=3,
            n_constr=2,
            xl=xl,
            xu=np.ones(12) * 0.99,
        )

    def _evaluate(self, X, out):
        n = X.shape[0]
        f1 = np.zeros(n)
        f2 = np.zeros(n)
        f3 = np.zeros(n)
        g1 = np.zeros(n)
        g2 = np.zeros(n)

        baseline_time = np.max(model_stats.get_inf_time(np.zeros_like(X[0]), DEVICE_PERF))
        max_reward = max(rewards.get_reward(x, GLOBAL_MIN_ACCURACY, MAX_MODEL_SIZES) for x in X) or 1.0

        for i in range(n):
            x = X[i]
            t = model_stats.get_inf_time(x, DEVICE_PERF)
            acc = model_stats.get_model_accuracy(x)[0]
            size = model_stats.get_model_size(x)

            f1[i] = np.max(t) / baseline_time
            diff = acc - GLOBAL_MIN_ACCURACY
            if diff > 0:
                f2[i] = diff * 100 * 10.0 
            else:
                f2[i] = abs(diff) * 100 * 8.0
            f3[i] = -1 * rewards.get_reward(x, GLOBAL_MIN_ACCURACY, MAX_MODEL_SIZES) / max_reward
            g1[i] = (GLOBAL_MIN_ACCURACY - acc) * 100
            g2[i] = np.max(size / MAX_MODEL_SIZES) - 1.0

        out['F'] = np.column_stack([f1, f2, f3])
        out['G'] = np.column_stack([g1, g2])

class FallBackProblem(Problem):
    def __init__(self, min_pruning=None):
        super().__init__(
            n_var=12,
            n_obj=1,
            n_constr=0,
            xl=min_pruning if min_pruning is not None else np.zeros(12),
            xu=np.ones(12) * 0.99,
        )

    def _evaluate(self, X, out):
        all_violations = []
        for x in X:
            acc = model_stats.get_model_accuracy(x)[0]
            sizes = model_stats.get_model_size(x)

            acc_violation = max(0, GLOBAL_MIN_ACCURACY * 100 - acc * 100) * 1000 * 12
            size_violation = sum(max(0, sizes[i] - MAX_MODEL_SIZES[i]) for i in range(len(sizes)))
            all_violations.append(acc_violation + size_violation)

        out['F'] = np.array(all_violations).reshape(-1, 1)


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
        ),
        eliminate_duplicates=True,
    )

    res = minimize(
        fallback_problem,
        algorithm,
        termination=('n_gen', N_GENERATIONS),
        verbose=False,
        tol=X_TOL,
    )

    best_solution = res.X
    min_violation_score = float(res.F)

    if min_violation_score < 1e-2:
        print("Fallback solution found with acceptable violations.")
        best = best_solution
    else:
        print("Fallback solution still violates constraints. Using minimum pruning.")
        best = min_pruning

    sizes = model_stats.get_model_size(best)
    headroom = (MAX_MODEL_SIZES - sizes) / MAX_MODEL_SIZES
    extra_budget = np.sum(np.clip(headroom, 0, 1)) * 0.5

    return distribute_pruning(best, feature_importances, device_perf, extra_budget=extra_budget)


def init():
    global MAX_MODEL_SIZES, GLOBAL_MIN_ACCURACY, DEVICE_PERF, p_min, model_stats, rewards

    MAX_MODEL_SIZES = [100] * 12
    GLOBAL_MIN_ACCURACY = 1.0
    DEVICE_PERF = np.ones(12)
    p_min = np.zeros(12)

    model_stats = ModelStats()
    rewards = Rewards()

    uniform_sample = np.random.uniform(1.0, 4.5, 12)
    MAX_MODEL_SIZES = np.array([MAX_MODEL_SIZES[i] * uniform_sample[i] for i in range(12)])

    max_possible_acc = model_stats.get_model_accuracy(np.zeros(12))
    GLOBAL_MIN_ACCURACY = float(np.round(
        np.random.uniform(0.7, max_possible_acc), 3
    ))

    DEVICE_PERF = np.random.uniform(0.0, 0.99, size=12)

    if np.round(max_possible_acc, 2) < np.round(GLOBAL_MIN_ACCURACY, 2):
        raise ValueError(
            "Maximum possible accuracy is lower than the global minimum accuracy requirement."
        )

    min_pruning = np.array([
        calculate_min_pruning_newton_newton_newton(MAX_MODEL_SIZES[i]) for i in range(12)
    ])

    min_acc = model_stats.get_model_accuracy(min_pruning)[0]
    print(f"Minimum pruning for size constraints would yield accuracy: {min_acc * 100:.2f}%")
    print(f"Required minimum accuracy: {GLOBAL_MIN_ACCURACY * 100:.2f}%")
    print(f"Max Size vector: {MAX_MODEL_SIZES}")

    smart_pruning = distribute_pruning(min_pruning, I, DEVICE_PERF)

    problem = MultiViewProblem(min_pruning)
    sampling = SmartSampling(I, min_pruning, smart_pruning, DEVICE_PERF)
    algorithm = NSGA2(pop_size=POP_SIZE, sampling=sampling)

    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', N_GENERATIONS),
        verbose=False,
        tol=X_TOL,
    )

    if res.G is None or res.X is None or res.F is None:
        print("WARNING: Optimization failed to converge. Using fallback solution.")
        P = fallback_GA_Optimization(min_pruning, I, DEVICE_PERF)
    else:
        feasible_mask = np.all(res.G <= 0, axis=1)
        if not np.any(feasible_mask):
            print("WARNING: No feasible solutions found. Using fallback solution.")
            P = fallback_GA_Optimization(min_pruning, I, DEVICE_PERF)
        else:
            feasible_indices = np.where(feasible_mask)[0]
            feasible_F = res.F[feasible_indices]
            feasible_X = res.X[feasible_indices]

            f2_vals = feasible_F[:, 1]
            best_idx = np.argmin(f2_vals)

            ties = np.where(f2_vals == f2_vals[best_idx])[0]
            if len(ties) > 1:
                sums = feasible_X[ties].sum(axis=1)
                best_idx = ties[np.argmax(sums)]

            P = feasible_X[best_idx]

    t = model_stats.get_inf_time(P, DEVICE_PERF)
    sizes = model_stats.get_model_size(P)
    acc = model_stats.get_model_accuracy(P)[0]
    accuracy_violated = np.round(acc * 100, 0) < np.round(GLOBAL_MIN_ACCURACY * 100, 0)

    table_data = []
    for i in range(len(P)):
        table_data.append([
            i,
            f",{P[i]:.3f}",
            f",{sizes[i]:.2f}MB",
            f",{MAX_MODEL_SIZES[i]:.2f}MB",
            f",{t[i]:.4f}ms",
            ",VIOLATED" if np.round(sizes[i], 0) > np.round(MAX_MODEL_SIZES[i], 0) else ",No",
            f",{DEVICE_PERF[i]:.2f}",
            f",{I[i]:.3f}"
        ])

    print("\nPer-view summary:")
    print(tabulate(
        table_data,
        headers=[
            "View", "Pruning Amount", "Size of Model", "Max Size",
            "Inference Time", "Violated Model Size", "Device Performance", "Importance"
        ],
        tablefmt="csv"
    ))

    print(f"\nRequired minimum accuracy: {np.round(GLOBAL_MIN_ACCURACY * 100, 0):.2f}%")
    print(f"Current accuracy: {np.round(acc * 100, 0):.2f}%")

    if accuracy_violated:
        print('Model does not meet minimum accuracy requirement.')
    else:
        print('Model meets minimum accuracy requirement.')

    inf_time = np.max(model_stats.get_inf_time(P, DEVICE_PERF))
    org_time = np.max(model_stats.get_inf_time(np.zeros_like(P), DEVICE_PERF))

    print(f"Original inference time: {org_time:.4f}ms")
    print(f"Optimized inference time: {inf_time:.4f}ms")
    print(f"{org_time / inf_time:.2f}x speedup in inference time")

    model_size_violated = any(
        np.round(sizes[i], 0) > np.round(MAX_MODEL_SIZES[i], 0) for i in range(len(sizes))
    )

    return (accuracy_violated, model_size_violated)