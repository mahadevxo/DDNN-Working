from StatsRewards import ModelStats
import numpy as np
import Optimize as optimizer

optim = optimizer.Optimizer()

results_file = "results.csv"
with open(results_file, 'w') as f:
    f.write("Adjusted p values,View accuracies %,Weighted accuracy %,Required global accuracy,Model sizes, Max sizes,Inference time\n")

def run():
    optim.init()
    test_result = optim.find_feasible_starting_point()
    if test_result[0] is not None:
        print("Running optimization with redistribution...")
        final_result = optim.optimize_pruning_with_redistribution()

        if final_result is not None:
            x = show_results(final_result)
            return x
    else:
        print("Constraints appear too tight even for redistribution approach.")
        return False


def show_results(final_result):
    
    passed=False
    
    # final_result: [adjusted_p_values, view_accuracies, model_sizes, weighted_accuracy, total_reward]
    
    statfn = ModelStats()
    i = optimizer.I
    
    print("\nFINAL OPTIMIZED SOLUTION WITH REDISTRIBUTION COMPLETE")
    print("Adjusted p values:        ", np.round(final_result[0], 3))
    print("View accuracies %:        ", np.round(final_result[1], 2))
    print("Weighted accuracy %:      ", np.round(final_result[3], 2))
    print("Required global accuracy: ", np.round(optimizer.GLOBAL_MIN_ACCURACY, 2))
    print("Model sizes:              ", np.round(final_result[2], 1))
    print("Max sizes:                ", np.round(optimizer.MAX_MODEL_SIZES, 1))
    print("Total reward:             ", np.round(final_result[4], 2)) if final_result[4] is not None else print("No total reward calculated.")
    
    print("-" * 110 + "\n")


    print(f"{'View':<6} {'Size':<10} {'Max Size':<12} {'Accuracy (%)':<20} {'Time':<14} {'Importance':<12} {'Prune':<10}, {'Device Perf':<12}")

    views_violated = []

    for view in range(12):
        size = final_result[2][view]
        max_size = optimizer.MAX_MODEL_SIZES[view]
        acc = final_result[1][view]
        importance = i[view]
        prune = final_result[0][view]
        if np.round(size, 1) > np.round(max_size, 1):
            views_violated.append(view + 1)
        print(f"{view+1:<6}, {size:<10.1f}, {max_size:<12.1f}, {acc:<16.2f}, {statfn.get_time(prune, device_perf=optimizer.DEVICES_PERF[view]):<14.2f}, {importance*100:<10.2f}, {prune*100:<10.3f}, {optimizer.DEVICES_PERF[view]:<10.3f}")
    print("\n" + "="*110)
    print("Final Weighted Accuracy: ", np.round(final_result[3], 1), "Required:", np.round(optimizer.GLOBAL_MIN_ACCURACY, 1))
    print("Views violating size constraints:", views_violated if views_violated else "None")
    if np.round(final_result[3], 1) >= np.round(optimizer.GLOBAL_MIN_ACCURACY, 1):
        print("Solution meets global accuracy requirement!")
        passed=True
    else:
        print("Solution does NOT meet global accuracy requirement.")
        passed=False

    avg_inf_time = np.mean([statfn.get_time(final_result[0][view], device_perf=optimizer.DEVICES_PERF[view]) for view in range(12)])
    org_inf_time = np.max([statfn.get_time(0.0, device_perf=d) for d in optimizer.DEVICES_PERF])
    max_inf_time = np.max([statfn.get_time(final_result[0][view], device_perf=optimizer.DEVICES_PERF[view]) for view in range(12)])

    print(f'Average Inference Time: {np.round(avg_inf_time, 2)} ms')
    print(f'Max Inference Time: {np.round(max_inf_time, 2)} ms')
    print(f"Original Inference Time: {np.round(org_inf_time, 2)} ms")
    print(f"Inference is {np.round(1+(org_inf_time - max_inf_time) / org_inf_time, 2)}x faster than original model.")
    
    with open(results_file, 'a') as f:
        f.write(f"{np.round(final_result[0], 3)}, {np.round(final_result[1], 2)}, {np.round(final_result[3], 2)}, {np.round(optimizer.GLOBAL_MIN_ACCURACY, 2)}, {np.round(final_result[2], 1)}, {np.round(optimizer.MAX_MODEL_SIZES, 1)}, {np.round(1+(org_inf_time - max_inf_time) / org_inf_time, 2)}\n\n")
    return passed

if __name__ == "__main__":
    iters = int(input("Enter number of iterations to run: ")) or 1
    print(f"Running optimization for {iters} iterations...")
    passed_count = 0
    for i in range(iters):
        print(f"\nIteration {i+1}/{iters}")
        passed = run()
        if passed:
            passed_count += 1
        print("-" * 110)
    print(f"Results saved to {results_file}")
    print("Optimization complete.")
    print(f"Total iterations passed: {passed_count}/{iters}")