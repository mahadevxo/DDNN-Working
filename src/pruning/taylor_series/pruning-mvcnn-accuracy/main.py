from PruneSaveModel import PruneSaveModel
from EvalCombinations import EvalCombinations
from CacheFeatures import CacheFeatures

def main():
    
    skip_save_model = int(input("Skip saving pruned model? (1 for yes, 0 for no): "))
    skip_cache_features = int(input("Skip caching features? (1 for yes, 0 for no): "))
    skip_eval = int(input("Skip evaluation? (1 for yes, 0 for no): "))
    iters=1000
    if skip_eval == 0:
        iters = int(input("Enter number of iterations for evaluation: "))
    
    print("-"*150)
    print(f"Skip save model: {bool(skip_save_model)}, Skip cache features: {bool(skip_cache_features)}, Skip eval: {bool(skip_eval)}")
    input("Press Enter to continue...")
    
    print("-"*150)
    
    if skip_save_model == 0:
        pruner = PruneSaveModel()
        pruner.run()

        print("Models pruned and saved successfully.")
    
    if skip_cache_features == 0:
        cacher = CacheFeatures()
        cacher.run()
    
        print("Features cached successfully.")
    
    if skip_eval == 0:
        evaluator = EvalCombinations(iters=iters)
        evaluator.run()
    
        print("Evaluation completed successfully.")

if __name__ == "__main__":
    main()