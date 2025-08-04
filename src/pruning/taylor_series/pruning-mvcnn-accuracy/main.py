from PruneSaveModel import PruneSaveModel
from EvalCombinations import EvalCombinations
from CacheFeatures import CacheFeatures

def main():
    
    skip_save_model = int(input("Skip saving pruned model? (1 for yes, 0 for no): "))
    skip_cache_features = int(input("Skip caching features? (1 for yes, 0 for no): "))
    skip_eval = int(input("Skip evaluation? (1 for yes, 0 for no): "))
    
    print("-"*300)
    
    if skip_save_model == 0:
        pruner = PruneSaveModel()
        pruner.run()

    print("Models pruned and saved successfully.")
    
    if skip_cache_features == 0:
        cacher = CacheFeatures()
        cacher.run()
    
    print("Features cached successfully.")
    
    if skip_eval == 0:
        evaluator = EvalCombinations()
        evaluator.run()
    
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    main()