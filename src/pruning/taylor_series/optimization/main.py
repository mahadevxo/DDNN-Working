import optimizer as opt
import warnings

def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iters = int(input("Enter number of iterations: "))
        failed = 0
        k=0
        i = 0
        while i < iters:
            print("-" * 50)
            print(f"\nIteration {i+1}/{iters}")
            a_v, s_v = opt.init()
            if a_v or s_v:
                failed += 1
                print(f"Iteration {i+1} failed. Total failures: {failed}")
            else:
                print(f"Iteration {i+1} succeeded.")
                i+=1
        print(f"\nTotal iterations: {iters}, Failed: {failed}, Errors: {k}")

if __name__ == "__main__":
    main()