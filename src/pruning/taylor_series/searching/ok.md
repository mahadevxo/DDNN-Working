# Gradient Descent Search Algorithm Flowchart

## Overview

The following flowchart illustrates the detailed operation of the pruning search algorithm implemented in `searchAlgo.py`. This algorithm combines grid search and gradient ascent methods to find the optimal pruning amount for neural networks while satisfying constraints on accuracy and model size.


## Detailed Component Explanations

### Prediction Functions

These functions predict model performance metrics for a given pruning amount:

1. **Accuracy Prediction**:
   - Uses a 5th degree polynomial: `9.04*x^5 - 11.10*x^4 - 1.86*x^3 + 4.13*x^2 - 1.17*x + 1.05`
   - Scaled by 75
   - Generally decreases as pruning increases

2. **Model Size Prediction**:
   - Linear equation: `-4.3237*x + 487.15` where x is pruning percentage
   - Decreases as pruning increases

3. **Computation Time Prediction**:
   - Linear equation: `-0.0134*x + 1.4681`
   - Slightly decreases as pruning increases

### Reward Calculation Process

The reward function combines multiple components:

1. **Accuracy Reward**:
   - Strong penalty for falling below minimum accuracy
   - Maximum reward at exactly minimum accuracy
   - Decreasing reward for excess accuracy (avoiding wasted capacity)

2. **Model Size Reward**:
   - Strong exponential penalty for exceeding maximum size
   - Hyperbolic reward for smaller models: `1.0/size_ratio` (capped at 5.0)
   - Special bonus for very small models (<30% of max size)

3. **Computation Time Reward**:
   - Reward based on improvement ratio
   - Uses tanh function to map improvements to reasonable reward values
   - Penalty for slower models

4. **Combined Reward**:
   - Weighted sum of component rewards: `x*acc_r + y*size_r + z*time_r`
   - Special bonus for solutions that are both accurate and small
   - Additional bonus for high pruning amounts that still meet constraints

### Numerical Gradient Calculation

The algorithm uses a finite difference method to approximate the gradient:
- Forward difference: `(f(x+h) - f(x))/h`
- Uses small epsilon (0.005) for step size
- The gradient indicates which direction to move for higher reward

### Adaptive Learning Rate

The learning rate decreases over iterations:
- `adaptive_lr = learning_rate / (1 + 0.1 * iteration)`
- Initial learning rate: 0.01
- This helps with convergence by taking smaller steps as we approach optimal points

### Early Stopping Conditions

Two mechanisms prevent unnecessary iterations:
1. **Patience-based stopping**:
   - If no improvement for 5 consecutive iterations, stop
   - Prevents wasting computation on plateaus

2. **Gradient-based stopping**:
   - If gradient magnitude < 0.001, stop
   - Indicates we've reached a local optimum

### Fine-tuning Process

After gradient ascent converges:
- Creates 7 evenly spaced points around the best point found
- Range: ±0.05 from best point, clamped to valid bounds
- Evaluates all points to find potentially better solutions
- Helps refine the solution and overcome limitations of gradient descent

### Constraint Management

The algorithm explicitly tracks:
- Valid solutions (meet both accuracy and size constraints)
- Invalid solutions
- It prioritizes valid solutions even if they have slightly lower rewards
- All visualizations use color-coding (green for valid, red for invalid)

### Visualizations Generated

The algorithm produces comprehensive visualizations:
1. **Reward vs Pruning Amount**:
   - Shows how reward varies with pruning
   - Highlights best solution

2. **Accuracy vs Pruning Amount**:
   - Shows accuracy trend
   - Marks minimum accuracy threshold

3. **Model Size vs Pruning Amount**:
   - Shows size reduction
   - Marks maximum size constraint

4. **Gradient Magnitudes**:
   - Shows how gradient changes during search
   - Helps diagnose convergence issues

These visualizations provide insights into the search process and solution quality.