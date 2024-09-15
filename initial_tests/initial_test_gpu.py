import numpy as np
import time
from numba import cuda
import matplotlib.pyplot as plt

# GPU kernel to compute a single Fibonacci number at a given index
@cuda.jit
def fibonacci_gpu_single(n, result):
    a = 0
    b = 1
    for i in range(n):
        a, b = b, a + b
    result[0] = a

# Define how many Fibonacci numbers to calculate
num_fibonacci_numbers = 93  # Limit to avoid overflow (int64 max limit)
num_iterations = 10000  # Number of iterations

# Initialize arrays to accumulate time records
time_records = [[] for _ in range(num_fibonacci_numbers)]

# Initialize a list to store total time per iteration
iteration_times = []

# Start total timer
start_total = time.perf_counter()

for iteration in range(num_iterations):
    # Start timer for this iteration
    iteration_start_time = time.perf_counter()

    # Preallocate device memory for result
    result_gpu = cuda.device_array(1, dtype=np.int64)
    
    # For each Fibonacci number index
    for i in range(num_fibonacci_numbers):
        # Measure time for GPU computation
        start_gpu = time.perf_counter()
        
        # Launch the GPU kernel to compute Fibonacci number at index i
        fibonacci_gpu_single[1, 1](i, result_gpu)
        cuda.synchronize()
        
        # Copy the result back from the GPU to the host
        result_host = result_gpu.copy_to_host()
        
        end_gpu = time.perf_counter()
        
        # Calculate time taken for this Fibonacci number
        computation_time = end_gpu - start_gpu
        time_records[i].append(computation_time)
        
        # Print the Fibonacci number and time taken for this index
        print(f"Iteration {iteration + 1}, GPU: Fibonacci number at {i}: {result_host[0]}")
    
    # End timer for this iteration
    iteration_end_time = time.perf_counter()
    
    # Calculate and store the total time for this iteration
    iteration_time = iteration_end_time - iteration_start_time
    iteration_times.append(iteration_time)
    
    # Print the total time for this iteration
    print(f"Iteration {iteration + 1}: Total time taken: {iteration_time:.5f} seconds")

# End the total time after all iterations
end_total = time.perf_counter()

# Print the total time for all iterations
print(f"Total time for {num_iterations} iterations (GPU): {end_total - start_total:.5f} seconds")

# Calculate average computation time per Fibonacci number
average_times = []
indices = []
for i in range(num_fibonacci_numbers):
    avg_time = sum(time_records[i]) / len(time_records[i])
    average_times.append(avg_time)
    indices.append(i)

# Plot the average computation times per Fibonacci number index
plt.figure(figsize=(10, 6))
plt.plot(indices, average_times, marker='o', label='Average Computation Time')

# Calculate the polynomial regression curve (degree 2)
coefficients = np.polyfit(indices, average_times, 2)
polynomial = np.poly1d(coefficients)
regression_curve = polynomial(indices)

# Plot the polynomial regression curve
plt.plot(indices, regression_curve, color='red', linestyle='--', label='Polynomial Regression Curve (Degree 2)')
plt.legend()
plt.xlabel('Fibonacci Number Index')
plt.ylabel('Average Computation Time (seconds)')
plt.title('Average Computation Time per Fibonacci Number (GPU)')
plt.grid(True)
# Do not call plt.show() yet

# Plot the total time per iteration
iteration_numbers = np.arange(1, num_iterations + 1)
plt.figure(figsize=(10, 6))
plt.plot(iteration_numbers, iteration_times, marker='o', label='Iteration Time')

# Calculate the polynomial regression curve (degree 2)
coefficients_iter = np.polyfit(iteration_numbers, iteration_times, 2)
polynomial_iter = np.poly1d(coefficients_iter)
regression_curve_iter = polynomial_iter(iteration_numbers)

# Plot the polynomial regression curve
plt.plot(iteration_numbers, regression_curve_iter, color='red', linestyle='--', label='Polynomial Regression Curve (Degree 2)')
plt.legend()
plt.xlabel('Iteration Number')
plt.ylabel('Time Taken (seconds)')
plt.title('Total Computation Time per Iteration (GPU)')
plt.grid(True)
plt.show()
