import time
import matplotlib.pyplot as plt
import numpy as np

# Fibonacci function using CPU
def fibonacci_cpu(num_terms, time_records):
    a, b = 0, 1
    fibonacci_sequence = [a, b]
    for i in range(2, num_terms):
        start_time = time.perf_counter()  # High-resolution timer
        a, b = b, a + b
        end_time = time.perf_counter()
        fibonacci_sequence.append(b)
        computation_time = end_time - start_time
        time_records[i].append(computation_time)
        # Print Fibonacci values with correct index
        print(f"Iteration {iteration + 1}: CPU: Fibonacci number at {i} is {b}")
    return fibonacci_sequence

# Define how many Fibonacci numbers to calculate
num_fibonacci_numbers = 93  # You can change this to any number of terms

# Initialize time records
time_records = [[] for _ in range(num_fibonacci_numbers)]

# Initialize a list to store total time per iteration
iteration_times = []

# Repeat Fibonacci calculation multiple times
num_iterations = 10000  # Number of iterations
for iteration in range(num_iterations):
    # Start timer for this iteration
    iteration_start_time = time.perf_counter()

    # Compute Fibonacci sequence and record times
    fib_sequence_cpu = fibonacci_cpu(num_fibonacci_numbers, time_records)

    # End timer for this iteration
    iteration_end_time = time.perf_counter()

    # Calculate and store the total time for this iteration
    iteration_time = iteration_end_time - iteration_start_time
    iteration_times.append(iteration_time)

    # Print the total time for this iteration
    print(f"Iteration {iteration + 1}: Total time taken: {iteration_time:.5f} seconds")

# Calculate total time across all iterations
total_time = sum(iteration_times)
print(f"Total time for {num_iterations} iterations: {total_time:.5f} seconds")

# Calculate average computation time for each Fibonacci number
average_times = []
indices = []
for i in range(2, num_fibonacci_numbers):
    avg_time = sum(time_records[i]) / len(time_records[i])
    average_times.append(avg_time)
    indices.append(i)

# Plot the average computation times per Fibonacci number index
plt.figure(figsize=(10, 6))
plt.plot(indices, average_times, marker='o', label='Average Computation Time')

# Calculate the polynomial regression curve (degree 2) for the first plot
coefficients = np.polyfit(indices, average_times, 2)
polynomial = np.poly1d(coefficients)
regression_curve = polynomial(indices)

# Plot the polynomial regression curve for the first plot
plt.plot(indices, regression_curve, color='red', linestyle='--', label='Polynomial Regression Curve (Degree 2)')
plt.legend()
plt.xlabel('Fibonacci Number Index')
plt.ylabel('Average Computation Time (seconds)')
plt.title('Average Computation Time per Fibonacci Number over Iterations (CPU)')
plt.grid(True)
# Do not call plt.show() yet

# Plot the total time per iteration
iteration_numbers = np.arange(1, num_iterations + 1)
plt.figure(figsize=(10, 6))
plt.plot(iteration_numbers, iteration_times, marker='o', label='Iteration Time')

# Calculate the polynomial regression curve (degree 2) for the second plot
coefficients_iter = np.polyfit(iteration_numbers, iteration_times, 2)
polynomial_iter = np.poly1d(coefficients_iter)
regression_curve_iter = polynomial_iter(iteration_numbers)

# Plot the polynomial regression curve for the second plot
plt.plot(iteration_numbers, regression_curve_iter, color='red', linestyle='--', label='Polynomial Regression Curve (Degree 2)')
plt.legend()
plt.xlabel('Iteration Number')
plt.ylabel('Time Taken (seconds)')
plt.title('Total Computation Time per Iteration (CPU)')
plt.grid(True)

# Display both plots
plt.show()
