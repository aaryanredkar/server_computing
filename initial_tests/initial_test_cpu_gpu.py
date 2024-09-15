import time
import numpy as np
from numba import cuda, njit
import threading
import gmpy2

# Define maximum number of bits based on the maximum n value
MAX_BITS = 7  # For n up to 127, since 2^7 = 128

# Function to compute Fibonacci using the fast doubling method on GPU
@cuda.jit
def fibonacci_gpu_fast_doubling(n_values, fib_results):
    idx = cuda.grid(1)
    if idx < n_values.size:
        n = n_values[idx]
        a = 0
        b = 1
        # Loop from the highest bit to the lowest
        for i in range(MAX_BITS - 1, -1, -1):
            c = a * (2 * b - a)
            d = a * a + b * b
            if ((n >> i) & 1):
                a = d
                b = c + d
            else:
                a = c
                b = d
        fib_results[idx] = a

# Function to compute Fibonacci using the fast doubling method on CPU
@njit
def fibonacci_cpu_fast_doubling(n):
    a, b = 0, 1
    for i in range(n.bit_length() - 1, -1, -1):
        c = a * (2 * b - a)
        d = a * a + b * b
        if ((n >> i) & 1):
            a, b = d, c + d
        else:
            a, b = c, d
    return a

# Function for arbitrary-precision Fibonacci using gmpy2
def fibonacci_cpu_gmpy2(n):
    a, b = gmpy2.mpz(0), gmpy2.mpz(1)
    for i in range(n.bit_length() - 1, -1, -1):
        c = a * (2 * b - a)
        d = a * a + b * b
        if ((n >> i) & 1):
            a, b = d, c + d
        else:
            a, b = c, d
    return a

# Function to compute and print Fibonacci numbers using CPU and GPU
def compute_and_print_fibonacci(num_terms):
    gpu_limit = 93  # Limit for GPU computation with int64
    large_num_threshold = 10000  # Threshold to switch to arbitrary-precision arithmetic

    # Prepare arrays for GPU computation
    n_values_gpu = np.arange(2, min(gpu_limit, num_terms), dtype=np.int64)
    fib_results_gpu = np.zeros_like(n_values_gpu)

    # Start GPU computation
    threadsperblock = 32
    blockspergrid = (n_values_gpu.size + (threadsperblock - 1)) // threadsperblock
    fibonacci_gpu_fast_doubling[blockspergrid, threadsperblock](n_values_gpu, fib_results_gpu)
    cuda.synchronize()  # Wait for GPU to finish

    # Prepare for CPU computation
    def cpu_compute():
        for n in range(max(2, gpu_limit), num_terms):
            if n < large_num_threshold:
                # Use njit-compiled function for medium-sized numbers
                fib_n = fibonacci_cpu_fast_doubling(n)
            else:
                # Use arbitrary-precision arithmetic for very large numbers
                fib_n = fibonacci_cpu_gmpy2(n)
            print(f"CPU: Fibonacci number at {n} is {fib_n}")

    # Start CPU computation in a separate thread
    cpu_thread = threading.Thread(target=cpu_compute)
    cpu_thread.start()

    # Print GPU results
    for idx in range(n_values_gpu.size):
        n = n_values_gpu[idx]
        fib_n = fib_results_gpu[idx]
        print(f"GPU: Fibonacci number at {n} is {fib_n}")

    # Wait for CPU computation to finish
    cpu_thread.join()

# Number of Fibonacci numbers to calculate
num_fibonacci_numbers = 93  # Adjust this number as needed

# Start the timer for total time across iterations
start_time = time.time()

# Repeat Fibonacci calculation multiple times
for _ in range(10000):  # Adjust the number of iterations as needed
    compute_and_print_fibonacci(num_fibonacci_numbers)

# End the timer after all iterations
end_time = time.time()

# Print the total time for all iterations
print(f"Total time for iterations: {end_time - start_time:.5f} seconds")
