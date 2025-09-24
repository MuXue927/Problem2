import numpy as np

def factorial(n):
    """Compute the factorial of a non-negative integer n."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def factorial_numpy(n):
    """Compute the factorial of a non-negative integer n using NumPy."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    return np.prod(np.arange(1, n + 1))



if __name__ == "__main__":
    # Example usage
    print(factorial(5))        # Output: 120
    print(factorial_numpy(5))  # Output: 120

    # Testing edge cases
    print(factorial(0))        # Output: 1
    print(factorial_numpy(0))  # Output: 1
  
    try:
        print(factorial(-1))   # Should raise ValueError
    except ValueError as e:
        print(f"Error: {e}")