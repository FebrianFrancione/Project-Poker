import numpy as np
from matplotlib import image
from matplotlib import pyplot
import timeit

# showing image using matplotlib images

image = image.imread('img/fig-ndarray-array-indexing-copy.png')
# printing image dtype
print(image.dtype)
# printing image shape
print(image.shape)
# display the array of pixels as an image
pyplot.imshow(image)
pyplot.show()

# numpy slicing and data sharing
x = np.arange(15, dtype=np.float32).reshape(3,5)
y  = x[:2,2:4].T
y[:] = -y
z = y.copy()
# this can also be done as
z2 = np.copy(y, order='C')

# Using Numpy to vectorize code
# x is 2-dimensional ndarray of numbers
def funct1(x):
    n, m = x.shape
    v = x[0, 0]
    for i in range(n):
        for j in range(m):
            if x[i, j] < v:
                v = x[i, j]

    y = np.empty_like(x)
    for i in range(n):
        for j in range(m):
            y[i, j] = x[i, j] - v

    return y

# we can do the same function as a vectorized numpy function

def funct1_vectorized(x):
    return x - x.min()

# testing these functions:
x = np.random.randint(1000, size=(200,200))
time_loop = timeit.timeit('funct1(x)',      setup="from __main__ import funct1, x", number=10)
vect_time_loop = timeit.timeit('funct1_vectorized(x)', setup="from __main__ import funct1_vectorized, x", number=10)

print("On a 200*200 matrix, Funct1 ran at: {:.1} and funct1_vectorized ran at: {:.1}. Therefore, the vectorized function ran {:.1} faster".format(time_loop, vect_time_loop,(time_loop/vect_time_loop)))

# check linear system solution quality

# Suppose you are given matrix  𝐴∈ℝ𝑚×𝑛  and vector  𝑏∈ℝ𝑚  and are told that  𝑥∈ℝ𝑛  is a solution to the system of linear equations  𝐴𝑥=𝑏 .
def is_solution(A, b, x, epsilon=1e-5):
    """Returns whether x is a solution to Ax=b, with all residuals below epsilon."""
    return np.abs(A @ x - b).max() < epsilon

A = np.array([[2., 0.5], [-5., 3.]])
b = np.array([5., 9.])
x = np.array([1.23529, 5.05882])
result = is_solution(A, b, x)
print(result)


# Shuffling pair of matrices together
# Suppose you are given a pair of matrices  𝑋∈ℝ𝑚×𝑛  and  𝑌∈ℝ𝑚×𝑘 , and you must 'shuffle' their rows by the same permutation.
