import numpy as np
from numpy.core.fromnumeric import shape
import easynn as nn

# Create a numpy array of 10 rows and 5 columns.
# Set the element at row i and column j to be i+j.
def Q1(): 
 y = np.random.rand(10,5)
 for i in range(10):
    for j in range(5):
        return (i + j)
        print(i+j)

# Add two numpy arrays together.
def Q2(a, b):
    return np.add(a,b)
a = np.random.rand(3,2)
b = np.random.rand(3,2)
    
    

# Multiply two 2D numpy arrays using matrix multiplication.
def Q3(a, b):
     return np.matmul(a,b)
a = np.random.rand(3,2)
b = np.random.rand(2,3)
   

# For each row of a 2D numpy array, find the column index
# with the maximum element. Return all these column indices.
def Q4(a):
    return np.argmax(a, axis = 1)
a = np.random.rand(3,2)
# maxIndices = np.argmax(a, axis = 1)
    

# Solve Ax = b.
def Q5(A, b):
    return np.linalg.solve(A,b)
A = np.random.rand(2,2)
b = np.random.rand(2,1)
    

# Return an EasyNN expression for a+b.
def Q6():
    a = nn.Input('a')
    b = nn.Input('b')
    return (a + b)
    

# Return an EasyNN expression for a+b*c.
def Q7():
    a = nn.Input('a')
    b = nn.Input('b')
    c = nn.Input('c')
    return a+b*c

# Given A and b, return an EasyNN expression for Ax+b.
def Q8(A, b):      
    A = nn.Const()
    b = nn.Const()
    x = nn.Input()
    return A*x+b

# Given n, return an EasyNN expression for x**n.
def Q9(n): 
    x = nn.Input('x')
    n = nn.Input('n')


# Return an EasyNN expression to compute
# the element-wise absolute value |x|.
def Q10():
    x = nn.Input('x')
    relu = nn.ReLU()
    m = relu(x)
    n = relu(-x)
    Abs_x = m + n
    return Abs_x
