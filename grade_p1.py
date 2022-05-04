import numpy as np
import easynn as nn
import easynn_golden as golden
import grade
import prj01


# create
def grade_Q1():
    a = prj01.Q1()
    if not hasattr(a, "shape"):
        return False
    if a.shape != (10, 5):
        return False
    return all([a[i, j] == i+j for i in range(10) for j in range(5)])


# add
def grade_Q2():
    a = np.random.rand(100, 10)
    b = np.random.rand(100, 10)
    c = prj01.Q2(a, b)
    if not hasattr(c, "shape"):
        return False
    if c.shape != (100, 10):
        return False
    return all([abs(c[i, j]-a[i, j]-b[i, j]) < 1e-9 for i in range(100) for j in range(10)])


# mul
def grade_Q3():
    a = np.array([[i for i in range(10)]])
    b = a.transpose()
    c = prj01.Q3(a, b)
    d = prj01.Q3(b, a)
    if not hasattr(c, "shape") or not hasattr(d, "shape"):
        return False
    if c.shape != (1, 1) or d.shape != (10, 10):
        return False
    return c[0, 0] == 285 and all([d[i, j] == i*j for i in range(10) for j in range(10)])


# max
def grade_Q4():
    a = np.random.rand(100, 10)
    b = prj01.Q4(a)
    if len(b) != 100:
        return False
    return all([a[i, b[i]] >= a[i, j] for i in range(100) for j in range(10)])


# solve
def grade_Q5():
    A = -np.random.rand(100, 100)+np.diag([100]*100)
    b = np.random.rand(100, 1)
    x = prj01.Q5(A, b)
    Ax = prj01.Q3(A, x)
    return np.allclose(Ax, b)


# a+b
def grade_Q6():
    a = np.random.rand(100, 10)
    b = np.random.rand(100, 10)
    c = prj01.Q6().compile(golden.Builder())(a = a, b = b)
    return np.allclose(c, prj01.Q2(a, b))


# a+b*c
def grade_Q7():
    a = np.random.rand(100, 50)
    b = np.random.rand(100, 10)
    c = np.random.rand(10, 50)
    d = prj01.Q7().compile(golden.Builder())(a = a, b = b, c = c)
    return np.allclose(d, prj01.Q2(a, prj01.Q3(b, c)))


# Ax+b
def grade_Q8():
    A = np.random.rand(100, 100)
    x = np.random.rand(100, 1)
    b = np.random.rand(100, 1)
    y = prj01.Q8(A, b).compile(golden.Builder())(x = x)
    return np.allclose(y, prj01.Q2(prj01.Q3(A, x), b))


# x**n
def grade_Q9():
    x = np.random.rand(100, 100)
    y = prj01.Q9(8).compile(golden.Builder())(x = x)
    x2 = prj01.Q3(x, x)
    x4 = prj01.Q3(x2, x2)
    x8 = prj01.Q3(x4, x4)
    return np.allclose(y, x8)


# |x|
def grade_Q10():
    x = np.random.randn(100, 100)
    y = prj01.Q10().compile(golden.Builder())(x = x)
    return np.allclose(y, np.abs(x))


grade.grade_all("p1", 1, 11, globals())
