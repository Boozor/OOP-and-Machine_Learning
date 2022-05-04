import numpy as np
import easynn as nn
import easynn_golden as golden
import easynn_cpp as cpp
import grade


def random_kwargs(kwargs):
    return {k: np.random.random(shape) if shape != None else np.random.random() for k, shape in kwargs.items()}


def is_same(p, n, **kwargs):
    e0 = p.compile(golden.Builder())
    e1 = p.compile(cpp.Builder())
    nkwargs = [random_kwargs(kwargs) for i in range(n)]
    return all([np.allclose(e0(**nkwargs[i]), e1(**nkwargs[i])) for i in range(n)])


def grade_Q1():
    x = nn.Input("x")
    return is_same(x, 1, x = None)


def grade_Q2():
    c = nn.Const(np.random.random())
    return is_same(c, 1)


def grade_Q3():
    x = nn.Input("x")
    c = nn.Const(np.random.random())
    y = x+c
    return is_same(y, 1, x = None)


def grade_Q4():
    x = nn.Input("x")
    c = nn.Const(np.random.random())
    y = x*c
    return is_same(y, 1, x = None)


def grade_Q5():
    x = nn.Input("x")
    y = nn.Input("y")
    z = x+y
    return is_same(z, 1, x = None, y = None)


def grade_Q6():
    x = nn.Input("x")
    y = nn.Input("y")
    z = x*y
    return is_same(z, 1, x = None, y = None)


def grade_Q7():
    x = nn.Input("x")
    y = nn.Input("y")
    z = nn.Input("z")
    r = x+y*z
    return is_same(r, 1, x = None, y = None, z = None)


def grade_Q8():
    x = nn.Input("x")
    y = nn.Input("y")
    z = nn.Input("z")
    r = (x-y)*z
    return is_same(r, 1, x = None, y = None, z = None)


def grade_Q9():
    x = nn.Input("x")
    y = nn.Input("y")
    c = nn.Const(np.random.random())
    z = x*y-x*c-y*c+c*c
    return is_same(z, 10, x = None, y = None)


def grade_Q10():
    x = nn.Input("x")
    y = nn.Input("y")
    c = nn.Const(np.random.random())
    z = x*y-x*c-y*c+c*c
    return is_same(z, 10, x = None, y = None)


grade.grade_all("p2", 1, 11, globals())
