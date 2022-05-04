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
    return is_same(x, 1, x = (9,)) and is_same(x, 1, x = (9, 9))


def grade_Q2():
    c1 = nn.Const(np.random.random((10,)))
    c2 = nn.Const(np.random.random((10, 10)))
    return is_same(c1, 1) and is_same(c2, 1)


def grade_Q3():
    x = nn.Input("x")
    y = nn.Input("y")
    z = x+y
    return all([
        is_same(z, 1, x = (11,), y = (11,)),
        is_same(z, 1, x = (11, 12), y = (11, 12)),
        is_same(z, 1, x = (11, 12, 13), y = (11, 12, 13)),
        is_same(z, 1, x = (11, 12, 13, 14), y = (11, 12, 13, 14))])


def grade_Q4():
    x = nn.Input("x")
    y = nn.Input("y")
    z = x-y
    return all([
        is_same(z, 1, x = (11,), y = (11,)),
        is_same(z, 1, x = (11, 12), y = (11, 12)),
        is_same(z, 1, x = (11, 12, 13), y = (11, 12, 13)),
        is_same(z, 1, x = (11, 12, 13, 14), y = (11, 12, 13, 14))])


def grade_Q5():
    x = nn.Input("x")
    y = nn.Input("y")
    z = x*y
    return is_same(z, 1, x = (11, 12), y = (12, 13))


def grade_Q6():
    x = nn.Input("x")
    y = nn.Input("y")
    z = x*y
    return is_same(z, 1, x = None, y = (12, 13)) and is_same(z, 1, x = (11, 12), y = None)


def grade_Q7():
    x = nn.Input("x")
    y = nn.Input("y")
    z = nn.Input("z")
    r = x+y*z
    return is_same(r, 1, x = (11, 13), y = (11, 12), z = (12, 13))


def grade_Q8():
    x = nn.Input("x")
    y = nn.Input("y")
    z = nn.Input("z")
    r = (x-y)*z
    return is_same(r, 1, x = (11, 12), y = (11, 12), z = (12, 13))


def grade_Q9():
    x = nn.Input("x")
    y = nn.Input("y")
    z = nn.Input("z")
    r = x*y*z
    return is_same(r, 1, x = (11, 12), y = (12, 13), z = (13, 14))


def grade_Q10():
    x = nn.Input("x")
    y = nn.Input("y")
    c1 = nn.Const(np.random.random((11, 12)))
    c2 = nn.Const(np.random.random((12, 13)))
    z = x*y-x*c2-c1*y+c1*c2
    return is_same(z, 1, x = (11, 12), y = (12, 13))


grade.grade_all("p3", 1, 11, globals())
