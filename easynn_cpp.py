import numpy as np
import ctypes
import time

_libeasynn = ctypes.CDLL("./libeasynn.so")

_libeasynn.create_program.restype = ctypes.c_void_p

_libeasynn.append_expression.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
    ctypes.c_int
]

_libeasynn.add_op_param_double.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_double
]

_libeasynn.add_op_param_ndarray.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p
]

_libeasynn.build.restype = ctypes.c_void_p
_libeasynn.build.argtypes = [
    ctypes.c_void_p
]

_libeasynn.add_kwargs_double.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_double
]

_libeasynn.add_kwargs_ndarray.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p
]

_libeasynn.execute.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p
]


def check_and_raise(ret):
    if ret != 0:
        raise Exception("libeasynn error: code %d" % ret)


def to_float_or_ndarray(c_dim, c_shape, c_data):
    if c_dim.value == 0:
        return c_data[0]
    shape = [c_shape[k] for k in range(c_dim.value)]
    N = 1
    for s in shape:
        N *= s
    print("shape =", shape, "N =", N)
    flat = np.array([c_data[i] for i in range(N)])
    return flat.reshape(shape)


class Eval:
    def __init__(self, evaluation):
        self.evaluation = evaluation
    
    def __call__(self, **kwargs):
        start = time.time()
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                _libeasynn.add_kwargs_double(
                    self.evaluation, k.encode(), ctypes.c_double(v))
            elif hasattr(v, "shape"):
                v = np.require(v, dtype=np.double, requirements=['C', 'O', 'W', 'A'])
                _libeasynn.add_kwargs_ndarray(
                    self.evaluation, k.encode(), v.ndim, v.ctypes.shape,
                    v.ctypes.data)
            else:
                raise Exception("%s: kwargs must be float or int or ndarray" % k)
        c_dim = ctypes.c_int()
        c_shape = ctypes.POINTER(ctypes.c_size_t)()
        c_data = ctypes.POINTER(ctypes.c_double)()
        check_and_raise(_libeasynn.execute(self.evaluation,
            ctypes.byref(c_dim), ctypes.byref(c_shape), ctypes.byref(c_data)))
        res = to_float_or_ndarray(c_dim, c_shape, c_data)
        t = time.time()-start
        if t > 0.1:
            print("c++ time %.2f" % t)
        return res


class Builder:
    def __init__(self):
        self.program = _libeasynn.create_program()
    
    def append(self, expr):
        inputs = [ex.id for ex in expr.inputs]
        num_inputs = len(inputs)
        op = expr.op
        _libeasynn.append_expression(
            self.program, expr.id,
            op.name.encode(), op.op_type.encode(),
            (ctypes.c_int*num_inputs)(*inputs), num_inputs)
        for k, v in op.parameters.items():
            if isinstance(v, (int, float)):
                check_and_raise(_libeasynn.add_op_param_double(
                    self.program, k.encode(), ctypes.c_double(v)))
            elif hasattr(v, "shape"):
                v = np.require(v, dtype=np.double, requirements=['C', 'O', 'W', 'A'])
                check_and_raise(_libeasynn.add_op_param_ndarray(
                    self.program, k.encode(), v.ndim, v.ctypes.shape,
                    v.ctypes.data))
            else:
                raise Exception("%s: op params must be float or int or ndarray: %s" % (expr, k))

    def build(self):
        return Eval(_libeasynn.build(self.program))
