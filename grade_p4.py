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
    relu = nn.ReLU()
    x = relu(nn.Input("x"))
    return is_same(x, 1, x = (10, 11, 12, 13))


def grade_Q2():
    flatten = nn.Flatten()
    x = flatten(nn.Input("x"))
    return is_same(x, 1, x = (10, 11, 12, 13))


def grade_Q3():
    x = nn.Input2d("images", 10, 11, 3)
    return is_same(x, 1, images = (50, 10, 11, 3))


def grade_Q4():
    f = nn.Linear("f", 100, 10)
    x = f(nn.Input("x"))
    x.resolve({
        "f.weight": np.random.random((10, 100)),
        "f.bias": np.random.random((10,))})
    return is_same(x, 1, x = (50, 100))


def grade_Q5():
    pool = nn.MaxPool2d(3, 3)
    x = pool(nn.Input2d("x", 12, 15, 3))
    return is_same(x, 1, x = (10, 12, 15, 3))


def grade_Q6():
    c = nn.Conv2d("c", 3, 16, 5)
    x = c(nn.Input2d("x", 15, 20, 3))
    x.resolve({
        "c.weight": np.random.random((16, 3, 5, 5)),
        "c.bias": np.random.random((16,))
    })
    return is_same(x, 1, x = (10, 15, 20, 3))


def grade_Q7():
    relu = nn.ReLU()
    flatten = nn.Flatten()
    f1 = nn.Linear("f1", 28*28, 100)
    f2 = nn.Linear("f2", 100, 10)
    x = nn.Input2d("images", 28, 28, 1)
    x = flatten(x)
    x = f2(relu(f1(x)))
    x.resolve(np.load("msimple_params.npz"))
    mnist_test = np.load("mnist_test.npz")
    images = mnist_test["images"][:20]

    infer0 = x.compile(golden.Builder())
    infer1 = x.compile(cpp.Builder())
    logit0 = infer0(images = images)
    logit1 = infer1(images = images)
    return np.allclose(logit0, logit1)


def grade_Q8():
    relu = nn.ReLU()
    flatten = nn.Flatten()
    f1 = nn.Linear("f1", 28*28, 100)
    f2 = nn.Linear("f2", 100, 10)
    x = nn.Input2d("images", 28, 28, 1)
    x = flatten(x)
    x = f2(relu(f1(x)))
    x.resolve(np.load("msimple_params.npz"))
    mnist_test = np.load("mnist_test.npz")
    images = mnist_test["images"][:1000]

    infer0 = x.compile(golden.Builder())
    infer1 = x.compile(cpp.Builder())
    label0 = infer0(images = images).argmax(axis = 1)
    label1 = infer1(images = images).argmax(axis = 1)
    return np.allclose(label0, label1)


def grade_Q9():
    pool = nn.MaxPool2d(2, 2)
    relu = nn.ReLU()
    flatten = nn.Flatten()

    x = nn.Input2d("images", 28, 28, 1)
    c1 = nn.Conv2d("c1", 1, 8, 3) # 28->26
    c2 = nn.Conv2d("c2", 8, 8, 3) # 26->24
    x = pool(relu(c2(relu(c1(x))))) # 24->12
    c3 = nn.Conv2d("c3", 8, 16, 3) # 12->10
    c4 = nn.Conv2d("c4", 16, 16, 3) # 10->8
    x = pool(relu(c4(relu(c3(x))))) # 8->4
    f = nn.Linear("f", 16*4*4, 10)
    x = f(flatten(x))

    x.resolve(np.load("mnist_params.npz"))
    mnist_test = np.load("mnist_test.npz")
    images = mnist_test["images"][:20]

    infer0 = x.compile(golden.Builder())
    infer1 = x.compile(cpp.Builder())
    logit0 = infer0(images = images)
    logit1 = infer1(images = images)
    return np.allclose(logit0, logit1)


def grade_Q10():
    pool = nn.MaxPool2d(2, 2)
    relu = nn.ReLU()
    flatten = nn.Flatten()

    x = nn.Input2d("images", 28, 28, 1)
    c1 = nn.Conv2d("c1", 1, 8, 3) # 28->26
    c2 = nn.Conv2d("c2", 8, 8, 3) # 26->24
    x = pool(relu(c2(relu(c1(x))))) # 24->12
    c3 = nn.Conv2d("c3", 8, 16, 3) # 12->10
    c4 = nn.Conv2d("c4", 16, 16, 3) # 10->8
    x = pool(relu(c4(relu(c3(x))))) # 8->4
    f = nn.Linear("f", 16*4*4, 10)
    x = f(flatten(x))

    x.resolve(np.load("mnist_params.npz"))
    mnist_test = np.load("mnist_test.npz")
    images = mnist_test["images"][:1000]

    infer0 = x.compile(golden.Builder())
    infer1 = x.compile(cpp.Builder())
    label0 = infer0(images = images).argmax(axis = 1)
    label1 = infer1(images = images).argmax(axis = 1)
    return np.allclose(label0, label1)


grade.grade_all("p4", 1, 11, globals())
