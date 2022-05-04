import numpy as np
import easynn as nn
import easynn_golden as golden
import grade

def grade_Q1():
    pool = nn.MaxPool2d(2, 2)
    relu = nn.ReLU()
    flatten = nn.Flatten()
    x = nn.Input2d("images", 28, 28, 1)
    c1 = nn.Conv2d("c1", 1, 8, 5) # 28->24
    x = pool(relu(c1(x))) # 24->12
    c2 = nn.Conv2d("c2", 8, 8, 5) # 12->8
    x = pool(relu(c2(x))) # 8->4
    f = nn.Linear("f", 8*4*4, 10)
    x = f(flatten(x))
    x.resolve(np.load("p6_params.npz"))
    mnist_test = np.load("mnist_test.npz")
    images = mnist_test["images"]
    labels = mnist_test["labels"]

    infer = x.compile(golden.Builder())
    pred_labels = infer(images = images).argmax(axis = 1)
    count = sum(labels == pred_labels)
    print(count)
    return count > 8500

def grade_Q2():
    pool = nn.MaxPool2d(2, 2)
    relu = nn.ReLU()
    flatten = nn.Flatten()
    x = nn.Input2d("images", 28, 28, 1)
    c1 = nn.Conv2d("c1", 1, 8, 5) # 28->24
    x = pool(relu(c1(x))) # 24->12
    c2 = nn.Conv2d("c2", 8, 8, 5) # 12->8
    x = pool(relu(c2(x))) # 8->4
    f = nn.Linear("f", 8*4*4, 10)
    x = f(flatten(x))
    x.resolve(np.load("p6_params.npz"))
    mnist_test = np.load("mnist_test.npz")
    images = mnist_test["images"]
    labels = mnist_test["labels"]

    infer = x.compile(golden.Builder())
    pred_labels = infer(images = images).argmax(axis = 1)
    count = sum(labels == pred_labels)
    return count > 9000

def grade_Q3():
    pool = nn.MaxPool2d(2, 2)
    relu = nn.ReLU()
    flatten = nn.Flatten()
    x = nn.Input2d("images", 28, 28, 1)
    c1 = nn.Conv2d("c1", 1, 8, 5) # 28->24
    x = pool(relu(c1(x))) # 24->12
    c2 = nn.Conv2d("c2", 8, 8, 5) # 12->8
    x = pool(relu(c2(x))) # 8->4
    f = nn.Linear("f", 8*4*4, 10)
    x = f(flatten(x))
    x.resolve(np.load("p6_params.npz"))
    mnist_test = np.load("mnist_test.npz")
    images = mnist_test["images"]
    labels = mnist_test["labels"]

    infer = x.compile(golden.Builder())
    pred_labels = infer(images = images).argmax(axis = 1)
    count = sum(labels == pred_labels)
    return count > 9300

def grade_Q4():
    pool = nn.MaxPool2d(2, 2)
    relu = nn.ReLU()
    flatten = nn.Flatten()
    x = nn.Input2d("images", 28, 28, 1)
    c1 = nn.Conv2d("c1", 1, 8, 5) # 28->24
    x = pool(relu(c1(x))) # 24->12
    c2 = nn.Conv2d("c2", 8, 8, 5) # 12->8
    x = pool(relu(c2(x))) # 8->4
    f = nn.Linear("f", 8*4*4, 10)
    x = f(flatten(x))
    x.resolve(np.load("p6_params.npz"))
    mnist_test = np.load("mnist_test.npz")
    images = mnist_test["images"]
    labels = mnist_test["labels"]

    infer = x.compile(golden.Builder())
    pred_labels = infer(images = images).argmax(axis = 1)
    count = sum(labels == pred_labels)
    return count > 9500

grade.grade_all("p6", 1, 5, globals())
