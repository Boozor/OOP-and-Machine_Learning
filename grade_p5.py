import numpy as np
import easynn as nn
import easynn_golden as golden
import grade

def grade_Q1():
    relu = nn.ReLU()
    flatten = nn.Flatten()
    f1 = nn.Linear("f1", 28*28, 32)
    f2 = nn.Linear("f2", 32, 10)
    x = nn.Input2d("images", 28, 28, 1)
    x = flatten(x)
    x = f2(relu(f1(x)))
    x.resolve(np.load("p5_params.npz"))
    mnist_test = np.load("mnist_test.npz")
    images = mnist_test["images"]
    labels = mnist_test["labels"]

    infer = x.compile(golden.Builder())
    pred_labels = infer(images = images).argmax(axis = 1)
    count = sum(labels == pred_labels)
    print(count)
    return count > 8500

def grade_Q2():
    relu = nn.ReLU()
    flatten = nn.Flatten()
    f1 = nn.Linear("f1", 28*28, 32)
    f2 = nn.Linear("f2", 32, 10)
    x = nn.Input2d("images", 28, 28, 1)
    x = flatten(x)
    x = f2(relu(f1(x)))
    x.resolve(np.load("p5_params.npz"))
    mnist_test = np.load("mnist_test.npz")
    images = mnist_test["images"]
    labels = mnist_test["labels"]

    infer = x.compile(golden.Builder())
    pred_labels = infer(images = images).argmax(axis = 1)
    count = sum(labels == pred_labels)
    return count > 9000

def grade_Q3():
    relu = nn.ReLU()
    flatten = nn.Flatten()
    f1 = nn.Linear("f1", 28*28, 32)
    f2 = nn.Linear("f2", 32, 10)
    x = nn.Input2d("images", 28, 28, 1)
    x = flatten(x)
    x = f2(relu(f1(x)))
    x.resolve(np.load("p5_params.npz"))
    mnist_test = np.load("mnist_test.npz")
    images = mnist_test["images"]
    labels = mnist_test["labels"]

    infer = x.compile(golden.Builder())
    pred_labels = infer(images = images).argmax(axis = 1)
    count = sum(labels == pred_labels)
    return count > 9300

def grade_Q4():
    relu = nn.ReLU()
    flatten = nn.Flatten()
    f1 = nn.Linear("f1", 28*28, 32)
    f2 = nn.Linear("f2", 32, 10)
    x = nn.Input2d("images", 28, 28, 1)
    x = flatten(x)
    x = f2(relu(f1(x)))
    x.resolve(np.load("p5_params.npz"))
    mnist_test = np.load("mnist_test.npz")
    images = mnist_test["images"]
    labels = mnist_test["labels"]

    infer = x.compile(golden.Builder())
    pred_labels = infer(images = images).argmax(axis = 1)
    count = sum(labels == pred_labels)
    return count > 9500

grade.grade_all("p5", 1, 5, globals())
