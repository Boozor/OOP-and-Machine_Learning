import numpy as np
import time
def save_theta(theta):
    f1_W, f1_b, f2_W, f2_b = theta

    np.savez_compressed("p5_params.npz", **{
        "f1.weight": f1_W,
        "f1.bias": f1_b,
        "f2.weight": f2_W,
        "f2.bias": f2_b
    })
def initialize_theta(bound):
    f1_W = np.random.uniform(-bound, bound, (32, 784))
    f1_b = np.random.uniform(-bound, bound, 32)
    f2_W = np.random.uniform(-bound, bound, (10, 32))
    f2_b = np.random.uniform(-bound, bound, 10)
    return (f1_W, f1_b, f2_W, f2_b)


def forward(images, theta):
    # number of samples
    N = images.shape[0]

    # unpack theta into f1 and f2
    f1_W, f1_b, f2_W, f2_b = theta

    # x = Flatten(images)
    x = images.astype(float).transpose(0, 3, 1, 2).reshape((N, -1))

    # g = Linear_f1(x)
    g = np.zeros((N, f1_b.shape[0]))

    for i in range(N):
        g[i, :] = np.matmul(f1_W, x[i]) + f1_b

    # h = ReLU(g)
    h = g * (g > 0)

    # z = Linear_f2(h)
    z = np.zeros((N, f2_b.shape[0]))
    for i in range(N):
        z[i, :] = np.matmul(f2_W, h[i]) + f2_b

    return (z, h, g, x)


def backprop(labels, theta, z, h, g, x):
    # number of samples
    N = labels.shape[0]

    # unpack theta into f1 and f2
    f1_W, f1_b, f2_W, f2_b = theta

    # nabla_J consists of partial J to partial f1_W, f1_b, f2_W, f2_b
    p_f1_W = np.zeros(f1_W.shape)
    p_f1_b = np.zeros(f1_b.shape)
    p_f2_W = np.zeros(f2_W.shape)
    p_f2_b = np.zeros(f2_b.shape)
    p_h = 0
    p_g = 0
    for i in range(N):
        expz = np.exp(z[i] - max(z[i]))
        p_z = expz / sum(expz)
        d_p_z = np.zeros(p_z.shape)
        d_p_z[batch_labels[i]] = 1
        diff_p_z = (p_z - d_p_z)
        p_f2_W = +np.matmul(diff_p_z.reshape(10, 1), h[i].reshape(32, 1).T)
        p_f2_b = +diff_p_z
        p_h = +np.matmul(f2_W.T, diff_p_z.reshape(10, 1))
        p_g = +np.multiply(p_h, g[i].reshape(32, 1) > 1)
        p_f1_W = +np.matmul(p_g, x[i].reshape(784, 1).T)
        p_f1_b = +p_g
    p_f1_W = p_f1_W / N
    p_f1_b = (p_f1_b / N).reshape(32, )
    p_f2_W = p_f2_W / N
    p_f2_b = (p_f2_b / N).reshape(10, )

    return (p_f1_W, p_f1_b, p_f2_W, p_f2_b)

def update(theta, nabla_J, epsilon):
 update=[]
 for i in range(0,4):
    update.append(theta[i]-epsilon*nabla_J[i])
 return update
np.random.seed(20480163)

# load training data and split them for validation/training
mnist_train = np.load("mnist_train.npz")
validation_images = mnist_train["images"][:1000]
validation_labels = mnist_train["labels"][:1000]
training_images = mnist_train["images"][1000:]
training_labels = mnist_train["labels"][1000:]
# print(validation_images.shape)
# print(validation_labels.shape)
# print(training_images.shape)
# print(training_labels.shape)

bound = 0.01
epsilon =0.00001
batch_size = 1

start = time.time()
theta = initialize_theta(bound)
batches = training_images.shape[0] // batch_size
for epoch in range(10):
    indices = np.arange(training_images.shape[0])
    np.random.shuffle(indices)
    for i in range(batches):
        batch_images = training_images[indices[i * batch_size:(i + 1) * batch_size]]
        batch_labels = training_labels[indices[i * batch_size:(i + 1) * batch_size]]

        z, h, g, x = forward(batch_images, theta)
        nabla_J = backprop(batch_labels, theta, z, h, g, x)
        theta = update(theta, nabla_J, epsilon)

    # check accuracy using validation examples
    z, _, _, _ = forward(validation_images, theta)
    pred_labels = z.argmax(axis=1)
    count = sum(pred_labels == validation_labels)
    print("epoch %d, accuracy %.3f, time %.2f" % (
        epoch, count / validation_images.shape[0], time.time() - start))

# save the weights to be submitted
save_theta(theta)