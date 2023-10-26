from random import uniform

x = [[uniform(-1, 1) for _ in range(5)] for _ in range(1000)]
wt = [uniform(-1, 1) for _ in range(5)]
bt = uniform(-1, 1)
y = [sum([j*k+bt for j,k in zip(i, wt)]) for i in x]

w = [uniform(-1, 1) for _ in range(5)]
b = uniform(-1, 1)

def forward(x, w, b):
    s = 0
    for i in range(len(x)):
        s += x[i] * w[i]
    return s + b

def mod_w(w, h, index):
    w[index] = w[index] + h
    return w

def loss(x, y, w, b, h, index):
    return (y - forward(x, mod_w(w, h, index), b)) ** 2

def loss_b(x, y, w, b, h):
    return (y - forward(x, w, b+h)) ** 2

def grad(x, y, w, b, h, index):
    return (loss(x, y, w, b, h, index) - loss(x, y, w, b, -h, index)) / (2 * h)

def grad_b(x, y, w, b, h):
    return (loss_b(x, y, w, b, h) - loss_b(x, y, w, b, -h)) / (2 * h)

def overall_loss(x, y, w, b):
    loss = 0
    for index, x_row in enumerate(x):
        s = 0
        for i in range(len(w)):
            s += w[i] * x_row[i]
        s += b
        loss += (y[index] - s) ** 2
    loss /= len(x)
    return loss

h = 0.001
lr = 0.001
epochs = 10
print(f'Initial Loss: {overall_loss(x, y, w, b)}')
for _ in range(epochs):
    for i in range(len(x)):
        dw = []
        for w_i in range(len(w)):
            dw.append(grad(x[i], y[i], w, b, h, w_i))
        b -= lr * grad_b(x[i], y[i], w, b, h)
        for w_i in range(len(w)):
            w[w_i] -= lr * dw[w_i]
print(f'Final Loss: {overall_loss(x, y, w, b)}')

print(w, b)