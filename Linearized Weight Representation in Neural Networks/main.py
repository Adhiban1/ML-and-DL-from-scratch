import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

L1 = 50
L2 = 5

x_matrix = np.random.randn(1000, L1)
wt = np.random.randn(L1*L2)
bt = np.random.randn(L2)
f2 = lambda x: 1 / (1 + np.exp(-x))
df2 = lambda x: f2(x) * (1 - f2(x))
y_matrix = f2(x_matrix @ wt.reshape(L1, L2) + bt)

del(wt)
del(bt)

w = np.random.randn(L1*L2)
dw = np.zeros_like(w)
b = np.random.randn(L2)
db = np.zeros_like(b)
zl2 = np.zeros(L2)

n = 0
lr = 0.1
epochs = 10
losses = []
for epoch in range(epochs):
    for x, y in zip(x_matrix, y_matrix):
        loss = 0
        for l2 in range(L2):
            s2 = b[l2]
            for l1 in range(L1):
                s2 += x[l1] * w[l1+L1*l2]
            zl2[l2] = f2(s2)

            loss += (y[l2] - zl2[l2]) ** 2

            for l1 in range(L1):
                dw[l1+L1*l2] = -2 * (y[l2] - zl2[l2]) * df2(s2) * x[l1]

            db[l2] = -2 * (y[l2] - zl2[l2]) * df2(s2)
        for l2 in range(L2):
            for l1 in range(L1):
                w[l1+L1*l2] = w[l1+L1*l2] - lr * dw[l1+L1*l2]
            b[l2] = b[l2] - lr * db[l2]

        loss /= L2
    print(f'Loss: {loss}')
    losses.append(loss)
plt.plot(losses)
plt.show()