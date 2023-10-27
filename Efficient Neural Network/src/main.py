from random import uniform
from copy import copy
import matplotlib.pyplot as plt
import sys


class NN:
    def __init__(self, layers):
        self.layers = layers
        self.wts = []
        self.w_size = 0
        for i in range(len(layers) - 1):
            temp = layers[i : 2 + i][::-1]
            self.w_size += temp[0] * temp[1]
            self.wts.append(temp)
        self.w = [uniform(-1, 1) for _ in range(self.w_size)]
        self.b = [uniform(-1, 1) for _ in range(len(self.layers) - 1)]
        self.losses = []

    def forward(self, x):
        shift = 0
        for index, (r, c) in enumerate(self.wts):
            z = []
            for i in range(r):
                s = 0
                for j in range(c):
                    s += x[j] * self.w[shift + i * c + j]
                s += self.b[index]
                z.append(s)
            # print(z)
            shift += r * c
            x = z
        return x

    def forward_w(self, x, w, b, wi, h):
        shift = 0
        for index, (r, c) in enumerate(self.wts):
            z = []
            for i in range(r):
                s = 0
                for j in range(c):
                    if wi == shift + i * c + j:
                        s += x[j] * (w[shift + i * c + j] + h)
                    else:
                        s += x[j] * w[shift + i * c + j]
                s += b[index]
                z.append(s)
            # print(z)
            shift += r * c
            x = z
        return x

    def forward_b(self, x, w, b, bi, h):
        shift = 0
        for index, (r, c) in enumerate(self.wts):
            z = []
            for i in range(r):
                s = 0
                for j in range(c):
                    s += x[j] * w[shift + i * c + j]
                if index == bi:
                    s += b[index] + h
                else:
                    s += b[index]
                z.append(s)
            # print(z)
            shift += r * c
            x = z
        return x

    def predict(self, x):
        return [self.forward(i) for i in x]

    def loss(self, yt, y):
        l = 0
        for i, j in zip(yt, y):
            l += (i - j) ** 2
        l /= len(yt)
        return l

    def overall_loss(self, yt, y):
        l = 0
        for i in range(len(yt)):
            for j in range(len(yt[0])):
                l += (yt[i][j] - y[i][j]) ** 2
        l /= len(yt) * len(yt[0])
        return l

    def fit(self, x, y, epochs, h, lr):
        for epoch in range(epochs):
            for xi in range(len(x)):
                dw = []
                for wi in range(self.w_size):
                    wl = copy(self.w)
                    wl[wi] += h
                    wr = copy(self.w)
                    wr[wi] -= h
                    dwi = (
                        self.loss(y[xi], self.forward_w(x[xi], self.w, self.b, wi, h))
                        - self.loss(
                            y[xi], self.forward_w(x[xi], self.w, self.b, wi, -h)
                        )
                    ) / (2 * h)
                    dw.append(dwi)

                db = []
                for bi in range(len(self.b)):
                    bl = copy(self.b)
                    bl[bi] += h
                    br = copy(self.b)
                    br[bi] -= h
                    dbi = (
                        self.loss(y[xi], self.forward_b(x[xi], self.w, self.b, bi, h))
                        - self.loss(
                            y[xi], self.forward_b(x[xi], self.w, self.b, bi, -h)
                        )
                    ) / (2 * h)
                    db.append(dbi)

                for i in range(len(dw)):
                    self.w[i] -= lr * dw[i]

                for i in range(len(db)):
                    self.b[i] -= lr * db[i]
            l = self.overall_loss(y, self.predict(x))
            self.losses.append(l)
            print(f"\r{epoch+1}/{epochs}: loss: {l}    ", end="", flush=True)
        print()

    def loss_plot(self, path=None):
        plt.plot(self.losses)
        if path:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()


layers = list(map(int, sys.argv[1:]))
x = [[uniform(0, 1) for _ in range(layers[0])] for _ in range(100)]

true_nn = NN(layers)
y = true_nn.predict(x)

nn = NN(layers)
print(f"Initial Loss: {nn.overall_loss(y, nn.predict(x))}")
nn.fit(x, y, 100, 0.1, 0.01)
print(f"Final Loss: {nn.overall_loss(y, nn.predict(x))}")
nn.loss_plot()
