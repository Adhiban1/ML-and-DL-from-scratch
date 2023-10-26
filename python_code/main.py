from random import uniform

def mat_dot(a, b):
    c = []
    for i in range(len(a)):
        temp = []
        for k in range(len(b[0])):
            s = 0
            for j in range(len(a[0])):
                s += a[i][j] * b[j][k]
            temp.append(s)
        c.append(temp)
    return c

def vec_dot_mat(a, b):
    c = []
    for k in range(len(b[0])):
        s = 0
        for j in range(len(a)):
            s += a[j] * b[j][k]
        c.append(s)
    return c

def rand_matrix(i, j):
    return [[uniform(-1, 1) for _ in range(j)] for _ in range(i)]

def rand_vec(i):
    return [uniform(-1, 1) for _ in range(i)]

x = rand_matrix(10, 10)
wt1 = rand_matrix(10, 5)
bt1 = uniform(-1, 1)
wt2 = rand_matrix(10, 1)
bt2 = uniform(-1, 1)

w1 = rand_matrix(10, 5)
b1 = uniform(-1, 1)
w2 = rand_matrix(10, 1)
b2 = uniform(-1, 1)

def forward(x, w1, w2, b1, b2):
    z1 = [i+b1 for i in vec_dot_mat(x, w1)]
    z2 = [i+b2 for i in vec_dot_mat(z1, w2)]
    return z2

y = [forward(i, wt1, wt2, bt1, bt2) for i in x]

def mod_w(w, index, h):
    w[index[0]][index[1]] += h
    return w

def loss(x, y, w1, w2, b1, b2):
    l = 0
    yp = forward(x, w1, w2, b1, b2)
    for i in range(len(y)):
        l += (y[i] - yp[i]) ** 2
    l /= len(y)
    return l

def overall_loss(x, y, w1, w2, b1, b2):
    yp = [forward(i, w1, w2, b1, b2) for i in x]
    l = 0
    for i in range(len(y)):
        for j in range(len(y[0])):
            l += (y[i][j] - yp[i][j]) ** 2
    l /= len(y) * len(y[0])
    return l

def grad(x, y, w1, w2, b1, b2, h, lr):
    dw1 = []
    for i in range(len(w1)):
        temp = []
        for j in range(len(w1[0])):
            temp.append((loss(x, y, mod_w(w1, (i, j), h), w2, b1, b2) - loss(x, y, mod_w(w1, (i, j), -h), w2, b1, b2)) / (2 * h))
        dw1.append(temp)
    
    dw2 = []
    for i in range(len(w2)):
        temp = []
        for j in range(len(w2[0])):
            temp.append((loss(x, y, w1, mod_w(w2, (i, j), h), b1, b2) - loss(x, y, w1, mod_w(w2, (i, j), -h), b1, b2)) / (2 * h))
        dw2.append(temp)
    
    db1 = (loss(x, y, w1, w2, b1+h, b2) - loss(x, y, w1, w2, b1-h, b2)) / (2 * h)
    db2 = (loss(x, y, w1, w2, b1, b2+h) - loss(x, y, w1, w2, b1, b2-h)) / (2 * h)

    for i in range(len(w1)):
        for j in range(len(w1[0])):
            w1[i][j] -= lr * dw1[i][j]

    for i in range(len(w2)):
        for j in range(len(w2[0])):
            w2[i][j] -= lr * dw2[i][j]
    
    b1 -= lr * db1
    b2 -= lr * db2

    return w1, w2, b1, b2

epochs = 1000
lr = 0.01
h = 0.001
opt_w_b = (w1, w2, b1, b2)
lowest_loss = overall_loss(x, y, w1, w2, b1, b2)
print('Inital loss: {}'.format(lowest_loss))
for _ in range(epochs):
    for x_row, y_row in zip(x,y):
        w1, w2, b1, b2 = grad(x_row, y_row, w1, w2, b1, b2, h, lr)
    l = overall_loss(x, y, w1, w2, b1, b2)
    if l < lowest_loss:
        lowest_loss = l
        opt_w_b = (w1, w2, b1, b2)
print('Lowest loss: {}'.format(lowest_loss))
print('Final loss: {}'.format(overall_loss(x, y, w1, w2, b1, b2)))