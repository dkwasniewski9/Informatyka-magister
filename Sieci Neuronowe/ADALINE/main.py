from keras.datasets import mnist
import numpy as np
import random

(train_X, train_y), (test_X, test_y) = mnist.load_data()
n = 28 * 28
w = np.random.uniform(-0.2, 0.2, [10, n + 1])
w[0] = 0.01
eta = 0.001
shuffled_examples = list(range(0, 60000))
random.shuffle(shuffled_examples)

train_X = train_X.reshape(60000, 784) / 255.0
test_X = test_X.reshape(10000, 784) / 255.0
for n in range(10):
    for i in shuffled_examples:
        O = np.dot(w[n][1:], train_X[i]) + w[n][0]
        C = 1 if n == train_y[i] else -1
        for j in range(28 * 28):
            w[n][j + 1] = w[n][j + 1] + eta * (C - O) * train_X[i][j]
        w[n][0] = w[n][0] + eta * (C - O)

results = np.zeros(10)
correct_answers = 0
for i in range(0, 10000):
    for n in range(10):
        results[n] = np.dot(w[n][1:], test_X[i]) + w[n][0]
    max_value = -1
    max_index = -1
    for j, value in enumerate(results):
        if value > max_value:
            max_value = value
            max_index = j
    if max_index == test_y[i]:
        correct_answers += 1

print(correct_answers / 10000 * 100, '%')
