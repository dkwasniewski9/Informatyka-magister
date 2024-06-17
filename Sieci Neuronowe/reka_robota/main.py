import tkinter as tk
import math
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return x * (1 - x)


def plot_errors(mean_errors):

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(mean_errors)
    plt.title('Training Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')

    plt.show()


class RoboticArmGUI:
    def __init__(self, master):
        self.master = master
        self.arm_length = 100
        self.arm = None
        self.target_point = None
        self.arm_starting_point = (0, 200)
        self.layers = []
        self.hidden_layers = 3
        self.neurons_per_layer = 10
        self.max_x = 0
        self.max_y = 0
        self.eta = 0.01
        self.errors = []
        self.examples = None
        self.example_number = 2000
        self.generate_examples()
        self.weights = []
        self.train(2000)
        self.canvas = tk.Canvas(master, width=self.max_x, height=self.max_y)
        self.canvas.bind("<Button-1>", self.move_arm)
        self.canvas.pack()
        self.draw_robot_arm(self.examples[0]['alpha'], self.examples[0]['beta'])

    def move_arm(self, event):
        activations = self.forward(event.x / self.max_x, event.y / self.max_y)
        self.draw_robot_arm(activations[-1][0][0] * np.pi, activations[-1][0][1] * np.pi)

    def calculate_new_point(self, x, y, angle):
        new_x = x + self.arm_length * math.sin(angle)
        new_y = y + self.arm_length * math.cos(angle)
        return new_x, new_y

    def generate_examples(self):
        self.examples = []
        self.max_x = 2 * self.arm_length + self.arm_starting_point[0]
        self.max_y = 2 * self.arm_length + self.arm_starting_point[1]
        for _ in range(0, self.example_number):
            example = {}
            x1, y1 = self.arm_starting_point
            alpha = np.random.uniform(0, np.pi)
            x2, y2 = self.calculate_new_point(x1, y1, alpha)
            beta = np.random.uniform(0, np.pi)
            x3, y3 = self.calculate_new_point(x2, y2, np.pi - beta + alpha)
            example['x'] = x3 / self.max_x
            example['y'] = y3 / self.max_y
            example['alpha'] = alpha / np.pi
            example['beta'] = beta / np.pi
            self.examples.append(example)

    def initialize_weights(self):
        self.layers.append(2)
        for i in range(self.hidden_layers):
            if i == self.hidden_layers // 2:
                self.layers.append(self.neurons_per_layer * 2)
            else:
                self.layers.append(self.neurons_per_layer)
        self.layers.append(2)
        self.weights = []
        for i in range(len(self.layers) - 1):
            self.weights.append((np.random.randn(self.layers[i], self.layers[i + 1]) - 0.5))

    def forward(self, x, y):
        activations = [np.array([[x, y]])]
        for i in range(len(self.weights)):
            inputs = np.dot(activations[-1], self.weights[i])
            activations.append(sigmoid(inputs))
        return activations

    def backward(self, alpha, beta, activations):
        delta = []
        epsilons = [np.array([[alpha, beta]]) - activations[-1]]
        self.errors = [epsilons[-1]]
        for i in range(len(self.weights) - 1):
            delta.append(epsilons[i] * sigmoid_prime(activations[-1 - i]))
            epsilons.append(np.dot(delta[i], self.weights[-1 - i].T))
        delta.append(epsilons[-1] * sigmoid_prime(activations[1]))
        delta.reverse()
        for i in range(len(self.weights)):
            self.weights[i] += self.eta * np.dot(activations[i].T, delta[i])

    def train(self, epochs):
        self.initialize_weights()
        mean_errors = []
        for epoch in range(epochs):
            print(epoch)
            self.errors = []
            for example in self.examples:
                activations = self.forward(example['x'], example['y'])
                self.backward(example['alpha'], example['beta'], activations)
            mean_error = np.mean(np.square(self.errors))
            mean_errors.append(mean_error)
        plot_errors(mean_errors)

    def draw_robot_arm(self, alpha, beta):
        self.canvas.delete("all")
        x2, y2 = self.calculate_new_point(self.arm_starting_point[0], self.arm_starting_point[1], alpha)

        self.canvas.create_line(self.arm_starting_point[0], self.arm_starting_point[1],
                                x2, y2, fill="black",
                                width=2)
        x3, y3 = self.calculate_new_point(x2, y2, np.pi - beta + alpha)
        self.canvas.create_line(x2, y2,
                                x3, y3, fill="black",
                                width=2)


root = tk.Tk()
app = RoboticArmGUI(root)
root.mainloop()
