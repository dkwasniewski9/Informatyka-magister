import os
import random
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QColor, QImage
from PyQt5.QtCore import Qt
import numpy as np


class GridCanvas(QWidget):
    def __init__(self, perceptrons):
        super().__init__()
        self.result = None
        self.save_grid = None
        self.check_button = None
        self.grid = None
        self.image_counter = 0
        self.perceptrons = perceptrons
        self.init()

    def paintEvent(self, event):
        painter = QPainter(self)
        for i in range(7):
            for j in range(5):
                color = Qt.white if self.grid[i][j] == 0 else Qt.black
                painter.fillRect(j * 50, i * 50, 50, 50, QColor(color))
        for i in range(1, 7):
            painter.drawLine(0, i * 50, 5 * 50, i * 50)
        for j in range(1, 5):
            painter.drawLine(j * 50, 0, j * 50, 50 * 7)

    def init(self):
        self.setWindowTitle('Perceptrony')
        width = 450
        height = 450
        self.setGeometry(100, 100, width, height)
        self.grid = [[0] * 5 for _ in range(7)]
        self.save_grid = [[0] * 5 for _ in range(7)]
        self.check_button = QPushButton('Sprawdź', self)
        self.check_button.clicked.connect(self.check_image)
        self.result = QLabel()
        self.result.setText(self.perceptrons.train_data)
        layout = QVBoxLayout()
        layout.addWidget(self.result, alignment=Qt.AlignRight)
        layout.addWidget(self.check_button, alignment=Qt.AlignBottom)
        self.setLayout(layout)

    def mousePressEvent(self, event):
        x = event.x() // 50
        y = event.y() // 50
        if 0 <= x < 5 and 0 <= y < 7:
            if event.buttons() & Qt.LeftButton:
                self.grid[y][x] = 1
                self.save_grid[y][x] = 1
            else:
                self.grid[y][x] = 0
                self.save_grid[y][x] = 0
            self.update()

    def clear(self):
        self.grid = [[0] * 5 for _ in range(7)]
        self.save_grid = [[0] * 5 for _ in range(7)]
        self.update()

    def check_image(self):
        user_image = np.empty(35)
        for i in range(7):
            for j in range(5):
                if self.save_grid[i][j] == 0:
                    user_image[i * 5 + j] = 0
                else:
                    user_image[i * 5 + j] = 1
        f = np.zeros(10, int)
        for n in range(10):
            f[n] = np.dot(user_image, self.perceptrons.w[n][1:])
        text = ''
        for i in range(10):
            text += str(i)
            text += ': false\n' if f[i] < 0 else ': true\n'
        self.result.setText(text)
        self.result.update()
        self.update()


def qimagetoarray(image):
    result = np.zeros(35)
    width, height = image.width(), image.height()
    for y in range(height):
        for x in range(width):
            color = image.pixelColor(x, y)
            result[y * width + x] = 1 if color.black() > 128 else 0
    return result


class Perceptrons:
    def __init__(self):
        self.w = np.random.uniform(-0.5, 0.5, [10, 36])
        self.eta = 0.05
        self.images = dict()
        self.load_images()
        self.train_data = ''
        self.train()

    def train(self):
        for n in range(10):
            pocket = np.zeros(38)
            current_streak = 0
            for i in range(20):
                random_list = list(range(0, 50))
                random.shuffle(random_list)
                for random_example in random_list:
                    answer = 1 if random_example // 5 == n else 0
                    example = qimagetoarray(self.images[str(random_example // 5)][random_example % 5])
                    O = np.dot(example, self.w[n][1:]).sum()
                    O = 0 if O < 0 else 1
                    ERR = answer - O
                    if ERR != 0:
                        for j in range(35):
                            self.w[n][j] = self.w[n][j] + self.eta * ERR * example[j]
                        self.w[n][0] = self.w[n][0] - self.eta * ERR
                    else:
                        current_streak += 1
                        if current_streak > pocket[0]:
                            correct = self.check_all_images(self.w[n], n)
                            if correct > pocket[1]:
                                pocket[0] = current_streak
                                pocket[1] = correct
                                pocket[2:] = self.w[n].copy()
            self.train_data += f'{n}: {self.check_all_images(self.w[n], n) / 0.5}%\n'
            self.train_data += f'with Pocket: {self.check_all_images(pocket[2:], n) / 0.5}%\n'
            self.w[n] = pocket[2:]

    def check_all_images(self, weights, number):
        correct = 0
        for i in range(10):
            for j in range(5):
                image = qimagetoarray(self.images[str(i)][j])
                output = np.dot(image, weights[1:]) + weights[0]
                answer = 1 if i == number else 0
                prediction = 1 if output >= 0 else 0
                correct += 1 if prediction == answer else 0
        return correct

    def load_images(self):
        path = os.path.join(os.getcwd(), 'images')
        directories = os.listdir(path)
        for directory in directories:
            self.images[directory] = []
            for file in os.listdir(os.path.join(path, directory)):
                file = os.path.join(path, directory, file)
                self.images[directory].append(QImage(file))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GridCanvas(Perceptrons())
    window.show()
    sys.exit(app.exec_())
