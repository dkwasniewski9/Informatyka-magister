import tkinter as tk
from PIL import Image, ImageTk
import os
import random
import numpy as np


class ImageWindow:
    def __init__(self, root):
        self.num_pixels = None
        self.eta = 0.01
        self.current_image = 0
        self.weights = None
        self.root = root
        self.root.title("Asocjator obrazkowy")

        self.loaded_image = None
        self.displayed_image = None
        self.original_image = None
        self.load_button = tk.Button(self.root, text="Wczytaj obrazek", command=self.load_image)
        self.load_button.grid(row=0, column=0, columnspan=2)

        self.image_label = tk.Label(self.root)
        self.image_label.grid(row=1, column=0, columnspan=2)

        self.load_image()
        self.button_frame = tk.Frame(self.root)
        self.button_frame.grid(row=2, column=0, columnspan=2)

        self.five_button = tk.Button(self.button_frame, text="Zaszum", command=lambda: self.make_noise(0.05))
        self.five_button.grid(row=0, column=0)

        self.start_button = tk.Button(self.root, text="Start", command=self.start)
        self.start_button.grid(row=1, column=2, rowspan=2, sticky="nsew")
        self.start_button.config(width=4)
        self.train()

    def start(self):
        if self.weights is None:
            return
        image = np.array(self.loaded_image.copy()).flatten()
        for i in range(self.num_pixels):
            output = np.dot(self.weights[i, 1:], image) + self.weights[i, 0]
            image[i] = 0 if output < 0 else 255

        reshaped_image = image.reshape(self.loaded_image.size[::-1])

        self.loaded_image = Image.fromarray(reshaped_image)

        new_size = (500, 500)
        displayed_image = self.loaded_image.resize(new_size)

        photo = ImageTk.PhotoImage(displayed_image)

        self.image_label.config(image=photo)
        self.image_label.image = photo

    def train(self):
        image_dir = "images"
        image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        image_arrays = []

        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path).convert('L')
            image_array = np.array(image).flatten()
            image_arrays.append(image_array)

        image_arrays = np.array(image_arrays)
        image_arrays = image_arrays.astype('float64')
        image_arrays /= 255

        self.num_pixels = image_arrays.shape[1]
        self.weights = np.zeros((self.num_pixels, self.num_pixels + 1))
        best_weights = self.weights.copy()
        best_streak = 0
        num_images = len(image_arrays)
        correct_streak = 0
        for epoch in range(40):
            shuffles = random.sample(range(num_images), num_images)
            example_array = [image_arrays[i].copy() for i in shuffles]
            for image in example_array:
                if random.random() < 0.05:
                    image = self.make_training_noise(image, 0.05)
                for i in range(self.num_pixels):
                    output = np.dot(self.weights[i, 1:], image) + self.weights[i, 0]
                    output = 0 if output < 0 else 1
                    expected_output = image[i]
                    error = expected_output - output

                    if error != 0:
                        if correct_streak > best_streak:
                            best_streak = correct_streak
                            best_weights = self.weights.copy()
                        self.weights[i, 1:] += self.eta * error * image
                        self.weights[i, 0] -= self.eta * error
                        correct_streak = 0
                    else:
                        correct_streak += 1
        self.weights = best_weights

    def load_image(self):
        image_dir = "images"
        image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        image_file = image_files[self.current_image % len(image_files)]
        random_image_path = os.path.join(image_dir, image_file)
        self.loaded_image = Image.open(random_image_path).convert('L')

        new_size = (500, 500)
        displayed_image = self.loaded_image.resize(new_size)

        photo = ImageTk.PhotoImage(displayed_image)

        self.image_label.config(image=photo)
        self.image_label.image = photo
        self.current_image += 1

    def make_noise(self, percentage):
        self.original_image = self.loaded_image.copy()
        pixels = self.original_image.load()
        width, height = self.original_image.size
        for x in range(width):
            for y in range(height):
                if random.random() < percentage:
                    if pixels[x, y] == 0:
                        pixels[x, y] = 255
                    else:
                        pixels[x, y] = 0
        self.loaded_image = self.original_image
        new_size = (500, 500)
        displayed_image = self.loaded_image.resize(new_size)

        photo = ImageTk.PhotoImage(displayed_image)

        self.image_label.config(image=photo)
        self.image_label.image = photo

    def make_training_noise(self, image, noise_level):
        for i in range(self.num_pixels):
            if random.random() < noise_level:
                if image[i] == 0:
                    image[i] = 1
                else:
                    image[i] = 0
        return image


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageWindow(root)
    root.mainloop()
