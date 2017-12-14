# coding=utf-8

# MNIST digits recognition using sklearn.neural_network.MLPClassifier
# The array is already divided by train and test set.
# All images of each digit are spliced in one picture.
# Size of one digit is 28х28 pixels.
# All pictures are already transferred to monochrome images.
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import metrics


def load(files, digits):
    # Наши цифры
    images = []
    # Сразу будем заполнять Y, чтобы потом не тратить время
    Y = []
    for i in range(0, len(files)):
        img = cv2.imread(files[i])
        height, width = img.shape[:2]
        img = img[0:height, 0:width - 28]  # Обрезаем неполный столбец справа
        # cv2.waitKey(0)
        height, width = img.shape[:2]
        h = 28
        w = 28
        for j in range(0, width, 28):
            for k in range(0, height, 28):
                crop_img = img[k:k + h, j:j + w]  # Берем 1 цифру
                images.append(crop_img)
                Y.extend([digits[i]])
    return images, Y


def prepare(images, features_extraction):
    features = []
    for image in images:
        features.append(features_extraction(image))
    return features


def projHist(image):
    # Для ускорения будем каждый раз просматривать в два направления подматрицы рангом k = r - 1, где
    # r - ранг исходной матрицы (28)
    features = []
    # Смещение
    offset = 0
    # При редуцировании матрицы теряются учтенные до этого пиксели - сохраняем
    col = np.empty(28)
    col.fill(0)
    row = np.empty(28)
    row.fill(0)
    for i in range(0, 28):
        count_row = 0
        count_col = 0
        for j in range(offset, 28):
            if image[i, j].any():
                count_row = count_row + 1
                # Сохраняем пиксель
                col[j] += 1
            if image[j, i].any():
                count_col = count_col + 1
                # Сохраняем пиксель
                row[j] += 1
        features.append(count_row + row[i])
        features.append(count_col + col[j])
        offset += 1
    return features


def zones(image):
    features = []
    # Зоны 4x4
    h = 4
    w = 4
    for j in range(0, 28, 4):
        for k in range(0, 28, 4):
            zone = image[k:k + h, j:j + w]  # Берем 1 зону
            count = 0
            for row in zone:
                for pixel in row:
                    if pixel.any():
                        count += 1
            features.append(count / 16.0)
    return features


def calculate(x_train, y_train, x_test, act_function, layer):
    clf = MLPClassifier(solver='lbfgs', activation=act_function, hidden_layer_sizes=layer, random_state=1)
    clf.fit(x_train, y_train)
    res = clf.predict(x_test)
    print act_function, "\t", layer, "\t", "\t", metrics.accuracy_score(Y_test, res)


layers = [(), (10), (50, 50), (60, 60, 60)]
functions = ["tanh", "relu"]
methods = [projHist, zones]
train, Y_train = load(
    ["mnist_train2.jpg", "mnist_train4.jpg", "mnist_train6.jpg"], [2, 4, 6])
test, Y_test = load(
    ["mnist_test2.jpg", "mnist_test4.jpg", "mnist_test6.jpg"], [2, 4, 6])
for method in methods:
    X_train = prepare(train, method)
    X_test = prepare(test, method)
    print method
    for layer in layers:
        for function in functions:
            calculate(X_train, Y_train, X_test, function, layer)
