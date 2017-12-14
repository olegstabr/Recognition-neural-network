# coding=utf-8

import cv2
import numpy as np
from math import sqrt
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

def load(files,digits):
    images = []
    Y = []
    for i in range(0,digits_len):
        img = cv2.imread(files[i], 0)
        height, width = img.shape[:2]
        img = img[0:height, 0:width-s]
        height, width = img.shape[:2]
        for j in range(0,width,s):
            for k in range(0,height,s):
                crop_img = img[k:k + s, j:j + s]
                images.append(crop_img)
                Y.extend([digits[i]])
    return images,Y

def prepare(images,features_extraction):
    features = []
    for image in images:
        features.append(features_extraction(image, k, c))
    return features

def zones(image, k, c):
    features = []
    w = s / k
    count = [0 for x in range (k*k)]
    for i in range (s):
        for j in range (s):
            if image[i][j] > c:
                count[i//w*k + j//w] += 1
    for l in range (k*k):
        count[l] /= float(w*w)
    features.extend(count)
    return features

def projections(image, k, c):
    features = []
    w = s / k
    count = [0 for x in range (k*s)]
    for i in range (s):
        for j in range (s):
            if image[i][j] > c:
                count[i*k + j//w] = 1
    features.extend(count)
    return features

def projhist(image, k, c):
    features = []
    count = [0 for x in range (s+s)]
    for i in range (s):
        for j in range (s):
            if image[i][j] > c:
                count[i] += 1
                count[s+j] +=1
    features.extend(count)
    return features

def calculate(X_train, Y_train, X_test, act_function, layer):
    clf = MLPClassifier(solver='lbfgs',
                        activation=act_function,
                        hidden_layer_sizes=layer,
                        random_state=1)
    clf.fit(X_train, Y_train)
    res = clf.predict(X_test)
    print (act_function + " (" + str(2+len(layer)) + " layers):  " + str(round(metrics.accuracy_score(Y_test, res),3)))

def makelayers(minl,maxl,method):
    n = 0
    layers = []
    if method == 'projections':
        n = int(round(sqrt(s*k*digits_len)))
    if method == 'zones':
        n = int(round(sqrt(k*k*digits_len)))
    if method == 'projhist':
        n = int(round(sqrt((s+s)*digits_len)))
    maxl -= 2
    minl -= 2
    for i in range(minl,maxl+1):
        el = []
        for j in range(i):
            el.append(n)
        layers.append(tuple(el))
    return layers, n

print ("Input recognizing digits separated with whitespaces: ")
digits = list(map(int,raw_input().split()))
print "Input min number of layers: "
min_layers = int(raw_input())
print "Input max number of layers: "
max_layers = int(raw_input())
print "Input activation functions separated with whitespaces: "
functions = raw_input().split()
print "Input preprocessing methods separated with whitespaces (zones, projections, projhist): "
methods = raw_input().split()
print "Input number of zones/areas separated with whitespaces (arrays/lists are allowed): "
num_of_areas = list(map(int,raw_input().split()))
print "Input color codes separated with whitespaces (arrays/lists are allowed). For example, 255 is white: "
colors = list(map(int,raw_input().split()))

digits_len = len(digits)

s = 28

images_train = []
images_test = []
for digit in digits:
    images_train.append("data/train/mnist_train"+str(digit)+".jpg")
    images_test.append("data/test/mnist_test"+str(digit)+".jpg")
train, Y_train = load(images_train,digits)
test, Y_test   = load(images_test,digits)

for k in num_of_areas:
    for c in colors:
        print "=============== k =",k," color >",c," ================"
        methods_dict = {'zones': zones,
                        'projections': projections,
                        'projhist': projhist}
        for m in methods:
            method = methods_dict[m]
            layers, num_of_neurons = makelayers(min_layers, max_layers, m)
            X_train = prepare(train,method)
            X_test = prepare(test,method)
            print "-----------------------------------------"
            print (m+" ("+str(num_of_neurons)+" neurons in each layer)")
            print "-----------------------------------------"
            for layer in layers:
                for function in functions:
                    calculate(X_train,Y_train,X_test,function,layer)

