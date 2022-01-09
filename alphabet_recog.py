import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps
import os
import time
import ssl

x = np.load('image.npz')['arr_0']
y = pd.read_csv('alphabets_labels.csv')['labels']

print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
#assigning the length of the classes 
nclasses = len(classes)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state = 9, train_size = 7500, test_size = 2500)

xtrain_scale = xtrain / 255.0
xtest_scale = xtest / 255.0

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(xtrain_scale, ytrain)

ypred = clf.predict(xtest_scale)
accuracy = accuracy_score(ytest, ypred)

print('accuracy is: ', accuracy * 100, '%')

capture = cv2.VideoCapture(0)

while True:
    try:
        ret, frame = capture.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape

        upper_left = (int(width / 2 - 56), int(height / 2 - 56))
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))

        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

        #roi = region of interest
        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
        
        im_pil = Image.fromarray(roi)
        image_bw = im_pil.convert('L')

        image_bw_resize = image_bw.resize((28, 28), Image.ANTIALIAS)

        image_bw_resize_inverted = PIL.ImageOps.invert(image_bw_resize)

        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resize_inverted, pixel_filter)

        image_bw_resize_inverted_scale = np.clip(image_bw_resize_inverted - min_pixel, 0, 255)

        max_pixel = np.max(image_bw_resize_inverted)

        image_bw_resize_inverted_scale = np.asarray(image_bw_resize_inverted_scale) / max_pixel

        test_sample = np.array(image_bw_resize_inverted_scale).reshape(1, 784)

        test_pred = clf.predict(test_sample)
        print('predicted alphabet is: ', test_pred)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
    except Exception as e:
        pass

capture.release()
cv2.destroyAllWindows()
