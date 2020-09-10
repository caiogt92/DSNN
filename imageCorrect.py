import cv2
import numpy as np
from keras.models import load_model
import os
from os import listdir
from os.path import isfile, join
import csv

#Path to the model created with model.py
model = load_model("")
#Path to the test folder containing the images you want to correct the orientation
path = ""
#Path to the folder where you want save the new corrected images
pathS = ""


def convert(input_data, direction, filename):
    if direction == 0:
        text = "rotated_left"
        input_data = cv2.rotate(input_data, cv2.ROTATE_90_CLOCKWISE)
    if direction == 1:
        text = "rotated_right"
        input_data = cv2.rotate(input_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if direction == 2:
        text = "upright"
    if direction == 3:
        text = "upside_down"
        input_data = cv2.rotate(input_data, cv2.ROTATE_180)    
    
    prediction_writer.writerow([filename, text])
    
    filename = os.path.splitext(filename)[0]+'.png'
    cv2.imwrite(pathS + filename, input_data)
    return input_data
    


listfiles = [f for f in listdir(path) 
             if isfile(join(path, f))]

mylist = []

with open('test.preds.csv', mode='w', newline='') as predictions:
    prediction_writer = csv.writer(predictions, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    prediction_writer.writerow(["fn", "label"])
    for image in listfiles:
        img = cv2.imread(path + image)
        img2 = np.reshape(img, [1, 64, 64, 3])       
        img3 = convert(img, model.predict_classes(img2), image)
        mylist.append(img3)
    
np.save('result', np.array(mylist))