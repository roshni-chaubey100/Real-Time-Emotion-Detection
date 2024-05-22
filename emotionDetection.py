import cv2
from keras.models import load_model
import numpy as np
import random
import csv

model = load_model(r'emotion_detector_model.h5')

model.load_weights("emotion_detector_model.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

webcam=cv2.VideoCapture(0)

#Mapping the emotion labels to traits and micro-traits as per the "Manwatching" book.
arr = []
with open('mapping.csv', mode ='r')as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
	    arr.append(lines)
         
angry = random.choice(arr[1:17])
disgust = random.choice(arr[17:32])
fear = random.choice(arr[32:47])
happy = random.choice(arr[47:62])
neutral = random.choice(arr[62:77])
sad = random.choice(arr[77:92])
surprised = random.choice(arr[92:])


labels = {0 : angry, 1 : disgust, 2 : fear, 3 : happy, 4 : neutral, 5 : sad, 6 : surprised}
while True:
    i,im=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(im,1.3,5)
    try: 
        for (p,q,r,s) in faces:
            image = gray[q:q+s,p:p+r]
            cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
            image = cv2.resize(image,(48,48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            
            cv2.putText(img = im, text = '% s' %(prediction_label), org = (p-10, q-10),fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale = 1, color = (125, 246, 55), thickness=1)
        # cv2.imshow("Output",im)
        cv2.imshow('Output', cv2.resize(im, (1500,960), interpolation=cv2.INTER_CUBIC))
        # cv2.waitKey(27)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except cv2.error:
        pass
