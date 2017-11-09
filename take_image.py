import numpy as np
import cv2
#from resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50, decode_predictions
import matplotlib.pyplot as plt
import numpy as np
import time
import schedule
from pythonosc import osc_message_builder
from pythonosc import udp_client

client = udp_client.SimpleUDPClient("10.196.9.201",12000)

def job():
    print("I'm working...")
    model = ResNet50(weights='imagenet', include_top=True)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    while(True):
        for i in range(10):
            cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
            ret,frame = cap.read() # return a single frame in variable `frame`
            #cv2.imshow('image',frame) #display the captured image
            cv2.imwrite('images/c1.png',frame)
            cv2.destroyAllWindows()
            img_path = 'images/c1.png'
            im = cv2.resize(cv2.imread(img_path), (224, 224))
            im = np.expand_dims (im, axis=0)
            out=model.predict(im)
            mes = decode_predictions(out, top=1)[0]
            output = str(mes[0][1])
            client.send_message("/filter", output)
            print(output)
            break
            cap.release()


schedule.every(1).seconds.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)