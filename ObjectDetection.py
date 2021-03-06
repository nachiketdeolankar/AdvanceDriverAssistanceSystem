import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

def objectDetection(ret, frame):
    option = {
      'model' : 'cfg/yolo.cfg',
      'load' : 'bin/yolo.weights',
      'threshold' : 0.2
    }

    tfnet = TFNet(option)

    colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

    capture = cv2.VideoCapture('out.mp4')


    while True:
        results = tfnet.return_predict(frame)

        if ret:
            for color, result in zip(colors, results):
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                confidence = result['confidence']
                text = '{}: {:.0f}%'.format(label, confidence * 100)
                frame = cv2.rectangle(frame, tl, br, color, 5)
                frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0) , 2)
            return frame
