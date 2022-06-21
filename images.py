import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

options = {
    'model' : 'cfg/yolo.cfg',
    'load' : 'bin/yolo.weights',
    'threshold' : 0.3
}

tfnet = TFNet(options)

img = cv2.imread('test_images/curvy1.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = tfnet.return_predict(img)
print(result)

tl1 = (result[0]['topleft']['x'], result[0]['topleft']['y'])
br1 = (result[0]['topleft']['x'], result[0]['topleft']['y'])
label = result[0]['label']
img = cv2.rectangle(img, tl1, br1, (0, 0, 0), 7)
img = cv2.putText(img, label, tl1, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)


tl2 = (result[0]['topleft']['x'], result[0]['topleft']['y'])
br2 = (result[0]['topleft']['x'], result[0]['topleft']['y'])
label = result[0]['label']
img = cv2.rectangle(img, tl2, br2, (0, 0, 0), 7)
img = cv2.putText(img, label, tl2, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)


tl3 = (result[0]['topleft']['x'], result[0]['topleft']['y'])
br3 = (result[0]['topleft']['x'], result[0]['topleft']['y'])
label = result[0]['label']
img = cv2.rectangle(img, tl3, br3, (0, 0, 0), 7)
img = cv2.putText(img, label, tl3, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)


plt.imshow(img)
plt.show()
