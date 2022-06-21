import numpy as np
import cv2
import time

stime = time.time()
cap = cv2.VideoCapture("out.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


while(cap.isOpened()):
    stime = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()

    #from line_fit_video import annotate
    #annotate(frame)

    from ObjectDetection import objectDetection
    objectDetection(ret, frame)


    # Display the resulting frame
    cv2.imshow('Driver Assistance System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
print('FPS {:.1f}'.format(1 / (time.time() - stime)))
