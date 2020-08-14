import cv2
import os
import time
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('http://192.168.43.1:8080/video')
cap = cv2.VideoCapture('http://192.168.1.16:8080/video')


import numpy as np

path = "vlong"
count = 0
while(cap.isOpened()):
    cnt = np.random.rand(1)[0]
    count += 1

    ret, frame = cap.read()
    frame = cv2.resize(frame, (640,480))
    # time.sleep(0.1)
    if ret == True:
        result = detector.detect_faces(frame)
        if len(result) == 1:
          for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
            cv2.imwrite(path + "/" + str(cnt) + ".jpg",frame) 

            # cv2.imwrite(os.path.join(path, str(cnt)), frame)
            if count %5 == 0:
              cv2.imwrite(os.path.join(path , str(cnt)) + ".jpg", frame)
            cnt += 1
            cv2.rectangle(frame,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
            cv2.putText(frame, "person", (bounding_box[0], bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 255, 30), 2, cv2.LINE_AA)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    else:
        break

cap.release()
cv2.destroyAllWindows()