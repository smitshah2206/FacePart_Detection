import cv2
eye_detector=cv2.CascadeClassifier('data/haarcascade_eye_tree_eyeglasses.xml')
face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
cap=cv2.VideoCapture(0)
while(cap.isOpened()):
    _,img=cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(gray,1.3,5)
    if len(faces)==1:
        for (x,y,w,h) in faces:
            hx=x
            hy=y-120
            hw=w
            hh=h-100
            cv2.rectangle(img,(hx,hy),(hx+hw,hy+hh),(0,255,0),2)
    cv2.imshow("Head Detector",img)
    k=cv2.waitKey(10)
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
