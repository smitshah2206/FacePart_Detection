import cv2
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('data/haarcascade_mcs_nose.xml')
cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feed = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        nose = nose_cascade.detectMultiScale(roi_gray)
        if len(nose)==1:
            for (nx,ny,nw,nh) in nose:
                cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),2)
    cv2.imshow('Nose Detector',img)
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
