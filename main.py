import cv2
import numpy as np
import tensorflow as tf

#facedetector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam  = cv2.VideoCapture(0)
model = tf.keras.models.load_model('nabin-010.model')
labels_dict={1:'MASK',0:'NO MASK'}
    
while(True):
    ret, frame = cam.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray,1.3,5)
   
    for (x,y,w,h) in face:  

        image = gray[y:y+w,x:x+w] 
        resized=cv2.resize(image,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result = np.argmax(model.predict(reshaped))
        print('this is result',result)

        if result==0:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.putText(frame, labels_dict[result],(x,y), color=(255,0,0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,thickness=2 )
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, labels_dict[result],(x,y), color=(255,0,0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,thickness=2 )    
    cv2.imshow("cam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()    
