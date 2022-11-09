# Facial-Recog
facial Recog project
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path='images'
images=[]
person_name=[]
my_list=os.listdir(path)
print(my_list)
for i in my_list:
    a=cv2.imread(f'{path}/{i}')
    images.append(a)
    person_name.append(os.path.splitext(i)[0])
print(person_name)

#finding of encodings

def face_encodings(images):
    encode_list=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list
encode_list_known=face_encodings(images)
print("all encodings done")

def attendence(names):
    with open('attendence.cv','r+') as f:
        mydatalist=f.readline()
        namelist=[]
        for line in mydatalist:
            entery=line.split(',')
            namelist.append(entery[0])
        if names not in namelist:
            time_now=datetime.now()
            tstr=time_now.strftime('%H:%M:%S')
            dstr=time_now.strftime('%D:%M:%Y')
            f.write(f'{name},{tstr},{dstr}')



#vedio capturings

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    faces=cv2.resize(frame,(0,0),None,0.25,0.25)
    faces=cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)
    faces_currentframe=face_recognition.face_locations(faces)
    encodes_currentframe=face_recognition.face_encodings(faces,faces_currentframe)

#fame matchings
    for encodeFace,faceloc in zip(encodes_currentframe,faces_currentframe):
        matches=face_recognition.compare_faces(encode_list_known,encodeFace)
        facedist=face_recognition.face_distance(encode_list_known,encodeFace)

        match_index=np.argmin(facedist)


    if matches[match_index]:
        name=person_name[match_index].upper()

        y1,x2,y2,x1 = faceloc
        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0))
        cv2.rectangle(frame,(x1,y1-35),(x2,y2),(0,255,0))
        cv2.putText(frame,name,(x1+6,y2+6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        attendence(name)


    cv2.imshow("camera",  frame)
    if cv2.waitKey(10000)==13:
        break
cap.release()
cv2.destroyAllWindows()


