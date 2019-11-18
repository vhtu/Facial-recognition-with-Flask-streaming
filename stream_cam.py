# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:57:44 2019

@author: seraj
"""
import time
import cv2
import os
from flask import Flask, render_template, Response
import pickle
import sqlite3

app = Flask(__name__)
sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor

# Khởi tạo bộ phát hiện khuôn mặt
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

# Khởi tạo bộ nhận diện khuôn mặt
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/trainner.yml')
id=0
#set text style
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (0,0,255)
fontcolor1 = (0,255,0)


# Hàm lấy thông tin người dùng qua ID
def getProfile(id):
    conn=sqlite3.connect("FaceBaseNew.db")
    cursor=conn.execute("SELECT * FROM People WHERE ID="+str(id))
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile



@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    """Video streaming generator function."""
    # Khởi tạo camera
    cap=cv2.VideoCapture(0)

    # Read until video is completed
    while(cap.isOpened()):
        ret, img = cap.read()  # import image
        # Lật ảnh cho đỡ bị ngược
        img = cv2.flip(img, -1)
        # Vẽ khung chữ nhật để định vị vùng người dùng đưa mặt vào
        centerH = img.shape[0] // 2;
        centerW = img.shape[1] // 2;
        sizeboxW = 300;
        sizeboxH = 400;
        #cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
        #              (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)

        # Chuyển ảnh về xám
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Phát hiện các khuôn mặt trong ảnh camera
        faces=faceDetect.detectMultiScale(gray,1.3,5);

        # Lặp qua các khuôn mặt nhận được để hiện thông tin
        for(x,y,w,h) in faces:
            # Vẽ hình chữ nhật quanh mặt
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            #cv2.imshow("countours", image)
            # Nhận diện khuôn mặt, trả ra 2 tham số id: mã nhân viên và dist (dộ sai khác)
            id,dist=recognizer.predict(gray[y:y+h,x:x+w])
            #print(id)
            profile=None

            # Nếu độ sai khác < 25% thì lấy profile
            if (dist<=90):
                profile=getProfile(id)

            dist = round(dist, 2)
            # Hiển thị thông tin tên người hoặc Unknown nếu không tìm thấy
            if(profile!=None):

                

                cv2.putText(img, "Ho Ten: " + str(profile[1]), (x,y+h+30), fontface, fontscale, fontcolor ,2)
                cv2.putText(img, "Tuoi: " + str(profile[2]), (x,y+h+60), fontface, fontscale, fontcolor ,2)
                cv2.putText(img, "Gioi tinh: " + str(profile[3]), (x,y+h+90), fontface, fontscale, fontcolor ,2)
                cv2.putText(img, "Phong ban: " + str(profile[5]), (x,y+h+120), fontface, fontscale, fontcolor ,2)
                cv2.putText(img, "%Confident: " + str(dist)+"%", (x,y+h+150), fontface, fontscale, fontcolor ,2)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        #time.sleep(0.1)
        if cv2.waitKey(1)==ord('q'):
            break;
   
        

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    #decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    #run the app locally on the givn port
    app.run(host='0.0.0.0', port=port)
    #optional if we want to run in debugging mode
    #app.run(debug=True)
    

