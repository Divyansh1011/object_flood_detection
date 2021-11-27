from flask import Flask,render_template, Response, redirect, request, session, abort, url_for
from flask import Flask, render_template, request, redirect, url_for, session
import tensorflow as tf
import cv2
import pafy
import time
import sys
import numpy as np 
import pandas as pd
from datetime import datetime 
import youtube_dl
from PIL import Image
from keras.preprocessing import image
from imutils.video import FPS
import os

app=Flask(__name__)
app.secret_key = 'verysecret'

def log_flood():
    with open('log/flood1.csv', 'r+') as f: 
        myDateList = f.readlines()
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S, %d/%m/%Y')
        f.writelines(f'\n{"Flooding"}, {dtString}')


def gen_object_detection():
    YOLO_PATH="yolo-coco"
    OUTPUT_FILE="output/outfile.avi"
# load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([YOLO_PATH, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    CONFIDENCE=0.5
    THRESHOLD=0.3
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
    weightsPath = os.path.sep.join([YOLO_PATH, "yolov3.weights"])
    configPath = os.path.sep.join([YOLO_PATH, "yolov3.cfg"])
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    url = "https://www.youtube.com/watch?v=U7HRKjlXK-Y&t=220s"
    video = pafy.new(url)    
    best = video.getbest(preftype="mp4")
    vs = cv2.VideoCapture()
    vs.open(best.url)
    time.sleep(2.0)
    fps = FPS().start()
    writer = None
    (W, H) = (None, None)

    cnt= 0 

    while True:
        cnt+= 1
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)  
        end = time.time()

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > CONFIDENCE:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
            
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		
        ima = cv2.resize(frame, (1280,720))

        ret, jpeg = cv2.imencode('.jpg', ima)

        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



def gen_flood():
    model = tf.keras.models.load_model('/home/divyansh/Divyansh/projects/res/fine_tuned_flood_detection_model (1)')
    f = 0
    url = "https://www.youtube.com/watch?v=XEpAgCnnYdY"
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    cap = cv2.VideoCapture()
    cap.open(best.url)
    while True:
        _,frame = cap.read()
        if not _:
            print("[INFO] no frame read from stream - exiting")
            sys.exit(0)
        
        im = Image.fromarray(frame, 'RGB')
        im = im.resize((224,224))
        img_array = image.img_to_array(im)
        img_array = np.expand_dims(img_array, axis=0)/265
        probabilties  = model.predict(img_array)[0]
        prediction = np.argmax(probabilties)

        if prediction == 0:
                log_flood()
                cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
                cv2.putText(frame, 'Flooding', (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 3)
                print(probabilties[prediction])
                f=f+1
        ima = cv2.resize(frame, (1280,720))

        ret, jpeg = cv2.imencode('.jpg', ima)

        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/video6')
def video6():
    return render_template('video6.html')


@app.route('/video7')
def video7():
    return render_template('video7.html')

@app.route('/video_flood')
def video_flood():
    return Response(gen_flood(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_object_detection')
def video_object_detection():
    return Response(gen_object_detection(),mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/details')
def details():  
    data = pd.read_csv('log/motion.csv', header=0)
    data_face=pd.read_csv('log/face.csv', header=0)
    data_act=pd.read_csv('log/activity.csv', header=0)
    data_fire=pd.read_csv('log/fire.csv', header=0)
    # myData = list(data.values)
    m1=str(data.tail(5).iloc[4][2])+" "+str(data.tail(5).iloc[4][1])+" "+str(data.tail(5).iloc[4][0])
    m2=str(data.tail(5).iloc[3][2])+" "+str(data.tail(5).iloc[3][1])+" "+str(data.tail(5).iloc[3][0])
    m3=str(data.tail(5).iloc[2][2])+" "+str(data.tail(5).iloc[2][1])+" "+str(data.tail(5).iloc[2][0])
    m4=str(data.tail(5).iloc[1][2])+" "+str(data.tail(5).iloc[1][1])+" "+str(data.tail(5).iloc[1][0])
    m5=str(data.tail(5).iloc[0][2])+" "+str(data.tail(5).iloc[0][1])+" "+str(data.tail(5).iloc[0][0])

    fa1=str(data_face.tail(5).iloc[4][2])+" "+str(data_face.tail(5).iloc[4][1])+" "+str(data_face.tail(5).iloc[4][0])
    fa2=str(data_face.tail(5).iloc[3][2])+" "+str(data_face.tail(5).iloc[3][1])+" "+str(data_face.tail(5).iloc[3][0])
    fa3=str(data_face.tail(5).iloc[2][2])+" "+str(data_face.tail(5).iloc[2][1])+" "+str(data_face.tail(5).iloc[2][0])
    fa4=str(data_face.tail(5).iloc[1][2])+" "+str(data_face.tail(5).iloc[1][1])+" "+str(data_face.tail(5).iloc[1][0])
    fa5=str(data_face.tail(5).iloc[0][2])+" "+str(data_face.tail(5).iloc[0][1])+" "+str(data_face.tail(5).iloc[0][0])

    a1=str(data_act.tail(5).iloc[4][2])+" "+str(data_act.tail(5).iloc[4][1])+" "+str(data_act.tail(5).iloc[4][0])
    a2=str(data_act.tail(5).iloc[3][2])+" "+str(data_act.tail(5).iloc[3][1])+" "+str(data_act.tail(5).iloc[3][0])
    a3=str(data_act.tail(5).iloc[2][2])+" "+str(data_act.tail(5).iloc[2][1])+" "+str(data_act.tail(5).iloc[2][0])
    a4=str(data_act.tail(5).iloc[1][2])+" "+str(data_act.tail(5).iloc[1][1])+" "+str(data_act.tail(5).iloc[1][0])
    a5=str(data_act.tail(5).iloc[0][2])+" "+str(data_act.tail(5).iloc[0][1])+" "+str(data_act.tail(5).iloc[0][0])

    f1=str(data_fire.tail(5).iloc[4][2])+" "+str(data_fire.tail(5).iloc[4][1])+ " "+str(data_fire.tail(5).iloc[4][0])
    f2=str(data_fire.tail(5).iloc[3][2])+" "+str(data_fire.tail(5).iloc[3][1])+" "+str(data_fire.tail(5).iloc[3][0])
    f3=str(data_fire.tail(5).iloc[2][2])+" "+str(data_fire.tail(5).iloc[2][1])+" "+str(data_fire.tail(5).iloc[2][0])
    f4=str(data_fire.tail(5).iloc[1][2])+" "+str(data_fire.tail(5).iloc[1][1])+" "+str(data_fire.tail(5).iloc[1][0])
    f5=str(data_fire.tail(5).iloc[0][2])+" "+str(data_fire.tail(5).iloc[0][1])+" "+str(data_fire.tail(5).iloc[0][0])

    return render_template('details.html', str_motion_1=m1, str_motion_2=m2, str_motion_3=m3, str_motion_4=m4, str_motion_5=m5,
                                           str_face_1=fa1, str_face_2=fa2, str_face_3=fa3, str_face_4=fa4, str_face_5=fa5,
                                           str_act_1=a1, str_act_2=a2, str_act_3=a3, str_act_4=a4, str_act_5=a5,
                                           str_fire_1=f1, str_fire_2=f2, str_fire_3=f3, str_fire_4=f4, str_fire_5=f5)

       

if __name__=="__main__":
    app.run(debug=True)