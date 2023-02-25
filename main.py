import cv2 as cv
import time
import datetime

cap=cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_fullbody.xml")

detection = False
detection_stopped_time = None
timer_started = False
Seconds_to_record_after_detection = 5

frame_size = (int(cap.get(3)),int(cap.get(4)))        #3 gives width and 4 gives height
fourcc = cv.VideoWriter_fourcc(*"mp4v")         # * decomposes ("m","p","4","v")
# out=cv.VideoWriter("Video.mp4",fourcc,20,frame_size)

while True:
    isTrue, frame = cap.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    bodies = face_cascade.detectMultiScale(gray,1.3,5)

    if len(faces)+len(bodies)>0:
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out=cv.VideoWriter(f"{current_time}.mp4",fourcc,20,frame_size)
            print("Recording Started!")
    elif detection:
        if timer_started:
            if time.time()-detection_stopped_time>= Seconds_to_record_after_detection:
                detection=False
                timer_started=False
                out.release()
                print("Recording Stopped!")
        else:
            timer_started=True
            detection_stopped_time=time.time()

    if detection:
        out.write(frame)

    # for (x,y,w,h) in faces:
    #     cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
    cv.imshow("Camera",frame)

    if cv.waitKey(1) & 0xFF==ord('q'):
        break

out.release()
cap.release()
cv.destroyAllWindows()
