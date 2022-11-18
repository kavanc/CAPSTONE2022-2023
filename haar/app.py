from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

cap = cv2.VideoCapture(0)
cap.set(10, 150)

def process_img(img):
    casc_path = "w_faces/classifier/cascade.xml"

    knife_cascade = cv2.CascadeClassifier(casc_path)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    knife = knife_cascade.detectMultiScale(img_gray, 1.3, 22)

    max_coords = [0, 0, 0, 0]
    for i, (x, y, w, h) in enumerate(knife):
        if w < 75:
            continue
        
        if w > max_coords[2]:
            max_coords[0] = x
            max_coords[1] = y
            max_coords[2] = w
            max_coords[3] = h
    
    x, y, w, h = max_coords[0], max_coords[1], max_coords[2], max_coords[3]
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0),  2)

    return img

def gen_frames():  
    while True:
        success, frame = cap.read()  # read the cap frame
        if not success:
            break
        else:
            frame = process_img(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)