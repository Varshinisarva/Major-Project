import os
import MySQLdb
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash,Response
 
import threading
from werkzeug.utils import secure_filename
import numpy as np
import joblib
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from database import *
from test import*
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from sendmail import sendmail
app = Flask(__name__)
app.secret_key='detection'
 
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
 


 
 
selected_features = ['Bear', 'Elephant', 'Leopard', 'Lion', 'Wolf']  # Change according to your dataset

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/registera")
def registera():
    return render_template("register.html")

@app.route("/camera")
def camera():
    return render_template("camera.html")

@app.route("/logina")
def logina():
    return render_template("login.html")
@app.route("/menua")
def menua():
    return render_template("menu.html")

@app.route("/register",methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username=request.form['username']
        email=request.form['email']
        password=request.form['password']
        status = user_reg(username,email,password)
        if status == 1:
            return render_template("/login.html")
        else:
            return render_template("/register.html",m1="failed")        
    

@app.route("/login",methods=['POST','GET'])
def login():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        status = user_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1:                                      
            return render_template("/menu.html", m1="sucess")
        else:
            return render_template("/login.html", m1="Login Failed")
             
# @app.route("/")
# def home():
#     return render_template("index.html")

@app.route('/logouta')
def logout():
    # Clear the session data
    session.clear()
    return redirect(url_for('logina'))






model = load_model('wild.h5')

# Define the activities (classes)
activities = ['Bear', 'Elephant', 'Leopard', 'Lion', 'Wolf'] # Replace with your actual activity labels

# Global variables for video streaming
video_frame = None
video_stream = cv2.VideoCapture()

process_thread = None  # Global variable for the process thread
stop_processing = False  # Flag variable to indicate when to stop processing




def process_video():
    global video_frame, video_stream, stop_processing

    while not stop_processing:
        ret, frame = video_stream.read()
        if not ret:
            break

        # Preprocess the frame (resize and normalize)
        resized_frame = cv2.resize(frame, (224, 224))
        normalized_frame = resized_frame / 255.0

        # Add the batch dimension
        input_frame = np.expand_dims(normalized_frame, axis=0)

        # Make prediction on the input frame
        pred = model.predict(input_frame)
        pred_label = np.argmax(pred)
        activity = activities[pred_label]

        # Draw the predicted activity label on the frame
        cv2.putText(frame, activity, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Update the global video frame for streaming
        video_frame = frame.copy()

@app.route('/predictpage')
def predictpage():
    # Stop the process thread if it's running
    global stop_processing, process_thread

    if process_thread and process_thread.is_alive():
        stop_processing = True
        process_thread.join()
        stop_processing = False

    return render_template('predictpage.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global video_frame

    while True:
        if video_frame is not None:
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', video_frame)
            frame = buffer.tobytes()

            # Yield the frame in the byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/predict', methods=['POST','GET'])
def predict():
    global video_stream, process_thread, stop_processing, video_frame

    # Get the uploaded video file
    video_file = request.files['video']

    # Save the uploaded video file
    video_path = 'static/uploads/' + video_file.filename
    video_file.save(video_path)

    # Release the previous video stream if any
    video_stream.release()

    # Reset the video_frame to None
    video_frame = None

    # Load the video
    video_stream = cv2.VideoCapture(video_path)

    # Stop the process thread if it's running
    if process_thread and process_thread.is_alive():
        stop_processing = True
        process_thread.join()
        stop_processing = False

    # Start processing the video in a separate thread
    process_thread = threading.Thread(target=process_video)
    process_thread.start()

    return render_template('result.html', video_path='/video_feed')

@app.route('/detect')
def detect():
    x=0
    y=0
    z=0
    e=0
    s=0
    # Load your pre-trained model (change the path to your model)
    model = tf.keras.models.load_model('wild.h5')

    # Load labels if necessary (optional, depending on your use case)
    labels = ['Bear', 'Elephant', 'Leopard', 'Lion', 'Wolf']  # Change according to your dataset
    
    # Set up the video capture from the webcam
    cap = cv2.VideoCapture(0)

    # Confidence threshold for displaying predictions
    CONFIDENCE_THRESHOLD = 0.95  # Set a threshold. If predictions are below this, show nothing.

    # Function to preprocess the frame before prediction (e.g., resizing, normalization)
    def preprocess_frame(frame):
        img_size = (224, 224)  # Example size, should match the input size expected by the model
        img = cv2.resize(frame, img_size)     # Resize frame to match model input size
        img = np.array(img) / 255.0           # Normalize pixel values to 0-1 range
        img = np.expand_dims(img, axis=0)     # Add batch dimension
        return img

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()

        # Preprocess the captured frame
        processed_frame = preprocess_frame(frame)
        
        # Make prediction using the pre-trained model
        predictions = model.predict(processed_frame)
        
        # Get the predicted label and confidence score
        max_confidence = np.max(predictions)  # Confidence of the highest predicted class
        predicted_label = labels[np.argmax(predictions)]  # The label of the highest predicted class
        if predicted_label=="Bear" and x==1:           
           sendmail("nikithamerlyn04@gmail.com","Bear Detected")
           print("Bear Detected")
           x=1
        if predicted_label=='Elephant' and y==0:           
            sendmail("nikithamerlyn04@gmail.com","Elephant Detected")  
            print("Elephant Detected")          
            y=1
        if predicted_label=='Lion' and z==0:           
            sendmail("nikithamerlyn04@gmail.com","Lion Detected")     
            print("Lion Detected")       
            z=1
        if predicted_label=='Leopard' and e==0:            
            sendmail("nikithamerlyn04@gmail.com","Leopard Detected")
            print("Leopard Detected")            
            e=1
        if predicted_label=='Wolf' and s==0:            
            sendmail("nikithamerlyn04@gmail.com","Wolf Detected")
            print("Wolf Detected")            
            s=1
        # Display label only if confidence is above the threshold
        if max_confidence >= CONFIDENCE_THRESHOLD:
            cv2.putText(frame, f'Prediction: {predicted_label} ({max_confidence:.2f})', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # If confidence is below the threshold, don't display any prediction
            cv2.putText(frame, 'No Prediction', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with the prediction or "No Prediction"
        cv2.imshow('Camera Feed', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return render_template("camera.html")

if __name__ == "__main__":
    app.run(debug=True)