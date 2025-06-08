import cv2
import numpy as np
import tensorflow as tf

# Load your pre-trained model (change the path to your model)
model = tf.keras.models.load_model('wild.h5')

# Load labels if necessary (optional, depending on your use case)
labels = ['Bear', 'Elephant', 'Leopard', 'Lion', 'Wolf']  # Change according to your dataset
 
# Set up the video capture from the webcam
cap = cv2.VideoCapture(0)

# Confidence threshold for displaying predictions
CONFIDENCE_THRESHOLD = 0.7  # Set a threshold. If predictions are below this, show nothing.

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
