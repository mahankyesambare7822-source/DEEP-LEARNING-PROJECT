import cv2
import numpy as np
import tensorflow as tf

# Load the brain we just saved
model = tf.keras.models.load_model('digit_model.h5')

cap = cv2.VideoCapture(0)
print("--- Camera active! Press 'q' to quit ---")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Draw a green box in the center
    cv2.rectangle(frame, (200, 200), (450, 450), (0, 255, 0), 2)
    
    # Preprocess what's inside the box
    roi = frame[200:450, 200:450]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    
    # Invert colors (AI likes white ink on black paper)
    inverted = cv2.bitwise_not(resized) 
    normalized = inverted / 255.0
    reshaped = normalized.reshape(1, 28, 28)

    # Ask the AI to guess
    prediction = model.predict(reshaped, verbose=0)
    digit = np.argmax(prediction)
    
    # Put the text on the screen
    cv2.putText(frame, f"I see a: {digit}", (210, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    cv2.imshow('Live AI Vision', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()