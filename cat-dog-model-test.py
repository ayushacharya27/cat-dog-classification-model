import tensorflow as tf
import cv2
import numpy as np

# Load the saved model
model = tf.keras.models.load_model(r"C:\Users\DELL\Desktop\cat-dog-ml-model\cat-dog-model.keras")

# Load the image using OpenCV
image_path = r"C:\Users\DELL\Desktop\cat-dog-ml-model\cat3.jpg"
image = cv2.imread(image_path)

# Check if the image was successfully loaded
if image is None:
    print(f"Failed to load image from {image_path}")
    exit(1)

# Preprocess the image: resize to (150, 150), convert to RGB, and normalize
image_resized = cv2.resize(image, (150, 150))
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
image_normalized = image_rgb / 255.0  # Normalize to [0, 1]

# Add a batch dimension (1, 150, 150, 3)
image_batch = np.expand_dims(image_normalized, axis=0)

# Make a prediction
prediction = model.predict(image_batch)

# Interpret the prediction (assuming binary classification: 0 = cat, 1 = dog)
if prediction[0] > 0.5:
    print("It's a dog!")
else:
    print("It's a cat!")

# Display the image with OpenCV
cv2.imshow("Test Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
