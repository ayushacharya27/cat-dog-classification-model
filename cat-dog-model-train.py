from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
from PIL import UnidentifiedImageError

# Path to the dataset
data_set_path = r"C:\Users\DELL\Desktop\kagglecatsanddogs_5340\PetImages"

# Function to check and remove corrupted images
def check_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                img_path = os.path.join(root, file)
                img = tf.keras.preprocessing.image.load_img(img_path)
            except (UnidentifiedImageError, IOError):
                print(f"Removing corrupted image: {img_path}")
                os.remove(img_path)

# Check and remove corrupted images
check_images(data_set_path)

# Create an ImageDataGenerator and set up the training data generator
train_data_gen = ImageDataGenerator(rescale=1./255)
train_generator = train_data_gen.flow_from_directory(
    data_set_path, target_size=(150, 150), batch_size=25, class_mode='binary'
)

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

# Train the model
history = model.fit(train_generator, epochs=3)

# Save the model
model.save(r'C:\Users\DELL\Desktop\cat-dog-ml-model\cat-dog-model.keras')
