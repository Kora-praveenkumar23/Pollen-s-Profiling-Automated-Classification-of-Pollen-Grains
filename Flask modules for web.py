# Import necessary Flask modules for web application development
from flask import Flask, request, jsonify, render_template
# Import Keras and TensorFlow for building and training the Convolutional Neural Network (CNN)
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from PIL import Image
import io

# Initialize the Flask application
app = Flask(__name__)

# Define the path where the trained model will be saved and loaded from
MODEL_PATH = 'cnn_model.h5'
# Define the target image size for the model input. Using a common size like 28x28 for demonstration.
IMG_HEIGHT, IMG_WIDTH = 28, 28
# Define the number of color channels (1 for grayscale, 3 for RGB)
IMG_CHANNELS = 1 # Assuming grayscale images for simplicity (e.g., MNIST-like data)
# Define the number of output classes for the classification task
NUM_CLASSES = 10 # Assuming 10 classes (e.g., digits 0-9)

# --- Model Building Function ---
def build_cnn_model():
    """
    Builds a Convolutional Neural Network (CNN) model using Keras.

    The model consists of:
    - Convolutional layers (Conv2D) for feature extraction.
    - Pooling layers (MaxPooling2D) for dimensionality reduction.
    - Flatten layer to convert 2D feature maps to 1D vector.
    - Fully connected (Dense) layers for classification.
    - Dropout layers for regularization to prevent overfitting.
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
        MaxPooling2D((2, 2)),
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        # Flatten the output to feed into fully connected layers
        Flatten(),
        # Fully Connected Layers
        Dense(128, activation='relu'),
        Dropout(0.5), # Dropout for regularization
        Dense(NUM_CLASSES, activation='softmax') # Output layer with softmax for multi-class classification
    ])

    # Compile the model
    # Optimizer: Adam is a good default for many tasks.
    # Loss function: sparse_categorical_crossentropy for integer labels (when not one-hot encoded).
    # Metrics: 'accuracy' to monitor performance during training.
    model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Model Training Function (Simulated/Placeholder) ---
def train_model():
    """
    Trains the CNN model.

    This function simulates data loading and augmentation using ImageDataGenerator.
    In a real application, you would load your actual dataset here.
    For this example, we create dummy data to allow the model to be built and saved.
    """
    model = build_cnn_model()
    print("CNN model built successfully.")

    # --- Data Collection (Conceptual) ---
    # In a real scenario, you would load your dataset here.
    # Example:
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # x_train = x_train.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype('float32') / 255.0
    # x_test = x_test.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype('float32') / 255.0

    # --- Data Preprocessing & Augmentation (Conceptual) ---
    # Create a dummy dataset for demonstration purposes as full dataset loading
    # and training is not feasible in this environment.
    # This dummy data allows the model to be 'trained' and saved.
    print("Generating dummy data for training simulation...")
    num_dummy_samples = 100
    dummy_x_train = np.random.rand(num_dummy_samples, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)
    dummy_y_train = np.random.randint(0, NUM_CLASSES, num_dummy_samples)
    dummy_x_val = np.random.rand(num_dummy_samples // 5, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)
    dummy_y_val = np.random.randint(0, NUM_CLASSES, num_dummy_samples // 5)

    # ImageDataGenerator for data augmentation (conceptual use)
    # This helps in increasing the diversity of your training data by applying random transformations.
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        rescale=1./255 # Normalization: Rescale pixel values from [0, 255] to [0, 1]
    )
    val_datagen = ImageDataGenerator(rescale=1./255) # Only rescale for validation data

    # Create data generators for training and validation
    train_generator = train_datagen.flow(dummy_x_train, dummy_y_train, batch_size=32)
    val_generator = val_datagen.flow(dummy_x_val, dummy_y_val, batch_size=32)

    # --- Model Training (Simulated) ---
    print("Starting simulated model training...")
    # For a real application, increase epochs and use real data.
    # Using a small number of epochs for quick demonstration.
    model.fit(train_generator,
              epochs=1, # Reduced epochs for faster execution in this environment
              validation_data=val_generator,
              verbose=1)
    print("Simulated model training complete.")

    # --- Model Evaluation (Conceptual) ---
    # In a real scenario, you would evaluate on a separate test set.
    # Example:
    # loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    # print(f"Model Test Accuracy: {accuracy*100:.2f}%")
    print("Model evaluation (conceptual) would happen here.")

    # Save the trained model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return model

# Load the model if it exists, otherwise train it
try:
    cnn_model = load_model(MODEL_PATH)
    print("Model loaded successfully from disk.")
except:
    print("Model not found. Training a new model...")
    cnn_model = train_model()

# --- Flask Routes ---

@app.route('/')
def index():
    """
    Renders the main HTML page for image upload.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image prediction requests.

    Expects an image file in the request.
    Preprocesses the image, makes a prediction using the loaded CNN model,
    and returns the prediction results as JSON.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Read the image file
            image_stream = io.BytesIO(file.read())
            img = Image.open(image_stream).convert('L') # Convert to grayscale

            # Preprocess the image for prediction
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = np.array(img) / 255.0 # Normalize pixel values to [0, 1]
            # Add batch dimension and channel dimension: (height, width) -> (1, height, width, 1)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = np.expand_dims(img_array, axis=-1)

            # Make prediction
            predictions = cnn_model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100

            return jsonify({
                'prediction': int(predicted_class),
                'confidence': float(confidence),
                'message': f'Image analyzed successfully. Predicted class: {predicted_class}'
            })

        except Exception as e:
            # Log the error for debugging
            print(f"Error during prediction: {e}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    return jsonify({'error': 'Something went wrong.'}), 500

if __name__ == '__main__':
    # When running locally, ensure 'templates' folder exists for index.html
    # and the model is trained/loaded.
    # For deployment in environments like Canvas, Flask is typically run by a WSGI server.
    app.run(debug=True) # debug=True enables auto-reloading and better error messages
