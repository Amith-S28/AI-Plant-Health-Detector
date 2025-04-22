import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import signal
import sys

# Global variable for shutdown
shutdown = False

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    global shutdown
    print("\nShutting down gracefully...")
    shutdown = True
    if 'cap' in globals():
        cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# Function to create CNN model
def create_cnn_model(input_shape=(128, 128, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to preprocess image for prediction
def preprocess_image(image, target_size=(128, 128)):
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to classify crop quality
def classify_crop(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return (f"Healthy {prediction} % Sure" ) if prediction[0] > 0.5 else (f"Diseased {prediction} % Sure!!")

# Function to load and prepare PlantVillage dataset
def load_plantvillage_dataset(dataset_dir, target_size=(128, 128), batch_size=32):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found. Please check the path.")
    
    image_paths = []
    labels = []
    
    for class_dir in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_dir)
        if os.path.isdir(class_path):
            label = "Healthy" if 'healthy' in class_dir.lower() else "Diseased"
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                image_paths.append(img_path)
                labels.append(label)
    
    df = pd.DataFrame({'image_path': image_paths, 'label': labels})
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='image_path',
        y_col='label',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col='image_path',
        y_col='label',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    return train_generator, val_generator, len(train_df), len(val_df)

# Main function
def main():
    global shutdown, cap
    signal.signal(signal.SIGINT, signal_handler)
    
    # Path to PlantVillage dataset and model save location
    dataset_dir = r'D:\Study mats\plantvillage dataset\color'
    model_dir = r'D:\Study mats\models'
    model_path = os.path.join(model_dir, 'plantvillage_crop_quality_model.h5')
    checkpoint_path = os.path.join(model_dir, 'plantvillage_crop_quality_checkpoint.h5')
    partial_path = os.path.join(model_dir, 'plantvillage_crop_quality_partial.h5')
    
    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory: {model_dir}")
    
    # Print current working directory for reference
    print("Current working directory:", os.getcwd())
    
    # Check if trained model exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = load_model(model_path)
    else:
        print(f"No existing model found at {model_path}. Training new model...")
        try:
            train_generator, val_generator, train_size, val_size = load_plantvillage_dataset(dataset_dir)
        except FileNotFoundError as e:
            print(e)
            return
        
        model = create_cnn_model()
        checkpoint = ModelCheckpoint(checkpoint_path,
                                    save_best_only=True, monitor='val_loss')
        try:
            model.fit(
                train_generator,
                steps_per_epoch=train_size // 32,
                epochs=10,
                validation_data=val_generator,
                validation_steps=val_size // 32,
                callbacks=[checkpoint]
            )
            print(f"Saving trained model to {model_path}")
        except KeyboardInterrupt:
            print(f"Training interrupted. Saving partial model to {partial_path}...")
            model.save(partial_path)
            return
        
        model.save(model_path)
    
    # Real-time analysis with webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while not shutdown:
        ret, frame = cap.read()
        if not ret or shutdown:
            print("Error: Failed to capture image or shutdown requested.")
            break
        
        result = classify_crop(frame, model)
        cv2.putText(frame, f"Quality: {result}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 255, 0) if result == "Healthy" else (0, 0, 255), 2)
        cv2.imshow('Crop Quality Analysis', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()