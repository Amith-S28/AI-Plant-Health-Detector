# Health-of-a-Plant-
AI model used to know the health of a plant based on picture of its leaf

What does the code do?
This code:

Trains a CNN model to classify plant leaves as "Healthy" or "Diseased" using images from the PlantVillage dataset.
Saves the trained model to avoid retraining every time.
Uses a webcam to capture live images of leaves and classify them in real-time as "Healthy" or "Diseased."
Displays the classification result on the webcam feed.
Key Libraries Used
OpenCV (cv2): For capturing webcam images, resizing, and displaying results.
NumPy (np): For handling arrays (images are stored as arrays).
TensorFlow/Keras (tf, tensorflow.keras): For building, training, and using the CNN model.
Pandas (pd): For organizing the dataset (image paths and labels).
Scikit-learn (sklearn): For splitting the dataset into training and validation sets.
OS, Signal, Sys: For handling file paths, graceful shutdown, and system operations.
Main Components of the Code

1. Graceful Shutdown (Signal Handler)
What it does: If you press Ctrl+C to stop the program, it cleanly closes the webcam and windows without crashing.
How it works:
A global variable shutdown tracks whether the program should stop.
The signal_handler function is triggered when Ctrl+C is pressed. It releases the webcam (cap.release()), closes all OpenCV windows (cv2.destroyAllWindows()), and exits the program.
Why it’s useful: Prevents the program from freezing or leaving resources (like the webcam) open.
Viva Tip: If asked, say, "The signal handler ensures the program shuts down gracefully when interrupted, releasing resources like the webcam to avoid crashes."

2. Creating the CNN Model (create_cnn_model)
What it does: Builds a CNN model to classify images as "Healthy" or "Diseased."
How it works:
The model has 3 convolutional layers (with 32, 64, and 128 filters) to extract features like edges and patterns from images.
Each convolutional layer is followed by a MaxPooling layer to reduce image size while keeping important features.
The output is flattened into a 1D array, passed through a Dense layer (128 neurons), and finally to a single neuron with a sigmoid activation to output a probability (0 = Diseased, 1 = Healthy).
The model uses the Adam optimizer and binary crossentropy loss (suitable for binary classification).
Input shape: Images are resized to 128x128 pixels with 3 color channels (RGB).
Viva Tip: Explain, "The CNN model has convolutional layers to detect features in leaf images, max-pooling to reduce size, and dense layers to classify the leaf as Healthy or Diseased based on a probability score."

3. Preprocessing Images (preprocess_image)
What it does: Prepares an image for the CNN model.
How it works:
Resizes the image to 128x128 pixels (same as the model’s input shape).
Converts the image from BGR (OpenCV’s default format) to RGB (Keras expects RGB).
Normalizes pixel values to the range [0, 1] by dividing by 255.
Adds an extra dimension to the image (to match the model’s expected input format: (1, 128, 128, 3)).
Why it’s needed: The model requires images in a specific format to make predictions.
Viva Tip: Say, "The preprocess_image function resizes the image, converts it to RGB, normalizes it, and prepares it for the CNN model to predict whether the leaf is Healthy or Diseased."

4. Classifying Crops (classify_crop)
What it does: Uses the trained model to classify a leaf image as "Healthy" or "Diseased."
How it works:
Takes an image, preprocesses it, and feeds it to the model.
The model outputs a probability (0 to 1). If it’s > 0.5, the leaf is "Healthy"; otherwise, it’s "Diseased."
Output: A string ("Healthy" or "Diseased").
Viva Tip: Explain, "This function uses the trained CNN to predict if a leaf is Healthy or Diseased based on the model’s output probability."

5. Loading the PlantVillage Dataset (load_plantvillage_dataset)
What it does: Loads images from the PlantVillage dataset and prepares them for training.
How it works:
The dataset is stored in a folder (dataset_dir), with subfolders for each class (e.g., "Tomato_Healthy," "Tomato_Bacterial_spot").
The code scans the folder, creates a list of image paths, and assigns labels ("Healthy" if the folder name contains "healthy," otherwise "Diseased").
Uses Pandas to create a DataFrame with image paths and labels.
Splits the data into 80% training and 20% validation sets using train_test_split.
Uses ImageDataGenerator to:
Rescale images (divide pixel values by 255).
Apply data augmentation (random rotations, flips, shifts) to the training set to make the model robust.
Load images in batches (32 images at a time) for efficient training.
Returns train_generator and val_generator (for training and validation) and the sizes of the datasets.
Viva Tip: Say, "This function loads the PlantVillage dataset, organizes images with labels, splits them into training and validation sets, and applies data augmentation to improve model performance."

6. Main Function (main)
This is the heart of the program, tying everything together. Let’s break it down step-by-step:

Setup Signal Handler:
Ensures the program can be stopped gracefully with Ctrl+C.
Define Paths:
dataset_dir: Path to the PlantVillage dataset.
model_dir: Where the trained model will be saved.
model_path: Path for the final trained model.
checkpoint_path: Path to save the best model during training.
partial_path: Path to save a partially trained model if training is interrupted.
Check for Existing Model:
If a trained model exists at model_path, it loads it using load_model.
If not, it trains a new model.
Training a New Model:
Calls load_plantvillage_dataset to prepare the dataset.
Creates a new CNN model with create_cnn_model.
Uses a ModelCheckpoint callback to save the best model (based on validation loss) during training.
Trains the model for 10 epochs (passes through the dataset).
If interrupted (e.g., by Ctrl+C), saves a partial model to partial_path.
Saves the final trained model to model_path.
Real-Time Webcam Analysis:
Opens the webcam using cv2.VideoCapture(0).
Captures frames in a loop until the program is stopped.
For each frame:
Calls classify_crop to predict if the leaf is "Healthy" or "Diseased."
Displays the result on the frame using cv2.putText (green text for "Healthy," red for "Diseased").
Shows the frame in a window named "Crop Quality Analysis."
Pressing q or triggering shutdown stops the loop.
Releases the webcam and closes windows when done.
Viva Tip: Explain, "The main function checks if a trained model exists. If not, it trains a new one using the PlantVillage dataset. Then, it uses the webcam to capture leaf images, classifies them as Healthy or Diseased, and displays the result live."

How It All Works Together
The program starts by checking if a trained model exists. If not, it trains one using the PlantVillage dataset.
During training, it processes images, applies data augmentation, and saves the best model.
Once trained (or loaded), it uses the webcam to capture live images of leaves.
Each image is preprocessed, classified by the CNN, and the result is shown on the screen.
The program stops cleanly if you press q or Ctrl+C.
Common Viva Questions and Answers
What is the purpose of this code?
It classifies plant leaves as Healthy or Diseased using a CNN model trained on the PlantVillage dataset and performs real-time analysis with a webcam.
What is a CNN, and why is it used here?
A Convolutional Neural Network (CNN) is a deep learning model designed for image processing. It’s used here because it can detect patterns (like disease spots) in leaf images effectively.
Why do you preprocess images?
Images are preprocessed to resize them, convert to RGB, normalize pixel values, and match the model’s input format for accurate predictions.
What is data augmentation, and why is it used?
Data augmentation applies random changes (like rotations, flips) to training images to make the model robust to variations in real-world images.
How does the webcam part work?
The webcam captures live images, which are preprocessed and fed to the CNN model. The model predicts if the leaf is Healthy or Diseased, and the result is displayed on the screen.
What happens if the program is interrupted during training?
If interrupted, the program saves a partially trained model to avoid losing progress and shuts down gracefully.
Why save the model?
Saving the model allows reusing it without retraining, which saves time since training is computationally expensive.
What is the PlantVillage dataset?
It’s a dataset of plant leaf images labeled as healthy or diseased, used to train the model to recognize plant conditions.
