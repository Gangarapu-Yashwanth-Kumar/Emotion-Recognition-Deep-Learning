# 🎭 Emotion-Recognition-Deep-Learning 🧠

## 1. Project Overview 🌟

This project presents an end-to-end solution for classifying human moods/emotions from images using a Convolutional Neural Network (CNN). It's designed to demonstrate how deep learning models can be trained and then deployed as an interactive web application, allowing users to upload images and receive instant mood predictions.

## 2. Key Features ✨

* **Deep Learning Model Training**: A comprehensive Jupyter Notebook (`Mood Classification using CNN.ipynb`) for building, training, and evaluating a CNN model on an image dataset for mood classification.
* **Flask Web Application**: An intuitive web interface (`Mood_Classifier.py`) built with Flask, enabling users to upload image files.
* **Real-time Mood Prediction**: The web application loads the trained CNN model to predict the mood from uploaded images and displays the result.
* **Image Preprocessing**: Handles necessary image resizing and normalization before feeding to the model for prediction.
* **Robust File Handling**: Includes secure file uploading and validation for allowed image formats.

## 3. Technology Stack 🛠️

The project leverages powerful tools for deep learning and web development:

* **Programming Language**: Python 🐍
* **Deep Learning Framework**: TensorFlow / Keras 🧠
* **Web Framework**: Flask 🌐
* **Numerical Operations**: NumPy
* **Image Processing**: OpenCV (`cv2`), Keras `image` utilities
* **File Management**: `os`, `werkzeug.utils`
* **Development Environment**: Jupyter Notebook 📓

## 4. How It Works 💡

The system operates in two main phases:

1.  **Model Training (Offline in Jupyter Notebook)**:
    * The `Mood Classification using CNN.ipynb` notebook defines and compiles a Convolutional Neural Network (CNN) architecture.
    * It uses `ImageDataGenerator` for efficient loading, preprocessing, and augmentation of image datasets.
    * The CNN is trained on the image data to learn patterns associated with different moods.
    * After training, the final model is saved as `Mood-Classifier.h5`.

2.  **Web Application (Online with Flask)**:
    * The `Mood_Classifier.py` Flask application loads the pre-trained `Mood-Classifier.h5` model.
    * Users access the web interface to upload an image file (JPG, JPEG, PNG, GIF, BMP, or WebP).
    * The uploaded image is securely saved and then preprocessed (resized to 200x200 and normalized) to match the model's input requirements.
    * The preprocessed image is fed into the loaded CNN model for prediction.
    * The model outputs a score, which is then interpreted into human-readable mood labels (e.g., "Happy", "Sad", "Neutral", "Angry", "Fearful", "Disgusted", "Surprised" based on score thresholds).
    * The predicted mood is displayed back to the user on a dedicated results page, along with the uploaded image.

## 5. How to Run the Project ▶️

Follow these steps to get the Mood Classification system running on your local machine:

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/YourUsername/Emotion-Recognition-Deep-Learning.git](https://github.com/YourUsername/Emotion-Recognition-Deep-Learning.git)
    cd Emotion-Recognition-Deep-Learning
    ```
    *(Remember to replace `YourUsername` with your actual GitHub username or the repository's path)*

2.  **Install Dependencies**:
    It's highly recommended to use a virtual environment.
    ```bash
    pip install Flask tensorflow numpy opencv-python Pillow werkzeug
    ```
    *(Note: `Pillow` is a dependency for `tensorflow.keras.preprocessing.image.load_img`)*

3.  **Obtain the Trained Model**:
    * Run all cells in the `Mood Classification using CNN.ipynb` notebook to train the model and save `Mood-Classifier.h5`. This file must be present in the same directory as `Mood_Classifier.py`.
    * Alternatively, you might find a pre-trained `Mood-Classifier.h5` model in the repository if provided by the author.

4.  **Create Upload Folder**:
    Ensure there's a folder named `static/uploads` in your project root, as this is where uploaded images will be stored.
    ```bash
    mkdir -p static/uploads
    ```

5.  **Set Flask Secret Key**:
    The `app.secret_key` in `Mood_Classifier.py` is crucial for session management. **Change `'your_secret_key'` to a strong, unique secret key for production.**
    ```python
    app.secret_key = 'a_very_strong_and_random_secret_key_here'
    ```

6.  **Launch the Web Application**:
    ```bash
    python Mood_Classifier.py
    ```
    Open your web browser and navigate to the address shown in your terminal (usually `http://127.0.0.1:5000/`).

## 6. Model Output Interpretation 📊

The model predicts a numerical score (e.g., between 0 and 1). This score is then mapped to specific mood labels based on predefined thresholds. For example:

* `score <= 0.2`: "😞 Sad"
* `0.2 < score < 0.4`: "😠 Angry"
* `0.4 <= score <= 0.6`: "🤔 Not Sure" (or Neutral)
* `0.6 < score <= 0.8`: "😄 Happy"
* `score > 0.8`: "😮 Surprised"

*(Note: The exact mood labels and thresholds are defined within the `predict_emotion` function in `Mood_Classifier.py`.)*

---

### 🙏 Thank You! 🙏

Thank you for exploring the Emotion-Recognition-Deep-Learning project! We hope this tool provides valuable insights into the fascinating world of AI and emotion detection. Your interest and feedback are greatly appreciated..
