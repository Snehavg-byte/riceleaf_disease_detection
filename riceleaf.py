import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk

# ✅ Step 1: Load and Preprocess Dataset
dataset_path = r"C:\Users\SNEHA\PycharmProjects\riceleaf\rice_leaf_dataset"


train_dataset = image_dataset_from_directory(
    os.path.join(dataset_path, "train"),
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)

validation_dataset = image_dataset_from_directory(
    os.path.join(dataset_path, "validation"),
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)

# ✅ Get class names
class_names = train_dataset.class_names
print(f"Class Names: {class_names}")

# ✅ Step 2: Define CNN Model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

# ✅ Step 3: Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Step 4: Train Model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=2,  # ✅ Adjust epochs as needed
    verbose=1
)

# ✅ Print final accuracy
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

print("Training completed! ✅")

# ✅ Step 5: Save Model
model_path = "C:/Users/SNEHA/PycharmProjects/riceleaf/rice_leaf_cnn_model.h5"
model.save(model_path)
print("Model saved successfully! ✅")

# ✅ Step 6: Plot Accuracy & Loss
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')

plt.show()

# ✅ Step 7: Load Model for GUI-based Prediction
model = keras.models.load_model(model_path)
print("Model reloaded successfully! ✅")


# ✅ Step 8: GUI for Prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img, img_array


def predict_disease():
    file_path = filedialog.askopenfilename(
        initialdir=r"C:\Users\SNEHA\PycharmProjects\riceleaf\rice_leaf_dataset\validation",
        title="Select Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        img, processed_img = preprocess_image(file_path)
        prediction = model.predict(processed_img)
        predicted_class = class_names[np.argmax(prediction)]

        # Display selected image
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

        # Show prediction result
        result_label.config(text=f"Predicted Disease: {predicted_class} ✅")


# ✅ Create GUI Window
root = tk.Tk()
root.title("Rice Leaf Disease Detection")
root.geometry("500x500")

title_label = tk.Label(root, text="Rice Leaf Disease Detection", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

select_button = tk.Button(root, text="Select Image", command=predict_disease, font=("Arial", 12), bg="lightblue")
select_button.pack(pady=10)

img_label = tk.Label(root)
img_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
