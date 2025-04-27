# cnn_fruits_classification.py
# Clean corrected version to run in Visual Studio Code (not Colab)

# Step 1: Import libraries
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import pathlib
import shutil

# Step 2: Helper Functions
def split_dir_to_train_test_val(directory="images/",
                                 train_size=0.7,
                                 test_size=0.2,
                                 val_size=0.1):
    """
    Split a main images directory into train/test/validation folders.
    """
    rng = random.Random(42)

    for root, folders, files in os.walk(directory):
        for folder in folders:
            list_of_files = os.listdir(os.path.join(root, folder))
            rng.shuffle(list_of_files)

            train_files = list_of_files[:int(len(list_of_files) * train_size)]
            test_files = list_of_files[int(len(list_of_files) * train_size):int(len(list_of_files) * (train_size + test_size))]
            val_files = list_of_files[int(len(list_of_files) * (train_size + test_size)):]

            for file_list, subfolder in zip([train_files, test_files, val_files], ["train", "test", "validation"]):
                dest_dir = os.path.join("files", subfolder, folder)
                os.makedirs(dest_dir, exist_ok=True)
                for one_file in file_list:
                    shutil.copy2(src=os.path.join(root, folder, one_file),
                                 dst=os.path.join(dest_dir, one_file))
            print(f"Folder '{folder}' split into train/test/val.")

def get_class_names_from_folder(directory):
    data_dir = pathlib.Path(directory)
    class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
    return class_names

def visualize_random_image(target_dir, target_class):
    """
    Visualizes a random image from a target class folder.
    """
    image_dir = os.path.join(target_dir, target_class)
    random_image = random.choice(os.listdir(image_dir))

    img = mpimg.imread(os.path.join(image_dir, random_image))
    plt.imshow(img)
    plt.title(target_class)
    plt.axis('off')
    plt.show()

# Step 3: Split the dataset (only if needed once)
split_dir_to_train_test_val(directory="images/")

# Step 4: Load class names
class_names = get_class_names_from_folder(directory="files/train/")
print("Class names:", class_names)

# Step 5: Visualize one random image
visualize_random_image("files/train/", class_names[0])

# Step 6: Create Data Generators
train_datagen = ImageDataGenerator(rescale=1/255.)
val_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

train_data = train_datagen.flow_from_directory(directory="files/train",
                                               target_size=(240, 240),
                                               batch_size=32,
                                               class_mode="categorical")

validation_data = val_datagen.flow_from_directory(directory="files/validation",
                                               target_size=(240, 240),
                                               batch_size=32,
                                               class_mode="categorical")

test_data = test_datagen.flow_from_directory(directory="files/test",
                                               target_size=(240, 240),
                                               batch_size=32,
                                               class_mode="categorical")

# Step 7: Build the CNN Model
model = Sequential([
    Conv2D(16, 3, activation="relu", input_shape=(240, 240, 3)),
    MaxPooling2D(pool_size=2),
    Conv2D(32, 3, activation="relu"),
    MaxPooling2D(pool_size=2),
    Conv2D(32, 3, activation="relu"),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation="softmax")  # Automatically matches number of classes
])

# Step 8: Compile the Model
model.compile(loss="categorical_crossentropy",
              optimizer=Adam(),
              metrics=["accuracy"])

# Step 9: Train the Model
history = model.fit(train_data,
                    epochs=5,
                    validation_data=validation_data)

# Step 10: Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_data)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Step 11: Plot Training History
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.show()

# Step 12: Confusion Matrix
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(test_data.classes, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:\n")
print(classification_report(test_data.classes, y_pred_classes, target_names=class_names))
model.save('my_cnn_model.h5')
print("Model saved successfully as 'my_cnn_model.h5'")

# DONE!
