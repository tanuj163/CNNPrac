from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

# Load the model
model = load_model('my_cnn_model.h5')

# Load class names (you must know them — save them separately or hardcode)
class_names = ['apple fruit', 'banana fruit', 'cheery fruit','orange fruit','chickoo fruit', 'grapes fruit' ,'kiwi fruit' ,'mango fruit', 'orange fruit' ,'strawberry fruit']  # Example

# Now you can use the model to predict images again
# Load and preprocess image
def load_and_prepare_image(image_path, img_size=240):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = img / 255.0
    return img

# Example image
image_path = "or.jpg"
img = load_and_prepare_image(image_path)
img = tf.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
predicted_class = class_names[np.argmax(prediction[0])]
print(f"Predicted class: {predicted_class}")
