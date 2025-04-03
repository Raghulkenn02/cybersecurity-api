import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model("D:/forensic_app/phishing_model.h5")

# Save the model in SavedModel format
model.save("D:/forensic_app/phishing_model_saved_model")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
