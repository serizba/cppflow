import tensorflow as tf

model = tf.keras.applications.EfficientNetB0()

# Export the model to a SavedModel
model.save('model', save_format='tf')


