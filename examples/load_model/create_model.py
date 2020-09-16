import tensorflow as tf
import numpy as np


input = tf.keras.Input(shape=(5,))

output = tf.keras.layers.Dense(5, activation=tf.nn.relu)(input)
output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(output)
model = tf.keras.Model(inputs=input, outputs=output)

model.compile()

# Export the model to a SavedModel
model.save('model', save_format='tf')