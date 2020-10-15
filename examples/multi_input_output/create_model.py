import tensorflow as tf
import numpy as np


input_1 = tf.keras.Input(shape=(5,), name='my_input_1')
input_2 = tf.keras.Input(shape=(5,), name='my_input_2')

x1 = tf.keras.layers.Dense(5, activation=tf.nn.relu)(input_1)
x2 = tf.keras.layers.Dense(5, activation=tf.nn.relu)(input_2)

output_1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name='my_outputs_1')(x1)
output_2 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name='my_outputs_2')(x2)

model = tf.keras.Model(inputs=[input_1, input_2], outputs=[output_1, output_2])

model.compile()

# Export the model to a SavedModel
model.save('model', save_format='tf')
