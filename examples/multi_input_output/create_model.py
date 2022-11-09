#!/usr/bin/env python
"""
    Example for a multiple inputs and outputs functionality.
"""

# MIT License
#
# Copyright (c) 2020 Sergio Izquierdo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# @file create_model.py
#
# @brief Creates and saves a simple multi input multi output Keras model.
#
# @section Creates and saves a simple multi input multi output Keras model.
#
# @section author_create_model Author(s)
# - Created  by Sergio Izquierdo

# Imports
import tensorflow as tf

input_1 = tf.keras.Input(shape=(5,), name='my_input_1')
input_2 = tf.keras.Input(shape=(5,), name='my_input_2')

x1 = tf.keras.layers.Dense(5, activation=tf.nn.relu)(input_1)
x2 = tf.keras.layers.Dense(5, activation=tf.nn.relu)(input_2)

output_1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,
                                 name='my_outputs_1')(x1)
output_2 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,
                                 name='my_outputs_2')(x2)

model = tf.keras.Model(inputs=[input_1, input_2], outputs=[output_1, output_2])

model.compile()

# Export the model to a SavedModel
model.save('model', save_format='tf')
