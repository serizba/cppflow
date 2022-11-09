#!/usr/bin/env python
"""
    Example for a load frozen tf graph functionality.
"""

# MIT License
#
# Copyright (c) 2021 Daisuke Kato
# Copyright (c) 2021 Paul
# Copyright (c) 2022 Sergio Izquierdo
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
# @brief Creates and saves a simple Keras model as a frozen graph.
#
# @section Creates and saves a simple Keras model as a frozen graph.
#
# @section author_create_model Author(s)
# - Created  by Daisuke Kato
# - Created  by Paul
# - Modified by Sergio Izquierdo

# Imports
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

input_1 = tf.keras.Input(shape=(5,))
output_1 = tf.keras.layers.Dense(5, activation=tf.nn.relu)(input_1)
output_1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(output_1)
model = tf.keras.Model(inputs=input_1, outputs=output_1)

# Create frozen graph
x = tf.TensorSpec(model.input_shape, tf.float32, name="x")
concrete_function = tf.function(lambda x: model(x)).get_concrete_function(x)
frozen_model = convert_variables_to_constants_v2(concrete_function)

# Check input/output node name
print(f"{frozen_model.inputs=}")
print(f"{frozen_model.outputs=}")

# Save the graph as protobuf format
directory = "."
tf.io.write_graph(frozen_model.graph, directory, "model.pb", as_text=False)
