import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

input = tf.keras.Input(shape=(5,))
output = tf.keras.layers.Dense(5, activation=tf.nn.relu)(input)
output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(output)
model = tf.keras.Model(inputs=input, outputs=output)

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