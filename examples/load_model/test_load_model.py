import tensorflow as tf
import numpy as np

input = np.ones((10,5), dtype=np.float32)
tensor = tf.convert_to_tensor(input, dtype_hint=tf.float32, name='inputs')
model = tf.saved_model.load("model")
output = model(tensor)

reference = np.ones((10,1)) * 0.9522027
np.testing.assert_almost_equal(output.numpy(), 0.9522027)
