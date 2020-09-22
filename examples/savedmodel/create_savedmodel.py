# Copyright 2018 The TensorFlow Authors.
# Copyright 2020 ljn917.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Obtained from https://github.com/tensorflow/docs/blob/master/site/en/guide/saved_model.ipynb
# With modifications by ljn917

import numpy as np
import tensorflow as tf

if int(tf.__version__.split('.')[0]) < 2:
    raise RuntimeError("Need tensorflow 2.0")

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

pretrained_model = tf.keras.applications.MobileNet()
mobilenet_save_path = "mobilenet/1/"
tf.saved_model.save(pretrained_model, mobilenet_save_path)

loaded_model = tf.saved_model.load(mobilenet_save_path)
print(list(loaded_model.signatures.keys()))  # ["serving_default"]

infer = loaded_model.signatures["serving_default"]
print(infer.structured_outputs)

# generate inputs and results
IMAGE_SIZE = 224
img_file = tf.keras.utils.get_file(
    "grace_hopper.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
img = tf.keras.preprocessing.image.load_img(img_file, target_size=[IMAGE_SIZE, IMAGE_SIZE])
x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.keras.applications.mobilenet.preprocess_input(x[tf.newaxis,...])
print("input tensor shape: ", x.shape) # (1, 224, 224, 3)

labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

result_before_save = pretrained_model(x)
print("output tensor value:")
print(result_before_save)
top5_idx = np.argsort(result_before_save)[0,::-1][:5]
print("Top 5 indices: ", top5_idx)
top5_prob = result_before_save.numpy()[0, top5_idx]
print("Top 5 probabilities: ", top5_prob)
# Top 5 indices:  [652 457 834 439 715]
# Top 5 probabilities:  [0.70124376 0.1869016  0.02683803 0.02179799 0.00899362]
decoded = imagenet_labels[top5_idx+1]
print("Result before saving:\n", decoded)
#  ['military uniform' 'bow tie' 'suit' 'bearskin' 'pickelhaube']

# write testcase
# the first line is x, 224x224x3 values
# the second line is result, 1000 values
# NOTE: DO *NOT* open this file, it is likely to freeze your editor
with open("testcase", "w") as f:
    [f.write('{} '.format(i)) for i in x.reshape(-1)]
    f.write('\n')
    [f.write('{} '.format(i)) for i in result_before_save.numpy().reshape(-1)]
    f.write('\n')
