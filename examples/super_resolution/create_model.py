import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class PointConv2D(tf.keras.layers.Layer):

    def __init__(self):
        super(PointConv2D, self).__init__()

    def build(self, input_shape):
        _, _, _, num_channel = input_shape
        self.conv = tf.keras.layers.Conv2D(filters = num_channel, kernel_size = 1, padding = 'same')

    def call(self, input):
        return self.conv(input)

class GroupConv2D(tf.keras.layers.Layer):

    def __init__(self, num_group, kernel_size = 3):
        super(GroupConv2D, self).__init__()
        self.num_group = num_group
        self.kernel_size = kernel_size

    def build(self, input_shape):

        _, _, _, num_channel = input_shape

        self.group_conv_list = []
        for _ in range(self.num_group):
            self.group_conv_list.append(tf.keras.layers.Conv2D(filters = int(num_channel / self.num_group), kernel_size = self.kernel_size, padding = 'same'))

    def call(self, input):

        sub_tensor_list = tf.split(input, num_or_size_splits = self.num_group, axis = -1)
        result = []

        for idx, sub_tensor in enumerate(sub_tensor_list):

            result.append(self.group_conv_list[idx](sub_tensor))

        return tf.concat(result, axis = -1)

class ResBlock_E(tf.keras.layers.Layer):
    #num_channel,
    def __init__(self, num_conv_block, kernel_size, group = True, num_group = 4, channel_multiplier = 1, padding = 'SAME', residual_scale = 0.1):
        super(ResBlock_E, self).__init__()
        #self.num_channel = num_channel
        self.num_conv_block = num_conv_block
        self.kernel_size = kernel_size
        self.group = group
        self.num_group = num_group
        self.channel_multiplier = channel_multiplier
        self.padding = padding
        self.residual_scale = residual_scale
        self.group_conv = GroupConv2D(self.num_group, self.kernel_size)

    def build(self, input_shape):

        _, _, _, num_channel = input_shape
        self.point = PointConv2D()
        self.num_channel = num_channel
        #self.depthwise_filter = tf.random.uniform([self.kernel_size, self.kernel_size, self.num_channel, self.channel_multiplier])

    def call(self, input):

        x = input
        #Residual-E
        for _ in range(self.num_conv_block):

            if self.group:
                x = self.group_conv(x)

            else:
                strides = [1, 1, 1, 1]
                x = tf.nn.depthwise_conv2d(x, self.depthwise_filter, strides, self.padding)

            x = tf.keras.layers.LeakyReLU()(x)

        #Pointwise Convolution
        x = self.point(x)
        x += self.residual_scale*input
        #Activation Function
        x = tf.keras.layers.LeakyReLU()(x)

        return x

class CasResBlock(tf.keras.layers.Layer):

    def __init__(self, num_res_block, num_conv_block, kernel_size, group = True, num_group = 4, channel_multiplier = 1, padding = 'SAME', residual_scale = 0.1):
        super(CasResBlock, self).__init__()
        self.num_res_block = num_res_block
        self.num_conv_block = num_conv_block
        self.kernel_size = kernel_size
        self.group = group
        self.num_group = num_group
        self.channel_multiplier = channel_multiplier
        self.padding = padding
        self.residual_scale = residual_scale
        self.res_list = [ResBlock_E(self.num_conv_block, self.kernel_size, self.group, self.num_group, self.channel_multiplier, self.padding, self.residual_scale) for idx in range(self.num_res_block)]
        self.point_list = [PointConv2D() for _ in range(self.num_res_block)]

    def call(self, input):

        x = input
        out = input

        for idx in range(self.num_res_block):

            res_out = self.res_list[idx](out)
            x = tf.concat([x, res_out], axis = -1)
            out = self.point_list[idx](x)

        return out

class ConvT2D(tf.keras.layers.Layer):

    def __init__(self, kernel_size, padding = 'same'):
        super(ConvT2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding

    def build(self, input_shape):

        _, _, _, num_channel = input_shape
        self.convt = tf.keras.layers.Conv2Conv2DTranspose(filters = num_channel, kernel_size = self.kernel_size, padding = self.padding)

    def call(self, input):
        return self.convt(input)

class Upsample(tf.keras.layers.Layer):

    def __init__(self, kernel_size, scale = 4):
        super(Upsample, self).__init__()
        self.scale = scale
        self.kernel_size = kernel_size
        self.conv_out = tf.keras.layers.Conv2D(filters = 3, kernel_size = kernel_size, padding = 'same')
        self.convt_list = []

    def build(self, input_shape):

        _, _, _, num_channel = input_shape

        for idx in range(int(np.log2(self.scale))):
            self.convt_list.append(tf.keras.layers.Conv2DTranspose(filters = int(num_channel / 2**(idx+1)), kernel_size = self.kernel_size, strides = 2, padding = 'same'))

    def call(self, input):

        x = input

        for idx in range(int(np.log2(self.scale))):
            x = self.convt_list[idx](x)

        out = self.conv_out(x)

        return out

class CasResNet(tf.keras.Model):

    def __init__(self, initial_filter_num, num_cas_block, num_res_block, num_conv_block, kernel_size, group = True, num_group = 4, channel_multiplier = 1, padding = 'SAME', residual_scale = 0.1):
        super(CasResNet, self).__init__()
        self.num_cas_block = num_cas_block
        self.num_res_block = num_res_block
        self.num_conv_block = num_conv_block
        self.kernel_size = kernel_size
        self.group = group
        self.num_group = num_group
        self.channel_multiplier = channel_multiplier
        self.padding = padding
        self.residual_scale = residual_scale
        self.conv_in = tf.keras.layers.Conv2D(filters = initial_filter_num, kernel_size = kernel_size, padding = 'same')
        self.conv_out = tf.keras.layers.Conv2D(filters = 3, kernel_size = kernel_size, padding = 'same')
        self.cas_list = []
        self.point_list = []

        for _ in range(num_cas_block):
            self.cas_list.append(CasResBlock(self.num_res_block, self.num_conv_block, self.kernel_size, self.group, self.num_group, self.channel_multiplier, self.padding, self.residual_scale))
            self.point_list.append(PointConv2D())

        self.upsample = Upsample(self.kernel_size)


    def build(self, input_shape):
        self.dim = input_shape[1:]

    def call(self, input):

        x = self.conv_in(input)
        out = x

        for idx in range(self.num_cas_block):

            cas_out = self.cas_list[idx](out)
            x = tf.concat([x, cas_out], axis = -1)
            out = self.point_list[idx](x)

        out = self.upsample(out)
        out = self.conv_out(out)

        out = tf.clip_by_value(out, 0.0, 255.0)

        return out

    def build_graph(self):
        x = tf.keras.Input(shape=(self.dim))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

initial_filter_num = 16
num_cas_block = 1
num_res_block = 2
num_conv_block = 2
kernel_size = 3
input = plt.imread('output.jpg')
input_small = cv2.resize(input, None, fx = 1/4, fy = 1/4).astype(np.float32)
epochs = 100

save_path = 'model'
if not os.path.exists(save_path):
    os.makedirs(save_path)

model = CasResNet(initial_filter_num, num_cas_block, num_res_block, num_conv_block, kernel_size)
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)

model.compile(optimizer, loss = tf.keras.losses.MeanSquaredError())
model.fit(input_small[np.newaxis, ...], input[np.newaxis, ...], epochs = epochs)
model.build_graph().summary()
tf.keras.utils.plot_model(model.build_graph(), show_shapes=True, dpi=256)

result = model(input_small[np.newaxis, ...])
result = tf.cast(tf.squeeze(result), tf.uint8).numpy()
plt.imsave('python_result.png', result)

model.save(save_path)

'''
#HOW TO LOAD THE PRETRAINED MODEL IN PYTHON
model = tf.keras.models.load_model(save_path)
result = model(input_small[np.newaxis, ...])
result = tf.cast(tf.squeeze(result), tf.uint8).numpy()
'''




































#end
