import tensorflow as tf
import numpy as np


class ACNN(object):
    def __init__(self,
                 filter_sizes):  # ,m1_phi_1, m1_phi_2, m1_phi_3, m1_phi_4, m2_phi_1, m2_phi_2, m2_phi_3, m2_phi_4):

        self.phim1_1 = tf.placeholder(tf.float32, [None, 3, 50, 1])
        self.phim1_2 = tf.placeholder(tf.float32, [None, 7, 50, 1])
        self.phim1_3 = tf.placeholder(tf.float32, [None, 5, 50, 1])
        self.phim1_4 = tf.placeholder(tf.float32, [None, 5, 50, 1])

        self.phim2_1 = tf.placeholder(tf.float32, [None, 3, 50, 1])
        self.phim2_2 = tf.placeholder(tf.float32, [None, 7, 50, 1])
        self.phim2_3 = tf.placeholder(tf.float32, [None, 5, 50, 1])
        self.phim2_4 = tf.placeholder(tf.float32, [None, 5, 50, 1])

        self.filter_sizes = filter_sizes  # [1,2,3]

        # conv layer1
        stacked_inputs = [[self.phim1_1, self.phim1_2, self.phim1_3, self.phim1_4],
                          [[self.phim2_1, self.phim2_2, self.phim2_3, self.phim2_4]]]
        data_dict = {
            "conv1_1": self.phim1_1,
            "conv1_2": self.phim1_2,
            "conv1_3": self.phim1_3,
            "conv1_4": self.phim1_4,

            "conv2_1": self.phim2_1,
            "conv2_2": self.phim2_2,
            "conv2_3": self.phim2_3,
            "conv2_4": self.phim2_4,

            "h_1": 3,
            "h_2": 7,
            "h_3": 5,
            "h_4": 5

        }

        self.pooled_outputs = {}
        for i, val1 in enumerate([1,2]):
            for j, val2 in enumerate([1,2,3,4]):
                with tf.name_scope("conv_" + str(val1) + str(val2)):
                    # Convolution Layer
                    conv_1 = tf.layers.conv2d(data_dict["conv" + str(val1) + "_" + str(val2)],
                                              filters=1,
                                              kernel_size=[1, 50],
                                              kernel_initializer=tf.constant_initializer(
                                                  np.arange(50, dtype=np.float32)),
                                              strides=(1, 1),
                                              padding='valid',
                                              activation=tf.nn.relu)

                    pool_1 = tf.layers.max_pooling2d(conv_1,
                                                     pool_size=(data_dict["h_"+str(val2)] - 1 + 1, 1),
                                                     strides=(1, 1),
                                                     padding='valid')

                    conv_2 = tf.layers.conv2d(data_dict["conv" + str(val1) + "_" + str(val2)],
                                              filters=1,
                                              kernel_size=[2, 50],
                                              kernel_initializer=tf.constant_initializer(
                                                  np.arange(50, dtype=np.float32)),
                                              strides=(1, 1),
                                              padding='valid',
                                              activation=tf.nn.relu)

                    pool_2 = tf.layers.max_pooling2d(conv_2,
                                                     pool_size=(data_dict["h_"+str(val2)] - 2 + 1, 1),
                                                     strides=(1, 1),
                                                     padding='valid')

                    conv_3 = tf.layers.conv2d(data_dict["conv" + str(val1) + "_" + str(val2)],
                                              filters=1,
                                              kernel_size=[3, 50],
                                              kernel_initializer=tf.constant_initializer(
                                                  np.arange(50, dtype=np.float32)),
                                              strides=(1, 1),
                                              padding='valid',
                                              activation=tf.nn.relu)

                    pool_3 = tf.layers.max_pooling2d(conv_3,
                                                     pool_size=(data_dict["h_"+str(val2)] - 3 + 1, 1),
                                                     strides=(1, 1),
                                                     padding='valid')
                    self.pooled_outputs["pool" + str(val1) + "_" + str(val2)] = tf.stack((tf.reshape(pool_1,[-1]),tf.reshape(pool_2,[-1]),tf.reshape(pool_3,[-1])))

        self.session = tf.Session()
        init_op = tf.initialize_all_variables()
        self.session.run(init_op)


a = np.reshape(np.arange(50), (1, 50))
b = np.reshape(np.arange(50, 100), (1, 50))
c = np.reshape(np.arange(100, 150), (1, 50))
temp = np.concatenate([a, b, c])
temp_1 = np.concatenate([a,b,a,c,a,c,c])
temp_2 = np.concatenate([a,b,a,c,a])
temp1 = np.reshape(temp, (3, 50, 1))
temp2 = np.reshape(temp_1, (7, 50, 1))
temp3 = 3*np.reshape(temp_2, (5, 50, 1))
temp4 = 2*np.reshape(temp_2, (5, 50, 1))
# temp1 = np.reshape(temp,(3,50))
model = ACNN(2)
# print(model.session.run(model.conv0_phim1_1, feed_dict={model.phim1_1: [temp1]}))
# print(model.session.run(model.pool0_phim1_1, feed_dict={model.phim1_1: [temp1]}))
# print("********************************************************")
# print(model.session.run(model.conv0_phim1_2, feed_dict={model.phim1_1: [temp1]}))
# print(model.session.run(model.pool0_phim1_2, feed_dict={model.phim1_1: [temp1]}))
# print("********************************************************")
# print(model.session.run(model.conv0_phim1_3, feed_dict={model.phim1_1: [temp1]}))
# print(model.session.run(model.pool0_phim1_3, feed_dict={model.phim1_1: [temp1]}))
print(model.session.run(model.pooled_outputs, feed_dict={model.phim1_1: [temp1], model.phim1_2:[temp2], model.phim1_3:[temp3],model.phim1_4:[temp4],
                                                         model.phim2_1: [temp1], model.phim2_2: [temp2],
                                                         model.phim2_3: [temp3], model.phim2_4: [temp4]}))