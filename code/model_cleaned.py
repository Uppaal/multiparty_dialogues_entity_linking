import tensorflow as tf
import numpy as np
import math


class ACNN(object):
    def __init__(self, filter_sizes, beta=0.1, mode='train'):
        if mode == 'train':
            self.mode = 1
        else:
            self.mode = 0
        self.beta = beta
        self.create_placeholders(filter_sizes)
        self.create_acnn_1_outs()
        self.create_mention_embeddings()
        self.create_mention_pair_embeddings()
        self.create_relu_layer()
        self.create_prediction()
        self.create_loss()
        self.training()
        self.init_op = tf.global_variables_initializer()
        self.init_op2 = tf.local_variables_initializer()
        self.sess = tf.Session()
        if mode == 'train':
            self.init3 = tf.variables_initializer(tf.all_variables())
            self.sess.run(self.init_op)
            self.sess.run(self.init3)

    def create_placeholders(self, filter_sizes):
        self.phim1_1 = tf.placeholder(tf.float32, [None, 3, 50, 1])
        self.phim1_2 = tf.placeholder(tf.float32, [None, 7, 50, 1])
        self.phim1_3 = tf.placeholder(tf.float32, [None, 5, 50, 1])
        self.phim1_4 = tf.placeholder(tf.float32, [None, 5, 50, 1])

        self.phim1_d = tf.placeholder(tf.float32, [None, 154, 1])

        self.phim2_1 = tf.placeholder(tf.float32, [None, 3, 50, 1])
        self.phim2_2 = tf.placeholder(tf.float32, [None, 7, 50, 1])
        self.phim2_3 = tf.placeholder(tf.float32, [None, 5, 50, 1])
        self.phim2_4 = tf.placeholder(tf.float32, [None, 5, 50, 1])

        self.phim2_d = tf.placeholder(tf.float32, [None, 154, 1])

        self.phi_p = tf.placeholder(tf.float32, [None, 4])

        self.filter_sizes = filter_sizes  # [1,2,3]
        self.label = tf.placeholder(tf.float32, [None, 1])

        self.data_dict = {
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

    def get_pooled_output(self, input_conv, kernel_size, pool_size, stride=(1, 1),
                          padding='valid', num_filters=1, activation=tf.nn.tanh, prob=0.8):
        conv = tf.layers.conv2d(input_conv,
                                filters=num_filters,
                                kernel_size=kernel_size,
                                strides=stride,
                                padding=padding,
                                activation=activation)
        if self.mode==1:
            drop = tf.nn.dropout(conv, keep_prob=prob)
            pool = tf.layers.max_pooling2d(drop,
                                        pool_size=pool_size,
                                        strides=stride,
                                        padding=padding)
        else:
            pool = tf.layers.max_pooling2d(conv,
                                           pool_size=pool_size,
                                           strides=stride,
                                           padding=padding)
        return pool

    def create_acnn_1_outs(self):
        pooled_outputs = []
        for i, val1 in enumerate([1, 2]):
            for j, val2 in enumerate([1, 2, 3, 4]):
                with tf.name_scope("conv_accn_1_out" + str(val1) + str(val2)):
                    pool_1 = self.get_pooled_output(self.data_dict["conv" + str(val1) + "_" + str(val2)],
                                                    kernel_size=[1, 50],
                                                    pool_size=(self.data_dict["h_" + str(val2)] - 1 + 1, 1),
                                                    num_filters=280)
                    pool_2 = self.get_pooled_output(self.data_dict["conv" + str(val1) + "_" + str(val2)],
                                                    kernel_size=[2, 50],
                                                    pool_size=(self.data_dict["h_" + str(val2)] - 2 + 1, 1),
                                                    num_filters=280)
                    pool_3 = self.get_pooled_output(self.data_dict["conv" + str(val1) + "_" + str(val2)],
                                                    kernel_size=[3, 50],
                                                    pool_size=(self.data_dict["h_" + str(val2)] - 3 + 1, 1),
                                                    num_filters=280)
                    pooled_outputs.append(
                        tf.reshape(tf.stack((tf.reshape(pool_1, [-1, 280]), tf.reshape(pool_2, [-1, 280]),
                                             tf.reshape(pool_3, [-1, 280]))), [-1, 3, 280]))
        self.out_accn1_m1 = tf.reshape(
            tf.stack((pooled_outputs[0], pooled_outputs[1], pooled_outputs[2], pooled_outputs[3])), [-1, 3, 4, 280])
        self.out_accn1_m2 = tf.reshape(
            tf.stack((pooled_outputs[4], pooled_outputs[5], pooled_outputs[6], pooled_outputs[7])), [-1, 3, 4, 280])

    def create_mention_embeddings(self):
        dic = {"accn1_out_m1": self.out_accn1_m1, "accn1_out_m2": self.out_accn1_m2,
               "phi_md1": self.phim1_d, "phi_md2": self.phim2_d}
        self.mention_embeddings = []
        for j, val2 in enumerate([1, 2]):
            with tf.name_scope("conv_accn_2_out" + str(val2)):
                pool = self.get_pooled_output(dic["accn1_out_m" + str(val2)],
                                              num_filters=280,
                                              kernel_size=[2, 2],
                                              stride=(1, 1),
                                              padding='valid',
                                              activation=tf.nn.tanh, pool_size=[2, 3])
                mention_emb = tf.concat((tf.reshape(pool, [-1, 280]), tf.reshape(dic["phi_md"+str(val2)], [-1, 154])), axis=1)
            self.mention_embeddings.append(mention_emb)

    def create_mention_pair_embeddings(self):
        input_mpair = tf.reshape(tf.stack((self.mention_embeddings[0], self.mention_embeddings[1])), [-1, 2, 1, 434])
        pool = self.get_pooled_output(input_mpair,
                                      num_filters=280,
                                      kernel_size=[1,1],
                                      pool_size=(2, 1))
        self.pool_test = pool
        self.mention_pair = tf.concat((tf.reshape(pool,[-1,280]), self.phi_p), axis=1)


    def create_relu_layer(self):
        self.relu_out = tf.layers.dense(self.mention_pair, 200, activation=tf.nn.relu, name="last_relu")

    def create_prediction(self):
        self.sigmoid_out = tf.layers.dense(self.relu_out,1, activation=tf.nn.sigmoid, name="sigmoid")

    def create_loss(self):
        f = tf.trainable_variables()
        weight_list = []
        for each in f:
            if 'bias' not in each.name:
                weight_list.append(each)
        self.loss = tf.reduce_mean(tf.squared_difference(self.sigmoid_out, self.label)) \
                    # +\
                    # tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.beta), weights_list= f)

    def predict(self, X):
        y_predict = self.sess.run(self.sigmoid_out, feed_dict={self.phim1_1: X[0],
                                                                   self.phim1_2: X[1],
                                                                   self.phim1_3: X[2],
                                                                   self.phim1_4: X[3],
                                                                   self.phim1_d: X[4],
                                                                   self.phim2_1: X[5],
                                                                   self.phim2_2: X[6],
                                                                   self.phim2_3: X[7],
                                                                   self.phim2_4: X[8],
                                                                   self.phim2_d: X[9],
                                                                   self.phi_p  : X[10]})
        predictor_c = lambda t: 1 if (t >= (1 - t)) else 0
        pred_func = np.vectorize(predictor_c)
        y_predict = pred_func(np.squeeze(y_predict))
        return y_predict

    def training(self):
        self.optimizer = tf.train.AdamOptimizer()
        self.saver = tf.train.Saver()
        self.train_op = self.optimizer.minimize(self.loss)