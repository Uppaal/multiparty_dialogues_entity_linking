import tensorflow as tf
import numpy as np
import math

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

        self.label = tf.placeholder(tf.float32, [None,1])

        self.filter_sizes = filter_sizes  # [1,2,3]

        # conv layer1
        stacked_inputs = [[self.phim1_1, self.phim1_2, self.phim1_3, self.phim1_4],
                          [[self.phim2_1, self.phim2_2, self.phim2_3, self.phim2_4]]]
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



        # self.pooled_outputs = {}
        self.pooled_outputs = []
        for i, val1 in enumerate([1,2]):
            for j, val2 in enumerate([1,2,3,4]):
                with tf.name_scope("conv_" + str(val1) + str(val2)):
                    # Convolution Layer
                    conv_1 = tf.layers.conv2d(self.data_dict["conv" + str(val1) + "_" + str(val2)],
                                              filters=1,
                                              kernel_size=[1, 50],
                                              # kernel_initializer=tf.constant_initializer(
                                              #     np.arange(50, dtype=np.float32)),
                                              strides=(1, 1),
                                              padding='valid',
                                              activation=tf.nn.relu)

                    pool_1 = tf.layers.max_pooling2d(conv_1,
                                                     pool_size=(self.data_dict["h_"+str(val2)] - 1 + 1, 1),
                                                     strides=(1, 1),
                                                     padding='valid')

                    conv_2 = tf.layers.conv2d(self.data_dict["conv" + str(val1) + "_" + str(val2)],
                                              filters=1,
                                              kernel_size=[2, 50],
                                              # kernel_initializer=tf.constant_initializer(
                                              #     np.arange(50, dtype=np.float32)),
                                              strides=(1, 1),
                                              padding='valid',
                                              activation=tf.nn.relu)

                    pool_2 = tf.layers.max_pooling2d(conv_2,
                                                     pool_size=(self.data_dict["h_"+str(val2)] - 2 + 1, 1),
                                                     strides=(1, 1),
                                                     padding='valid')

                    conv_3 = tf.layers.conv2d(self.data_dict["conv" + str(val1) + "_" + str(val2)],
                                              filters=1,
                                              kernel_size=[3, 50],
                                              # kernel_initializer=tf.constant_initializer(
                                              #     np.arange(50, dtype=np.float32)),
                                              strides=(1, 1),
                                              padding='valid',
                                              activation=tf.nn.relu)

                    pool_3 = tf.layers.max_pooling2d(conv_3,
                                                     pool_size=(self.data_dict["h_"+str(val2)] - 3 + 1, 1),
                                                     strides=(1, 1),
                                                     padding='valid')
                    # self.pooled_outputs["pool" + str(val1) + "_" + str(val2)] = tf.transpose(tf.stack((tf.reshape(pool_1,[-1]),tf.reshape(pool_2,[-1]),tf.reshape(pool_3,[-1]))))
                    self.pooled_outputs.append(
                        tf.stack((tf.reshape(pool_1, [-1]), tf.reshape(pool_2, [-1]), tf.reshape(pool_3, [-1]))))

        # for pool_out in self.pooled_outputs:
        self.dat_conv2_1 = tf.expand_dims(tf.transpose(tf.stack((self.pooled_outputs[0],
                                   self.pooled_outputs[1],
                                   self.pooled_outputs[2],
                                   self.pooled_outputs[3]
                                   ))),axis=3)
        self.dat_conv2_2 = tf.expand_dims(tf.transpose(tf.stack((self.pooled_outputs[4],
                                   self.pooled_outputs[5],
                                   self.pooled_outputs[6],
                                   self.pooled_outputs[7]
                                   ))),axis=3)

        conv2_data = {"conv_2_1": self.dat_conv2_1, "conv_2_2": self.dat_conv2_2}
        self.mention_embeddings = []
        #COnv_2
        for j, val2 in enumerate([1,2]):
            with tf.name_scope("conv_2_" + str(val1) + str(val2)):
                # Convolution Layer
                conv_1 = tf.layers.conv2d(conv2_data["conv_2_" + str(val2)],
                                          filters=1,
                                          kernel_size=[1, 4],
                                          # kernel_initializer=tf.constant_initializer(
                                          #     np.arange(4, dtype=np.float32)),
                                          strides=(1, 1),
                                          padding='valid',
                                          activation=tf.nn.relu)

                pool_1 = tf.layers.max_pooling2d(conv_1,
                                                 pool_size=(3 - 1 + 1, 1),
                                                 strides=(1, 1),
                                                 padding='valid')

                conv_2 = tf.layers.conv2d(conv2_data["conv_2_" + str(val2)],
                                          filters=1,
                                          kernel_size=[2, 4],
                                          # kernel_initializer=tf.constant_initializer(
                                          #     np.arange(4, dtype=np.float32)),
                                          strides=(1, 1),
                                          padding='valid',
                                          activation=tf.nn.relu)

                pool_2 = tf.layers.max_pooling2d(conv_2,
                                                 pool_size=(3 - 2 + 1, 1),
                                                 strides=(1, 1),
                                                 padding='valid')

                conv_3 = tf.layers.conv2d(conv2_data["conv_2_" + str(val2)],
                                          filters=1,
                                          kernel_size=[3, 4],
                                          # kernel_initializer=tf.constant_initializer(
                                          #     np.arange(4, dtype=np.float32)),
                                          strides=(1, 1),
                                          padding='valid',
                                          activation=tf.nn.relu)

                pool_3 = tf.layers.max_pooling2d(conv_3,
                                                 pool_size=(3 - 3 + 1, 1),
                                                 strides=(1, 1),
                                                 padding='valid')

                self.mention_embeddings.append(tf.stack((tf.reshape(pool_1,[-1]),
                                                         tf.reshape(pool_2,[-1]),
                                                         tf.reshape(pool_3,[-1]))))

        #Concatenate and pass through hidden layer to create a mention pair embedding
        self.me_concat = tf.transpose(tf.concat((self.mention_embeddings[0], self.mention_embeddings[1]),axis=0))
        self.w1 = tf.Variable(tf.truncated_normal([6, 24],
                                                  stddev=1.0 / math.sqrt(float(6))),
                              name='w1')
        self.b1 = tf.Variable(tf.zeros([24]),
                              name='b1')
        self.hidden_layer1 = tf.nn.relu(tf.matmul(self.me_concat, self.w1) + self.b1)
        self.w2 = tf.Variable(tf.truncated_normal([24, 12],
                                                  stddev=1.0 / math.sqrt(float(24))),
                              name='w2')
        self.b2 = tf.Variable(tf.zeros([12]),
                              name='b2')
        self.mention_pair = tf.nn.relu(tf.matmul(self.hidden_layer1, self.w2) + self.b2)

        self.w3 = tf.Variable(tf.truncated_normal([12, 1],
                                                  stddev=1.0 / math.sqrt(float(12))),
                              name='w3')
        self.b3 = tf.Variable(tf.zeros([1]), name='b3')
        self.sigmoid_out = tf.nn.sigmoid(tf.matmul(self.mention_pair, self.w3) + self.b3)
        self.loss = tf.reduce_mean(tf.squared_difference(self.sigmoid_out, self.label))

        init_op = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init_op)

    def predict(self, X):
        y_predict = self.sess.run( self.sigmoid_out, feed_dict={self.phim1_1: X[0],
                                                                    self.phim1_2: X[1],
                                                                    self.phim1_3: X[2],
                                                                    self.phim1_4: X[3],
                                                                    self.phim2_1: X[4],
                                                                    self.phim2_2: X[5],
                                                                    self.phim2_3: X[6],
                                                                    self.phim2_4: X[7]})
        predictor_c = lambda t:1 if(t >= (1-t)) else 0
        pred_func = np.vectorize(predictor_c)
        y_predict = pred_func(np.squeeze(y_predict))
        return y_predict


    def train(self, X, Y, X_test, Y_test):
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(self.loss)
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)
        cost = float('inf')
        weights = []
        for epoch in range(3000):
            _, c  = self.sess.run([train_op, self.loss], feed_dict={self.phim1_1: X[0],
                                                                    self.phim1_2: X[1],
                                                                    self.phim1_3: X[2],
                                                                    self.phim1_4: X[3],
                                                                    self.phim2_1: X[4],
                                                                    self.phim2_2: X[5],
                                                                    self.phim2_3: X[6],
                                                                    self.phim2_4: X[7],
                                                                    self.label  : Y})
            # print (self.objective(X,y))
            # print (self.sess.run([self.prob], feed_dict={self.X:X}))
            if((epoch+1)%100):
                error = self.predict(X_test)
                print(sum(error == Ytest))
            # print (c)
        #     if c < cost:
        #         cost = c
        #         weights = self.get_model_params()
        #     # print("epoch %d :avg COST is %f" % (epoch, c))
        # # print (cost)
        # self.set_model_params(*weights)

model = ACNN(2)
data_all =  np.load('../pairs_500.npy')
data_first = data_all[0]
# Datasets
phim1_1 = []
phim1_2 = []
phim1_3 = []
phim1_4 = []
phim2_1 = []
phim2_2 = []
phim2_3 = []
phim2_4 = []
# Labels
labels = []

for i,each in enumerate(data_first):
    phim1_1.append(each[0][0])
    phim1_2.append(each[0][1])
    phim1_3.append(each[0][2])
    phim1_4.append(each[0][3])

    phim2_1.append(each[1][0])
    phim2_2.append(each[1][1])
    phim2_3.append(each[1][2])
    phim2_4.append(each[1][3])

    labels.append(each[2])

np_phim1_1 = np.asarray(phim1_1)
np_phim1_2 = np.asarray(phim1_2)
np_phim1_3 = np.asarray(phim1_3)
np_phim1_4 = np.asarray(phim1_4)
np_phim2_1 = np.asarray(phim2_1)
np_phim2_2 = np.asarray(phim2_2)
np_phim2_3 = np.asarray(phim2_3)
np_phim2_4 = np.asarray(phim2_4)
np_labels = np.asarray(labels)

np_phim1_1_train = np_phim1_1[0:400]
np_phim1_2_train = np_phim1_2[0:400]
np_phim1_3_train = np_phim1_3[0:400]
np_phim1_4_train = np_phim1_4[0:400]
np_phim2_1_train = np_phim2_1[0:400]
np_phim2_2_train = np_phim2_2[0:400]
np_phim2_3_train = np_phim2_3[0:400]
np_phim2_4_train = np_phim2_4[0:400]
np_labels_train = np_labels[0:400]

Xtrain = [np_phim1_1_train, np_phim1_2_train, np_phim1_3_train, np_phim1_4_train, np_phim2_1_train, np_phim2_2_train, np_phim2_3_train, np_phim2_4_train]
Ytrain = np_labels_train

np_phim1_1_test = np_phim1_1[400:]
np_phim1_2_test = np_phim1_2[400:]
np_phim1_3_test = np_phim1_3[400:]
np_phim1_4_test = np_phim1_4[400:]
np_phim2_1_test = np_phim2_1[400:]
np_phim2_2_test = np_phim2_2[400:]
np_phim2_3_test = np_phim2_3[400:]
np_phim2_4_test = np_phim2_4[400:]
np_labels_test = np_labels[400:]

Xtest = [np_phim1_1_test, np_phim1_2_test, np_phim1_3_test, np_phim1_4_test, np_phim2_1_test, np_phim2_2_test, np_phim2_3_test, np_phim2_4_test]
Ytest = np_labels_test

model.train(Xtrain, Ytrain[..., np.newaxis], Xtest, Ytest)
y = model.predict(Xtest)
print (sum(y == Ytest))
print ("OK")

# print(model.session.run(model.pooled_outputs, feed_dict={model.phim1_1: np_phim1_1,
#                                                          model.phim1_2: np_phim1_2,
#                                                          model.phim1_3: np_phim1_3,
#                                                          model.phim1_4: np_phim1_4,
#                                                          model.phim2_1: np_phim2_1,
#                                                          model.phim2_2: np_phim2_2,
#                                                          model.phim2_3: np_phim2_3,
#                                                          model.phim2_4: np_phim2_4}))