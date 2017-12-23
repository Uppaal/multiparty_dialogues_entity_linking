import tensorflow as tf

class SC(object):
    def __init__(self, mode='train'):
        if mode == 'train':
            self.mode = 1
        else:
            self.mode = 0
        self.create_placeholders()
        self.create_pooled_outputs_on_word_image()
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

    def create_placeholders(self):
        self.input_word_image = tf.placeholder(tf.float32, [None, 25, 50, 1])
        self.label = tf.placeholder(tf.float32, [None,1])

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

    def create_pooled_outputs_on_word_image(self):
        pool_1 = self.get_pooled_output(self.input_word_image,
                                        kernel_size=[3, 50],
                                        pool_size=(25 - 3 + 1, 1),
                                        num_filters=100,
                                        activation=tf.nn.relu)
        pool_1 = tf.reshape(pool_1, shape=[-1, 100])
        pool_2 = self.get_pooled_output(self.input_word_image,
                                        kernel_size=[4, 50],
                                        pool_size=(25 - 4 + 1, 1),
                                        num_filters=100,
                                        activation=tf.nn.relu)
        pool_2 = tf.reshape(pool_2, shape=[-1, 100])
        pool_3 = self.get_pooled_output(self.input_word_image,
                                        kernel_size=[5, 50],
                                        pool_size=(25 - 5 + 1, 1),
                                        num_filters=100,
                                        activation=tf.nn.relu)
        pool_3 = tf.reshape(pool_3, shape=[-1, 100])
        self.cnn_flatten = tf.concat((pool_1,pool_2,pool_3),axis=1)

    def create_relu_layer(self):
        self.relu_out = tf.layers.dense(self.cnn_flatten, 100, activation=tf.nn.relu)
        if self.mode==1:
            self.relu_out = tf.nn.dropout(self.relu_out, keep_prob=0.8)

    def create_prediction(self):
        self.sigmoid_out = tf.layers.dense(self.relu_out,1, activation=tf.nn.sigmoid, name="sigmoid")

    def create_loss(self):
        # self.loss = tf.reduce_mean(self.label*tf.log(self.sigmoid_out) +
        #                            (1.0 - self.label)*tf.log(1.0 - self.sigmoid_out))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.sigmoid_out))

    def training(self):
        self.optimizer = tf.train.AdamOptimizer()
        self.saver = tf.train.Saver()
        self.train_op = self.optimizer.minimize(self.loss)


