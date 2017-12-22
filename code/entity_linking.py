import tensorflow as tf
import numpy as np


class EntityLinking(object):
	def __init__(self, embedding_size, n_classes, mode):
		if mode=='train':
			self.train_mode = True
		else:
			self.train_mode = False
		self.embedding_size = embedding_size
		self.n_classes = n_classes
		self.create_placeholders()
		self.create_make_embeddings_layer()
		self.create_relu_layer()
		# self.create_prediction()
		self.create_loss()
		self.training()
		self.init_op = tf.global_variables_initializer()
		self.sess = tf.Session()
		if self.train_mode:
			self.sess.run(self.init_op)
		# self.variables = tf.global_variables()


	def create_placeholders(self):
		self.mention = tf.placeholder(tf.float32, [None,self.embedding_size], name='mention') 
			# batch_size x embeddings_size
		self.mentions_in_cluster = tf.placeholder(tf.float32, [None,None,self.embedding_size,1], name='mentions_in_cluster')  
			# batch_size x height x width x 1
		self.mention_pairs_in_cluster = tf.placeholder(tf.float32, [None,None,self.embedding_size,1], name='mention_pairs_in_cluster') 
			# batch_size x height x width x 1
		self.y = tf.placeholder(tf.float32, [None,self.n_classes], name='y')
	

	def get_rep_embeddings(self,list_of_embeddings, n_filters, filter_size, conv_stride = [1,1], keep_prob = 0.8):
		avg_pool = tf.reduce_mean(list_of_embeddings, axis=1, keep_dims=True)
		max_pool = tf.reduce_max(list_of_embeddings, axis=1, keep_dims=True)
		pooled = tf.concat([avg_pool, max_pool], axis=1, name='pooled')
			# batch_size x 2 x embeddings_size x 1
		conv = tf.layers.conv2d(inputs = pooled,
								filters = n_filters,
								kernel_size = filter_size,
								strides = conv_stride,
								padding = 'valid',
								activation = tf.nn.tanh,
								name='conv'
								)
			# batch_size x 1 x 1 x embeddings_size
		if self.train_mode:
			conv = tf.nn.dropout(conv, keep_prob = keep_prob)
		cluster_emb = tf.reshape(conv, [-1, self.embedding_size], name='cluster_emb')
			# batch_size x embeddings_size
		return cluster_emb

	def create_make_embeddings_layer(self):
		with tf.name_scope('mention_cluster'):
			with tf.variable_scope('mention_cluster'):
				self.cluster_emb_m = self.get_rep_embeddings(list_of_embeddings = self.mentions_in_cluster, 
														n_filters = self.embedding_size,
														filter_size = [2,self.embedding_size])
		
		with tf.name_scope('mention_pairs_cluster'):
			with tf.variable_scope('mention_pairs_cluster'):
				self.cluster_emb_p = self.get_rep_embeddings(list_of_embeddings = self.mention_pairs_in_cluster, 
														n_filters = self.embedding_size,
														filter_size = [2,self.embedding_size])
		
		self.emb_features = tf.concat([self.mention,self.cluster_emb_m,self.cluster_emb_p], axis=1, name='emb_features') 
			# batch_size x (3 * embeddings_size)

	
	def create_relu_layer(self):
		self.relu_1_out = tf.layers.dense(self.emb_features, self.embedding_size, activation=tf.nn.relu, name="relu_1")
    		# batch_size x embeddings_size
		self.preds = tf.layers.dense(self.relu_1_out, self.n_classes, activation=tf.nn.relu, name="relu_preds")
    		# batch_size x n_classes


    # def create_preds(self):
        # self.preds = tf.nn.softmax(self.relu_2_out) 
	    	# batch_size x n_classes

	def prediction(self, mentions, mention_clusters, mention_p_clusters, y):
		ret = self.sess.run(self.preds, 
			feed_dict = {self.mention:mentions,
						self.mentions_in_cluster:mention_clusters,
						self.mention_pairs_in_cluster:mention_p_clusters,
						self.y:y})
		return np.argmin(ret,axis=1)


	def create_loss(self):
		self.softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.preds)
		self.loss = tf.reduce_mean(self.softmax_cross_entropy)
	
	def training(self):
		# self.optimizer = tf.train.AdamOptimizer()
		self.optimizer = tf.train.GradientDescentOptimizer(0.01)
		self.saver = tf.train.Saver()
		self.train_op = self.optimizer.minimize(self.loss)

print()
print('OK')