import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt



class siamese:
	"""Encapsulates the functions that are unique to siamese networks."""

	def __init__(self, learning_rate):
		#placeholder objects for inputs
		self.x1 = tf.placeholder(tf.float32, [None, 135])
		self.x2 = tf.placeholder(tf.float32, [None, 135])
		#placeholder objects for categorical targets
		self.y1 = tf.placeholder(tf.float32, [None])
		self.y2 = tf.placeholder(tf.float32, [None])
		self.nrows = tf.placeholder(tf.int32, [None])

		with tf.variable_scope("siamese") as scope:
			#output from network 1
			self.o1 = self.encoder(self.x1, 2)
			scope.reuse_variables()
			#output from the same network but different input
			self.o2 = self.encoder(self.x2, 2)

		#main target: the identity or otherwise of the two categorical targets
		self.y_ = tf.placeholder(tf.float32, [None])
		#loss computed from the two outputs of the network and the main target
		self.loss = self.loss_function()
		#training function
		self.train = tf.train.MomentumOptimizer(learning_rate, 0.05).minimize(self.loss)
		self.accuracy = self.evaluate()


	def fc_layer(self, input_layer, name, output_size, activation):
		"""Creates a single fully connected feedforward layer. Takes an input
		layer, a name, an output size, and an activation function."""

		with tf.name_scope(name):
			input_size = int(input_layer.shape[1])
			initer = tf.truncated_normal_initializer(stddev=0.01)
			W = tf.get_variable(name+"weights", dtype=tf.float32, shape=[input_size, output_size], initializer=initer)
			b = tf.get_variable(name+"biases", dtype=tf.float32, initializer=tf.constant(0.01, shape=[output_size], dtype=tf.float32))

			raw_out = tf.add(tf.matmul(input_layer, W), b)

			if activation:
				return( activation(raw_out))

			return(raw_out)


	def encoder(self, input_, output_size):
		"""Constructs a fully connected neural network. Takes an input
		placeholder object, an output size and a name."""

		layer_1 = self.fc_layer(input_, "layer1", 50, tf.nn.relu)

		layer_2 = self.fc_layer(layer_1, "layer2", 40, tf.nn.relu)

		encoded = self.fc_layer(layer_2, "output", output_size, tf.nn.relu)

		return(encoded)

	def evaluate(self):
		with tf.name_scope('evaluation'):

			#concatenate categorical target variables into one vector
			all_labels = tf.concat([self.y1, self.y2], 0)

			#create a matrix of distances between all pairs of points in all_labels
			#bit of algebra off the internet
			A = tf.concat([self.o1, self.o2], 0)
			r = tf.reduce_sum(A*A, 1)
			r = tf.reshape(r, [-1, 1])
			D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)

			#set diagonal, i.e. the distance of a point to itself, to a large number
			D_normalised = tf.matrix_set_diag(D, tf.fill(self.nrows, value=1000.0))

			#take the index number of the point closest to the point in question 
			closest = tf.argmin(D_normalised, 1)

			#create a vector of the label number of the closest point to a given point
			closest_label = tf.gather(params=all_labels, indices=closest)

			#score 1 if the target label and the closest label are identical, 0 otherwise, and take the average
			correct = tf.equal(all_labels, closest_label)
			accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

			#write to summary object
			tf.summary.scalar('accuracy', accuracy)

			return(accuracy)


	def loss_function(self):

		with tf.name_scope("loss"):
			#bit vector where 1 iff label 1 == label 2
			labels_t = self.y_
			#bit vector where 1 iff label 1 != label 2
			labels_f = tf.subtract(1.0, self.y_, name="labels_false")

			#subtracts the two coordinates of output 2 from those of output 1, squares the results and sums them up
			euclidean_squared = tf.reduce_sum(tf.pow(tf.subtract(self.o1, self.o2), 2), 1)

			#takes the square root of the above, which is the euclidean distance between each pair of points
			euclidean = tf.sqrt(euclidean_squared + 1e-6, name="distance")

			#the margin. pairs of labels that were mapped to positions further removed from each other than this are discarded
			C = tf.constant(5.0, name="C")

			#pairwise multiplication of true label indicator with squared distance. Large value = same label but large distance
			self.pos = tf.multiply(labels_t, euclidean_squared, name="positive_loss")
			#pairwise multiplication of false label indicator with C - distance. Large value = different labels but small distance
			#different labels with a distance larger than C in the output space are ignored
			self.neg = tf.multiply(tf.pow(tf.maximum(tf.subtract(C, euclidean), 0), 2), labels_f)
			#total loss for each pair of predictions
			losses = tf.add(self.pos, self.neg, name="losses")
			#total loss
			loss = tf.reduce_mean(losses, name="loss")
			#added to the summary object
			tf.summary.scalar('loss', loss)

			return(loss)


def split_sample_contrastive(X, Y, number_dict):

	if X.shape[0] % 2 != 0:
		X = X.iloc[:-1,:]
		Y = Y[:-1]

	X = X.reset_index(drop=True)
	Y = Y.reset_index(drop=True)

	#sequence in which data will be sampled from the training data set
	sample_sequence = np.arange(X.shape[0])
	np.random.shuffle(sample_sequence)
	#create the index for values in X_train that go into the first and second network
	batch_ind = sample_sequence >= X.shape[0]/2
	#these unwieldy things first reorder X_train or Y_train, respectively, 
	#according to the random sequence, then sample them into batch 1 or 2
	x1 = X.T[sample_sequence].T[batch_ind == True].values
	y1 = Y[sample_sequence][batch_ind == True].values
	x2 = X.T[sample_sequence].T[batch_ind == False].values
	y2 = Y[sample_sequence][batch_ind == False].values

	#Main target variable values 
	y = [float(x == y) for x, y in zip(y1, y2)]

	if 0 not in y1:
		y1 = pd.Series(np.array([number_dict[x] for x in y1]))
		y2 = pd.Series(np.array([number_dict[x] for x in y2]))

	return(x1, y1, x2, y2, y)



if __name__ == "__main__":

	LOGDIR = "./graphs"

	print(''.join(["\n"]*100))

	#start tensorflow session
	sess = tf.Session()
	with sess.as_default():

		tf.set_random_seed(1234)


		#initialise siamese network with learning rate
		siam = siamese(0.01)

		#initialise tensorflow summary writers
		train_writer = tf.summary.FileWriter(os.path.join(LOGDIR, "train"), sess.graph)
		test_writer = tf.summary.FileWriter(os.path.join(LOGDIR, "test"), sess.graph)

		summary_op = tf.summary.merge_all()

		saver = tf.train.Saver()

		#initialise variables
		init = tf.global_variables_initializer()
		sess.run(init)

		test_persons = ["p1", "p3", "p6"]

		prev_loss = [50]*50
		loss_threshold = 0
		loss_delta, loss = 3, 50

		#train the network
		np.random.seed(101)
		for step in range(10000):

			if loss_delta > loss_threshold:

				if step % 10 == 0:
					for root, dirs, files in os.walk("/home/leon/Documents/projects/data/sports/data_three_back/"):
						file = np.random.choice(files)
						data = pd.read_csv(os.path.join(root, file), header=None)

						while not any([x in data[:-1].values for x in test_persons]):
							file = np.random.choice(files)
							data = pd.read_csv(os.path.join(root, file), header=None)

					#separate data into train and test sets
					test_ind = np.array([x in test_persons for x in data.iloc[:,-1].values])

					#creating test data and target objects
					test = data[test_ind]
					X_test = test.iloc[:,:-2]
					Y_test = test.iloc[:,-1]

					train = data[test_ind == False]

					X_train = train.iloc[:,:-2]
					Y_train = train.iloc[:,-1]

					Y_values = list(set(np.concatenate([Y_train.values, Y_test.values])))
					number_dict = {x:y for x,y in zip(Y_values, range(len(Y_values)))}

					X_test1, Y_test1, X_test2, Y_test2, Y_test = split_sample_contrastive(X_test, Y_test, number_dict)

				X_train = train.iloc[:,:-2]
				Y_train = train.iloc[:,-1]
				X_train1, Y_train1, X_train2, Y_train2, Y_train = split_sample_contrastive(X_train, Y_train, number_dict)

				#train the network
				summary, _, loss = sess.run([summary_op, siam.train, siam.loss], feed_dict={
																		siam.x1: X_train1,
																		siam.x2: X_train2,
																		siam.y_: Y_train,
																		siam.y1: Y_train1,
																		siam.y2: Y_train2,
																		siam.nrows: [len(Y_train)*2]
																		})




				#test the network
				if step % 20 == 0:

					loss_delta = np.mean(prev_loss) - loss
					prev_loss = prev_loss[1:] + [loss]


					summary_result, acc = sess.run([summary_op, siam.accuracy], feed_dict={
																siam.x1: X_test1,
																siam.x2: X_test2,
																siam.y_: Y_test,
																siam.y1: Y_test1,
																siam.y2: Y_test2,
																siam.nrows: [len(Y_test)*2]
																})


					print("Step: {0}, loss: {1:.2}, accuracy {2:.2}".format(step, loss, acc))

					#write loss and accuracy to summary objects
					train_writer.add_summary(summary, step)
					test_writer.add_summary(summary_result, step)
			else:
				pass

		saver.save(sess, './model')

		embed = siam.o1.eval({siam.x1: np.concatenate([X_test1, X_test2], axis=0)})

		label_dict = {v: k for k, v in number_dict.items()}

		targets = np.concatenate([Y_test1, Y_test2], axis=0)
		targets = pd.Series(np.array([label_dict[x] for x in targets]))

		embed = pd.concat([pd.DataFrame(embed), targets], axis=1)


	pd.DataFrame(embed).to_csv("./embed.txt", header=False, index=False)

	train_writer.close()
	test_writer.close()
	sess.close()



