import numpy as np
import pandas as pd
import tensorflow as tf
import siamese_op
import os

if __name__ == '__main__':
	sess = tf.InteractiveSession()
	# setup siamese network
	siam = siamese_op.siamese(0.02);
	saver = tf.train.Saver()
	tf.global_variables_initializer().run()

	saver.restore(sess, './model')


	with sess.as_default():
		np.random.seed(105)
		files_ = []
		for root, dirs, files in os.walk("/home/leon/Documents/projects/data/sports/data_three_back/"):
			files_ = np.random.choice(files, 200, False)

		data_list = []
		for file in files_:
			data = pd.read_csv(os.path.join("/home/leon/Documents/projects/data/sports/data_three_back", file), header=None)
			data = data.sample(n=20)

			data_list += [data]


		data = pd.concat(data_list, axis=0)

		test_activities = ["a19"]

		#data = data[[x not in test_activities for x in data.iloc[:,-2]]]

		embed = siam.o1.eval({siam.x1: data.iloc[:,:-2]})

		embed = pd.concat([pd.DataFrame(embed), data.iloc[:,-2].reset_index(drop=True)], axis=1)
		embed.to_csv("./embed.txt", header=False, index=False)
