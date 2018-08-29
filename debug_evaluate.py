			y1 = tf.placeholder(tf.float32, [None])
			y2 = tf.placeholder(tf.float32, [None])
			o1 = tf.placeholder(tf.float32, [None, 2])
			o2 = tf.placeholder(tf.float32, [None, 2])
			nrows = tf.placeholder(tf.int32, [None])

			#concatenate categorical target variables into one vector
			all_labels = tf.concat([y1, y2], 0)

			#create a matrix of distances between all pairs of points in all_labels
			#bit of algebra off the internet
			A = tf.concat([o1, o2], 0)
			r = tf.reduce_sum(A*A, 1)
			r = tf.reshape(r, [-1, 1])
			D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)

			#set diagonal, i.e. the distance of a point to itself, to a large number
			D_normalised = tf.matrix_set_diag(D, tf.fill(nrows, value=1000.0))

			#take the index number of the point closest to the point in question 
			closest = tf.argmin(D_normalised, 0)

			#create a vector of the label number of the closest point to a given point
			closest_label = tf.gather(params=all_labels, indices=closest)

			#score 1 if the target label and the closest label are identical, 0 otherwise, and take the average
			correct = tf.equal(all_labels, closest_label)
			accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
