

saver.restore(sess, './model')


for root, dirs, files in os.walk("/home/leon/Documents/projects/data/sports/data_three_back/"):
	files_ = np.random.choice(files, 20, False)
	
	for file in files:
		data = pd.read_csv(os.path.join(root, file), header=None)