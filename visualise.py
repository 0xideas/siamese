import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
	embeds = pd.read_csv("/home/leon/Documents/projects/siamese/embed.txt")

	colours = ['red', 'green', 'blue', 'yellow', 'purple']
	fig, ax = plt.subplots()
	for activity, colour in zip(sorted(list(set(embeds.iloc[:,2]))), colours):
		em = embeds[embeds.iloc[:,2] == activity]
		ax.scatter(em.iloc[:,0], em.iloc[:,1], c=colour,label=activity,alpha=0.3, edgecolors='none')

	ax.legend()
	ax.grid(True)

	plt.show()
