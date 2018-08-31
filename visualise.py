import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
	embeds = pd.read_csv("/home/leon/Documents/projects/siamese/embed.txt")

	colours = [(1,0,0), (0,1,0),(0,0,1), (1,1,0), (1,0,1), (0,1,1)]
	colours += [(0.5,0,0), (0,0.5,0), (0,0,0.5), (0.5,1,0), (0.5,0,1)]
	colours += [(1,0.5,0), (0,0.5,1), (1, 0, 0.5), (0,1,0.5), (0.5,0.5,0)]
	colours += [(0.5, 0, 0.5), (0,0.5,0.5), (0.6,0.3,0), (0.6,0,0.3)]
	colours += [(0.3,0.6,0), (0,0.6,0.3), (0.3,0,0.6), (0,0.3,0.6)]
	fig, ax = plt.subplots()
	for activity, colour in zip(sorted(list(set(embeds.iloc[:,2]))), colours):
		print(activity, colour)
		em = embeds[embeds.iloc[:,2] == activity]
		ax.scatter(em.iloc[:,0], em.iloc[:,1], c=colour,label=activity,alpha=0.3, edgecolors='none')

	ax.legend()
	ax.grid(True)

	plt.show()
