import matplotlib.pyplot as plt
import pandas as pd

embeds = pd.read_csv("/home/leon/Documents/projects/siamese/embed.txt")
print(embeds)
print(embeds.iloc[:,0])

plt.scatter(x=embeds.iloc[:,0], y=embeds.iloc[:,1], c=embeds.iloc[:,2])
plt.show()