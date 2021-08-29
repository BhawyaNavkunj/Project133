import pandas as pd
from sklearn.cluster import KMeans
import csv
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("star_with_gravity.csv")

X = df.iloc[:,3:5]
print(X)

wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters = i,init = "k-means++", random_state=42)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
sns.lineplot(range(1,11),wcss,marker="o",color="red")
plt.title("Elbow method")
plt.xlabel("number of clusters")
plt.ylabel("wcss")
plt.show()

print("Number of clusters : 3")