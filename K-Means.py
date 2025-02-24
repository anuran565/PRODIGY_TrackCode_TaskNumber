import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans #Import KMeans
from sklearn.metrics import silhouette_score

try:
    df_scaled=pd.read_csv("Scaled_Mall_Customers.csv")
    print("The scaled data set.")
    print(df_scaled.info())
except FileNotFoundError:
    print("File Not Found Error")
    exit()

#K-means clustering
#1.Determine Optimal K (Number of clusters) -- Elbow Method
inertia = []
for k in range(2, 11):   #Experiment with a range of K-values
    kmeans=KMeans(n_clusters=k , random_state=42)
    kmeans.fit(df_scaled)  #Fit K-means to the scaled data
    inertia.append(kmeans.inertia_)

plt.plot(range(2,11) , inertia, marker='o')
plt.xlabel('Number of Clusters (k) ')
plt.ylabel('WCSS (Inertia)')  # Correct label
plt.title('Elbow Method for Optimal k')
plt.show()

#Apply K-Means with the chosen K
optimal_k=4 #By elbow method
kmeans=KMeans(n_clusters=optimal_k, random_state=42)
df_scaled['Cluster']=kmeans.fit_predict(df_scaled)  #Fit and get cluster labels in one step

#Analyze the clusters
cls=df_scaled['Cluster'].value_counts()   #Count the number of customers in each cluster.
print("Number of customers in each segment : ",cls)
#Cluster Centers (Useful for understanding the characteristics of clusters.)
centroids=kmeans.cluster_centers_
print("Cluster Centers : ",centroids)

#Visualize the Cluster (e.g:Scatter plot of two important features)
plt.figure(figsize=(8,6))
sns.scatterplot(x=df_scaled['Annual Income (k$)'], 
                y=df_scaled['Spending Score (1-100)'], 
                hue=df_scaled['Cluster'], palette='viridis')
plt.title('Customer Segments')
plt.show()




    
    
