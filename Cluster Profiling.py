import pandas as pd
from sklearn.cluster import KMeans
try:
    df_scaled=pd.read_csv("Scaled_Mall_Customers.csv")
    print("The Scaled Dataset : ")
except FileNotFoundError:
    print("File Not Found Error.")


optimal_k=4 
kmeans=KMeans(n_clusters=optimal_k, random_state=42)
df_scaled['Cluster']=kmeans.fit_predict(df_scaled)

#Group Data By Cluster
grouped_data=df_scaled.groupby('Cluster')

#Calculate Descriptive Statistics for Each Cluster
#mean(The average value of each feature within the cluster.)
cluster_means=grouped_data.mean()
print("Mean of each cluster : ",cluster_means)

#median
cluster_medians=grouped_data.median()
print("Median of each cluster : ",cluster_medians)

#standard deviation
cluster_std=grouped_data.std()
print("Standard Deviation of each cluster : ",cluster_std)

#combine and present the results
cluster_profiles=pd.concat([cluster_means,cluster_medians,cluster_std],axis=1, keys=['Mean','Median','Standard Deviation'])
print("Cluster Profiles : ",cluster_profiles)

cluster_profiles.to_csv("cluster_profiles.csv", index=True, float_format="%.2f")
print("Cluster profiles saved to cluster_profiles.csv")


