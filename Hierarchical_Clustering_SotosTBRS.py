# %%
# importing required libraries
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as shc
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

# %matplotlib inline
# %%
# reading the data
data_orig = pd.read_csv("data.csv")
# %%
head = data_orig.head()  # checking the first 5 rows of the data.
# %% dropping some columns, e.g., diagnosis
data = data_orig.drop(["Diagnosis"], axis=1)
# %%
data = data.fillna(data.median())  # imputing NaNs with the median of column.
# %%
data = normalize(data, axis=0)
# %%
# SP2 Sensory + SBQ
data = pd.DataFrame(
    data,
    columns=[
        "SBQ_Frequency",
        "SBQ_Impact",
        "sp2_aud_raw",
        "sp2_vis_raw",
        "sp2_tou_raw",
        "sp2_mov_raw",
        "sp2_bod_raw",
        "sp2_ora_raw",
    ],
)
# %%
# draws dendrogram
plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data, method="ward"))
# %%
# running hierarchical clustering
cluster = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="ward")
clusters_predicted = cluster.fit_predict(data)
# %%
# putting the results in data frame format & combining 'Diagnosis' column
clustersDf = pd.DataFrame(clusters_predicted)
clustersDf.columns = ["clusters_predicted"]
# %%
combinedDf = pd.concat([data, clustersDf], axis=1).reset_index()
# %%
data_final = combinedDf.join(data_orig["Diagnosis"])
data_final = data_final.drop(["index"], axis=1)
# %%
# visualising the two clusters
plt.subplots(figsize=(15, 5))
sns.countplot(
    x=data_final["Diagnosis"],
    order=data_final["Diagnosis"].value_counts().index,
    hue=data_final["clusters_predicted"],
)
plt.show()
# %%
# exporting results
dataexport = data_final.to_excel("results.xlsx", index=False)

# %%
mut = (
    data_final.Diagnosis
)  # using the 'Diagnosis' column of the pandas df 'data_final'.
lut = dict(
    zip(mut.unique(), "rbg")
)  # linking each unique item in 'lut' to an rbg colour. 2 items (0,1) are now linked to red and blue.
row_colors = mut.map(
    lut
)  # linking the rgb to each corresponding item in the Diagnosis column.
sns.set(font_scale=12)
plot = sns.clustermap(
    data,
    metric="euclidean",
    standard_scale=None,
    figsize=(100, 100),
    method="ward",
    cmap="viridis",
    row_colors=row_colors,
    tree_kws=dict(linewidths=6),
)  # running the plot with Diagnosis column included. cmap options e.g., 'mako', 'viridis', 'Blues'.
plt.show()
# %%
plot.savefig("heatmap.tiff")
