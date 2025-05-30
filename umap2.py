import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans,HDBSCAN,SpectralClustering,AgglomerativeClustering
from os import listdir
import pandas as pd
clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
folder_umap='D:/Data/umap_fig/'
results = []
results += [each for each in listdir(folder_umap) if each.endswith('.npy')]
filename = r"D:/Data/bovine_ML.xlsx"
df = pd.read_excel(filename, header=0, index_col=None).iloc[:, :4]
def plotting_kmeans(reduced_data,stringa):
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "K-means clustering on the 2D reduced dataset\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    nome=stringa.split('.')
    plt.savefig('D:/Data/umap_fig/'+nome[0]+'png',dpi=300, bbox_inches='tight')
    plr.close()
# Prepare data
X = df.iloc[:, :3].to_numpy()  # Features
y = df.iloc[:, 3].to_numpy()   # Labels
cartella='D:/Data/umap_fig/'

cl_type=['Kmeans']
cols=['Metric','NN','MinDist','Inertia','homogeneity_score','completeness_score','v_measure_score','adjusted_rand_score','adjusted_mutual_info_score','silhouette_score']
for the_cluster in cl_type:
    tot=[]
    for outfile in results:
        centro=[]
        print(outfile)
        umap_red=np.load(cartella+outfile)    
        for tipo in np.unique(y):
            print(tipo)
            indici=np.where(y==tipo)[0]
            centro.append(np.mean(umap_red[indici,:], axis=0))
        stringhe=outfile.split('_')
        stringa=stringhe[-1].split('.')[0].replace('-', '.')
        if the_cluster==cl_type[0]:
            kmeans = KMeans(init=np.asarray(centro), n_clusters=6, random_state=0,max_iter=500,verbose=1)
            estimator=kmeans.fit(umap_red)        
            results = [stringhe[3],stringhe[5],stringa,  estimator.inertia_]
            results += [m(y, estimator.labels_) for m in clustering_metrics]
            results += [ metrics.silhouette_score(umap_red, estimator.labels_ ,random_state=0   )]
        elif the_cluster==cl_type[1]:
            sc=SpectralClustering(n_clusters=6,random_state=0)
        elif the_cluster==cl_type[2]:
        tot.append(results) # Show the results
    df = pd.DataFrame(tot, columns=cols)    
    df.to_csv('file_'+the_cluster+'.csv')
