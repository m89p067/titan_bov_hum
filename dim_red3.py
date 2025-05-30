import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS,KMeans

cell_lines=['Cell Line1', 'Cell Line2', 'Cell Line3']
folder ="D:/Data/datasets_analysis/"
filenameHs='Human_ML3'
H_df=pd.read_csv(folder+filenameHs+'.csv',index_col=False,header=0)
X = H_df.loc[:,cell_lines].to_numpy()
y = H_df['Cond'].tolist()


colori=['royalblue','mediumseagreen','orangered']
markers=['d','s','o','X','v','*']

do_save=True
if do_save:
    un_y=(list(set(y)))
    metric_n_neighbors_min_dist = [
        ("hamming", 5, 0.8),
        ("hamming", 15, 0.25),
        ("hamming", 50, 0.99),
        ("jaccard", 5, 0.8),("jaccard", 15, 0.8),("jaccard", 25, 0.8),
        ("jaccard", 50, 0.1),
        ("dice", 5, 0.25),
        ("dice", 15, 0.5),("dice", 25, 0.5),
        ("dice", 50, 0.99),
        ("russellrao", 5, 0.8),
        ("russellrao", 15, 0.99),
        ("russellrao", 50, 0.0),
        ("kulsinski", 5, 0.8),("kulsinski", 20, 0.25),
        ("kulsinski", 50, 0.25),
        ("rogerstanimoto", 5, 0.25),
        ("rogerstanimoto", 15, 0.99),
        ("sokalmichener", 5, 0.99),
        ("sokalmichener", 15, 0.8),
        ("sokalsneath", 5, 0.25),("sokalsneath", 10, 0.25),
        ("yule", 5, 0.0),("yule", 20, 0.0),
        ("yule", 50, 0.0),
        ("euclidean", 5, 0.001),
        ("euclidean", 10, 0.01),
        ("euclidean", 30, 0.1),
        ("cosine", 15, 0.01),
        ("cosine", 30, 0.1),
        ("cosine", 50, 0.5),
        ("manhattan", 5, 0.001),
        ("manhattan", 10, 0.05),
        ("correlation", 15, 0.1),
        ("correlation", 30, 0.05),
        ("chebyshev", 10, 0.001),
        ("hamming", 50, 0.1),
        ("minkowski", 15, 0.01),("minkowski", 5, 0.01),("minkowski", 10, 0.01),
        ("mahalanobis", 30, 0.5),
        ("mahalanobis", 30, 0.01),
        ("mahalanobis", 30, 0.1),
        ("mahalanobis", 15, 0.5),
        ("mahalanobis", 15, 0.01),
        ("mahalanobis", 15, 0.1),
        ("mahalanobis", 5, 0.5),
        ("mahalanobis", 5, 0.01),
        ("mahalanobis", 5, 0.1)
    ]
    alt_mahalanobis =False
    results = []  # List to store results
    folder_umap="D:/Data/datasets_analysis/UMAP_H/"
    # Loop through each combination of metric, n_neighbors, and min_dist
    for metric, n_neighbors, min_dist in metric_n_neighbors_min_dist:
        print(f"Processing: Metric={metric}, n_neighbors={n_neighbors}, min_dist={min_dist}")
        try:

            if  alt_mahalanobis ==False:
                mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric=metric, random_state=42)
            else:
                # Assuming 'data' is your input dataset (rows are samples, columns are features)
                cov_matrix = np.cov(X, rowvar=False)  # Compute the covariance matrix
                inv_cov_matrix = inv(cov_matrix)  # Invert the covariance matrix
                #standard Mahalanobis distance uses the full sample covariance matrix 
                mapper = umap.UMAP( metric=mahalanobis_metric, metric_kwds={'VI': inv_cov_matrix},  n_neighbors=n_neighbors,   min_dist=min_dist,  n_components=2,  random_state=42 )
            out_umap = mapper.fit_transform(X)

            # Check if out_umap is empty
            if out_umap.size == 0:
                print("Empty array for current parameter combination. Skipping plot...")
                continue

            
            # Evaluating the 2D components through Silhouette Score
            centro = []
            for tipo in un_y:
                indici =[i for i in range(len(y)) if y[i] == tipo]
                centro.append(np.mean(out_umap[indici, :], axis=0))
            kmeans = KMeans(init=np.asarray(centro), n_clusters=len(un_y), random_state=0, max_iter=500, verbose=1)
            estimator = kmeans.fit(out_umap)
            silhouette = silhouette_score(out_umap, estimator.labels_, random_state=0)
            print("Silhouette Score:", silhouette)
            # Plot
            fig, ax = plt.subplots(figsize=(6, 6))
            for target_class, color, marker in zip(un_y, colori, markers):
                idx=[i for i in range(len(y)) if y[i] == target_class]
                ax.scatter(x = out_umap[idx,0], y = out_umap[idx,1],label=target_class,color=color,marker=marker)
            #plt.title(f'UMAP (Metric={metric}, n_neighbors={n_neighbors}, min_dist={min_dist})')
            plt.title("Human UMAP")
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            stringa="HUMAN_"+metric+"_"+str(n_neighbors)+"_"+str(min_dist)+"_"+str(round(silhouette,2))
            ax.grid()
            plt.savefig(folder_umap+stringa+'.png',dpi=300,bbox_inches='tight')
            plt.close()
            results.append([metric, n_neighbors, min_dist,round(silhouette,2)])
            # Store results
            np.save(folder_umap+stringa, out_umap)
            
        except Exception as e:
            print(f"Error occurred: {e}")
            continue
    df = pd.DataFrame(results, columns=["Metric", "N-Neigh", "Min-Dist","Sil."])
    df.to_csv(folder_umap+"Results.csv",index=False,header=True)
else:
    metric=np.nan
    n_neighbors=np.nan
    min_dist=np.nan
    silhouette=np.nan
    stringa="HUMAN_"+metric+"_"+str(n_neighbors)+"_"+str(min_dist)+"_"+str(round(silhouette,2))
    
