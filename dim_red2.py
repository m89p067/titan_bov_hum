import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import balanced_accuracy_score
#from sklearn.cross_decomposition import PLSRegression
#from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS,KMeans
import umap
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import inv

# Define a function to compute Mahalanobis distance between two points
def mahalanobis_metric(x, y, inv_cov_matrix):
    return mahalanobis(x, y, inv_cov_matrix)

alt_mahalanobis=False
print('Data created by merge_H_B_ver1.py')
folder='D:/Parinaz/datasets_analysis/'
filenameH='Human_ML2'
filenameB='Bovine_ML2'
filenameBH='Bovine_and_Human2'

filenameHs='Human_ML3'
filenameBs='Bovine_ML3'
filenameBHs='Bovine_and_Human3'

filenameB_FC='BovineFC'
filenameBB='Bovine_ML4'
filenameBH1='Bovine_and_Human4'
cell_lines=['Cell Line1', 'Cell Line2', 'Cell Line3']

B_df=pd.read_csv(folder+filenameBs+'.csv',index_col=False,header=0)
H_df=pd.read_csv(folder+filenameHs+'.csv',index_col=False,header=0)
BH_df=pd.read_csv(folder+filenameBHs+'.csv',index_col=False,header=0)
X = B_df.loc[:,cell_lines].to_numpy()
y = B_df['Cond'].tolist()
X2 = H_df.loc[:,cell_lines].to_numpy()
y2 = H_df['Cond'].tolist()

ML_test=False
if ML_test:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #sc = StandardScaler() #PCA performs best with a normalized feature set
    #X_train = sc.fit_transform(X_train)
    #X_test = sc.transform(X_test)

    print('Use 1 principal component to train our algorithm. BOVINE')
    pca1 = PCA(n_components=1)
    X_train1 = pca1.fit_transform(X_train)
    X_test1 = pca1.transform(X_test)

    classifier1 = RandomForestClassifier(max_depth=2, random_state=0)
    classifier1.fit(X_train1, y_train)

    # Predicting the Test set results
    y_pred1 = classifier1.predict(X_test1)

    cm = confusion_matrix(y_test, y_pred1)
    print(cm)
    print('Bal. Accuracy:' , balanced_accuracy_score(y_test, y_pred1))

    print('Use 2 principal component to train our algorithm. BOVINE')
    pca2 = PCA(n_components=2)
    X_train2 = pca2.fit_transform(X_train)
    X_test2 = pca2.transform(X_test)

    classifier2 = RandomForestClassifier(max_depth=2, random_state=0)
    classifier2.fit(X_train2, y_train)

    # Predicting the Test set results
    y_pred2 = classifier2.predict(X_test2)

    cm = confusion_matrix(y_test, y_pred2)
    print(cm)
    print('Bal. Accuracy:' , balanced_accuracy_score(y_test, y_pred2))
PCA_BH=False
if PCA_BH:

    ex_var=[]
    pca = PCA(n_components=2)
    X_red=pca.fit_transform(X)
    ex_var.append(pca.explained_variance_)
    X_red2=pca.fit_transform(X2)
    ex_var.append(pca.explained_variance_)
    # PLSDA is not recommended for multiclass problem. See paper Partial least squares discriminant analysis: taking the magic away for detail discussion
    #le = LabelEncoder()
    #y_encoded = le.fit_transform(y)
    #mapping = dict(zip(le.classes_, range(len(le.classes_))))
    #plsr = PLSRegression(n_components=2, scale=False).fit_transform(X, y_encoded)
    #target_classes2 = list(mapping.values())

    transformer = KernelPCA(n_components=2, kernel='rbf')
    plsr = transformer.fit_transform(X)

    plsr2 = transformer.fit_transform(X2)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))

    target_classes = B_df['Cond'].unique().tolist()
    target_classes2 = H_df['Cond'].unique().tolist()
    colors = ("blue", "red", "green")
    markers = ("^", "s", "o")

    #for idx_n,(target_class, color, marker,idx_tc) in enumerate(zip(target_classes, colors, markers,target_classes2)):
    for target_class, color, marker in zip(target_classes, colors, markers):
        idx=[i for i in range(len(y)) if y[i] == target_class]
        ax1.scatter(
            x=X_red[idx, 0],
            y=X_red[idx, 1],
            color=color,
            #label=target_class,
            alpha=0.5,
            marker=marker,
        )
        #idx=[i for i in y_encoded if y_encoded[i] == idx_tc]
        #for mykey, myval in mapping.items():
        #    if myval == idx_tc:
        #        variabile=mykey
        ax2.scatter(
            x=plsr[idx, 0],
            y=plsr[idx, 1],
            color=color,
            #label=variabile,
            label=target_class,
            alpha=0.5,
            marker=marker,
        )
        
        ax3.scatter(
            x=X_red2[idx, 0],
            y=X_red2[idx, 1],
            color=color,
            #label=target_class,
            alpha=0.5,
            marker=marker,
        )
        #idx=[i for i in y_encoded if y_encoded[i] == idx_tc]
        #for mykey, myval in mapping.items():
        #    if myval == idx_tc:
        #        variabile=mykey
        ax4.scatter(
            x=plsr2[idx, 0],
            y=plsr2[idx, 1],
            color=color,
            #label=variabile,
            #label=target_class,
            alpha=0.5,
            marker=marker,
        )
    ax1.set_title("PCA [B]")
    #ax2.set_title("PLS-DA")
    ax2.set_title("k-PCA [B]")

    ax3.set_title("PCA [H]")
    ax4.set_title("k-PCA [H]")

    ax3.set_xlabel("1st comp.")
    ax4.set_xlabel("1st comp.")
    ax1.set_ylabel("2nd comp.")
    ax3.set_ylabel("2nd comp.")

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()

    ax1.set_xlim([-12, 22])
    ax1.set_ylim([-3, 4.5])
    ax3.set_xlim([-12, 22])
    ax3.set_ylim([-3, 4.5])

    ax2.set_xlim([-0.8, 0.8])
    ax2.set_ylim([-0.6, 0.85])
    ax4.set_xlim([-0.8, 0.8])
    ax4.set_ylim([-0.6, 0.85])
    fig.legend(loc=7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    plt.savefig(folder+'PCA_BH.png',dpi=300,bbox_inches='tight')
    plt.close()
    print(ex_var)
    ##########################################################################
    fig, axs = plt.subplots(3,sharex=True)
    X=BH_df.loc[:,cell_lines].to_numpy()
    pca = PCA(n_components=2)
    X_red=pca.fit_transform(X)
    un_col=[i+' '+y for i,y in zip(BH_df["Cond"],BH_df["Type"])]
    colori=['r','g','b','c','m','orange']
    markers=['d','s','o','X','v','*']
    for target_class, color, marker in zip((list(set(un_col))), colori, markers):
        idx=[i for i in range(len(un_col)) if un_col[i] == target_class]
        indici=target_class.split(" ")
        if 'TI_CT_PPHE' in indici[0] :
            axs[0].scatter(x = X_red[idx,0], y = X_red[idx,1],label=indici[1],color=color,marker=marker)
            if indici[1]=="H":
                axs[0].set_title(indici[0])            
        elif 'TI_CT' in indici[0]:
            axs[1].scatter(x = X_red[idx,0], y = X_red[idx,1],label=indici[1],color=color,marker=marker)
            if indici[1]=="H":
                axs[1].set_title(indici[0])
        elif 'TI_PPHE' in indici[0]:
            axs[2].scatter(x = X_red[idx,0], y = X_red[idx,1],label=indici[1],color=color,marker=marker)
            if indici[1]=="H":
                axs[2].set_title(indici[0])
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()    
    axs[0].legend(bbox_to_anchor=(1.15, 1.075))    
    axs[1].legend(bbox_to_anchor=(1.15, 1.075))    
    axs[2].legend(bbox_to_anchor=(1.15, 1.075))
    plt.xlabel("Second comp.")
    axs[0].set_ylabel("First comp.")
    axs[1].set_ylabel("First comp.")
    axs[2].set_ylabel("First comp.")
    #plt.title("Bovine & Human PCA")
    plt.tight_layout()
    plt.savefig(folder+'PCA_Both.png',dpi=300,bbox_inches='tight')
    plt.close() 
    un_col2=['TI_PPHE B', 'TI_CT H', 'TI_CT_PPHE H', 'TI_CT B', 'TI_CT_PPHE B', 'TI_PPHE H']
    colori=['dodgerblue','orangered','red','lightseagreen','royalblue','crimson']
    fig, ax = plt.subplots(figsize=(6, 6))
    for target_class, color, marker in zip(un_col2, colori, markers):
        idx=[i for i in range(len(un_col)) if un_col[i] == target_class]
        ax.scatter(x = X_red[idx,0], y = X_red[idx,1],label=target_class,color=color,marker=marker)
    plt.title('Bovine and Human PCA')
    plt.xlabel('First comp.')
    plt.ylabel('Second comp.')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
    plt.savefig(folder+'PCA_BOTH2.png',dpi=300,bbox_inches='tight')
    plt.close()

BB_df=pd.read_csv(folder+filenameBB+'.csv',index_col=False,header=0)
un_col=BB_df.loc[:,"Cond"].tolist()
colori=['r','g','b','c','m','orange']
markers=['d','s','o','X','v','*']
is_pca=False
if is_pca:
    fig, ax = plt.subplots(figsize=(6, 6))
    X = BB_df.loc[:,cell_lines].to_numpy()
    pca = PCA(n_components=2)
    X_red=pca.fit_transform(X)
    for target_class, color, marker in zip((list(set(un_col))), colori, markers):
        idx=[i for i in range(len(un_col)) if un_col[i] == target_class]
        ax.scatter(x = X_red[idx,0], y = X_red[idx,1],label=target_class,color=color,marker=marker)
    ax.grid()
    ax.legend(loc="best")
    plt.xlabel("Second comp.")
    plt.ylabel("First comp.")
    plt.title("Bovine PCA")
    plt.tight_layout()
    plt.savefig(folder+'PCA_Bovine.png',dpi=300,bbox_inches='tight')
    plt.close() 

TNSE_BOTH=False
if TNSE_BOTH:
    all_res=[]
    tsne_res={}
    # Define parameters to explore
    perplexities = [5, 10, 15, 20,25,  30,35,  40,45, 50,60,70,75]
    early_exaggerations = [8, 12, 16]
    fig, ax = plt.subplots(figsize=(6, 6))
    X = BB_df.loc[:,cell_lines].to_numpy()
    un_col=BB_df.loc[:,"Cond"].tolist()
    min_kl_divergence = float('inf')
    best_tsne = None
    for perplexity in perplexities:
        for early_exaggeration in early_exaggerations:
            tsne = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration, n_iter=3000,random_state=42,init='pca')
            X_reduced = tsne.fit_transform(X)
            kl_divergence = tsne.kl_divergence_
            if kl_divergence < min_kl_divergence:
                min_kl_divergence = kl_divergence
                best_tsne = X_reduced
                var1=perplexity
                var2=early_exaggeration
                var3=kl_divergence
    
    for target_class, color, marker in zip((list(set(un_col))), colori, markers):
        idx=[i for i in range(len(un_col)) if un_col[i] == target_class]
        ax.scatter(x = best_tsne[idx,0], y = best_tsne[idx,1],label=target_class,color=color,marker=marker)
    
    print(f"Perplexity: {var1}, Early Exaggeration: {var2}")
    print(f"KL Divergence: {var3}")
    ax.grid()
    ax.legend(loc="best")
    plt.xlabel("Second dim.")
    plt.ylabel("First dim.")
    plt.title("Bovine t-SNE")
    plt.tight_layout()
    plt.savefig(folder+'TSNE_Bovine.png',dpi=300,bbox_inches='tight')
    plt.close()
    np.save(folder+"tsne_bovine",best_tsne)
TNSE_BOTH=True
if TNSE_BOTH:
    X=BH_df.loc[:,cell_lines].to_numpy()
    all_res=[]
    tsne_res={}
    target_classes = [i2+' '+y2 for i2,y2 in zip(BH_df["Cond"],BH_df["Type"])]
    colori=['orangered','forestgreen','navy','lightseagreen','purple','goldenrod']
    markers=['d','s','o','P','H','D']   
    perplexities = [5, 10, 15, 20,25,  30,35,  40,45, 50,60,70,75]
    early_exaggerations = [8, 12, 16]
    fig, axs = plt.subplots(3,sharex=True)
    min_kl_divergence = float('inf')
    best_tsne = None
    for perplexity in perplexities:
        for early_exaggeration in early_exaggerations:
            tsne = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration, n_iter=3000,random_state=42,init='pca')
            X_reduced = tsne.fit_transform(X)
            kl_divergence = tsne.kl_divergence_
            if kl_divergence < min_kl_divergence:
                min_kl_divergence = kl_divergence
                best_tsne = X_reduced
                var1=perplexity
                var2=early_exaggeration
                var3=kl_divergence
    
    for target_class, color, marker in zip((list(set(target_classes))), colori, markers):
        idx=[i for i in range(len(target_classes)) if target_classes[i] == target_class]
        indici=target_class.split(" ")
        if 'TI_CT_PPHE' in indici[0] :
            axs[0].scatter(x = best_tsne[idx,0], y = best_tsne[idx,1],label=indici[1],color=color,marker=marker)
            if indici[1]=="H":
                axs[0].set_title(indici[0])            
        elif 'TI_CT' in indici[0]:
            axs[1].scatter(x = best_tsne[idx,0], y = best_tsne[idx,1],label=indici[1],color=color,marker=marker)
            if indici[1]=="H":
                axs[1].set_title(indici[0])
        elif 'TI_PPHE' in indici[0]:
            axs[2].scatter(x = best_tsne[idx,0], y = best_tsne[idx,1],label=indici[1],color=color,marker=marker)
            if indici[1]=="H":
                axs[2].set_title(indici[0])
    
    print(f"Perplexity: {var1}, Early Exaggeration: {var2}")
    print(f"KL Divergence: {var3}")
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()    
    axs[0].legend(bbox_to_anchor=(1.15, 1.075))    
    axs[1].legend(bbox_to_anchor=(1.15, 1.075))    
    axs[2].legend(bbox_to_anchor=(1.15, 1.075))
    plt.xlabel("First dim.")
    axs[0].set_ylabel("Second dim.")
    axs[1].set_ylabel("Second dim.")
    axs[2].set_ylabel("Second dim.")
    #plt.title("Bovine and Human t-SNE")
    plt.tight_layout()
    plt.savefig(folder+'TSNE_HB.png',dpi=300,bbox_inches='tight')
    plt.close()            
    np.save(folder+"tsne_HB",best_tsne)
do_UMAP=False
if do_UMAP:
    colori=['darkred','greenyellow','mediumturquoise','crimson','deepskyblue','rebeccapurple']
    markers=['d','s','o','X','v','*']
    X = BB_df.loc[:,cell_lines].to_numpy()
    y=BB_df.loc[:,"Cond"].tolist()
    un_y=(list(set(y)))
    metric_n_neighbors_min_dist = [
        ("hamming", 5, 0.8),
        ("hamming", 15, 0.25),
        ("hamming", 50, 0.99),
        ("jaccard", 5, 0.8),
        ("jaccard", 50, 0.1),
        ("dice", 5, 0.25),
        ("dice", 15, 0.5),
        ("dice", 50, 0.99),
        ("russellrao", 5, 0.8),
        ("russellrao", 15, 0.99),
        ("russellrao", 50, 0.0),
        ("kulsinski", 5, 0.8),
        ("kulsinski", 50, 0.25),
        ("rogerstanimoto", 5, 0.25),
        ("rogerstanimoto", 15, 0.99),
        ("sokalmichener", 5, 0.99),
        ("sokalmichener", 15, 0.8),
        ("sokalsneath", 5, 0.25),
        ("yule", 5, 0.0),
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
        ("minkowski", 15, 0.01),
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

    results = []  # List to store results
    folder_umap="D:/Parinaz/datasets_analysis/UMAP/"
    # Loop through each combination of metric, n_neighbors, and min_dist
    for metric, n_neighbors, min_dist in metric_n_neighbors_min_dist:
        print(f"Processing: Metric={metric}, n_neighbors={n_neighbors}, min_dist={min_dist}")
        try:
            
            if  alt_mahalanobis ==False:
                mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric=metric, random_state=42)
            else:
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
            plt.title(f'UMAP (Metric={metric}, n_neighbors={n_neighbors}, min_dist={min_dist})')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            stringa="BOV_"+metric+"_"+str(n_neighbors)+"_"+str(min_dist)+"_"+str(round(silhouette,2))
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
do_UMAP=False
if do_UMAP:
    colori=['orangered','forestgreen','navy','lightseagreen','purple','goldenrod']
    markers=['d','s','o','P','H','D'] 
    X=BH_df.loc[:,cell_lines].to_numpy()
    y = [i2+' '+y2 for i2,y2 in zip(BH_df["Cond"],BH_df["Type"])]    
    un_y=(list(set(y)))
    metric_n_neighbors_min_dist = [
        ("hamming", 5, 0.8),
        ("hamming", 15, 0.25),
        ("hamming", 50, 0.99),
        ("jaccard", 5, 0.8),
        ("jaccard", 50, 0.1),
        ("dice", 5, 0.25),
        ("dice", 15, 0.5),
        ("dice", 50, 0.99),
        ("russellrao", 5, 0.8),
        ("russellrao", 15, 0.99),
        ("russellrao", 50, 0.0),
        ("kulsinski", 5, 0.8),
        ("kulsinski", 50, 0.25),
        ("rogerstanimoto", 5, 0.25),
        ("rogerstanimoto", 15, 0.99),
        ("sokalmichener", 5, 0.99),
        ("sokalmichener", 15, 0.8),
        ("sokalsneath", 5, 0.25),
        ("yule", 5, 0.0),
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
        ("minkowski", 15, 0.01),
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

    results = []  # List to store results
    folder_umap="D:/Parinaz/datasets_analysis/UMAP_HB/"
    # Loop through each combination of metric, n_neighbors, and min_dist
    for metric, n_neighbors, min_dist in metric_n_neighbors_min_dist:
        print(f"Processing: Metric={metric}, n_neighbors={n_neighbors}, min_dist={min_dist}")
        try:

            if alt_mahalanobis ==False:
                mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric=metric, random_state=42)
            else:
                cov_matrix = np.cov(X, rowvar=False)  # Compute the covariance matrix
                inv_cov_matrix = inv(cov_matrix)  # Invert the covariance matrix
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
            results.append([metric, n_neighbors, min_dist,round(silhouette,2)])
            # Store results
            stringa="HB_"+metric+"_"+str(n_neighbors)+"_"+str(min_dist)+"_"+str(round(silhouette,2))
            np.save(folder_umap+stringa, out_umap)

            fig, axs = plt.subplots(3,sharex=True)
            for target_class, color, marker in zip(un_y, colori, markers):
                idx=[i for i in range(len(y)) if y[i] == target_class]
                indici=target_class.split(" ")
                if 'TI_CT_PPHE' in indici[0] :
                    axs[0].scatter(x = out_umap[idx,0], y = out_umap[idx,1],label=indici[1],color=color,marker=marker)
                    if indici[1]=="H":
                        axs[0].set_title(indici[0])            
                elif 'TI_CT' in indici[0]:
                    axs[1].scatter(x = out_umap[idx,0], y = out_umap[idx,1],label=indici[1],color=color,marker=marker)
                    if indici[1]=="H":
                        axs[1].set_title(indici[0])
                elif 'TI_PPHE' in indici[0]:
                    axs[2].scatter(x = out_umap[idx,0], y = out_umap[idx,1],label=indici[1],color=color,marker=marker)
                    if indici[1]=="H":
                        axs[2].set_title(indici[0])
            
            axs[0].grid()
            axs[1].grid()
            axs[2].grid()    
            axs[0].legend(bbox_to_anchor=(1.15, 1.075))    
            axs[1].legend(bbox_to_anchor=(1.15, 1.075))    
            axs[2].legend(bbox_to_anchor=(1.15, 1.075))
            plt.xlabel("First dim.")
            axs[0].set_ylabel("Second dim.")
            axs[1].set_ylabel("Second dim.")
            axs[2].set_ylabel("Second dim.")
            #plt.title("Bovine and Human t-SNE")
            plt.tight_layout()
            plt.savefig(folder_umap+stringa+'.png',dpi=300,bbox_inches='tight')
            plt.close()
            un_y=['TI_PPHE B', 'TI_CT H', 'TI_CT_PPHE H', 'TI_CT B', 'TI_CT_PPHE B', 'TI_PPHE H']
            colori=['dodgerblue','orangered','red','lightseagreen','royalblue','crimson']
            fig, ax = plt.subplots(figsize=(6, 6))
            for target_class, color, marker in zip(un_y, colori, markers):
                idx=[i for i in range(len(y)) if y[i] == target_class]
                ax.scatter(x = out_umap[idx,0], y = out_umap[idx,1],label=target_class,color=color,marker=marker)
            plt.title(f'UMAP (Metric={metric}, n_neighbors={n_neighbors}, min_dist={min_dist})')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            stringa="HB2_"+metric+"_"+str(n_neighbors)+"_"+str(min_dist)+"_"+str(round(silhouette,2))
            plt.savefig(folder_umap+stringa+'.png',dpi=300,bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error occurred: {e}")
            continue
    df = pd.DataFrame(results, columns=["Metric", "N-Neigh", "Min-Dist","Sil."])
    df.to_csv(folder_umap+"Results_HB.csv",index=False,header=True)
redo_old=False
if redo_old:
    X=BH_df.loc[:,cell_lines].to_numpy()
    target_classes = [i2+' '+y2 for i2,y2 in zip(BH_df["Cond"],BH_df["Type"])]
    un_y=['TI_PPHE B', 'TI_CT H', 'TI_CT_PPHE H', 'TI_CT B', 'TI_CT_PPHE B', 'TI_PPHE H']
    colori=['dodgerblue','orangered','red','lightseagreen','royalblue','crimson']
    best_tsne=np.load(folder+"tsne_HB.npy")
    fig, ax = plt.subplots(figsize=(6, 6))
    for target_class, color, marker in zip(un_y, colori, markers):
        idx=[i for i in range(len(target_classes)) if target_classes[i] == target_class]
        ax.scatter(x = best_tsne[idx,0], y = best_tsne[idx,1],label=target_class,color=color,marker=marker)
    plt.title('Bovine and Human t-SNE')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
    plt.savefig(folder+'TSNE_BOTH2.png',dpi=300,bbox_inches='tight')
    plt.close()
    stringa="HB_mahalanobis_30_0.5_0.52"
    folder_umap="D:/Parinaz/datasets_analysis/UMAP_HB/"
    out_umap=np.load(folder_umap+stringa+".npy")
    fig, ax = plt.subplots(figsize=(6, 6))
    for target_class, color, marker in zip(un_y, colori, markers):
        idx=[i for i in range(len(target_classes)) if target_classes[i] == target_class]
        ax.scatter(x = out_umap[idx,0], y = out_umap[idx,1],label=target_class,color=color,marker=marker)
    plt.title('Bovine and Human UMAP')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.savefig(folder+'UMAP_BOTH2.png',dpi=300,bbox_inches='tight')
    plt.close()
redo_old=False
if redo_old:
    colori=['r','g','b','c','m','orange']
    markers=['d','s','o','X','v','*']
    X = BB_df.loc[:,cell_lines].to_numpy()
    y=BB_df.loc[:,"Cond"].tolist()
    un_y=(list(set(y)))
    folder_umap="D:/Parinaz/datasets_analysis/UMAP/"
    stringa="BOV_sokalsneath_5_0.25_0.35"
    tmp=stringa.split("_")
    metric=tmp[1]
    n_neighbors=tmp[2]
    min_dist=tmp[3]
    silhouette=tmp[-1]
    out_umap=np.load(folder_umap+stringa+".npy")
    fig, ax = plt.subplots(figsize=(6, 6))
    for target_class, color, marker in zip(un_y, colori, markers):
        idx=[i for i in range(len(y)) if y[i] == target_class]
        ax.scatter(x = out_umap[idx,0], y = out_umap[idx,1],label=target_class,color=color,marker=marker)
    #plt.title(f'UMAP (Metric={metric}, n_neighbors={n_neighbors}, min_dist={min_dist})')
    plt.title("Bovine UMAP")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    ax.legend(loc="best")

    stringa="BOV_"+metric+"_"+n_neighbors+"_"+min_dist+"_"+silhouette
    ax.grid()
    plt.savefig(folder+stringa+'.png',dpi=300,bbox_inches='tight')
    plt.close()    

make_lists=False
if make_lists:
    folder='D:/Parinaz/datasets_analysis/'
    filenameH='Human_ML2'
    filenameB='Bovine_ML2'
    filenameBH='Bovine_and_Human2'

    filenameHs='Human_ML3'
    filenameBs='Bovine_ML3'
    filenameBHs='Bovine_and_Human3'

    filenameB_FC='BovineFC'
    filenameBB='Bovine_ML4'
    filenameBH1='Bovine_and_Human4'
    
    BH_df=pd.read_csv(folder+filenameBHs+'.csv',index_col=False,header=0)

    subset=BH_df.loc[:,['Protein','Gene','Descrip','Type','Cond']]
    subset.to_csv(folder+'out_genes.csv', sep=';',index=False)
