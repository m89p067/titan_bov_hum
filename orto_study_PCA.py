import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import mahalanobis
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import balanced_accuracy_score
#from sklearn.cross_decomposition import PLSRegression
#from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
#from sklearn.metrics import silhouette_score
#from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS,KMeans
import umap
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
print('Data created by merge_H_B_ver1.py')
folder='D:/Parinaz/datasets_analysis/'
filenameH='Human_ML2'
filenameB='Bovine_ML2'
filenameBH='Bovine_and_Human2'
save_folder='D:/Parinaz/datasets_analysis/ortho/'
filenameHs='Human_ML3'
filenameBs='Bovine_ML3'
filenameBHs='Bovine_and_Human3'

filenameB_FC='BovineFC'
filenameBB='Bovine_ML4'
filenameBH1='Bovine_and_Human4'
cell_lines=['Cell Line1', 'Cell Line2', 'Cell Line3']

##B_df=pd.read_csv(folder+filenameBs+'.csv',index_col=False,header=0)
##H_df=pd.read_csv(folder+filenameHs+'.csv',index_col=False,header=0)
BH_df=pd.read_csv(folder+filenameBHs+'.csv',index_col=False,header=0)
##X = B_df.loc[:,cell_lines].to_numpy()
##y = B_df['Cond'].tolist()
##X2 = H_df.loc[:,cell_lines].to_numpy()
##y2 = H_df['Cond'].tolist()


##########################################################################
fig, axs = plt.subplots(3,sharex=True)
X=BH_df.loc[:,cell_lines].to_numpy()
pca = PCA(n_components=2)
X_red=pca.fit_transform(X)
do_plot=False
if do_plot:
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
    axs[0].set(yticklabels=[])
    axs[1].set(yticklabels=[])
    axs[2].set(yticklabels=[])
    axs[0].set(xticklabels=[])
    axs[0].legend(bbox_to_anchor=(1.15, 1.075))    
    axs[1].legend(bbox_to_anchor=(1.15, 1.075))    
    axs[2].legend(bbox_to_anchor=(1.15, 1.075))
    plt.xlabel("First comp.")
    axs[0].set_ylabel("Second comp.")
    axs[1].set_ylabel("Second comp.")
    axs[2].set_ylabel("Second comp.")
    print(pca.explained_variance_)
    plt.tight_layout()
    plt.savefig(save_folder+'PCA_Both.png',dpi=300,bbox_inches='tight')
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
    plt.savefig(save_folder+'PCA_BOTH2.png',dpi=300,bbox_inches='tight')
    plt.close()


##########################################################################
print('Working on a B and H as one dataset, different approach')

clf = IsolationForest(random_state=42, verbose=0)
predAll =clf.fit(X).predict(X)

# classified -1 are anomalous
BH_df['anomaly']=predAll

outliers=BH_df.loc[BH_df['anomaly']==-1]
outlier_index=list(outliers.index)
X_reduce=BH_df.copy()

BH_df['PCA1']=X_red[:,0]
BH_df['PCA2']=X_red[:,1]

dfB = BH_df.loc[BH_df['Type'] == 'B'].reset_index(drop=True)
dfH = BH_df.loc[BH_df['Type'] == 'H'].reset_index(drop=True)

XB=dfB.loc[:,cell_lines].to_numpy()
XH=dfH.loc[:,cell_lines].to_numpy()

outliersB=dfB.loc[dfB['anomaly']==-1]
outlier_indexB=list(outliersB.index)
X_reduceB=XB.copy()

outliersH=dfH.loc[dfH['anomaly']==-1]
outlier_indexH=list(outliersH.index)
X_reduceH=XH.copy()

do_plot=True
if do_plot:
    ###################################################################
    fig = plt.figure(figsize=plt.figaspect(0.5))
    #fig, (ax1,ax2) = plt.subplots(1,2,figsize=(5,10),subplot_kw=dict(projection='3d'))

    # =============
    # First subplot
    # =============
    # set up the Axes for the first plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(X_reduceB[:, 0], X_reduceB[:, 1], zs=X_reduceB[:, 2], s=4, lw=1, label="inliers",c="green")# Plot x's for the ground truth outliers
    ax1.scatter(X_reduceB[outlier_indexB,0],X_reduceB[outlier_indexB,1], X_reduceB[outlier_indexB,2], lw=2, s=60, marker="x", c="red", label="outliers")
    ax1.legend()
    ax1.set_title('Bovine')
    ax1.set_xlim([np.min(X[:,0]),np.max(X[:,0])])
    ax1.set_ylim([np.min(X[:,1]),np.max(X[:,1])])
    ax1.set_zlim([np.min(X[:,2]),np.max(X[:,2])])
    ax1.set(yticklabels=[])
    ax1.set(xticklabels=[])
    ax1.set(zticklabels=[])
    #ax1.set_xlabel("Cell  Line 1",labelpad=None)
    #ax1.set_ylabel("Cell  Line 2",labelpad=None)
    #ax1.set_zlabel("Cell  Line 3",labelpad=None)# Plot the compressed data points
    # ==============
    # Second subplot
    # ==============
    # set up the Axes for the second plot

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(X_reduceH[:, 0], X_reduceH[:, 1], zs=X_reduceH[:, 2], s=4, lw=1, label="inliers",c="green")# Plot x's for the ground truth outliers
    ax2.scatter(X_reduceH[outlier_indexH,0],X_reduceH[outlier_indexH,1], X_reduceH[outlier_indexH,2], lw=2, s=60, marker="x", c="red", label="outliers")
    ax2.legend()
    ax2.set_title('Human')
    ax2.set_xlim([np.min(X[:,0]),np.max(X[:,0])])
    ax2.set_ylim([np.min(X[:,1]),np.max(X[:,1])])
    ax2.set_zlim([np.min(X[:,2]),np.max(X[:,2])])
    ax2.set(yticklabels=[])
    ax2.set(xticklabels=[])
    ax2.set(zticklabels=[])
    #ax2.set_xlabel("Cell  Line 1",labelpad=0.1)
    #ax2.set_ylabel("Cell  Line 2",labelpad=0.1)
    #ax2.set_zlabel("Cell  Line 3",labelpad=0.1)# Plot the compressed data points
    fig.subplots_adjust( left=None, bottom=None,  right=None, top=None, wspace=None, hspace=None)
    fig.tight_layout()
    plt.savefig(save_folder+'Cell_Lines.png',dpi=300,bbox_inches='tight')
    plt.close()

