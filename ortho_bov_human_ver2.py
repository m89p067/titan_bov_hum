# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import re
from google.colab import drive
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import OneClassSVM
import matplotlib.lines as mlines
!pip install https://github.com/Phlya/adjustText/archive/master.zip
from adjustText import adjust_text
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
!pip install gprofiler-official
from gprofiler import GProfiler
!pip install wordcloud
from wordcloud import (WordCloud, get_single_color_func)
!pip install umap-learn
import umap
filenameBHs='Bovine_and_Human3'
drive.mount('/content/drive/', force_remount=False)
# %cd /content/drive/MyDrive/Ortho_BOV_HUMAN/
#my_genes=pd.read_csv('/content/drive/MyDrive/Ortho_BOV_HUMAN/out_genes.csv',index_col=False,sep=';')
cols=['Protein','Gene','Descrip','Type','Cond']
cond=['TI_CT_PPHE', 'TI_CT', 'TI_PPHE'] # ,'TI_P4000','TI_CT_CH','TI_CT_CH_TPP']
cell_lines=['Cell Line1', 'Cell Line2', 'Cell Line3']
BH_df=pd.read_csv('/content/drive/MyDrive/Ortho_BOV_HUMAN/'+filenameBHs+'.csv',index_col=False,header=0)
my_genes=BH_df.loc[:,cols]

class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.

       Uses wordcloud.get_single_color_func

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)
def separa(s1):
  result1= re.search('; (.*)\[', s1)
  if result1==None:
    return s1
  else:
    return result1.group(1)

#BH_df['Cond'] = BH_df['Cond'].map({'TI_PPHE': 'Ti_CT_CH_TPP_PPHE','TI_CT_PPHE':'pippo','TI_CT':'Ti_CT'})
#BH_df['Type'] = BH_df['Type'].map({'H': 'hBM-MSCs','B':'FBS'})
#BH_df = BH_df[BH_df['Cond']!='pippo'].reset_index()

#dfB = BH_df.loc[BH_df['Type'] == 'FBS'].reset_index(drop=True)
#dfH = BH_df.loc[BH_df['Type'] == 'hBM-MSCs'].reset_index(drop=True)
dfB = BH_df.loc[BH_df['Type'] == 'B'].reset_index(drop=True)
dfH = BH_df.loc[BH_df['Type'] == 'H'].reset_index(drop=True)

XB=dfB.loc[:,cell_lines].to_numpy()
XH=dfH.loc[:,cell_lines].to_numpy()

clf = IsolationForest(random_state=42, verbose=0)
predB =clf.fit(XB).predict(XB)
predH=clf.fit(XH).predict(XH)

# classified -1 are anomalous
dfB['anomaly']=predB
dfH['anomaly']=predH

outliersB=dfB.loc[dfB['anomaly']==-1]
outlier_indexB=list(outliersB.index)
X_reduceB=XB.copy()

outliersH=dfH.loc[dfH['anomaly']==-1]
outlier_indexH=list(outliersH.index)
X_reduceH=XH.copy()

fig = plt.figure(figsize=plt.figaspect(0.5))
X=BH_df.loc[:,cell_lines].to_numpy()
# =============
# First subplot
# =============
# set up the Axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(X_reduceB[:, 0], X_reduceB[:, 1], zs=X_reduceB[:, 2], s=4, lw=1, label="inliers",c="green")# Plot x's for the ground truth outliers
ax.scatter(X_reduceB[outlier_indexB,0],X_reduceB[outlier_indexB,1], X_reduceB[outlier_indexB,2], lw=2, s=60, marker="x", c="red", label="outliers")
ax.legend()
ax.set_title('Bovine')
ax.set_xlim([np.min(X[:,0]),np.max(X[:,0])])
ax.set_ylim([np.min(X[:,1]),np.max(X[:,1])])
ax.set_zlim([np.min(X[:,2]),np.max(X[:,2])])
ax.set(yticklabels=[])
ax.set(xticklabels=[])
ax.set(zticklabels=[])
ax.set_xlabel("Cell  Line 1")
ax.set_ylabel("Cell  Line 2")
# ==============
# Second subplot
# ==============
# set up the Axes for the second plot

ax = fig.add_subplot(1, 2, 2, projection='3d')

ax.scatter(X_reduceH[:, 0], X_reduceH[:, 1], zs=X_reduceH[:, 2], s=4, lw=1, label="inliers",c="green")# Plot x's for the ground truth outliers
ax.scatter(X_reduceH[outlier_indexH,0],X_reduceH[outlier_indexH,1], X_reduceH[outlier_indexH,2], lw=2, s=60, marker="x", c="red", label="outliers")
ax.legend()
ax.set_title('Human')
ax.set_xlim([np.min(X[:,0]),np.max(X[:,0])])
ax.set_ylim([np.min(X[:,1]),np.max(X[:,1])])
ax.set_zlim([np.min(X[:,2]),np.max(X[:,2])])
ax.set(yticklabels=[])
ax.set(xticklabels=[])
ax.set(zticklabels=[])
ax.set_xlabel("Cell  Line 1")
ax.set_ylabel("Cell  Line 2")
ax.set_zlabel("Cell  Line 3")# Plot the compressed data points
plt.savefig('Cell_Lines.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()

is_tsne=False
if is_tsne:
  perplexity =75
  early_exaggeration =8
  tsne = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration, max_iter=3000,random_state=42,init='pca')
  PCA_reduceB = tsne.fit_transform(XB)
  PCA_reduceH = tsne.fit_transform(XH)
  tipo='tSNE'
else:
  metric, n_neighbors, min_dist="mahalanobis", 30, 0.5
  mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric=metric, random_state=42)
  PCA_reduceB = mapper.fit_transform(XB)
  PCA_reduceH = mapper.fit_transform(XH)
  tipo='UMAP'
clf = IsolationForest(random_state=42, verbose=0)
predB=clf.fit(PCA_reduceB).predict(PCA_reduceB)
predH=clf.fit(PCA_reduceH ).predict(PCA_reduceH)

# classified -1 are anomalous
dfB['anomaly']=predB
dfH['anomaly']=predH

dfB['DimRed1']=PCA_reduceB[:,0]
dfB['DimRed2']=PCA_reduceB[:,1]

dfH['DimRed1']=PCA_reduceH[:,0]
dfH['DimRed2']=PCA_reduceH[:,1]

outliersB=dfB.loc[dfB['anomaly']==-1]
outlier_indexB=list(outliersB.index)
X_reduceB=XB.copy()

outliersH=dfH.loc[dfH['anomaly']==-1]
outlier_indexH=list(outliersH.index)
X_reduceH=XH.copy()

fig = plt.figure(figsize=plt.figaspect(0.5))

# =============
# First subplot
# =============
# set up the Axes for the first plot
ax = fig.add_subplot(1, 2, 1)
ax.set_xlabel("Dim. 1")
ax.set_ylabel("Dim. 2")
ax.scatter(PCA_reduceB[:, 0], PCA_reduceB[:, 1],  s=4, lw=1, label="inliers",c="blue")# Plot x's for the ground truth outliers
ax.scatter(PCA_reduceB[outlier_indexB,0],PCA_reduceB[outlier_indexB,1],  lw=2, s=60, marker="x", c="crimson", label="outliers")
ax.legend()
ax.set_title('Bovine '+tipo)
ax.axhline(linewidth=1, color='k',linestyle='--')
ax.axvline(linewidth=1, color='k',linestyle='--')
# ==============
# Second subplot
# ==============
# set up the Axes for the second plot

ax = fig.add_subplot(1, 2, 2)
ax.set_xlabel("Dim. 1")
ax.set_ylabel("Dim. 2")
ax.scatter(PCA_reduceH[:, 0], PCA_reduceH[:, 1],  s=4, lw=1, label="inliers",c="green")# Plot x's for the ground truth outliers
ax.scatter(PCA_reduceH[outlier_indexH,0],PCA_reduceH[outlier_indexH,1], lw=2, s=60, marker="x", c="red", label="outliers")
ax.legend()
ax.set_title('Human '+tipo)

ax.axhline(linewidth=1, color='k',linestyle='--')
ax.axvline(linewidth=1, color='k',linestyle='--')
plt.savefig(tipo+'_outl.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.set_xlabel("Dim. 1")
ax.set_ylabel("Dim. 2")
ax.scatter(PCA_reduceH[:, 0], PCA_reduceH[:, 1],  s=4, lw=1, label="inliers H",c="green")# Plot x's for the ground truth outliers
ax.scatter(PCA_reduceH[outlier_indexH,0],PCA_reduceH[outlier_indexH,1], lw=2, s=60, marker="x", c="red", label="outliers H")
ax.scatter(PCA_reduceB[:, 0], PCA_reduceB[:, 1],  s=4, lw=1, label="inliers B",c="blue")# Plot x's for the ground truth outliers
ax.scatter(PCA_reduceB[outlier_indexB,0],PCA_reduceB[outlier_indexB,1],  lw=2, s=60, marker="x", c="crimson", label="outliers B")
ax.axhline(linewidth=1, color='k',linestyle='--')
ax.axvline(linewidth=1, color='k',linestyle='--')
ax.legend()
ax.set_title(tipo+': Human and Bovine')
plt.savefig(tipo+'_outlBoth.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()

dfB.to_csv('IF_'+tipo+'_Bovine.csv', index=False)
dfH.to_csv('IF_'+tipo+'_Human.csv', index=False)

estimators = {
    "Empirical Covariance": EllipticEnvelope(support_fraction=1.0, contamination=0.25),
    "Robust Covariance\n(Minimum Covariance Determinant)": EllipticEnvelope(
        contamination=0.25
    ),
    #"OCSVM": OneClassSVM(nu=0.25, gamma=0.35),
}

fig = plt.figure(figsize=plt.figaspect(0.5))

# =============
# First subplot
# =============
# set up the Axes for the first plot
ax = fig.add_subplot(1, 2, 1)
colors = ["tab:blue", "tab:orange", "tab:red"]
# Learn a frontier for outlier detection with several classifiers
legend_lines = []
for color, (name, estimator) in zip(colors, estimators.items()):
    estimator.fit(PCA_reduceB)
    DecisionBoundaryDisplay.from_estimator(
        estimator,
        PCA_reduceB,
        response_method="decision_function",
        plot_method="contour",
        levels=[0],
        colors=color,
        ax=ax,
    )
    legend_lines.append(mlines.Line2D([], [], color=color, label=name))


ax.scatter(PCA_reduceB[~np.asarray(outlier_indexB), 0], PCA_reduceB[~np.asarray(outlier_indexB), 1], color="black",alpha=0.5)
ax.scatter(PCA_reduceB[outlier_indexB, 0], PCA_reduceB[outlier_indexB, 1], color="crimson",alpha=0.5)
ax.set_xlabel("Dim. 1")
ax.set_ylabel("Dim. 2")
ax.set_title("Bovine "+tipo)
# ==============
# Second subplot
# ==============
# set up the Axes for the second plot

ax = fig.add_subplot(1, 2, 2)
legend_lines = []
for color, (name, estimator) in zip(colors, estimators.items()):
    estimator.fit(PCA_reduceH)
    DecisionBoundaryDisplay.from_estimator(
        estimator,
        PCA_reduceH,
        response_method="decision_function",
        plot_method="contour",
        levels=[0],
        colors=color,
        ax=ax,linewidths=2
    )
    legend_lines.append(mlines.Line2D([], [], color=color, label=name))


ax.scatter(PCA_reduceH[~np.asarray(outlier_indexH), 0], PCA_reduceH[~np.asarray(outlier_indexH), 1], color="black",alpha=0.5)
ax.scatter(PCA_reduceH[outlier_indexH, 0], PCA_reduceH[outlier_indexH, 1], color="red",alpha=0.5)
ax.legend(handles=legend_lines, bbox_to_anchor=(1.05, 1.0),loc="upper left")
_ = ax.set(
    xlabel="Dim. 1",
    ylabel="Dim. 2",
    title="Human "+tipo,
)

plt.savefig(tipo+'cov.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()

"""**Most frequent keywords among outliers**"""

list1 = ['OS=Bos','taurus','OS=Homo','sapiens','PE=1','SV=1','PE=2','SV=2','SV=3','of','protein','PE=3','chain',
         'SV=4','1','2','3','4','5','6','Gene','factor','gene','cytoskeletal','subunit','type','I','II','Isoform']

outB=pd.DataFrame()
outH=pd.DataFrame()
for the_cond in cond:
  print('- - - ',the_cond,' - - -')
  tmpB = outliersB.loc[((outliersB['Type'] == 'B') & (outliersB['Cond'] == the_cond))]
  tmpH = outliersH.loc[((outliersH['Type'] == 'H') & (outliersH['Cond'] == the_cond))]

  #tmpB=tmpB.loc[:,cols]
  #tmpH=tmpH.loc[:,cols]

  #wordB=dfB.Descrip.str.split(expand=True).stack().value_counts()
  #wordH=dfH.Descrip.str.split(expand=True).stack().value_counts()

  #wordB=pd.Series(np.concatenate([x.split() for x in dfB.Descrip])).value_counts()
  #wordH=pd.Series(np.concatenate([x.split() for x in dfH.Descrip])).value_counts()

  wordB=pd.Series(' '.join(tmpB.Descrip).split()).value_counts()
  wordH=pd.Series(' '.join(tmpH.Descrip).split()).value_counts()

  #wordB=dfB.Descrip.str.split().explode().value_counts()
  #wordH=dfH.Descrip.str.split().explode().value_counts()

  resB=wordB[wordB.index.isin(list1) == False]
  resH=wordH[wordH.index.isin(list1) == False]

  idxB=resB[resB > 2].index
  idxH=resH[resH > 2].index

  for i in idxB:
    print('Bovine - KEYWORD:',i)
    stringa=tmpB[tmpB['Descrip'].str.contains(i)]
    #print(stringa)
    outB= pd.concat([outB, stringa], ignore_index=True)
  #print('\n\n')
  for i in idxH:
    print('Human - KEYWORD:',i)
    stringa=tmpH[tmpH['Descrip'].str.contains(i)]
    #print(stringa)
    outH= pd.concat([outH, stringa], ignore_index=True)

outB.to_csv('B_sel_'+tipo+'_outl.csv', index=False)

outH.to_csv('H_sel_'+tipo+'_outl.csv', index=False)

bov_text=[]
human_text=[]
#colori=['navy','dimgray','darkorange']
colori=['navy','darkorange']
custom_lines = [Line2D([0], [0], color=colori[0], lw=4),
                Line2D([0], [0], color=colori[1], lw=4),
               # Line2D([0], [0], color=colori[2], lw=4)
               ]
cond=['Ti_CT_CH_TPP_PPHE','Ti_CT']
def add_patch(legend):

    ax1 = legend.axes

    handles, labels = ax1.get_legend_handles_labels()
    handles.append(custom_lines[0])
    labels.append(cond[0])
    handles.append(custom_lines[1])
    labels.append(cond[1])
#    handles.append(custom_lines[2])
#    labels.append(cond[2])
    legend._legend_box = None
    legend._init_legend_box(handles, labels)
    legend._set_loc(legend._loc)
    legend.set_title(legend.get_title().get_text())


fig = plt.figure(figsize=plt.figaspect(0.5))

# =============
# First subplot
# =============
# set up the Axes for the first plot
ax = fig.add_subplot(1, 2, 1)
ax.set_xlabel("Dim. 1")
ax.set_ylabel("Dim. 2")
ax.scatter(PCA_reduceB[:, 0], PCA_reduceB[:, 1],  s=4, lw=1, label="inliers FBS",c="blue")# Plot x's for the ground truth outliers
ax.scatter(PCA_reduceB[outlier_indexB,0],PCA_reduceB[outlier_indexB,1],  lw=2, s=60, marker="x", c="crimson", label="outliers FBS")
ax.set_title('Bovine '+tipo)

for index, row in outB.iterrows():
  if row['Cond']==cond[0]:
    colore=colori[0]
  elif row['Cond']==cond[1]:
    colore=colori[1]
  elif row['Cond']==cond[2]:
    colore=colori[2]
  testo=ax.text(row['DimRed1'],row['DimRed2'],row['Gene'], ha='center', va='center',color=colore)
  bov_text.append(testo)
adjust_text(bov_text, ax=ax)
# ==============
# Second subplot
# ==============
# set up the Axes for the second plot

ax = fig.add_subplot(1, 2, 2)
ax.set_xlabel("Dim. 1")
ax.set_ylabel("Dim. 2")
ax.scatter(PCA_reduceH[:, 0], PCA_reduceH[:, 1],  s=4, lw=1, label="inliers BM-MSC",c="green")# Plot x's for the ground truth outliers
ax.scatter(PCA_reduceH[outlier_indexH,0],PCA_reduceH[outlier_indexH,1], lw=2, s=60, marker="x", c="red", label="outliers BM-MSC")
lgd =ax.legend(bbox_to_anchor=(1.05, 1.0),loc="upper left")
ax.set_title('Human '+tipo)

for index, row in outH.iterrows():
  if row['Cond']==cond[0]:
    colore=colori[0]
  elif row['Cond']==cond[1]:
    colore=colori[1]
#  elif row['Cond']==cond[2]:
#    colore=colori[2]
  testo=ax.text(row['DimRed1'],row['DimRed2'],row['Gene'], ha='center', va='center',color=colore)
  human_text.append(testo)
adjust_text(human_text, ax=ax)

add_patch(lgd)
plt.savefig(tipo+'_named_outl.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.set_xlabel("Dim. 1")
ax.set_ylabel("Dim. 2")
ax.scatter(PCA_reduceB[:, 0], PCA_reduceB[:, 1],  s=4, lw=1, label="B inl.",c="blue")# Plot x's for the ground truth outliers
ax.scatter(PCA_reduceB[outlier_indexB,0],PCA_reduceB[outlier_indexB,1],  lw=2, s=60, marker="x", c="crimson", label="B outl.")
ax.set_title('Bovine and Human '+tipo)
ax.scatter(PCA_reduceH[:, 0], PCA_reduceH[:, 1],  s=4, lw=1, label="B inl.",c="green")# Plot x's for the ground truth outliers
ax.scatter(PCA_reduceH[outlier_indexH,0],PCA_reduceH[outlier_indexH,1], lw=2, s=60, marker="x", c="red", label="H outl.")
adjust_text(bov_text+human_text, ax=ax)

lgd =ax.legend(bbox_to_anchor=(1.05, 1.0),loc="upper left")
add_patch(lgd)
plt.savefig(tipo+'_named_outl_HB.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()

"""**Gene Ontology**"""

#Most frequent keywords and related genes among outliers
use_go=False
dictH={}
gp = GProfiler(return_dataframe=True)
gp.profile(organism='hsapiens',  query=outH['Gene'].tolist())
for the_cond in cond:
  print('- - - ',the_cond,' - - -')
  out_tmp2=outH.loc[outH['Cond']==the_cond]
  out_tmp=gp.profile(organism='hsapiens',  query=out_tmp2['Gene'].tolist())
  if use_go:
    str_des='only_go'
    dictH[the_cond]=out_tmp.loc[out_tmp['source'].str.contains('GO:'), 'name'].tolist()
  else:
    dictH[the_cond]=out_tmp['name'].tolist()
    str_des='all_sources'
  out_tmp.sort_values(by=['source']).to_csv(str_des+'_selected_H_'+tipo+'_outl_'+the_cond+'.csv', index=False)
  with pd.option_context('display.max_rows', None,'display.max_columns', None,'display.precision', 3,'display.width', None,'display.max_colwidth', None):
    display(out_tmp.sort_values(by=['source']))

tmp1=[x.lower() for x in dictH[cond[0]]]
tmp2=[x.lower() for x in dictH[cond[1]]]
tmp3=[x.lower() for x in dictH[cond[2]]]
tmp1=[ separa(s) for s in tmp1]
tmp2=[ separa(s) for s in tmp2]
tmp3=[ separa(s) for s in tmp3]
tmp1=[words for segments in tmp1 for words in segments.split()]
tmp2=[words for segments in tmp2 for words in segments.split()]
tmp3=[words for segments in tmp3 for words in segments.split()]
#generic_terms=['synapse','membrane','cell','binding','regulation','plasma','secretory','organelle','layer','complex','activity','protein',
#               'basal','expression','bounded','cells','positive','short','containing','repair','organization','in','of','gene','processing',
#               'leading','edge',  'periphery',  'protein-containing','component','space','region','match']
generic_terms=[]
l1 = [x for x in tmp1 if x not in generic_terms]
l2 = [x for x in tmp2 if x not in generic_terms]
l3 = [x for x in tmp3 if x not in generic_terms]

text = l1+l2+l3
text3=' '.join(str(e) for e in text)
# Since the text is small collocations are turned off and text is lower-cased
wc = WordCloud(collocations=False).generate(text3)

color_to_words = {
    'green': l1,
     'red': l2,
    'blue':l3
}

# Words that are not in any of the color_to_words values
# will be colored with a grey single color function
default_color = 'grey'

# Create a color function with single tone
# grouped_color_func = SimpleGroupedColorFunc(color_to_words, default_color)

# Create a color function with multiple tones
grouped_color_func = GroupedColorFunc(color_to_words, default_color)

# Apply our color function
wc.recolor(color_func=grouped_color_func)

# Plot
plt.figure()
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.savefig(tipo+'_sel_wc_outl.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()

use_go=True
#All outliers
dictH={}
gp = GProfiler(return_dataframe=True)
for the_cond in cond:
  print('- - - ',the_cond,' - - -')
  tmp=outliersH.loc[outliersH['Cond']==the_cond]
  out_tmp=gp.profile(organism='hsapiens',  query=tmp['Gene'].tolist())
  out_tmp.sort_values(by=['source']).to_csv('H_'+tipo+'_outl_'+the_cond+'.csv', index=False)

  if use_go:
    str_des='only_go'
    dictH[the_cond]=out_tmp.loc[out_tmp['source'].str.contains('GO:'), 'name'].tolist()
  else:
    str_des='all_sources'
    dictH[the_cond]=out_tmp['name'].tolist()
  with pd.option_context('display.max_rows', None,'display.max_columns', None,'display.precision', 3,'display.width', None,'display.max_colwidth', None):
    display(out_tmp.sort_values(by=['source']))

tmp1=[x.lower() for x in dictH[cond[0]]]
tmp2=[x.lower() for x in dictH[cond[1]]]
tmp3=[x.lower() for x in dictH[cond[2]]]
tmp1=[ separa(s) for s in tmp1]
tmp2=[ separa(s) for s in tmp2]
tmp3=[ separa(s) for s in tmp3]
tmp1=[words for segments in tmp1 for words in segments.split()]
tmp2=[words for segments in tmp2 for words in segments.split()]
tmp3=[words for segments in tmp3 for words in segments.split()]
#generic_terms=['synapse','membrane','cell','binding','regulation','plasma','secretory','organelle','layer','complex','activity','protein',
#               'basal','expression','bounded','cells','positive','short','containing','repair','organization','in','of','gene','processing',
#               'leading','edge',  'periphery',  'protein-containing','component','space','region']
generic_terms=[]
l1 = [x for x in tmp1 if x not in generic_terms]
l2 = [x for x in tmp2 if x not in generic_terms]
l3 = [x for x in tmp3 if x not in generic_terms]

text = l1+l2+l3
text3=' '.join(str(e) for e in text)
# Since the text is small collocations are turned off and text is lower-cased
wc = WordCloud(collocations=False).generate(text3)

color_to_words = {
    'green': l1,
     'red': l2,
    'blue':l3
}

# Words that are not in any of the color_to_words values
# will be colored with a grey single color function
default_color = 'grey'

# Create a color function with single tone
# grouped_color_func = SimpleGroupedColorFunc(color_to_words, default_color)

# Create a color function with multiple tones
grouped_color_func = GroupedColorFunc(color_to_words, default_color)

# Apply our color function
wc.recolor(color_func=grouped_color_func)

# Plot
plt.figure()
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.savefig(tipo+'_'+str_des+'_wc_outl.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()
