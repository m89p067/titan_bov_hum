# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import re
from google.colab import drive
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
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
filenameBHs='Bovine_and_Human3'
drive.mount('/content/drive/', force_remount=False)
# %cd /content/drive/MyDrive/Ortho_BOV_HUMAN/
#my_genes=pd.read_csv('/content/drive/MyDrive/Ortho_BOV_HUMAN/out_genes.csv',index_col=False,sep=';')
cols=['Protein','Gene','Descrip','Type','Cond']
cond=['TI_CT_PPHE', 'TI_CT', 'TI_PPHE'] # ,'TI_P4000','TI_CT_CH','TI_CT_CH_TPP']
cell_lines=['Cell Line1', 'Cell Line2', 'Cell Line3']
BH_df=pd.read_csv('/content/drive/MyDrive/Ortho_BOV_HUMAN/'+filenameBHs+'.csv',index_col=False,header=0)
my_genes=BH_df.loc[:,cols]

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

# =============
# First subplot
# =============
# set up the Axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_xlabel("Cell  Line 1")
ax.set_ylabel("Cell  Line 2")
ax.set_zlabel("Cell  Line 3")# Plot the compressed data points
ax.scatter(X_reduceB[:, 0], X_reduceB[:, 1], zs=X_reduceB[:, 2], s=4, lw=1, label="inliers",c="green")# Plot x's for the ground truth outliers
ax.scatter(X_reduceB[outlier_indexB,0],X_reduceB[outlier_indexB,1], X_reduceB[outlier_indexB,2], lw=2, s=60, marker="x", c="red", label="outliers")
ax.legend()
ax.set_title('Bovine')

# ==============
# Second subplot
# ==============
# set up the Axes for the second plot

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_xlabel("Cell  Line 1")
ax.set_ylabel("Cell  Line 2")
ax.set_zlabel("Cell  Line 3")# Plot the compressed data points
ax.scatter(X_reduceH[:, 0], X_reduceH[:, 1], zs=X_reduceH[:, 2], s=4, lw=1, label="inliers",c="green")# Plot x's for the ground truth outliers
ax.scatter(X_reduceH[outlier_indexH,0],X_reduceH[outlier_indexH,1], X_reduceH[outlier_indexH,2], lw=2, s=60, marker="x", c="red", label="outliers")
ax.legend()
ax.set_title('Human')

plt.show()

PCA_reduceB = PCA(n_components=2).fit_transform(XB)
PCA_reduceH = PCA(n_components=2).fit_transform(XH)

clf = IsolationForest(random_state=42, verbose=0)
predB=clf.fit(PCA_reduceB).predict(PCA_reduceB)
predH=clf.fit(PCA_reduceH ).predict(PCA_reduceH)

# classified -1 are anomalous
dfB['anomaly']=predB
dfH['anomaly']=predH

dfB['PCA1']=PCA_reduceB[:,0]
dfB['PCA2']=PCA_reduceB[:,1]

dfH['PCA1']=PCA_reduceH[:,0]
dfH['PCA2']=PCA_reduceH[:,1]

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
ax.set_xlabel("First comp.")
ax.set_ylabel("Second comp.")
ax.scatter(PCA_reduceB[:, 0], PCA_reduceB[:, 1],  s=4, lw=1, label="inliers",c="green")# Plot x's for the ground truth outliers
ax.scatter(PCA_reduceB[outlier_indexB,0],PCA_reduceB[outlier_indexB,1],  lw=2, s=60, marker="x", c="red", label="outliers")
ax.set_title('Bovine PCA')
ax.axhline(linewidth=1, color='k',linestyle='--')
ax.axvline(linewidth=1, color='k',linestyle='--')
# ==============
# Second subplot
# ==============
# set up the Axes for the second plot

ax = fig.add_subplot(1, 2, 2)
ax.set_xlabel("First comp.")
ax.set_ylabel("Second comp.")
ax.scatter(PCA_reduceH[:, 0], PCA_reduceH[:, 1],  s=4, lw=1, label="inliers",c="green")# Plot x's for the ground truth outliers
ax.scatter(PCA_reduceH[outlier_indexH,0],PCA_reduceH[outlier_indexH,1], lw=2, s=60, marker="x", c="red", label="outliers")
ax.legend( bbox_to_anchor=(1.05, 1.0),loc="upper left")
ax.set_title('Human PCA')

ax.axhline(linewidth=1, color='k',linestyle='--')
ax.axvline(linewidth=1, color='k',linestyle='--')
plt.savefig('BH_outPCA.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.set_xlabel("First comp.")
ax.set_ylabel("Second comp.")
ax.scatter(PCA_reduceB[:, 0], PCA_reduceB[:, 1],  s=4, lw=1, label="inliers",c="green")# Plot x's for the ground truth outliers
ax.scatter(PCA_reduceB[outlier_indexB,0],PCA_reduceB[outlier_indexB,1],  lw=2, s=60, marker="x", c="red", label="outliers")
ax.set_title('Human and Bovine PCA')
ax.axhline(linewidth=1, color='k',linestyle='--')
ax.axvline(linewidth=1, color='k',linestyle='--')
ax.scatter(PCA_reduceH[:, 0], PCA_reduceH[:, 1],  s=4, lw=1, label="inliers",c="green")# Plot x's for the ground truth outliers
ax.scatter(PCA_reduceH[outlier_indexH,0],PCA_reduceH[outlier_indexH,1], lw=2, s=60, marker="x", c="red", label="outliers")
ax.legend( bbox_to_anchor=(1.05, 1.0),loc="upper left")
plt.savefig('BH_outPCA2.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()

dfB.to_csv('IF_PCA_Bovine.csv', index=False)
dfH.to_csv('IF_PCA_Human.csv', index=False)

estimators = {
    "Empirical Covariance": EllipticEnvelope(support_fraction=1.0, contamination=0.25),
    "Robust Covariance (Minimum Covariance Determinant)": EllipticEnvelope(
        contamination=0.25
    ),
  #  "OCSVM": OneClassSVM(nu=0.25, gamma=0.35),
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
ax.scatter(PCA_reduceB[outlier_indexB, 0], PCA_reduceB[outlier_indexB, 1], color="magenta",alpha=0.5)
ax.set_xlabel("First comp.")
ax.set_ylabel("Second comp.")
ax.set_title("Bovine PCA")
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
ax.scatter(PCA_reduceH[outlier_indexH, 0], PCA_reduceH[outlier_indexH, 1], color="magenta",alpha=0.5)
ax.legend(handles=legend_lines, bbox_to_anchor=(1.05, 1.0),loc="upper left")
_ = ax.set(
    xlabel="First comp.",
    ylabel="Second comp.",
    title="Human PCA",
)

plt.savefig('BH_outPCA3.png',dpi=300,bbox_inches='tight')
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

outB

outH

bov_text=[]
human_text=[]
colori=['navy','dimgray','darkorange']

custom_lines = [Line2D([0], [0], color=colori[0], lw=4),
                Line2D([0], [0], color=colori[1], lw=4),
                Line2D([0], [0], color=colori[2], lw=4)]

def add_patch(legend):

    ax1 = legend.axes

    handles, labels = ax1.get_legend_handles_labels()
    handles.append(custom_lines[0])
    labels.append(cond[0])
    handles.append(custom_lines[1])
    labels.append(cond[1])
    handles.append(custom_lines[2])
    labels.append(cond[2])
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
ax.set_xlabel("First comp.")
ax.set_ylabel("Second comp.")
ax.scatter(PCA_reduceB[:, 0], PCA_reduceB[:, 1],  s=4, lw=1, label="inliers",c="green")# Plot x's for the ground truth outliers
ax.scatter(PCA_reduceB[outlier_indexB,0],PCA_reduceB[outlier_indexB,1],  lw=2, s=60, marker="x", c="red", label="outliers")
ax.set_title('Bovine PCA')

for index, row in outB.iterrows():
  if row['Cond']==cond[0]:
    colore=colori[0]
  elif row['Cond']==cond[1]:
    colore=colori[1]
  elif row['Cond']==cond[2]:
    colore=colori[2]
  testo=ax.text(row['PCA1'],row['PCA2'],row['Gene'], ha='center', va='center',color=colore)
  bov_text.append(testo)
adjust_text(bov_text, ax=ax)
# ==============
# Second subplot
# ==============
# set up the Axes for the second plot

ax = fig.add_subplot(1, 2, 2)
ax.set_xlabel("First comp.")
ax.set_ylabel("Second comp.")
ax.scatter(PCA_reduceH[:, 0], PCA_reduceH[:, 1],  s=4, lw=1, label="inliers",c="green")# Plot x's for the ground truth outliers
ax.scatter(PCA_reduceH[outlier_indexH,0],PCA_reduceH[outlier_indexH,1], lw=2, s=60, marker="x", c="red", label="outliers")
lgd =ax.legend(bbox_to_anchor=(1.05, 1.0),loc="upper left")
ax.set_title('Human PCA')

for index, row in outH.iterrows():
  if row['Cond']==cond[0]:
    colore=colori[0]
  elif row['Cond']==cond[1]:
    colore=colori[1]
  elif row['Cond']==cond[2]:
    colore=colori[2]
  testo=ax.text(row['PCA1'],row['PCA2'],row['Gene'], ha='center', va='center',color=colore)
  human_text.append(testo)
adjust_text(human_text, ax=ax)

add_patch(lgd)
plt.show()

"""**Gene Ontology**"""

#Most frequent keywords among outliers
gp = GProfiler(return_dataframe=True)
gp.profile(organism='hsapiens',  query=outH['Gene'].tolist())

#All outliers
dictH={}
gp = GProfiler(return_dataframe=True)
for the_cond in cond:
  print('- - - ',the_cond,' - - -')
  tmp=outliersH.loc[outliersH['Cond']==the_cond]
  out_tmp=gp.profile(organism='hsapiens',  query=tmp['Gene'].tolist())
  out_tmp.sort_values(by=['source']).to_csv('H_all_outl_'+the_cond+'.csv', index=False)
  #dictH[the_cond]=out_tmp['name'].tolist()
  dictH[the_cond]=out_tmp.loc[out_tmp['source'].str.contains('GO:'), 'name'].tolist()
  with pd.option_context('display.max_rows', None,'display.max_columns', None,'display.precision', 3,'display.width', None,'display.max_colwidth', None):
    display(out_tmp.sort_values(by=['source']))

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
plt.show()
