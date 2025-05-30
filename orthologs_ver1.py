import pandas as pd
import numpy as np
import re
from google.colab import drive
import matplotlib.pyplot as plt
!pip install gprofiler-official
from gprofiler import GProfiler
from matplotlib.lines import Line2D
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

dfB['Cond'].unique()

"""**g:Orth (orth)**"""

gp = GProfiler(return_dataframe=True)
rslt_df = dfB.loc[dfB['Cond'] == 'TI_CT_PPHE']
out_tmp=gp.orth(organism='btaurus',
            query=rslt_df['Gene'].tolist(),
            target='hsapiens')
out_tmp = out_tmp.loc[out_tmp['ortholog_ensg'] != 'N/A']
out_tmp

gp1 = GProfiler(return_dataframe=True)
rslt_df1 = dfB.loc[dfB['Cond'] == 'TI_PPHE']
out_tmp1=gp1.orth(organism='btaurus',
            query=rslt_df1['Gene'].tolist(),
            target='hsapiens')
out_tmp1 = out_tmp1.loc[out_tmp1['ortholog_ensg'] != 'N/A']
out_tmp1

gp2 = GProfiler(return_dataframe=True)
rslt_df2 = dfB.loc[dfB['Cond'] == 'TI_CT']
out_tmp2=gp2.orth(organism='btaurus',
            query=rslt_df2['Gene'].tolist(),
            target='hsapiens')
out_tmp2 = out_tmp2.loc[out_tmp2['ortholog_ensg'] != 'N/A']
out_tmp2

print('Sanity check')
out_tmp.equals(out_tmp1)
out_tmp.equals(out_tmp2)
out_tmp1.equals(out_tmp2)

out_tmp.to_csv('Ortho_TI_CT_PPHE.csv', index=False)
out_tmp1.to_csv('Ortho_TI_PPHE.csv', index=False)
out_tmp2.to_csv('Ortho_TI_CT.csv', index=False)
tipo=['tSNE','PCA','UMAP'][-1]
dfB=pd.read_csv('IF_'+tipo+'_Bovine.csv', index_col=False)
dfH=pd.read_csv('IF_'+tipo+'_Human.csv', index_col=False)

orto_H=out_tmp['name'].unique().tolist()

oH=dfH[dfH['Gene'].isin(orto_H)]

oH

orto_HinB=oH['Gene'].unique().tolist()
tmp_B=out_tmp[out_tmp['name'].isin(orto_HinB)]
orto_B=tmp_B['incoming'].unique().tolist()
oB=dfB[dfB['Gene'].isin(orto_B)]
oB

!pip install https://github.com/Phlya/adjustText/archive/master.zip
from adjustText import adjust_text

outliersB=dfB.loc[dfB['anomaly']==-1]
outliersH=dfH.loc[dfH['anomaly']==-1]
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

red_lines=['DimRed1', 	'DimRed2']
reduceB=dfB.loc[:,red_lines].to_numpy()
reduceH=dfH.loc[:,red_lines].to_numpy()
fig = plt.figure(figsize=plt.figaspect(0.5))
outlier_indexB=list(outliersB.index)
outlier_indexH=list(outliersH.index)
# =============
# First subplot
# =============
# set up the Axes for the first plot
ax = fig.add_subplot(1, 2, 1)
ax.set_xlabel("Dim. 1")
ax.set_ylabel("Dim. 2")
ax.scatter(reduceB[:, 0], reduceB[:, 1],  s=4, lw=1, label="inliers B",c="blue")# Plot x's for the ground truth outliers
ax.scatter(reduceB[outlier_indexB,0],reduceB[outlier_indexB,1],  lw=2, s=60, marker="x", c="crimson", label="outliers B")
ax.set_title('Bovine '+tipo)
geni_unici=oB.Gene.unique()
for gene,index, anom,xr,yr in zip(oB['Gene'],oB['Cond'],oB['anomaly'],oB['DimRed1'],oB['DimRed2']):
  if index==cond[0] and anom==-1:
    colore=colori[0]
  elif index==cond[1] and anom==-1:
    colore=colori[1]
  elif index==cond[2] and anom==-1:
    colore=colori[2]
  testo=ax.text(xr,yr,gene, ha='center', va='center',color=colore)
  bov_text.append(testo)
adjust_text(bov_text, ax=ax)
# ==============
# Second subplot
# ==============
# set up the Axes for the second plot

ax = fig.add_subplot(1, 2, 2)
ax.set_xlabel("Dim. 1")
ax.set_ylabel("Dim. 2")
ax.scatter(reduceH[:, 0], reduceH[:, 1],  s=4, lw=1, label="inliers H",c="green")# Plot x's for the ground truth outliers
ax.scatter(reduceH[outlier_indexH,0],reduceH[outlier_indexH,1], lw=2, s=60, marker="x", c="red", label="outliers H")
lgd =ax.legend(bbox_to_anchor=(1.05, 1.0),loc="upper left")
ax.set_title('Human '+tipo)

for gene,index, anom,xr,yr in zip(oH['Gene'],oH['Cond'],oH['anomaly'],oH['DimRed1'],oH['DimRed2']):
  if index==cond[0] and anom==-1:
    colore=colori[0]
  elif index==cond[1] and anom==-1:
    colore=colori[1]
  elif index==cond[2] and anom==-1:
    colore=colori[2]
  testo=ax.text(xr,yr,gene, ha='center', va='center',color=colore)
  human_text.append(testo)
adjust_text(human_text, ax=ax)

add_patch(lgd)
plt.savefig(tipo+'_ORTHO.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()
