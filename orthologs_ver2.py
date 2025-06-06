
import pandas as pd
import numpy as np
from google.colab import drive
import matplotlib.pyplot as plt
drive.mount('/content/drive/', force_remount=False)
# %cd /content/drive/MyDrive/Ortho_BOV_HUMAN/

out_tmp=pd.read_csv('Ortho_TI_CT_PPHE.csv', index_col=False)
out_tmp1=pd.read_csv('Ortho_TI_PPHE.csv', index_col=False)
out_tmp=pd.read_csv('Ortho_TI_CT.csv', index_col=False)
tipo=['tSNE','PCA','UMAP'][-1]
dfB=pd.read_csv('IF_'+tipo+'_Bovine.csv', index_col=False)
dfH=pd.read_csv('IF_'+tipo+'_Human.csv', index_col=False)

orto_H=out_tmp['name'].unique().tolist()
oH=dfH[dfH['Gene'].isin(orto_H)]
orto_HinB=oH['Gene'].unique().tolist()
tmp_B=out_tmp[out_tmp['name'].isin(orto_HinB)]
orto_B=tmp_B['incoming'].unique().tolist()
oB=dfB[dfB['Gene'].isin(orto_B)]

outliersB=dfB.loc[dfB['anomaly']==-1]
outliersH=dfH.loc[dfH['anomaly']==-1]

"""**incoming** is the input gene, **converted** is the canonical Ensembl ID for the input gene, **ortholog_ensg** is the canonical Ensembl ID for the orthologous gene in the target organism."""

out_tmp
