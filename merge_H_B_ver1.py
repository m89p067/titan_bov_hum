import pandas as pd
import numpy as np
import pdb
pd.options.mode.chained_assignment = None  # default='warn'
folder='D:/Data/datasets_analysis/'
filenameB='Cartel1.xlsx'
raw_datasetB=pd.read_excel(filenameB,header=0,index_col=None)
raw_datasetB[['idx','Protein1']] = raw_datasetB['Peak Name'].str.split('|',n=1,expand=True)
raw_datasetB[['Protein','Descrip2']] = raw_datasetB['Protein1'].str.split('|',n=1,expand=True)
raw_datasetB[['Gene','Descrip']] = raw_datasetB['Descrip2'].str.split('_BOVIN',n=1,expand=True).fillna('')
raw_datasetB = raw_datasetB.drop(['idx', 'Descrip2','Protein1','Peak Name'], axis=1)
raw_datasetBB = raw_datasetB.copy(deep=True)
cols_to_drop=['TI_P4000_1', 'TI_P4000_2', 'TI_P4000_3','TI_CT_CH_1', 'TI_CT_CH_2',
       'TI_CT_CH_3','TI_CT_CH_TPP_1', 'TI_CT_CH_TPP_2', 'TI_CT_CH_TPP_3']
raw_datasetB = raw_datasetB.drop(cols_to_drop, axis=1)



folderH='D:/Parinaz/Umani/'
filenameH='AreeNormOdonto_08112021.xlsx' # Filename of the proteomic dataset
raw_datasetH=pd.read_excel(folderH+filenameH)
raw_datasetH[['idx','Protein1']] = raw_datasetH['Peak Name'].str.split('|',n=1,expand=True)
raw_datasetH[['Protein','Descrip2']] = raw_datasetH['Protein1'].str.split('|',n=1,expand=True)
raw_datasetH[['Gene','Descrip3']] = raw_datasetH['Descrip2'].str.split('_H',n=1,expand=True).fillna('')
for index, row in raw_datasetH.iterrows():
    if '_HUMAN' in row["Group"]:
        idx=row["Group"].index('_HUMAN')
        raw_datasetH.loc[index, "Group"] = row["Group"][idx+7:]        
    else:
        raw_datasetH.loc[index, "Group"] = row["Group"]
cols_to_drop=['idx', 'Descrip2','Protein1','Peak Name','Descrip3']
raw_datasetH.rename(columns = {'Group':'Descrip'}, inplace = True)
raw_datasetH = raw_datasetH.drop(cols_to_drop, axis=1)

colonne={'ODONTO_TICTCHTPPPPHE_100nguL_03112021_SWATH1':'TI_PPHE_1',
       'ODONTO_TICTCHTPPPPHE_100nguL_03112021_SWATH2':'TI_PPHE_2',
       'ODONTO_TICTCHTPPPPHE_100nguL_03112021_SWATH3':'TI_PPHE_3',
       'ODONTO_TICTP_200nguL_04112021_SWATH1':'TI_CT_PPHE_1',
       'ODONTO_TICTP_200nguL_04112021_SWATH2':'TI_CT_PPHE_2',
       'ODONTO_TICTP_200nguL_04112021_SWATH3':'TI_CT_PPHE_3',
       'ODONTO_TICT_200nguL_05112021_SWATH1':'TI_CT_1',
       'ODONTO_TICT_200nguL_05112021_SWATH2':'TI_CT_2',
       'ODONTO_TICT_200nguL_05112021_SWATH3':'TI_CT_3'}
raw_datasetH.rename(colonne,  axis = "columns", inplace = True)

# REMOVE PEAKS THAT ARE ZEROS
raw_datasetH=raw_datasetH[raw_datasetH.select_dtypes(include=[np.number]).ne(0.0).all(1)]
raw_datasetB=raw_datasetB[raw_datasetB.select_dtypes(include=[np.number]).ne(0.0).all(1)]
raw_datasetBB=raw_datasetBB[raw_datasetBB.select_dtypes(include=[np.number]).ne(0.0).all(1)]
# LOG TRANSFORMATION
for col_i in list(raw_datasetH.select_dtypes(include=[np.number]).columns.values):
    raw_datasetH.loc[:,col_i]=raw_datasetH.loc[:,col_i].map(np.log2)
for col_i in list(raw_datasetB.select_dtypes(include=[np.number]).columns.values):
    raw_datasetB.loc[:,col_i]=raw_datasetB.loc[:,col_i].map(np.log2)
for col_i in list(raw_datasetBB.select_dtypes(include=[np.number]).columns.values):
    raw_datasetBB.loc[:,col_i]=raw_datasetBB.loc[:,col_i].map(np.log2)
raw_datasetB['TI_PPHE'] = raw_datasetB[['TI_PPHE_1', 'TI_PPHE_2', 'TI_PPHE_3']].mean(axis=1)
raw_datasetH['TI_PPHE'] = raw_datasetH[['TI_PPHE_1', 'TI_PPHE_2', 'TI_PPHE_3']].mean(axis=1)

raw_datasetB['TI_CT_PPHE'] = raw_datasetB[['TI_CT_PPHE_1', 'TI_CT_PPHE_2', 'TI_CT_PPHE_3']].mean(axis=1)
raw_datasetH['TI_CT_PPHE'] = raw_datasetH[['TI_CT_PPHE_1', 'TI_CT_PPHE_2', 'TI_CT_PPHE_3']].mean(axis=1)

raw_datasetB['TI_CT'] = raw_datasetB[['TI_CT_1', 'TI_CT_2', 'TI_CT_3']].mean(axis=1)
raw_datasetH['TI_CT'] = raw_datasetH[['TI_CT_1', 'TI_CT_2', 'TI_CT_3']].mean(axis=1)


raw_datasetB['Type'] = 'B'
raw_datasetH['Type'] = 'H'

#ind = raw_datasetB.Protein.isin(raw_datasetH.Protein) & raw_datasetH.Protein.isin(raw_datasetB.Protein)
#out_common=pd.concat([raw_datasetH[ind],raw_datasetB[ind]],ignore_index=True)
ind = raw_datasetB.Gene.isin(raw_datasetH.Gene) & raw_datasetH.Gene.isin(raw_datasetB.Gene)
out_common=pd.concat([raw_datasetH[ind],raw_datasetB[ind]],ignore_index=True)
print('Found ',out_common.shape[0],' genes in common between Human and Bovine')

out=pd.concat([raw_datasetB, raw_datasetH],ignore_index=True)
out = out.reset_index(drop=True)

list_subset1=['TI_CT_PPHE_1', 'TI_CT_PPHE_2', 'TI_CT_PPHE_3']
list_subset2=['TI_CT_1', 'TI_CT_2','TI_CT_3']
list_subset3=['TI_PPHE_1', 'TI_PPHE_2', 'TI_PPHE_3']
list_subset4=['TI_P4000_1', 'TI_P4000_2', 'TI_P4000_3']
list_subset5=['TI_CT_CH_1', 'TI_CT_CH_2','TI_CT_CH_3']
list_subset6=['TI_CT_CH_TPP_1', 'TI_CT_CH_TPP_2', 'TI_CT_CH_TPP_3']
list_renamed=['Cell Line1','Cell Line2','Cell Line3']
other_cols=['Protein', 'Gene','Descrip', 'Type']
subset1= out[list_subset1+other_cols]
subset2= out[list_subset2+other_cols]
subset3= out[list_subset3+other_cols]
subset1.loc[:,'Cond']='TI_CT_PPHE'
subset2.loc[:,'Cond']='TI_CT'
subset3.loc[:,'Cond']='TI_PPHE'
subset1.rename(dict(zip(list_subset1, list_renamed)), axis=1, inplace=True)
subset2.rename(dict(zip(list_subset2, list_renamed)), axis=1, inplace=True)
subset3.rename(dict(zip(list_subset3, list_renamed)), axis=1, inplace=True)
out2=pd.concat([subset1, subset2, subset3],ignore_index=True)

subset1= raw_datasetH[list_subset1+other_cols]
subset2= raw_datasetH[list_subset2+other_cols]
subset3= raw_datasetH[list_subset3+other_cols]
subset1.loc[:,'Cond']='TI_CT_PPHE'
subset2.loc[:,'Cond']='TI_CT'
subset3.loc[:,'Cond']='TI_PPHE'
subset1.rename(dict(zip(list_subset1, list_renamed)), axis=1, inplace=True)
subset2.rename(dict(zip(list_subset2, list_renamed)), axis=1, inplace=True)
subset3.rename(dict(zip(list_subset3, list_renamed)), axis=1, inplace=True)
raw_datasetH2=pd.concat([subset1, subset2, subset3],ignore_index=True)

subset1= raw_datasetB[list_subset1+other_cols]
subset2= raw_datasetB[list_subset2+other_cols]
subset3= raw_datasetB[list_subset3+other_cols]
subset1.loc[:,'Cond']='TI_CT_PPHE'
subset2.loc[:,'Cond']='TI_CT'
subset3.loc[:,'Cond']='TI_PPHE'
subset1.rename(dict(zip(list_subset1, list_renamed)), axis=1, inplace=True)
subset2.rename(dict(zip(list_subset2, list_renamed)), axis=1, inplace=True)
subset3.rename(dict(zip(list_subset3, list_renamed)), axis=1, inplace=True)
raw_datasetB2=pd.concat([subset1, subset2, subset3],ignore_index=True)

other_cols=['Protein', 'Gene','Descrip']
subset1= raw_datasetBB[list_subset1+other_cols]
subset2= raw_datasetBB[list_subset2+other_cols]
subset3= raw_datasetBB[list_subset3+other_cols]
subset4= raw_datasetBB[list_subset4+other_cols]
subset5= raw_datasetBB[list_subset5+other_cols]
subset6= raw_datasetBB[list_subset6+other_cols]
subset1.loc[:,'Cond']='TI_CT_PPHE'
subset2.loc[:,'Cond']='TI_CT'
subset3.loc[:,'Cond']='TI_PPHE'
subset4.loc[:,'Cond']='TI_P4000'
subset5.loc[:,'Cond']='TI_CT_CH'
subset6.loc[:,'Cond']='TI_CT_CH_TPP'
subset1.rename(dict(zip(list_subset1, list_renamed)), axis=1, inplace=True)
subset2.rename(dict(zip(list_subset2, list_renamed)), axis=1, inplace=True)
subset3.rename(dict(zip(list_subset3, list_renamed)), axis=1, inplace=True)
subset4.rename(dict(zip(list_subset4, list_renamed)), axis=1, inplace=True)
subset5.rename(dict(zip(list_subset5, list_renamed)), axis=1, inplace=True)
subset6.rename(dict(zip(list_subset6, list_renamed)), axis=1, inplace=True)
raw_datasetBB2=pd.concat([subset1, subset2, subset3,subset4, subset5, subset6],ignore_index=True)

save=False
if save:
    out.to_csv(folder+'Bovine_and_Human2.csv', index=False)
    out2.to_csv(folder+'Bovine_and_Human3.csv', index=False)
    raw_datasetH.to_csv(folder+'Human_ML2.csv', index=False)
    raw_datasetB.to_csv(folder+'Bovine_ML2.csv', index=False)
    raw_datasetH2.to_csv(folder+'Human_ML3.csv', index=False)
    raw_datasetB2.to_csv(folder+'Bovine_ML3.csv', index=False)
    raw_datasetBB2.to_csv(folder+'Bovine_ML4.csv', index=False)
