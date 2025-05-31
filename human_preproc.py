import pandas as pd
import numpy as np
print('HUMAN and BOVINE RAW Datasets')
folder='D:/Data/'
filename='AreeNormOdonto_08112021.xlsx' # Filename of the proteomic dataset
raw_dataset=pd.read_excel(folder+filename)
raw_dataset[['idx','Protein1']] = raw_dataset['Peak Name'].str.split('|',n=1,expand=True)
raw_dataset[['Protein','Descrip']] = raw_dataset['Protein1'].str.split('|',n=1,expand=True)

colonne={'ODONTO_TICTCHTPPPPHE_100nguL_03112021_SWATH1':'TI_PPHE_1',
       'ODONTO_TICTCHTPPPPHE_100nguL_03112021_SWATH2':'TI_PPHE_2',
       'ODONTO_TICTCHTPPPPHE_100nguL_03112021_SWATH3':'TI_PPHE_3',
       'ODONTO_TICTP_200nguL_04112021_SWATH1':'TI_CT_PPHE_1',
       'ODONTO_TICTP_200nguL_04112021_SWATH2':'TI_CT_PPHE_2',
       'ODONTO_TICTP_200nguL_04112021_SWATH3':'TI_CT_PPHE_3',
       'ODONTO_TICT_200nguL_05112021_SWATH1':'TI_CT_1',
       'ODONTO_TICT_200nguL_05112021_SWATH2':'TI_CT_2',
       'ODONTO_TICT_200nguL_05112021_SWATH3':'TI_CT_3'}

raw_dataset['Description'] = raw_dataset['Group'] + raw_dataset['Descrip']
raw_dataset = raw_dataset.drop(columns=['Peak Name','idx', 'Protein1','Descrip','Group'])
raw_dataset.rename(colonne,  axis = "columns", inplace = True)
raw_dataset['Description'] = raw_dataset['Description'].map(lambda x: x.lstrip('sp|'))



new1=raw_dataset[['TI_CT_PPHE_1', 'TI_CT_PPHE_2', 'TI_CT_PPHE_3','Protein', 'Description']].copy()
new4=raw_dataset[['TI_CT_1', 'TI_CT_2', 'TI_CT_3','Protein', 'Description']].copy()
new5=raw_dataset[['TI_PPHE_1', 'TI_PPHE_2', 'TI_PPHE_3','Protein', 'Description']].copy()

print('Saving Human data')

new4.to_pickle(folder+"H_oxidized_titanium.pkl") #TI_CT
new1.to_pickle(folder+"H_oxidized_titanium_with_polyphenol.pkl") #TI_CT_PPHE
new5.to_pickle(folder+"H_oxidized_titanium_with_chitosan_and_crosslinker_and_polyphenol.pkl") #TI_PPHE

