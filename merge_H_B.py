import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
filenameB='bovine_ML.xlsx'
dfB=pd.read_excel(filenameB,header=0,index_col=None)
dfB['Specie'] = 'B'
mypath="D:/Data/Umani/"
appended_data = []
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for i in onlyfiles:
    save=False
    if '.pkl' in i:
        print('Processing:'+i)
        df=pd.read_pickle(mypath+i)
        df2=df.iloc[: , :3]
        df2.columns = ['Cell Line 1', 'Cell Line 2', 'Cell Line 3']
        classe=i.split('_')[1:-1]
        df2['Class']='-'.join(classe)
        df2['Protein']=df['Protein']
        df2['Description']=df['Description']
        appended_data.append(df2)
appended_data = pd.concat(appended_data)
appended_data.reset_index(drop=True,inplace=True)
save=False
if save:
    appended_data.to_excel('Human_ML.xlsx', index=False)
appended_data['Specie'] = 'H'
appended_data['Class'] = appended_data['Class'].map({'oxidized':'TI-CT', 'oxidized-titanium-with':'TI-CT-PPHE',
       'oxidized-titanium-with-chitosan-and-crosslinker-and':'TI-PPHE'})
out=pd.concat([dfB, appended_data],ignore_index=True)
out['GN'] = out['Description'].apply(lambda st: st[st.find("GN=")+3:st.find("PE=")])
values = ['TI-CT-CH', 'TI-CT-CH-TPP',  'TI-P4000']
out= out[out.Class.isin(values) == False]
out = out.reset_index(drop=True)
out.to_csv('Bovine_and_Human.csv', index=False)
