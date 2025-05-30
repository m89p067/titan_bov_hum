import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import pdb

mypath="D:/Parinaz/Bovine data for ML/"
appended_data = []
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for i in onlyfiles:
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
appended_data.to_excel('bovine_ML.xlsx', index=False)
