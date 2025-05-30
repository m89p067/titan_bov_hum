import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
def log2_with_epsilon(arr,EPSILON=1e-10):
    result = np.log2(arr + EPSILON)
    return result
def log10_with_epsilon(arr,EPSILON=1e-10):
    result = np.log10(arr + EPSILON)
    return result
import pdb
print('Data created by merge_H_B_ver1.py')
folder='D:/Parinaz/new_img/'
filenameH='Human_ML2'
filenameB='Bovine_ML2'
filenameBH='Bovine_and_Human2'

filenameHs='Human_ML3'
filenameBs='Bovine_ML3'
filenameBHs='Bovine_and_Human3'

filenameB_FC='BovineFC.csv'
cell_lines=['Cell Line1', 'Cell Line2', 'Cell Line3']
    
save_plot=False
if save_plot: # CREATES THE FILE 
    merged_df=pd.read_csv(folder+filenameBHs+'.csv',index_col=False,header=0)
    tmp=pd.melt(merged_df, id_vars=['Protein', 'Gene', 'Descrip',
           'Type', 'Cond'], value_vars=cell_lines,  var_name='Cell Line', value_name='Expression',ignore_index=True)
    
    # Before LOG-TRANSFORM
    g = sns.FacetGrid(tmp, col="Type",hue="Cond")
    g.map(sns.kdeplot, "Expression")
    g.set(xlim=(0, 40))
    g.add_legend()
    plt.savefig(folder+'KDE_log.png',dpi=300,bbox_inches='tight')
    plt.close()
    sns.stripplot(data=tmp, x="Expression", y="Cond", hue="Type")
    plt.savefig(folder+'Strip_log.png',dpi=300,bbox_inches='tight')
    plt.close()
    tmp.to_csv(folder+'Bovine_and_Human4.csv', index=False)    
    dataH=pd.read_csv(folder+filenameH+'.csv',index_col=False,header=0)
    dataB=pd.read_csv(folder+filenameB+'.csv',index_col=False,header=0)
    #list_subset1=['TI_CT_PPHE_1', 'TI_CT_PPHE_2', 'TI_CT_PPHE_3']
    list_subset2=['TI_CT_1', 'TI_CT_2','TI_CT_3']
    list_subset3=['TI_PPHE_1', 'TI_PPHE_2', 'TI_PPHE_3']
    #list_avg=['TI_PPHE','TI_CT_PPHE','TI_CT']
    list_avg=['Ti-CT-CH-TPP-PPHE','TI_CT']
    other_cols=['Protein', 'Gene','Descrip', 'Type']
    for i in list_avg:
        x=dataH.loc[:,i].to_numpy()
        print( 'HUMAN:'+i,
        stats.kurtosis(x, bias=False), pd.DataFrame(x).kurtosis()[0],
        stats.skew(x, bias=False), pd.DataFrame(x).skew()[0],
        sep='\n')
    for i in list_avg:
        x=dataB.loc[:,i].to_numpy()
        print( 'BOVINE:'+i,
        stats.kurtosis(x, bias=False), pd.DataFrame(x).kurtosis()[0],
        stats.skew(x, bias=False), pd.DataFrame(x).skew()[0],
        sep='\n')
else:
    filenameBHs='Bovine_and_Human4'
    init_df=pd.read_csv(folder+filenameBHs+'.csv',index_col=False,header=0)
    init_df['Cond'] = init_df['Cond'].map({'TI_PPHE': 'Ti_CT_CH_TPP_PPHE','TI_CT_PPHE':'pippo','TI_CT':'Ti_CT'})
    init_df['Type'] = init_df['Type'].map({'H': 'hBM-MSCs','B':'FBS'})
    print('Initially:',init_df.shape)
    merged_df = init_df[init_df['Cond']!='pippo'].reset_index()
    print('Then:',merged_df.shape)
    y = [i2+' '+y2 for i2,y2 in zip(merged_df["Cond"],merged_df["Type"])]
    merged_df['Dataset']=y
    un_y=(list(set(y)))
    for i in un_y:
        idx=[i2 for i2 in range(len(y)) if y[i2] == i]
        x = merged_df.loc[idx,"Expression"].to_numpy()
        print( 'DATASET: '+i,
        'KURTOSIS:',stats.kurtosis(x, bias=False), 'KURTOSIS (2):',pd.DataFrame(x).kurtosis()[0],
        'SKEWNESS:',stats.skew(x, bias=False), 'SKEWNESS (2):',pd.DataFrame(x).skew()[0],
        sep='\n')
        print('\n')
    g=sns.kdeplot(data=merged_df, x="Expression", hue="Dataset")
    g.set(xlim=(0, 40))
    #g.add_legend()
    plt.savefig(folder+'KDE_log2.png',dpi=300,bbox_inches='tight')
    plt.close()
    
    B_df=pd.read_csv(folder+'Bovine_ML4'+'.csv',index_col=False,header=0)
    y = B_df["Cond"].tolist()    
    un_y=(list(set(y)))
    tmp=pd.melt(B_df, id_vars=['Protein', 'Gene', 'Descrip',
           'Cond'], value_vars=cell_lines,  var_name='Cell Line', value_name='Expression',ignore_index=True)
    for i in un_y:
        idx=[i2 for i2 in range(len(y)) if y[i2] == i]
        x = tmp.loc[idx,"Expression"].to_numpy()
        print( 'DATASET: '+i,
        'KURTOSIS:',stats.kurtosis(x, bias=False), 'KURTOSIS (2):',pd.DataFrame(x).kurtosis()[0],
        'SKEWNESS:',stats.skew(x, bias=False), 'SKEWNESS (2):',pd.DataFrame(x).skew()[0],
        sep='\n')
        print('\n')
    g=sns.kdeplot(data=tmp, x="Expression", hue="Cond")
    g.set(xlim=(0, 40))
    #g.add_legend()
    plt.savefig(folder+'KDE_logBov.png',dpi=300,bbox_inches='tight')
    plt.close()
