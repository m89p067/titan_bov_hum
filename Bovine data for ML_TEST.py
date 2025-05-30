import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

filename='bovine_ML.xlsx'
df=pd.read_excel(filename,header=0,index_col=None).iloc[: , :4]
prot_name=pd.read_excel(filename,header=0,index_col=None,usecols=['Protein'])
df2=df.copy()
X=df.iloc[: , :3].to_numpy() # The DATA (3 cell lines or also called features)
y=df.iloc[:,3].to_numpy()    #The LABELS (6 experimental conditions)
label=pd.factorize(df.Class)[0]
X_pca = PCA(n_components=2).fit(X)
X_reduced=X_pca.transform(X) # Reduced dataset to 2 components
expl_var=X_pca.explained_variance_ratio_
df_view=pd.DataFrame(X_reduced,columns=['PC 1','PC 2'])
df_view['Class']=df['Class']
df2.columns = [c.replace(' ', '_') for c in df2.columns]
fig = plt.figure(figsize=(8,16))
ax = fig.add_subplot(2, 1, 1, projection='3d')
for s in df.Class.unique():
    ax.scatter(df2.Cell_Line_1[df2.Class==s],df2.Cell_Line_2[df2.Class==s],df2.Cell_Line_3[df2.Class==s],label=s)
ax.set_title('Proteomic data')
ax.set_xlabel('Cell Line 1')
ax.set_ylabel('Cell Line 2')
ax.set_zlabel('Cell Line 3') 
ax.legend(loc='center left', bbox_to_anchor=(1.05, 1.0))
ax = fig.add_subplot(2, 1, 2)
sns.scatterplot(data=df_view, x="PC 1", y="PC 2", hue="Class",ax=ax)
ax.set_title('Principal component analysis')
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
plt.show()
