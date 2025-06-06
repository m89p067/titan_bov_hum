# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import re
from google.colab import drive
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
!pip install https://github.com/Phlya/adjustText/archive/master.zip
from functools import reduce
!pip install wordcloud
from wordcloud import (WordCloud, get_single_color_func)
#!pip install matplotlib_set_diagrams #https://github.com/paulbrodersen/matplotlib_set_diagrams
#from matplotlib_set_diagrams import VennDiagram
!pip install matplotlib_venn_wordcloud #https://github.com/paulbrodersen/matplotlib_venn_wordcloud
from matplotlib_venn_wordcloud import venn3_wordcloud,venn2_wordcloud
drive.mount('/content/drive/', force_remount=False)
# %cd /content/drive/MyDrive/Ortho_BOV_HUMAN/

#!pip install nltk
#import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('all')
#nltk.download("popular")

cols=['Protein','Gene','Descrip','Type','Cond']
#cond=['TI_CT_PPHE', 'TI_CT', 'TI_PPHE'] # ,'TI_P4000','TI_CT_CH','TI_CT_CH_TPP']
cell_lines=['Cell Line1', 'Cell Line2', 'Cell Line3']

"""**Most frequent keywords among outliers**"""

tipo=['tSNE','UMAP']
outB=pd.read_csv('B_sel_'+tipo[1]+'_outl.csv', index_col=False)
outH=pd.read_csv('H_sel_'+tipo[1]+'_outl.csv', index_col=False)
cond=['Ti_CT_CH_TPP_PPHE','Ti_CT']
outH['Cond'] = outH['Cond'].map({'TI_PPHE': 'Ti_CT_CH_TPP_PPHE','TI_CT_PPHE':'pippo','TI_CT':'Ti_CT'})
outH['Type'] = outH['Type'].map({'H': 'hBM-MSCs','B':'FBS'})
outH = outH[outH['Cond']!='pippo'].reset_index()

outH

def flatten(xss):
  myout=[]
  for xs in xss:
      for x in xs:
        if x != "":
          #x=x.lower()
          myout.append(x.replace(',', ''))
  return myout
def compare_list_of_words__to_another_list_of_words(from_strA, to_strB):
  fromA = list(set(from_strA))
  for word_to_match in fromA:
    totalB = len(to_strB)
    number_of_match = (to_strB).count(word_to_match)
    data = str((((to_strB).count(word_to_match))/totalB)*100)
    print('words: -- ' + word_to_match + ' --' + '\n'
    '       number of match    : ' + number_of_match + ' from ' + str(totalB) + '\n'
    '       percent of match   : ' + data + ' percent')

words_no_want=['protein','chain','Protein','Isoform',  'isoform','subunit','A','B','I','II','of','beta','alpha','cognate',
               '1','2','3','4','5','6','7','8','9','type','71','beta/alpha','theta','factor','type-2','alpha-1B','S9','10','F',
                 '90-alpha', 'acidic', 'Beta','Short','U', 'beta-4', '14-3-3','105','14','Heterogeneous','90-beta','beta-6','78','S3','kDa',
               'cytoskeletal']

text_1=outH.loc[outH['Cond'] == cond[0], 'Descrip'].tolist()
text_2=outH.loc[outH['Cond'] == cond[1], 'Descrip'].tolist()
#text_3=outH.loc[outH['Cond'] == cond[2], 'Descrip'].tolist()
text_1h=[text.partition('OS=')[0] for text in text_1]
text_2h=[text.partition('OS=')[0] for text in text_2]
#text_3h=[text.partition('OS=')[0] for text in text_3]
# Tokenize strings.
tokens1 = [text.split(' ') for text in text_1h]
tokens2 = [text.split(' ') for text in text_2h]
#tokens3 = [text.split(' ') for text in text_3h]
tokens1 =set(flatten(tokens1)) - set(words_no_want)
tokens2 =set( flatten(tokens2))- set(words_no_want)
#tokens3 = set(flatten(tokens3 ))- set(words_no_want)

scaler_factor=1000
def normalize(xval,x_min,x_max):
  return  (xval-x_min)/(x_max-x_min)
def normalize2(xval, lower_bound,upper_bound,x_min,x_max):
  valore=lower_bound + (xval - x_min) * (upper_bound - lower_bound) / (x_max- x_min)
  return valore
#colori=['navy','dimgray','darkorange']
colori=['navy','darkorange']
#set_list=[tokens1,tokens2,tokens3]
set_list=[tokens1,tokens2]
#word_to_frequency =dict(Counter(tokens1)+Counter(tokens2)+Counter(tokens3))
word_to_frequency =dict(Counter(tokens1)+Counter(tokens2))
count_freq=list(word_to_frequency.values())
word_to_frequency.update((x, normalize2(y,scaler_factor*min(count_freq),scaler_factor*max(count_freq),min(count_freq),max(count_freq))) for x, y in word_to_frequency.items())

def color_func(word, *args, **kwargs):
  if word in tokens1:
    # return "#000080" # navy blue
    return colori[0]
  elif word in tokens2:
    # return "#8b0000" # red4
    return colori[1]
  elif word in tokens3:
    return colori[2]
  else:
    return "black" # gray6 (aka off-black)
f = lambda x: x.replace("_", "-")
cond_names=list(map(f, cond))
fig, ax = plt.subplots(1,1)
#ax.set_title("UMAP Outliers gene descriptions word counts", fontsize=36)
#venn3_wordcloud(set_list,set_labels=cond_names,
venn2_wordcloud(set_list,set_labels=cond_names,
                set_edgecolors=colori,
                word_to_frequency=word_to_frequency ,
                #word_to_frequency=count_dict ,
                wordcloud_kwargs=dict(color_func=color_func, relative_scaling=.85),
                #max_font_size=10, min_font_size=6),
                ax=ax)
#plt.savefig('WC_Descr.png',dpi=300,bbox_inches='tight')
plt.savefig('WC_Descr2.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()

#count_dict=dict(Counter(tokens1)+Counter(tokens2)+Counter(tokens3))
count_dict=dict(Counter(tokens1)+Counter(tokens2))
filtered_dict = {key: value for key, value in count_dict.items() if value > 1}
for keyz in filtered_dict:
  print('Working on keyword: ',keyz)
  trovata=False
  for s in (list(set(text_1h))):
    if (keyz in s):
      riga=outH.loc[(outH['Descrip'].str.contains(s) ) & (outH['Cond'].str.contains(cond[0]))].drop_duplicates()
      #riga=outH.loc[(outH['Descrip'].apply(lambda x: keyz in x)) & (outH['Cond']==cond[0])]
      for index, row in riga.iterrows():
        print('Exp :',cond[0],' -  Gene :',row['Gene'],' - keyword :',keyz,'  - description :',row['Descrip'].partition('OS=')[0])

      trovata=True
  for s in (list(set(text_2h))):
    if (keyz in s):
      riga=outH.loc[(outH['Descrip'].str.contains(s) ) & (outH['Cond'].str.contains(cond[1]))].drop_duplicates()
      #riga=outH.loc[(outH['Descrip'].apply(lambda x: keyz in x)) & (outH['Cond']==cond[1])]
      for index, row in riga.iterrows():
        print('Exp :',cond[1],' -  Gene :',row['Gene'],' - keyword :',keyz,'  - description :',row['Descrip'].partition('OS=')[0])

      trovata=True
#  for s in (list(set(text_3h))):
#    if (keyz in s):
#      riga=outH.loc[(outH['Descrip'].str.contains(s) ) & (outH['Cond'].str.contains(cond[2]))].drop_duplicates()
#      #riga=outH.loc[(outH['Descrip'].apply(lambda x: keyz in x)) & (outH['Cond']==cond[2])]
#      for index, row in riga.iterrows():
#        print('Exp :',cond[2],' -  Gene :',row['Gene'],' - keyword :',keyz,'  - description :',row['Descrip'].partition('OS=')[0])

      trovata=True
  if trovata:
    print('\n')

genes1,genes2,genes3=[],[],[]
#filtered_dict = {key: value for key, value in count_dict.items() if value >= 1}
for keyz in count_dict:
  for s in (list(set(text_1h))):
    if (keyz in s):
      riga=outH.loc[(outH['Descrip'].str.contains(s) ) & (outH['Cond'].str.contains(cond[0]))].drop_duplicates()
      #riga=outH.loc[(outH['Descrip'].apply(lambda x: keyz in x)) & (outH['Cond']==cond[0])]
      for index, row in riga.iterrows():
        #print('Exp :',cond[0],' -  Gene :',row['Gene'],' - keyword :',keyz,'  - description :',row['Descrip'].partition('OS=')[0])
        genes1.append(row['Gene'])

  for s in (list(set(text_2h))):
    if (keyz in s):
      riga=outH.loc[(outH['Descrip'].str.contains(s) ) & (outH['Cond'].str.contains(cond[1]))].drop_duplicates()
      #riga=outH.loc[(outH['Descrip'].apply(lambda x: keyz in x)) & (outH['Cond']==cond[1])]
      for index, row in riga.iterrows():
        #print('Exp :',cond[1],' -  Gene :',row['Gene'],' - keyword :',keyz,'  - description :',row['Descrip'].partition('OS=')[0])
        genes2.append(row['Gene'])

#  for s in (list(set(text_3h))):
#    if (keyz in s):
#      riga=outH.loc[(outH['Descrip'].str.contains(s) ) & (outH['Cond'].str.contains(cond[2]))].drop_duplicates()
#      #riga=outH.loc[(outH['Descrip'].apply(lambda x: keyz in x)) & (outH['Cond']==cond[2])]
#      for index, row in riga.iterrows():
#        #print('Exp :',cond[2],' -  Gene :',row['Gene'],' - keyword :',keyz,'  - description :',row['Descrip'].partition('OS=')[0])
#        genes3.append(row['Gene'])

def color_func2(word, *args, **kwargs):
  if word in genes1:
    # return "#000080" # navy blue
    return colori[0]
  elif word in genes2:
    # return "#8b0000" # red4
    return colori[1]
  elif word in genes3:
    return colori[2]
  else:
    return "black" # gray6 (aka off-black)
#gene_to_frequency =dict(Counter(set(genes1))+Counter(set(genes2))+Counter(set(genes3)))
gene_to_frequency =dict(Counter(set(genes1))+Counter(set(genes2)))
count_freq=list(gene_to_frequency.values())
scaler_factor=1000
gene_to_frequency.update((x, normalize2(y,scaler_factor*min(count_freq),scaler_factor*max(count_freq),min(count_freq),max(count_freq))) for x, y in gene_to_frequency.items())
fig, ax = plt.subplots(1,1)
#ax.set_title("UMAP Outliers gene names counts", fontsize=36)
#venn3_wordcloud([set(genes1),set(genes2),set(genes3)],set_labels=cond_names,
venn2_wordcloud([set(genes1),set(genes2)],set_labels=cond_names,
                set_edgecolors=colori,
                    #word_to_frequency=gene_to_frequency ,
                    wordcloud_kwargs=dict(color_func=color_func2,relative_scaling=0.65),
                                          #max_font_size=10, min_font_size=6),
                    ax=ax)
#plt.savefig('WC_Genes.png',dpi=300,bbox_inches='tight')
plt.savefig('WC_Genes2.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()

"""# **Only one keyword**"""

text_1=outH.loc[outH['Cond'] == cond[0], 'Descrip'].tolist()
text_2=outH.loc[outH['Cond'] == cond[1], 'Descrip'].tolist()
#text_3=outH.loc[outH['Cond'] == cond[2], 'Descrip'].tolist()
text_1h=[text.partition('OS=')[0] for text in text_1]
text_2h=[text.partition('OS=')[0] for text in text_2]
#text_3h=[text.partition('OS=')[0] for text in text_3]
# Tokenize strings.
tokens1 = [text.split(' ') for text in text_1h]
tokens2 = [text.split(' ') for text in text_2h]
#tokens3 = [text.split(' ') for text in text_3h]
tokens1 =set(flatten(tokens1))
tokens2 =set( flatten(tokens2))
#tokens3 = set(flatten(tokens3 ))

#count_dict=dict(Counter(tokens1)+Counter(tokens2)+Counter(tokens3))
count_dict=dict(Counter(tokens1)+Counter(tokens2))
filtered_dict = {key: value for key, value in count_dict.items() if value == 1}
for keyz in filtered_dict:
  print('Working on keyword: ',keyz)
  trovata=False
  for s in (list(set(text_1h))):
    if (keyz in s):
      riga=outH.loc[(outH['Descrip'].str.contains(s) ) & (outH['Cond'].str.contains(cond[0]))].drop_duplicates()
      #riga=outH.loc[(outH['Descrip'].apply(lambda x: keyz in x)) & (outH['Cond']==cond[0])]
      for index, row in riga.iterrows():
        print('Exp :',cond[0],' -  Gene :',row['Gene'],' - keyword :',keyz,'  - description :',row['Descrip'].partition('OS=')[0])

      trovata=True
  for s in (list(set(text_2h))):
    if (keyz in s):
      riga=outH.loc[(outH['Descrip'].str.contains(s) ) & (outH['Cond'].str.contains(cond[1]))].drop_duplicates()
      #riga=outH.loc[(outH['Descrip'].apply(lambda x: keyz in x)) & (outH['Cond']==cond[1])]
      for index, row in riga.iterrows():
        print('Exp :',cond[1],' -  Gene :',row['Gene'],' - keyword :',keyz,'  - description :',row['Descrip'].partition('OS=')[0])

      trovata=True
#  for s in (list(set(text_3h))):
#    if (keyz in s):
#      riga=outH.loc[(outH['Descrip'].str.contains(s) ) & (outH['Cond'].str.contains(cond[2]))].drop_duplicates()
#      #riga=outH.loc[(outH['Descrip'].apply(lambda x: keyz in x)) & (outH['Cond']==cond[2])]
#      for index, row in riga.iterrows():
#        print('Exp :',cond[2],' -  Gene :',row['Gene'],' - keyword :',keyz,'  - description :',row['Descrip'].partition('OS=')[0])

      trovata=True
  if trovata:
    print('\n')
