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

tipo=['tSNE','UMAP']
outB=pd.read_csv('B_sel_'+tipo[1]+'_outl.csv', index_col=False)
outH=pd.read_csv('H_sel_'+tipo[1]+'_outl.csv', index_col=False)
cond=['Ti_CT_CH_TPP_PPHE','Ti_CT']

#outH['Cond'] = outH['Cond'].map({'TI_PPHE': 'Ti_CT_CH_TPP_PPHE','TI_CT_PPHE':'pippo','TI_CT':'Ti_CT'})
#outH['Type'] = outH['Type'].map({'H': 'hBM-MSCs','B':'FBS'})
#outH = outH[outH['Cond']!='pippo'].reset_index()

outB['Cond'] = outB['Cond'].map({'TI_PPHE': 'Ti_CT_CH_TPP_PPHE','TI_CT_PPHE':'pippo','TI_CT':'Ti_CT'})
outB['Type'] = outB['Type'].map({'H': 'hBM-MSCs','B':'FBS'})
outB = outB[outB['Cond']!='pippo'].reset_index()

outH=outB.copy()

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
colori=['darkcyan','yellow']
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
plt.savefig('WC_DescrB.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()
