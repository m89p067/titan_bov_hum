import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import os
os.environ["KERAS_BACKEND"] = "torch"
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.manifold import TSNE
def plot_my_clusters(Xe,labels,names, title, vtitle):
    plt.figure()
    
    # Select the colours of the clusters
    colors = ['red', 'forestgreen', 'yellow','blue','pink','cyan','lime']
    lw = 2
    plt.figure(figsize=(9,7));
    for color, i, target_name in zip(colors, [0, 1, 2,3,4,5], np.unique(target_names)):
        idx=np.where(labels==i)[0]
        plt.scatter(Xe[idx, 0], Xe[idx, 1], color=color, alpha=1., lw=lw, label=target_name);
   
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title);
    plt.xlabel(vtitle + "1")
    plt.ylabel(vtitle + "2")
    plt.show();
    
filename = "D:/Data/bovine_ML.xlsx"

df=pd.read_excel(filename,header=0,index_col=None)
df.Class.value_counts().reset_index(name='Sum of Accidents')


save=False
if save:
    crosstab_result = pd.crosstab(df['Protein'], df['Class'], margins=True, margins_name="Total")
    print(crosstab_result)
    crosstab_result.to_csv('occorrenze.csv')


X = df.iloc[:, :3].to_numpy()  # Features
y=pd.factorize(df.iloc[:, 3])[0].reshape(-1,1) # changing string class label to int
target_names =  df.iloc[:, 3].to_numpy()   # Labels


# Normalize the data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# PCA of the STANDARDIZED DATA
pca_xy = PCA(n_components=2).fit_transform(X_normalized)
with plt.style.context("seaborn-white"):
    fig, ax = plt.subplots()
    ax.scatter(pca_xy[:,0], pca_xy[:,1], c=y, cmap=plt.cm.Set2)
    ax.set_title("PCA | Bovine proteomics [normalized]")
plt.show()
print('Hints:')
print('keras.compile() Compiles the layers into a network.')
print('keras.Sequential() Models a sequential neural network.')
print('keras.Dense() A regular densely-connected NN layer.')
print('tf.keras.Input() Used to instantiate a Keras tensor.')
print('Simulate dimensionality reduction using autoencoders')


# Create an AE and fit it with our data using 2 neurons in the dense layer 
# using keras' functional API

# Get the number of data samples i.e. the number of columns
input_dim = X_normalized.shape[1]
output_dim = 2

# Specify the number of neurons for the dense layers
encoding_dim = 5  # Number of neurons in the hidden layer 

# Specify the input layer
input_features = Input(shape=(input_dim,))

# Add a denser layer as the encode layer following the input layer 
# with 2 neurons and no activation function
encoded = Dense(encoding_dim)(input_features)

# Add a denser layer as the decode layer following the encode layer 
# with output_dim as a parameter and no activation function
decoded = Dense(output_dim)(encoded)

# Create an autoencoder model with
# input as input_features and outputs decoded
autoencoder = Model(inputs=input_features, outputs=decoded)

# Complile the autoencoder model (specify the training configuration)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# View the summary of the autoencoder
autoencoder.summary()

# Get the history of the model to plot
history = autoencoder.fit(X_normalized, y,
                epochs=200,
                batch_size=16,
                shuffle=False,
                validation_split=0.15,
                verbose = 0)

# Plot the loss 
plt.plot(history.history['loss'], color='#FF7E79',linewidth=3, alpha=0.5)
plt.plot(history.history['val_loss'], color='#007D66', linewidth=3, alpha=0.4)
plt.title('Model train vs Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()

# Create a model which has input as input_features and 
# output as encoded
do_enc=True
if do_enc:
    encoder = Model(inputs=input_features, outputs=encoded)
else:
    encoder = autoencoder


# Predict on the entire data using the encoder model, 
# remember to use X_scaled 
encoded_data = encoder.predict(X_normalized)

# Apply TSNE for dimensionality reduction
n_comp=1
tsne = TSNE(n_components=n_comp, perplexity=50, learning_rate=3000, init='pca')
tsne_features = tsne.fit_transform(encoded_data)

if n_comp==2:
    # Plot the TSNE results without label
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], cmap='viridis')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')

    # Plot the TSNE results with label
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=y, cmap='viridis')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(label='Protein Class')
    plt.show()
elif n_comp==1:
    cars=pd.DataFrame({'t-SNE':tsne_features[:, 0],'Class':df.iloc[:, 3]})
    sns.stripplot(x='t-SNE', y='Class', data=cars)
    plt.show()
