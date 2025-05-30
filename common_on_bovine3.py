import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import os
os.environ["KERAS_BACKEND"] = "torch"
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.manifold import TSNE
    
filename = "D:/Parinaz/bovine_ML.xlsx"

df=pd.read_excel(filename,header=0,index_col=None)

X = df.iloc[:, :3].to_numpy()  # Features
y=pd.factorize(df.iloc[:, 3])[0].reshape(-1,1) # changing string class label to int
target_names =  df.iloc[:, 3].to_numpy()   # Labels


# Normalize the data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# PCA of the NORMALIZED DATA
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
encoder = Model(inputs=input_features, outputs=encoded)


# Predict on the entire data using the encoder model, 
# remember to use X_scaled 
encoded_data = encoder.predict(X_normalized)

# Apply TSNE for dimensionality reduction
n_comp=1
tsne = TSNE(n_components=n_comp, perplexity=50, learning_rate=3000, init='pca')
tsne_features = tsne.fit_transform(encoded_data)


my_df=pd.DataFrame({'t-SNE':tsne_features[:, 0],'Class':df.iloc[:, 3]})
sns.stripplot(x='t-SNE', y='Class', data=my_df)
plt.show()
