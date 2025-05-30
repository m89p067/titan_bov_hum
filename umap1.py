import pandas as pd
import numpy as np
import umap
import seaborn as sns
import matplotlib.pyplot as plt

filename = r"D:/Data/bovine_ML.xlsx"
df = pd.read_excel(filename, header=0, index_col=None).iloc[:, :4]

# Prepare data
X = df.iloc[:, :3].to_numpy()  # Features
y = df.iloc[:, 3].to_numpy()   # Labels

# Define combinations of parameters to process
parameter_combinations = [
    {'metric': 'cosine', 'n_neighbors': 5, 'min_dist': 0.5},
    {'metric': 'cosine', 'n_neighbors': 15, 'min_dist': 0.5},
    {'metric': 'cosine', 'n_neighbors': 15, 'min_dist': 0.15},
    {'metric': 'cosine', 'n_neighbors': 5, 'min_dist': 0.15},
    {'metric': 'correlation', 'n_neighbors': 5, 'min_dist': 0.0},
    {'metric': 'correlation', 'n_neighbors': 15, 'min_dist': 0.99},
    {'metric': 'hamming', 'n_neighbors': 5, 'min_dist': 0.99},
    {'metric': 'hamming', 'n_neighbors': 50, 'min_dist': 0.0},
    {'metric': 'hamming', 'n_neighbors': 15, 'min_dist': 0.5},
    {'metric': 'hamming', 'n_neighbors': 50, 'min_dist': 0.5},
    {'metric': 'hamming', 'n_neighbors': 20, 'min_dist': 0.25},
    {'metric': 'jaccard', 'n_neighbors': 5, 'min_dist': 0.0},
    {'metric': 'jaccard', 'n_neighbors': 50, 'min_dist': 0.8},
    {'metric': 'jaccard', 'n_neighbors': 15, 'min_dist': 0.4},
    {'metric': 'dice', 'n_neighbors': 5, 'min_dist': 0.0},
    {'metric': 'dice', 'n_neighbors': 5, 'min_dist': 0.1},
    {'metric': 'dice', 'n_neighbors': 10, 'min_dist': 0.1},
    {'metric': 'dice', 'n_neighbors': 10, 'min_dist': 0.2},
    {'metric': 'russellrao', 'n_neighbors': 50, 'min_dist': 0.1},
    {'metric': 'russellrao', 'n_neighbors': 50, 'min_dist': 0.5},
    {'metric': 'russellrao', 'n_neighbors': 15, 'min_dist': 0.5},
    {'metric': 'russellrao', 'n_neighbors': 15, 'min_dist': 0.1},
    {'metric': 'kulsinski', 'n_neighbors': 5, 'min_dist': 0.0},
    {'metric': 'kulsinski', 'n_neighbors': 5, 'min_dist': 0.5},
    {'metric': 'kulsinski', 'n_neighbors': 15, 'min_dist': 0.2},
    {'metric': 'sokalmichener', 'n_neighbors': 5, 'min_dist': 0.1},
    {'metric': 'sokalmichener', 'n_neighbors': 15, 'min_dist': 0.2},
    {'metric': 'sokalsneath', 'n_neighbors': 50, 'min_dist': 0.1},
    {'metric': 'sokalsneath', 'n_neighbors': 15, 'min_dist': 0.1},
    {'metric': 'sokalsneath', 'n_neighbors': 15, 'min_dist': 0.2},
    {'metric': 'yule', 'n_neighbors': 5, 'min_dist': 0.5},
    {'metric': 'yule', 'n_neighbors': 15, 'min_dist': 0.5},
    {'metric': 'yule', 'n_neighbors': 5, 'min_dist': 0.1}
]

# Loop through each specified combination of parameters
for params in parameter_combinations:
    metric = params['metric']
    n_neighbors = params['n_neighbors']
    min_dist = params['min_dist']
    
    print(f"Processing: Metric={metric}, n_neighbors={n_neighbors}, min_dist={min_dist}")
    try:
        mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric=metric,random_state=42)
        out_umap = mapper.fit_transform(X)

        # Check for NaN values in out_umap
        if np.isnan(out_umap).any():
            print("NaN values in the output array. Skipping plot...")
            continue

        # Plot
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111)
        sns.scatterplot(x=out_umap[:,0], y=out_umap[:,1], hue=y)
        plt.title(f'UMAP (Metric={metric}, n_neighbors={n_neighbors}, min_dist={min_dist})')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

        # Save the plot to a file
        plot_filename = f'umap_plot_metric_{metric}_n_{n_neighbors}_dist_{min_dist}'.replace(".", "-")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # Display the plot
        plt.savefig('D:/Data/umap_fig/'+plot_filename+'.png',dpi=300, bbox_inches='tight')
        plt.close()
        np.save('D:/Data/umap_fig/'+plot_filename, out_umap, allow_pickle=True)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error occurred: {e}")
        continue
