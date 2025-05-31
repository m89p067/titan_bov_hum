# Effect of titanium treatments on Human vs. Bovine proteomics: insights from data science
Partially embedded in *Polyphenols Promote Stem Cell Surface Adaptation onto low-adhesion Chitosan-Titanium Through Protein Synthesis, Cytoskeletal Remodeling, and Potential Anti-Inflammatory Effects*

The collection encloses a data science investigation of the dataset:

File *Bovine data for ML_TEST.py* performs PCA on *bovine_ML.xlsx*
File *bovine_for_ML.py* creates *bovine_ML.xlsx*
File *common_on_bovine2.py* performs ML tests on *bovine_ML.xlsx* 
File *common_on_bovine3.py* runs ML-related analysis on *bovine_ML.xlsx* 
File *dim_red1.py* is a routine for dimensionality reduction and stats calculation on Human or Bovine data on data in *Human_ML2,Bovine_ML2,Bovine_and_Human2,Human_ML3,Bovine_ML3,Bovine_and_Human3,BovineFC,Bovine_and_Human4,Bovine_ML4* csv files
File *dim_red2.py* runs clustering and visualizations on reduced data 
File *dim_red3.py* for clustering results evalution
File *human_preproc.py* contains preprocessing and reorganization of the raw data
File *merge_H_B.py* is for additional data management script
File *merge_H_B_ver1.py* script for data restructuration
File *ortho_bov_human_ver1.py* analyzes data for anomaly detection and wordcloud visualization (data *Bovine_and_Human3*)
File *ortho_bov_human_ver2.py* is for gene ontology (data in *Bovine_and_Human3*)
File *ortho_bov_human_ver3.py* is a script for gene ontology terms summarization (data is the output from *ortho_bov_human_ver2.py*)
File *orthologs_ver1.py* script of orthologs computation
File *orthologs_ver2.py* manages orthologous genes outcomes can creates tables
File *orto_study_PCA.py* is for outliers analysis and visualization (data *BovineFC, Bovine_ML4, Bovine_and_Human4*)
File *umap1.py* contains a UMAP dimensionality reduction analuysis on *bovine_ML.xlsx*
File *umap2.py* tests clustering on umap based low dimensional embeddings (data *bovine_ML.xlsx*)
File *venn_bovine.py* creates Venn diagrams on low-dimensional embeddings outliers
