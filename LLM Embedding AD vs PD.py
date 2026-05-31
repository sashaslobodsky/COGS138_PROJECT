#imports
import anndata as ad
import os
import scanpy as sc
import pandas as pd
os.listdir('.')
meta = pd.read_csv('GSE174367_snRNA-seq_cell_meta.csv.gz')
print(meta.head())
adata = sc.read_10x_h5('GSE174367_snRNA-seq_filtered_feature_bc_matrix.h5')
#fixing warning regarding no unique names
adata.var_names_make_unique()
print(adata.var_names.is_unique)
print(adata)
PD_meta = pd.read_csv('IPDCO_hg_midbrain_cell.tsv', sep='\t')
print(PD_meta.head())

#making sure AD metadata line up with AD matrix

print(adata.shape)
print(meta.shape)
print(adata.obs_names[:5])
print(meta['Barcode'].head())

print(meta.columns)
print(PD_meta.columns)


#importing metadata to the AData object
meta = meta.set_index('Barcode')
common_cells = adata.obs_names.intersection(meta.index)
adata = adata[common_cells].copy()
adata.obs = meta.loc[common_cells]

print(adata.obs.head())
print(adata.obs['Diagnosis'].value_counts())

pd_count_pre = pd.read_csv('IPDCO_hg_midbrain_UMI.tsv',
                        sep = '\t',
                        nrows = 5
)


print(pd_count_pre.head())
print(pd_count_pre.shape)
print(pd_count_pre.index[:5])
print(pd_count_pre.columns[:5])
print(PD_meta['barcode'].head())

pd_genes = pd.read_csv('IPDCO_hg_midbrain_genes.tsv',
                     sep = '\t',
)
print(pd_genes.head())
print(pd_genes.shape)

# AD preprocessing
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata_hvg = adata[:, adata.var["highly_variable"]].copy()

sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg)
sc.pp.neighbors(adata_hvg)
sc.tl.umap(adata_hvg)

sc.pl.umap(adata_hvg, color="Diagnosis")
sc.pl.umap(adata_hvg, color="Cell.Type")

print("AD preprocessing done!")

pd_count_sub = pd.read_csv('IPDCO_hg_midbrain_UMI.tsv',
                           sep = '\t',
                           nrows = 5000,
                           usecols = range(10000),
                           dtype = 'float32'
                           )
print(pd_count_sub.shape)
print('small pd subset loaded')

#creating temp PD AnnData because it keeps crashing\
pd_gene_sub = pd_genes.iloc[:5000].copy()
PD_meta_indexed = PD_meta.set_index("barcode")

pd_adata_sub = ad.AnnData(
    X=pd_count_sub.T.values,
    obs=PD_meta_indexed.loc[pd_count_sub.columns],
    var=pd_gene_sub
)

pd_adata_sub.var_names = pd_gene_sub["gene"].astype(str).values
pd_adata_sub.var_names_make_unique()
pd_adata_sub.obs["Disease"] = "PD"

print(pd_adata_sub)

#loading PD as anndata
# print('starting full PD count load..')#trying to see if its fully loading the matrix

# pd_count = pd.read_csv('IPDCO_hg_midbrain_UMI.tsv',
#                        sep = '\t',
#                        dtype = 'float32'#because file is too big use float32 to reduce memory
#                        )
# print('finsished full PD count load')
# pd_genes = pd.read_csv('IPDCO_hg_midbrain_genes.tsv',
#                       sep = '\t'
#                        )
# PD_meta = PD_meta.set_index('barcode')

# pd_adata = ad.AnnData(
#     X = pd_count.T.values,
#     obs = PD_meta.loc[pd_count.columns],
#     var = pd_genes
# )

# pd_adata.var_names = pd_genes['gene'].astype(str).values
# pd_adata.var_names_make_unique()
# print(pd_adata)
# print(pd_adata.obs.head())
# print(pd_adata.var.head())

# #adding disease labels now to PD
# pd_adata.obs['Diagnosis']='PD'

# ad_adata = adata
# #matching column names for ad and pd
# ad_adata.obs['Disease']= ad_adata.obs['Diagnosis']
# pd_adata.obs['Disease']= 'PD'

# #finding overlap of genes in AD and PD
# com_genes = ad_adata.var_names.intersection(pd_adata.var_names)

# ad_adata = ad_adata[:,com_genes].copy()
# pd_adata = pd_adata[:,com_genes].copy()

# print(ad_adata.shape)
# print(pd_adata.shape)

#combining the shared genes into one dataset
# combine = ad.concat(
#     [ad_adata, pd_adata],
#     label="dataset",
#     keys=["AD_dataset", "PD_dataset"]
# )

# print(combine)
# print(combine.obs["Disease"].value_counts())

print('Done!') 