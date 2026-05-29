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

#loading PD as anndata
pd_count = pd.read_csv('IPDCO_hg_midbrain_UMI.tsv',
                       sep = '\t',
                       dtype = 'float32'#because file is too big use float32 to reduce memory
                       )
pd_genes = pd.read_csv('IPDCO_hg_midbrain_genes.tsv',
                       sep = '\t'
                       )
PD_meta = PD_meta.set_index('barcode')

pd_adata = ad.AnnData(
    X = pd_count.T.values,
    obs = PD_meta.loc[pd_count.columns],
    var = pd_genes
)
pd_adata.var_names = pd_genes['genes'].astype(str).values
pd_adata.var_names_make_unique()
print(pd_adata)
print(pd_adata.obs.head())
print(pd_adata.var.head())

