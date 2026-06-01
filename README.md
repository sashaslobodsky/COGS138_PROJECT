# COGS138_PROJECT

Final project analysis for COGS 138.

The completed notebook is:

- `Final_Project_Alzheimers_AD_vs_Control.ipynb`

The notebook analyzes the Alzheimer disease single-nucleus RNA-seq dataset
GSE174367 using donor-level pseudobulk expression profiles, PCA/UMAP
visualization, cell-type composition, and leave-one-donor-out classification.

## Results summary

The final analysis uses 18 donors from GSE174367: 11 Alzheimer disease donors
and 7 control donors, totaling 61,472 matched nuclei. Because diagnosis is a
donor-level label, nuclei were aggregated into donor-level pseudobulk expression
profiles before classification.

The clearest result is that AD and control samples show partial transcriptomic
separation, but the signal is not strong enough to claim a robust diagnostic
classifier from this small dataset.

- Logistic regression on PCA features: 0.667 accuracy, 0.649 balanced accuracy.
- Random forest on highly variable genes: 0.611 accuracy, 0.578 balanced
  accuracy.
- The PCA logistic regression baseline performed better than the random forest,
  suggesting that a low-dimensional expression representation captured more
  useful disease signal than the more flexible tree model in this small-sample
  setting.

Cell-type composition also differed somewhat by diagnosis. Excitatory neurons
had a higher mean proportion in AD donors than controls in this dataset
(`p = 0.039` by exploratory Welch t-test). Other broad cell types were not
clearly different at this level.

The top exploratory differentially expressed highly variable genes included
`ADCY10`, `AC018742.1`, `CNP`, `EGF`, `C5orf64`, `LINC01608`, `HSPH1`, `RHOG`,
`GREB1L`, and `OPALIN`. These should be interpreted as candidate signals from
this dataset, not validated biomarkers, because there are only 18 donors.

## Writeup guidance

The notebook already contains the main writeup sections needed for submission:
research question, background, workflow, figures, classifier results,
interpretation, limitations, and future work.

The most important conclusion to state in a presentation or final explanation is:
single-nucleus expression profiles contain some AD-versus-control structure, but
the small number of donors, possible batch/age/sex/cell-type confounds, and the
absence of the originally proposed scGPT comparison mean the results should be
presented as a transparent classical baseline rather than a final foundation
model result.

To reproduce locally:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/final_ad_analysis.py
python3 scripts/build_final_notebook.py
```

Raw and processed data are ignored by Git because the expression matrix is large.
