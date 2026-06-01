from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Final_Project_Alzheimers_AD_vs_Control.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

nb.cells = [
    md(
        """
        # COGS 138 Final Project: Alzheimer Disease Transcriptomics

        **Research question:** Can single-nucleus RNA-seq expression profiles distinguish Alzheimer disease (AD)
        donors from healthy controls, and which biological signals support that separation?

        The original project proposal aimed to compare transformer-based gene-expression embeddings against
        classical machine-learning baselines across AD and Parkinson disease. The Parkinson dataset was too large
        to load reliably in the available environment, so this final notebook focuses on a complete, reproducible
        AD-versus-control analysis using the Alzheimer disease dataset from GEO accession **GSE174367**.

        The final workflow uses donor-level pseudobulk profiles rather than treating each nucleus as an independent
        sample. This avoids data leakage from training and testing on cells from the same person.
        """
    ),
    md(
        """
        ## Background and Motivation

        Alzheimer disease is a neurodegenerative disorder involving changes in neuronal and glial biology.
        Single-nucleus RNA sequencing is useful for this question because it measures gene expression in individual
        brain-cell nuclei, allowing us to ask whether disease state is reflected in transcriptomic structure.

        Recent foundation-model papers such as scGPT, Geneformer, and scBERT motivate the broader idea that learned
        gene-expression representations may capture biological state. This notebook does not claim to complete that
        full foundation-model comparison. Instead, it builds the necessary classical baseline: a transparent,
        reproducible AD-versus-control analysis that can later be compared against scGPT or Geneformer embeddings.

        References from the proposal: Cui et al. 2024, *Nature Methods*; Theodoris et al. 2023, *Nature*; Yang et al.
        2022, *Nature Machine Intelligence*.
        """
    ),
    md(
        """
        ## Computational Workflow

        1. Download the GSE174367 single-nucleus RNA-seq metadata and 10x expression matrix.
        2. Align cell barcodes between metadata and the expression matrix.
        3. Aggregate nuclei into donor-level pseudobulk expression profiles.
        4. Normalize expression to log counts per million and select highly variable genes.
        5. Visualize donor profiles using PCA and UMAP.
        6. Compare AD and control donors using leave-one-donor-out classifiers.
        7. Summarize cell-type composition and highly variable genes associated with diagnosis.
        """
    ),
    code(
        """
        from pathlib import Path
        import pandas as pd
        from IPython.display import Image, display

        from scripts.final_ad_analysis import run_analysis

        processed = Path("data/processed")
        required = [
            processed / "ad_sample_metadata.csv",
            processed / "classifier_results.csv",
            processed / "top_hvg_differential_expression.csv",
            processed / "cell_type_composition_tests.csv",
        ]

        if not all(path.exists() for path in required):
            run_analysis()

        sample_meta = pd.read_csv(processed / "ad_sample_metadata.csv", index_col=0)
        classifier_results = pd.read_csv(processed / "classifier_results.csv")
        de = pd.read_csv(processed / "top_hvg_differential_expression.csv")
        composition = pd.read_csv(processed / "cell_type_composition_tests.csv")
        pca_variance = pd.read_csv(processed / "pca_variance.csv")
        """
    ),
    md(
        """
        ## Dataset

        The analysis uses GSE174367, a single-nucleus transcriptomic dataset with AD and control nuclei.
        The raw cell-level matrix contains tens of thousands of nuclei, but model evaluation is done at the
        donor level because diagnosis is a donor-level label.
        """
    ),
    code(
        """
        print(f"Number of donors: {sample_meta.shape[0]}")
        print(f"Total matched nuclei: {sample_meta['n_cells'].sum():,}")
        display(sample_meta)
        display(sample_meta['Diagnosis'].value_counts().rename_axis('Diagnosis').to_frame('n_donors'))
        """
    ),
    code('display(Image(filename="figures/diagnosis_counts.png"))'),
    md(
        """
        ## Cell-Type Composition

        Before looking at expression, we checked whether AD and control samples differ in broad cell-type
        composition. This is useful because disease-related expression differences can be confounded by different
        proportions of neurons, glia, and vascular cells.
        """
    ),
    code(
        """
        display(Image(filename="figures/cell_type_composition.png"))
        display(composition)
        """
    ),
    md(
        """
        ## Expression-Space Structure

        PCA and UMAP were run on donor pseudobulk log-CPM expression for the top highly variable genes.
        PCA is the main baseline representation because it is transparent and appropriate for a small number of
        donor-level observations.
        """
    ),
    code(
        """
        display(pca_variance.head(10))
        display(Image(filename="figures/pca_pseudobulk.png"))
        display(Image(filename="figures/umap_pseudobulk.png"))
        """
    ),
    md(
        """
        ## Classification Baselines

        Two classical baselines were evaluated with leave-one-donor-out cross-validation:

        - Logistic regression trained on PCA features.
        - Random forest trained on the selected highly variable genes.

        Leave-one-donor-out cross-validation is strict for this dataset because the model must predict a donor
        it did not see during training.
        """
    ),
    code(
        """
        display(classifier_results)
        display(Image(filename="figures/confusion_logistic_regression_on_pca_features.png"))
        display(Image(filename="figures/confusion_random_forest_on_hvgs.png"))
        """
    ),
    md(
        """
        ## Differential Expression Screen

        The table below is an exploratory Welch t-test screen across the selected highly variable genes.
        Because there are only 18 donors, these genes should be interpreted as candidates that help describe the
        dataset, not as validated biomarkers.
        """
    ),
    code("display(de.head(20))"),
    md(
        """
        ## Interpretation

        The donor-level expression profiles contain AD-related signal, but the separation is not perfect.
        Logistic regression on PCA features achieved approximately 0.67 accuracy and 0.65 balanced accuracy under
        leave-one-donor-out validation, while the random forest baseline performed slightly worse. This suggests
        that a low-dimensional expression representation captures some disease structure, but the small number of
        donors limits classification performance.

        The cell-type analysis showed a higher mean excitatory-neuron proportion in AD donors in this dataset,
        while most other broad cell types were not clearly different. This matters because disease status may be
        reflected both in gene-expression changes within cell types and in shifts in cell-type composition.

        Overall, the completed analysis supports a cautious conclusion: AD and control samples are partially
        separable from single-nucleus expression data using transparent classical methods, but stronger claims
        would require more donors, better covariate control, and ideally a transformer embedding comparison such
        as scGPT or Geneformer.
        """
    ),
    md(
        """
        ## Limitations and Future Work

        - The donor sample size is small: 11 AD donors and 7 controls.
        - Age, sex, batch, and cell-type composition may confound some disease signal.
        - This notebook uses pseudobulk profiles, which are statistically safer for donor-level prediction but
          lose some cell-state resolution.
        - The proposed scGPT comparison was not completed here because installing and running the model is a larger
          engineering task than the available project time allowed.
        - A stronger extension would run cell-type-specific pseudobulk models and compare PCA features directly
          against pretrained scGPT or Geneformer embeddings.
        """
    ),
]

nbf.write(nb, NOTEBOOK)
print(f"Wrote {NOTEBOOK}")
