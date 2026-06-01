from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "figures"

AD_META = RAW_DIR / "GSE174367_snRNA-seq_cell_meta.csv.gz"
AD_H5 = RAW_DIR / "GSE174367_snRNA-seq_filtered_feature_bc_matrix.h5"

URLS = {
    AD_META: "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE174367&format=file&file=GSE174367%5FsnRNA%2Dseq%5Fcell%5Fmeta%2Ecsv%2Egz",
    AD_H5: "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE174367&format=file&file=GSE174367%5FsnRNA%2Dseq%5Ffiltered%5Ffeature%5Fbc%5Fmatrix%2Eh5",
}


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def download_inputs() -> None:
    ensure_dirs()
    for path, url in URLS.items():
        if path.exists() and path.stat().st_size > 0:
            print(f"{path.name} already exists ({path.stat().st_size / 1e6:.1f} MB).")
            continue
        print(f"Downloading {path.name}...")
        urlretrieve(url, path)
        print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB).")


def decode(values: np.ndarray) -> np.ndarray:
    return np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def make_unique(names: np.ndarray) -> np.ndarray:
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        count = seen.get(name, 0)
        out.append(name if count == 0 else f"{name}_{count}")
        seen[name] = count + 1
    return np.array(out)


def load_10x_h5(path: Path) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        group = h5["matrix"]
        data = group["data"][:]
        indices = group["indices"][:]
        indptr = group["indptr"][:]
        shape = tuple(group["shape"][:])
        barcodes = decode(group["barcodes"][:])
        genes = make_unique(decode(group["features"]["name"][:]))
    matrix = sparse.csc_matrix((data, indices, indptr), shape=shape)
    return matrix, genes, barcodes


def build_pseudobulk() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading metadata and expression matrix...")
    meta = pd.read_csv(AD_META)
    matrix, genes, barcodes = load_10x_h5(AD_H5)

    barcode_to_idx = pd.Series(np.arange(len(barcodes)), index=barcodes)
    common = meta["Barcode"].isin(barcode_to_idx.index)
    meta = meta.loc[common].copy()
    cell_idx = barcode_to_idx.loc[meta["Barcode"]].to_numpy()

    sample_ids, sample_codes = np.unique(meta["SampleID"], return_inverse=True)
    cell_to_sample = sparse.csr_matrix(
        (np.ones(len(sample_codes), dtype=np.float32), (np.arange(len(sample_codes)), sample_codes)),
        shape=(len(sample_codes), len(sample_ids)),
    )

    print("Aggregating cells into donor-level pseudobulk profiles...")
    counts = matrix[:, cell_idx] @ cell_to_sample
    pseudobulk = pd.DataFrame(
        counts.T.toarray().astype(np.float32),
        index=sample_ids,
        columns=genes,
    )

    sample_meta = (
        meta.groupby("SampleID")
        .agg(
            Diagnosis=("Diagnosis", "first"),
            Age=("Age", "first"),
            Sex=("Sex", "first"),
            Batch=("Batch", "first"),
            n_cells=("Barcode", "size"),
        )
        .loc[sample_ids]
    )

    cell_counts = pd.crosstab(meta["SampleID"], meta["Cell.Type"]).loc[sample_ids]
    return pseudobulk, sample_meta, cell_counts


def normalize_and_select_hvgs(pseudobulk: pd.DataFrame, n_genes: int = 3000) -> pd.DataFrame:
    detected = (pseudobulk > 0).sum(axis=0)
    keep = (detected >= 3) & (pseudobulk.sum(axis=0) >= 10)
    filtered = pseudobulk.loc[:, keep]

    library_size = filtered.sum(axis=1)
    log_cpm = np.log1p(filtered.div(library_size, axis=0) * 1_000_000)
    variances = log_cpm.var(axis=0).sort_values(ascending=False)
    selected = variances.head(min(n_genes, len(variances))).index
    return log_cpm.loc[:, selected]


def plot_diagnosis_counts(sample_meta: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=sample_meta.reset_index(), x="Diagnosis", hue="Diagnosis", palette="Set2", legend=False, ax=ax)
    ax.set_title("Donor diagnosis counts")
    ax.set_xlabel("")
    ax.set_ylabel("Number of donors")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "diagnosis_counts.png", dpi=200)
    plt.close(fig)


def plot_cell_type_composition(cell_counts: pd.DataFrame, sample_meta: pd.DataFrame) -> pd.DataFrame:
    proportions = cell_counts.div(cell_counts.sum(axis=1), axis=0)
    plot_df = proportions.join(sample_meta["Diagnosis"]).reset_index().melt(
        id_vars=["SampleID", "Diagnosis"],
        var_name="Cell type",
        value_name="Proportion",
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.boxplot(data=plot_df, x="Cell type", y="Proportion", hue="Diagnosis", palette="Set2", ax=ax)
    sns.stripplot(
        data=plot_df,
        x="Cell type",
        y="Proportion",
        hue="Diagnosis",
        dodge=True,
        color="black",
        alpha=0.45,
        size=3,
        legend=False,
        ax=ax,
    )
    ax.set_title("Cell-type composition by diagnosis")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cell_type_composition.png", dpi=200)
    plt.close(fig)

    rows = []
    for cell_type in proportions.columns:
        ad = proportions.loc[sample_meta["Diagnosis"] == "AD", cell_type]
        control = proportions.loc[sample_meta["Diagnosis"] == "Control", cell_type]
        stat, p_value = stats.ttest_ind(ad, control, equal_var=False)
        rows.append(
            {
                "Cell.Type": cell_type,
                "AD_mean": ad.mean(),
                "Control_mean": control.mean(),
                "Difference_AD_minus_Control": ad.mean() - control.mean(),
                "Welch_t": stat,
                "p_value": p_value,
            }
        )
    summary = pd.DataFrame(rows).sort_values("p_value")
    summary.to_csv(PROCESSED_DIR / "cell_type_composition_tests.csv", index=False)
    return summary


def plot_embeddings(log_cpm_hvg: pd.DataFrame, sample_meta: pd.DataFrame) -> pd.DataFrame:
    scaled = StandardScaler().fit_transform(log_cpm_hvg)
    pca = PCA(n_components=min(10, scaled.shape[0] - 1), random_state=42)
    pcs = pca.fit_transform(scaled)
    embedding = pd.DataFrame(pcs[:, :2], index=log_cpm_hvg.index, columns=["PC1", "PC2"]).join(sample_meta)
    embedding.to_csv(PROCESSED_DIR / "pca_embedding.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=embedding.reset_index(),
        x="PC1",
        y="PC2",
        hue="Diagnosis",
        style="Sex",
        size="n_cells",
        sizes=(50, 180),
        palette="Set2",
        ax=ax,
    )
    ax.set_title("PCA of donor pseudobulk expression")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pca_pseudobulk.png", dpi=200)
    plt.close(fig)

    try:
        import umap

        reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
        coords = reducer.fit_transform(scaled)
        umap_df = pd.DataFrame(coords, index=log_cpm_hvg.index, columns=["UMAP1", "UMAP2"]).join(sample_meta)
        umap_df.to_csv(PROCESSED_DIR / "umap_embedding.csv")

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=umap_df.reset_index(),
            x="UMAP1",
            y="UMAP2",
            hue="Diagnosis",
            style="Sex",
            size="n_cells",
            sizes=(50, 180),
            palette="Set2",
            ax=ax,
        )
        ax.set_title("UMAP of donor pseudobulk expression")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "umap_pseudobulk.png", dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"UMAP failed; PCA results are still available. Error: {exc}")

    variance = pd.DataFrame(
        {
            "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    variance.to_csv(PROCESSED_DIR / "pca_variance.csv", index=False)
    return embedding


def evaluate_classifiers(log_cpm_hvg: pd.DataFrame, sample_meta: pd.DataFrame) -> pd.DataFrame:
    labels = sample_meta.loc[log_cpm_hvg.index, "Diagnosis"]
    y = LabelEncoder().fit_transform(labels)
    loo = LeaveOneOut()

    models = {
        "Logistic regression on PCA features": Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=min(5, log_cpm_hvg.shape[0] - 2), random_state=42)),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
            ]
        ),
        "Random forest on HVGs": RandomForestClassifier(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
    }

    rows = []
    for name, model in models.items():
        y_pred = cross_val_predict(model, log_cpm_hvg, y, cv=loo)
        cm = confusion_matrix(y, y_pred)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y, y_pred),
                "confusion_matrix": json.dumps(cm.tolist()),
            }
        )

        fig, ax = plt.subplots(figsize=(4.8, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["AD", "Control"], yticklabels=["AD", "Control"], ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        fig.savefig(FIGURE_DIR / f"confusion_{safe_name}.png", dpi=200)
        plt.close(fig)

    results = pd.DataFrame(rows)
    results.to_csv(PROCESSED_DIR / "classifier_results.csv", index=False)
    return results


def differential_expression(log_cpm_hvg: pd.DataFrame, sample_meta: pd.DataFrame) -> pd.DataFrame:
    ad = log_cpm_hvg.loc[sample_meta["Diagnosis"] == "AD"]
    control = log_cpm_hvg.loc[sample_meta["Diagnosis"] == "Control"]
    rows = []
    for gene in log_cpm_hvg.columns:
        stat, p_value = stats.ttest_ind(ad[gene], control[gene], equal_var=False)
        rows.append(
            {
                "gene": gene,
                "AD_mean": ad[gene].mean(),
                "Control_mean": control[gene].mean(),
                "logCPM_difference_AD_minus_Control": ad[gene].mean() - control[gene].mean(),
                "Welch_t": stat,
                "p_value": p_value,
            }
        )
    de = pd.DataFrame(rows).sort_values("p_value")
    de.to_csv(PROCESSED_DIR / "top_hvg_differential_expression.csv", index=False)
    return de


def run_analysis() -> dict[str, pd.DataFrame]:
    download_inputs()
    pseudobulk, sample_meta, cell_counts = build_pseudobulk()
    pseudobulk.to_csv(PROCESSED_DIR / "ad_pseudobulk_counts.csv")
    sample_meta.to_csv(PROCESSED_DIR / "ad_sample_metadata.csv")
    cell_counts.to_csv(PROCESSED_DIR / "ad_cell_type_counts.csv")

    log_cpm_hvg = normalize_and_select_hvgs(pseudobulk)
    log_cpm_hvg.to_csv(PROCESSED_DIR / "ad_log_cpm_hvg.csv")

    plot_diagnosis_counts(sample_meta)
    composition_tests = plot_cell_type_composition(cell_counts, sample_meta)
    embedding = plot_embeddings(log_cpm_hvg, sample_meta)
    classifier_results = evaluate_classifiers(log_cpm_hvg, sample_meta)
    de = differential_expression(log_cpm_hvg, sample_meta)

    print("\nSample metadata:")
    print(sample_meta)
    print("\nClassifier results:")
    print(classifier_results)
    print("\nTop differential HVGs:")
    print(de.head(10))

    return {
        "sample_meta": sample_meta,
        "cell_counts": cell_counts,
        "composition_tests": composition_tests,
        "embedding": embedding,
        "classifier_results": classifier_results,
        "differential_expression": de,
    }


if __name__ == "__main__":
    run_analysis()
