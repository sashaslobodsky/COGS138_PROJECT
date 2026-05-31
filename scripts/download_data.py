from pathlib import Path
import urllib.request
import tarfile

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

urls = {
    # Parkinson's GSE157783
    "GSE157783_IPDCO_hg_midbrain_UMI.tar.gz":
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE157783&format=file&file=GSE157783%5FIPDCO%5Fhg%5Fmidbrain%5FUMI%2Etar%2Egz",

    "GSE157783_IPDCO_hg_midbrain_cell.tar.gz":
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE157783&format=file&file=GSE157783%5FIPDCO%5Fhg%5Fmidbrain%5Fcell%2Etar%2Egz",

    "GSE157783_IPDCO_hg_midbrain_genes.tar.gz":
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE157783&format=file&file=GSE157783%5FIPDCO%5Fhg%5Fmidbrain%5Fgenes%2Etar%2Egz",

    # Alzheimer's GSE174367 snRNA-seq only
    "GSE174367_snRNA-seq_cell_meta.csv.gz":
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE174367&format=file&file=GSE174367%5FsnRNA%2Dseq%5Fcell%5Fmeta%2Ecsv%2Egz",

    "GSE174367_snRNA-seq_filtered_feature_bc_matrix.h5":
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE174367&format=file&file=GSE174367%5FsnRNA%2Dseq%5Ffiltered%5Ffeature%5Fbc%5Fmatrix%2Eh5",
}

for filename, url in urls.items():
    output_path = DATA_DIR / filename

    if output_path.exists():
        print(f"{filename} already exists, skipping download.")
    else:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Saved to {output_path}")

    if filename.endswith(".tar.gz"):
        print(f"Extracting {filename}...")
        with tarfile.open(output_path, "r:gz") as tar:
            tar.extractall(DATA_DIR)
        print(f"Extracted {filename}")

print("All required data files are downloaded and extracted.")