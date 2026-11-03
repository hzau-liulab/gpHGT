# gpHGT

Gene–pathway heterogeneous graph learning for infection-associated cell-state prediction.

## Model

Each cell is represented as a graph of gene and pathway nodes. The implementation integrates gene expression, precomputed pathway activity, Gene2Vec embeddings, pathway embeddings, protein interactions, and pathway membership. Gated heterogeneous graph attention and attention-guided clustering are combined through Mamba-like linear attention and type-specific attention pooling.

## Installation

Install compatible builds of PyTorch, PyTorch Geometric, and torch-scatter for your platform, then install the dependencies listed in `requirements.txt`. Original dependency versions were not supplied.

## Input data

The cohort directory must contain:

| File | Content |
| --- | --- |
| `data/NEW_training_sample.csv.gz` | Training-pool cell-by-gene expression matrix |
| `data/NEW_testing_sample.csv.gz` | Test cell-by-gene expression matrix |
| `data/rna_names.csv` | Gene names in the first column after a header |
| `data/NEW_training_label.csv.gz` | Training-pool labels in the first column |
| `data/NEW_testing_label.csv.gz` | Test labels in the first column |
| `HTG/ppi_filtered.txt` | Headerless tab-separated Gene1, Gene2, Score |
| `pathway_gsva/gene_all.txt` | Pathway identifier followed by tab-separated member genes |
| `pathway_gsva/pathway_all.txt` | Cell-by-pathway score matrix with a row-index column |
| `gene2vec_dim_200_iter_9.txt` | Gene identifiers and 200-dimensional embeddings |
| `pathway_gsva/keen/KEGG_embeddings.csv` | Pathway embeddings with a header |
| `pathway_gsva/keen/Reactome_embeddings.csv` | Pathway embeddings with a header |
| `pathway_gsva/keen/Wikipathways_embeddings.csv` | Pathway embeddings with a header |

Pathway embedding identifiers must contain `pathway:`. All graph genes must occur in the expression matrices. Pathway-score rows must match expression rows: training-pool cells first, then test cells. Class `1` is the positive class.

## Run

```bash
GPHGT_DATA_DIR=/path/to/cohort python scripts/train_gphgt.py
```

Alternatively, change the default `./data` path in the script. Input files are not bundled. The supplied settings use seed 42, validation size 4,310, batch size 32, 20 epochs, and AdamW with learning rate 0.0006 and weight decay 0.001. Adjust cohort-dependent settings for your input data.

## Outputs

- `results/models/last_model_msg.pt`: checkpoint from the final completed training epoch; testing uses this same model.
- `results/per_cell_predictions_msg.csv`: test labels, predictions, and positive-class probabilities.
- `results/signals_msg/batch*.json`: node pooling attention and cell representations.

## Scope and validation

This package covers the supplied single-cell training script. Preprocessing, embedding generation, transfer learning, and downstream analyses are separate workflows and are not included. Syntax and structural checks were performed; training was not run because inputs and the original environment were not supplied.

See [CHANGES.md](CHANGES.md) for edits and retained behavior. Authors should add their chosen license and final manuscript citation before public release.
