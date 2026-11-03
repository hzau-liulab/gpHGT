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
| `data/NEW_training_sample.csv.gz` | Train cell-by-gene expression matrix |
| `data/NEW_testing_sample.csv.gz` | Test cell-by-gene expression matrix |
| `data/rna_names.csv` | Gene names |
| `data/NEW_training_label.csv.gz` | Train labels  |
| `data/NEW_testing_label.csv.gz` | Test labels  |
| `HTG/ppi_filtered.txt` |  Gene1, Gene2, Score |
| `pathway_gsva/gene_all.txt` | Pathway identifier followed by member genes |
| `gene2vec_dim_200_iter_9.txt` |  200-dimensional embeddings |
| `pathway_gsva/keen/KEGG_embeddings.csv` | Pathway embeddings  |
| `pathway_gsva/keen/Reactome_embeddings.csv` | Pathway embeddings  |
| `pathway_gsva/keen/Wikipathways_embeddings.csv` | Pathway embeddings  |


## Run

```bash
GPHGT_DATA_DIR=/path/to/cohort python scripts/train_gphgt.py
```

Alternatively, change the default `./data` path in the script. Input files are not bundled. The supplied settings use seed 42, validation size 4,310, batch size 32, 20 epochs, and AdamW with learning rate 0.0006 and weight decay 0.001. Adjust cohort-dependent settings for your input data.

## Outputs

- `results/models/last_model_msg.pt`: checkpoint from the final completed training epoch; testing uses this same model.
- `results/per_cell_predictions_msg.csv`: test labels, predictions, and positive-class probabilities.
- `results/signals_msg/batch*.json`: node pooling attention and cell representations.

