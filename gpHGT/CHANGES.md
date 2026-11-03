# Changes from the uploaded script

## Formatting and organization

- Normalized Python formatting and removed obsolete commented-out code.
- Added a module description and section comments.
- Added the `GPHGT_DATA_DIR` environment variable with `./data` as the default.
- Moved cuBLAS configuration before the PyTorch import.
- Automatically create checkpoint output directories.

## Explicit implementation fixes

- Register HGT relation projections, priors, and gates before constructing AdamW, so these parameters participate in optimization.
- Preserve final-epoch checkpoint saving and test evaluation, as requested. The checkpoint is named `last_model_msg.pt` to reflect its contents.
- Initialize the early-stopping counter explicitly.

## Retained behavior

The architecture, graph construction, loss, learning-rate schedule, node features, cell-level split, fixed validation size, and attention calculations are preserved. Standardization still occurs before the validation split, and clustering initialization remains random on every forward pass, including evaluation. Prediction and attention export remain separate passes. These choices were not changed as part of code cleanup.

The former export-device concern was incorrect: `HGTNet.forward()` already moves the graph to the model device.

## Verification

Python syntax parsing and structural checks passed. Full training and numerical equivalence were not tested. The parameter-registration fix can change training results; existing published numerical results should not be described as reproduced by this edited version without rerunning it.
