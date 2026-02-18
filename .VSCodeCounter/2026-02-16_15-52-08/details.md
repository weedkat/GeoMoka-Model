# Details

Date : 2026-02-16 15:52:08

Directory /home/uwu/Projects/GeoMoka-Model

Total : 42 files,  3451 codes, 714 comments, 816 blanks, all 4981 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [README.md](/README.md) | Markdown | 58 | 0 | 18 | 76 |
| [config/ISPRS-Postdam/metadata.yaml](/config/ISPRS-Postdam/metadata.yaml) | YAML | 19 | 9 | 1 | 29 |
| [config/ISPRS-Postdam/train\_dpt.yaml](/config/ISPRS-Postdam/train_dpt.yaml) | YAML | 82 | 11 | 14 | 107 |
| [config/ISPRS-Postdam/train\_smp.yaml](/config/ISPRS-Postdam/train_smp.yaml) | YAML | 78 | 11 | 16 | 105 |
| [pyproject.toml](/pyproject.toml) | TOML | 40 | 0 | 5 | 45 |
| [requirements.txt](/requirements.txt) | pip requirements | 12 | 0 | 1 | 13 |
| [scripts/download\_isprs\_postdam.py](/scripts/download_isprs_postdam.py) | Python | 19 | 0 | 6 | 25 |
| [scripts/download\_pretrained.py](/scripts/download_pretrained.py) | Python | 21 | 0 | 6 | 27 |
| [src/geomoka/cli/calibrate.py](/src/geomoka/cli/calibrate.py) | Python | 138 | 44 | 35 | 217 |
| [src/geomoka/cli/datasplit.py](/src/geomoka/cli/datasplit.py) | Python | 42 | 0 | 16 | 58 |
| [src/geomoka/cli/scaffold.py](/src/geomoka/cli/scaffold.py) | Python | 18 | 1 | 8 | 27 |
| [src/geomoka/cli/train.py](/src/geomoka/cli/train.py) | Python | 11 | 0 | 5 | 16 |
| [src/geomoka/dataloader/build\_dataset.py](/src/geomoka/dataloader/build_dataset.py) | Python | 8 | 1 | 3 | 12 |
| [src/geomoka/dataloader/dataset.py](/src/geomoka/dataloader/dataset.py) | Python | 64 | 26 | 16 | 106 |
| [src/geomoka/dataloader/mask\_converter.py](/src/geomoka/dataloader/mask_converter.py) | Python | 50 | 17 | 16 | 83 |
| [src/geomoka/dataloader/transform.py](/src/geomoka/dataloader/transform.py) | Python | 80 | 72 | 26 | 178 |
| [src/geomoka/dataloader/wrapper.py](/src/geomoka/dataloader/wrapper.py) | Python | 84 | 16 | 30 | 130 |
| [src/geomoka/eval/evaluate.py](/src/geomoka/eval/evaluate.py) | Python | 100 | 49 | 28 | 177 |
| [src/geomoka/eval/metrics.py](/src/geomoka/eval/metrics.py) | Python | 122 | 64 | 41 | 227 |
| [src/geomoka/inference/engine.py](/src/geomoka/inference/engine.py) | Python | 200 | 94 | 51 | 345 |
| [src/geomoka/inference/rasterio.py](/src/geomoka/inference/rasterio.py) | Python | 24 | 3 | 2 | 29 |
| [src/geomoka/losses/ohem.py](/src/geomoka/losses/ohem.py) | Python | 49 | 3 | 8 | 60 |
| [src/geomoka/model/backbone/dinov2.py](/src/geomoka/model/backbone/dinov2.py) | Python | 315 | 49 | 55 | 419 |
| [src/geomoka/model/backbone/dinov2\_layers/\_\_init\_\_.py](/src/geomoka/model/backbone/dinov2_layers/__init__.py) | Python | 5 | 5 | 2 | 12 |
| [src/geomoka/model/backbone/dinov2\_layers/attention.py](/src/geomoka/model/backbone/dinov2_layers/attention.py) | Python | 58 | 9 | 22 | 89 |
| [src/geomoka/model/backbone/dinov2\_layers/block.py](/src/geomoka/model/backbone/dinov2_layers/block.py) | Python | 183 | 22 | 48 | 253 |
| [src/geomoka/model/backbone/dinov2\_layers/drop\_path.py](/src/geomoka/model/backbone/dinov2_layers/drop_path.py) | Python | 17 | 9 | 10 | 36 |
| [src/geomoka/model/backbone/dinov2\_layers/layer\_scale.py](/src/geomoka/model/backbone/dinov2_layers/layer_scale.py) | Python | 16 | 6 | 7 | 29 |
| [src/geomoka/model/backbone/dinov2\_layers/mlp.py](/src/geomoka/model/backbone/dinov2_layers/mlp.py) | Python | 26 | 8 | 8 | 42 |
| [src/geomoka/model/backbone/dinov2\_layers/patch\_embed.py](/src/geomoka/model/backbone/dinov2_layers/patch_embed.py) | Python | 53 | 18 | 19 | 90 |
| [src/geomoka/model/backbone/dinov2\_layers/swiglu\_ffn.py](/src/geomoka/model/backbone/dinov2_layers/swiglu_ffn.py) | Python | 48 | 5 | 11 | 64 |
| [src/geomoka/model/build\_model.py](/src/geomoka/model/build_model.py) | Python | 99 | 2 | 29 | 130 |
| [src/geomoka/model/semseg/dpt.py](/src/geomoka/model/semseg/dpt.py) | Python | 136 | 0 | 38 | 174 |
| [src/geomoka/model/util/blocks.py](/src/geomoka/model/util/blocks.py) | Python | 83 | 27 | 38 | 148 |
| [src/geomoka/model/wrapper.py](/src/geomoka/model/wrapper.py) | Python | 94 | 40 | 25 | 159 |
| [src/geomoka/train/supervised.py](/src/geomoka/train/supervised.py) | Python | 153 | 21 | 36 | 210 |
| [src/geomoka/train/tester.py](/src/geomoka/train/tester.py) | Python | 15 | 0 | 2 | 17 |
| [src/geomoka/train/trainer.py](/src/geomoka/train/trainer.py) | Python | 138 | 54 | 26 | 218 |
| [src/geomoka/train/unimatch\_v2.py](/src/geomoka/train/unimatch_v2.py) | Python | 219 | 0 | 63 | 282 |
| [src/geomoka/util/dist\_helper.py](/src/geomoka/util/dist_helper.py) | Python | 30 | 5 | 7 | 42 |
| [src/geomoka/util/utils.py](/src/geomoka/util/utils.py) | Python | 76 | 13 | 17 | 106 |
| [tests/experiment.ipynb](/tests/experiment.ipynb) | JSON | 368 | 0 | 1 | 369 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)