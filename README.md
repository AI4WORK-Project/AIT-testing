# WISDOM Package

WISDOM is a PyTorch library for importance-driven, internal-activation coverage
testing of classification, YOLO detection and pose-estimation models.

<div align="center">
  <img src="figs/wisdom_overview.png" alt="Wisdom Overview diagram" width="1500"/>
</div>

## Quick validation

From the source checkout, use the existing `.venv` with uv:

```shell
cd /shared/storage/cs/scratch/lrr550/package_wisdom/Wisdom
uv sync --locked --group test --extra detection --extra bo

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --offline --no-sync pytest -q tests/test_wisdom_e2e.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --offline --no-sync pytest -q
uv run --offline --no-sync python run_wisdom.py --help
uv lock --check --offline
uv pip check
```

The standalone suite uses tiny synthetic data, CPU and random weights, with no
model/dataset downloads. It covers all three tasks, including the actual
packaged NVIDIA PyTorch pose architecture. No external dataset or sibling
`trt_pose` clone is required. The full suite also checks two YOLO scales and
the sklearn, BoTorch and automatic BO backends.

Packaging tests build/install a wheel into a temporary directory, verify
imports resolve to that installation, then run the runtime suite outside the
checkout. Source-policy checks and packaging itself are not rerun recursively.
Initial dependency installation may require the network; tests do not.
`--no-sync` preserves the extras installed by the explicit sync above.

## Build and install a wheel

```shell
uv build --wheel --config-setting=--build-option=--keep-temp --out-dir ../wisdom-wheelhouse

# In the Python environment where you want to use WISDOM:
python -m pip install ../wisdom-wheelhouse/wisdom-0.1.0-py3-none-any.whl
# Optional detection and BoTorch backends:
python -m pip install '../wisdom-wheelhouse/wisdom-0.1.0-py3-none-any.whl[detection,bo]'
python -m run_wisdom --help
```

With uv, use `uv pip install` instead of `python -m pip install`. Rebuild the
wheel after changing source. Build outputs are deliberately outside the
checkout. `--keep-temp` avoids a setuptools temporary-directory cleanup failure
on this shared filesystem; it leaves an ignored `build/` directory. Start with
a fresh build tree when changing package contents so stale files cannot enter
the wheel. Datasets, research model weights and CIFAR model factories are not
included in the wheel.

The base package includes PyTorch/torchvision, Captum and sklearn. Detection,
BoTorch and plotting (`[plot]`) remain optional; pose needs no TensorRT extra.
For an explicit installed-wheel regression run:

```shell
uv run --offline --no-sync pytest -q -s tests/packaging/test_wheel_install.py
```

## Source layout

- `wisdom/`: shared attribution, consensus pretraining, task adapters, clustering and coverage.
- `models_info/trt_pose/`: NVIDIA PyTorch model definitions and license, without plugins or weights.
- `tests/`: synthetic runtime, task, script and wheel regression tests.
- `run_wisdom.py`: inference and coverage entry point.
- `wisdom_*_train.py`: task-specific neuron-score pretraining scripts.
- `convert_torchvision_dataset.py`: offline CIFAR-10/100 and MNIST converter.
- `pyproject.toml` / `uv.lock`: authoritative dependencies and package metadata.
- `setup.py`: compatibility metadata shim; builds use `uv build`.
- `Docker/Dockerfile`: uv-managed runtime and CPU test targets.

## Inference and coverage

`run_wisdom.py` loads a model, evaluates the test data and calculates coverage.
Only WISDOM mode with a missing/empty score CSV starts score pretraining.
An existing valid CSV is reused. This does not train the model's weights.

```shell
uv run python run_wisdom.py --help
uv run python wisdom_classification_train.py --help
uv run python wisdom_yolo_train.py --help
uv run python wisdom_pose_train.py --help
```

After wheel installation, use `python -m run_wisdom` and
`python -m wisdom_classification_train` (likewise for the other scripts).
There is no separate `wisdom` console command.

### Classification and offline dataset conversion

The runner reads ImageFolder data: one subdirectory per class under each split.
The converter decodes already-downloaded CIFAR-10, CIFAR-100 or MNIST, reserves
a deterministic stratified 10% of the official training split for validation,
and preserves the official test split:

```shell
uv run python convert_torchvision_dataset.py --help
uv run python convert_torchvision_dataset.py \
  --dataset cifar10 \
  --data-root /shared/storage/cs/scratch/lrr550/datasets \
  --output-root /shared/storage/cs/scratch/lrr550/datasets/cifar-10-imagefolder
uv run python convert_torchvision_dataset.py \
  --dataset cifar100 --data-root /shared/storage/cs/scratch/lrr550/datasets
uv run python convert_torchvision_dataset.py \
  --dataset mnist --data-root /shared/storage/cs/scratch/lrr550/datasets
```

Default output names are `cifar10-imagefolder`, `cifar100-imagefolder` and
`mnist-imagefolder` under the supplied root. The CIFAR-10 command above
explicitly uses the existing `cifar-10-imagefolder` naming convention.
Each export contains `build/`, `validation/`, `test/` and `conversion.json`.
Conversion uses `download=False` and refuses any existing destination,
including a final symlink. It never edits the native dataset. Do not rerun it
on an existing export or run simultaneous conversions to the same destination.

For the trusted CIFAR-10 ResNet18 supplied in the source checkout:

```shell
uv run python run_wisdom.py \
  --mode wisdom --task classification \
  --weights-path ./models_info/saved_models/resnet18_CIFAR10_whole.pth \
  --checkpoint-format module \
  --build-data-path /shared/storage/cs/scratch/lrr550/datasets/cifar-10-imagefolder/build \
  --validation-data-path /shared/storage/cs/scratch/lrr550/datasets/cifar-10-imagefolder/validation \
  --test-data-path /shared/storage/cs/scratch/lrr550/datasets/cifar-10-imagefolder/test \
  --image-size 32 --normalize custom \
  --normalize-mean 0.4914,0.4822,0.4465 \
  --normalize-std 0.2023,0.1994,0.2010 \
  --wisdom-csv ./artifacts/resnet18_cifar10_wisdom.csv \
  --output-json ./results/resnet18_cifar10.json \
  --device cpu
```

Only use `--checkpoint-format module` for trusted serialized modules.
For raw CIFAR ResNet18 state dictionaries in a source checkout, use
`--checkpoint-format state-dict --model-factory models_info.models_cv.resnet:ResNet18`.
For wheel-only usage, supply an importable factory from your own package.
A state dictionary does not identify its architecture or task; explicit
`--task` is required, and `.pth` never implies classification.

The following commands are templates: replace the model, factory and data
paths with your own local files. The classification factory must take no arguments.

```shell
uv run python run_wisdom.py \
  --mode wisdom --task classification \
  --weights-path ./classifier.pth --checkpoint-format state-dict \
  --model-factory my_package.models:make_classifier \
  --build-data-path ./data/build --test-data-path ./data/test \
  --image-size 32 --normalize none \
  --wisdom-csv ./artifacts/classifier_wisdom.csv \
  --output-json ./results/classifier_coverage.json --device cpu
```

Preprocessing must match model training, not merely the dataset's name.
`--normalize` supports `none`, `imagenet`, `cifar`, `mnist` and `custom`.
Custom RGB statistics require three finite values each; `--grayscale` requires
one. Standard deviations must be positive:

```shell
--normalize custom \
--normalize-mean 0.40,0.42,0.39 \
--normalize-std 0.20,0.21,0.19
```

The identical resize, channel conversion and normalization are applied to
build, explicit BO validation, the automatic holdout and test data.

### YOLO detection

Use a local Ultralytics YAML plus matching raw state dictionary:

```shell
uv run python run_wisdom.py \
  --mode wisdom --task detection \
  --model-path ./models/yolo.yaml --weights-path ./models/yolo_state_dict.pth \
  --checkpoint-format state-dict \
  --build-data-path ./data/yolo/build/images \
  --test-data-path ./data/yolo/test/images --imgsz 640 \
  --selection-mode per-group --num-groups 3 --num-layers 9 \
  --wisdom-csv ./artifacts/yolo_wisdom.csv \
  --output-json ./results/yolo_coverage.json --device cpu
```

Detection uses RGB inputs scaled to `[0,1]`; classification normalization flags
do not change this path. For metrics, provide matching YOLO `.txt` labels in
`labels/` next to `images/`. Empty label files mean no objects; absent labels
mean metrics are unavailable.

### Pose / trt_pose

```shell
uv run python run_wisdom.py \
  --mode wisdom --task pose \
  --weights-path ./models/random_pose.pth --checkpoint-format state-dict \
  --pose-topology ./configs/human_pose.json \
  --pose-architecture resnet18_baseline_att \
  --build-data-path ./data/pose/build --test-data-path ./data/pose/test \
  --image-size 224 \
  --wisdom-csv ./artifacts/pose_wisdom.csv \
  --output-json ./results/pose_coverage.json --device cpu
```

Supply a topology JSON with a `keypoints` list and one-based `skeleton` links.
WISDOM constructs the explicit architecture with `len(keypoints)` confidence
channels and `2 * len(skeleton)` PAF channels, then strictly loads its state
dictionary on the requested device. `--pose-model-kwargs` accepts a JSON object
for nondefault architecture parameters. Pretrained downloads are disabled.

Model definitions come from `models_info/trt_pose` inside WISDOM.
`--model-path` is an optional explicit external-source override for pose;
no sibling clone is needed normally. Unused ImageNet classifier parameters
remain in the checkpoint but are excluded from attribution, as are the final
pose output heads.

For NVIDIA pose, use the state-dict reconstruction route shown above (or
`build_trt_pose_model()` in the API) so the required layer metadata is attached.
A legacy pickled/directly constructed NVIDIA module without that metadata is
not covered by this guarantee and can still expose an unused classifier.

The built-in pose loader uses resized RGB `[0,1]` images. Classification
normalization flags do not alter it. For checkpoints requiring additional
normalization or supervised heatmap/PAF/mask targets, supply a matching
transformed loader through the Python API.

Random state-dict weights prove integration only. Meaningful pose-quality
evaluation requires trained weights, matching preprocessing and pose ground truth.

### CSV, clustering and BO lifecycle

WISDOM and IDC score artifacts are separate. `--mode wisdom` accepts
`--wisdom-csv`: a valid nonempty CSV is reused, while a missing/empty CSV is
generated by WISDOM pretraining. `--mode idc` requires an existing valid
`--idc-csv` and never starts WISDOM pretraining. Malformed nonempty CSVs are
errors. The schema is `LayerName,NeuronIndex,Score`.

Build data supplies pretraining and activation-cluster fitting. BO uses
validation data only, and the final result uses test data. These paths must
be distinct. With `--bo` and no `--validation-data-path`, the runner removes a
deterministic 10% holdout from build data using `--seed` (default 42).
Those samples do not enter score pretraining or cluster fitting. Tiny datasets
reserve at least one sample; JSON reports the realized fraction. If using a
converted dataset's existing validation split, supply its explicit path to
avoid reserving another holdout.

CSV, trainer checkpoint and cluster cache formats do not encode full dataset,
model and preprocessing provenance. Use new artifact/cache paths when changing
weights, data, preprocessing or split seed; do not reuse artifacts built using
samples that are now validation data.

`--top-m-neurons` means M per scope: globally, per dynamic group, or per
considered layer. The same per-scope budget is respected in consensus fusion.
`--num-groups` chooses the group count; `--num-layers` evenly limits discovered
eligible layers before partitioning. Omit the latter to consider all eligible
layers. Impossible counts are errors; no YOLO-specific layer ranges are needed.

BO is available in WISDOM mode. All controls below have defaults:

```shell
uv run python run_wisdom.py \
  --mode wisdom --task classification \
  --weights-path ./classifier.pth --checkpoint-format state-dict \
  --model-factory my_package.models:make_classifier \
  --build-data-path ./data/build --test-data-path ./data/test \
  --output-json ./results/classifier_bo.json \
  --bo --bo-backend auto --bo-init 3 --bo-iter 3 \
  --bo-candidate-pool-size 32 \
  --bo-cluster-methods KMeans,MiniBatchKMeans,Birch \
  --bo-n-clusters 2,3,4 --device cpu
```

BO maximizes Pearson correlation between coverage and the task metric over
up to five ordered, nested validation subsets. It tunes clustering, not model
weights. Constant metric/coverage series yield objective zero. The chosen
configuration and BO history path appear in terminal output and JSON; without
BO the fixed clustering configuration is printed instead.

### Honest task metrics

Classification reports accuracy, mean cross-entropy and weighted F1. Detection
reports precision/recall/F1 using confidence >= 0.25, class-aware NMS at IoU 0.45
and one-to-one same-class matching at IoU >= 0.5; this is not COCO mAP.

Programmatic supervised pose batches report peak PCK at 5% of the heatmap
diagonal, counting only unmasked, nonzero target keypoints. This is not COCO
OKS/AP or multiperson association accuracy. Image-only pose instead reports
`pose_confidence_surrogate`: the average per-keypoint maximum sigmoid confidence.
It is bounded but not calibrated as a probability (zero maps give 0.5).
Image-only pose BO uses this surrogate, never fake F1 or accuracy.

Unavailable metrics are `null` in JSON and `N/A` in the terminal.
Coverage measures diversity of joint cluster assignments of selected internal
neuron activations. For pose it is not coverage of keypoint coordinates,
anatomical correctness or a guarantee of accuracy. Correlation-based BO using
unlabeled confidence does not validate pose quality.

### Direct neuron-score pretraining

These scripts generate the consensus CSV only; they do not train model weights
or calculate final coverage. Inputs below are local placeholders as above.
`--top-m` here corresponds to the runner's `--top-m-neurons`.

```shell
uv run python wisdom_classification_train.py \
  --model-path ./classifier.pth --checkpoint-format state-dict \
  --model-factory my_package.models:make_classifier \
  --imagefolder-root ./data/build --image-size 32 --normalize none \
  --methods la --top-m 1 --num-layers 1 \
  --out-csv ./artifacts/classifier_wisdom.csv --device cpu

uv run python wisdom_yolo_train.py \
  --weights ./models/yolo.yaml --img-dir ./data/yolo/build/images \
  --imgsz 32 --num-images 4 --methods la --top-m 1 \
  --selection-mode per-group --num-groups 2 --num-layers 2 \
  --out-csv ./artifacts/yolo_wisdom.csv --device cpu

uv run python wisdom_pose_train.py \
  --model-path ./models/random_pose.pth --checkpoint-format state-dict \
  --pose-topology ./configs/human_pose.json \
  --pose-architecture resnet18_baseline_att \
  --img-dir ./data/pose/build --image-size 32 --methods lgxa \
  --top-m 1 --num-layers 1 \
  --out-csv ./artifacts/pose_wisdom.csv --device cpu
```

The direct YOLO script accepts a local YAML for random initialization or a
trusted local Ultralytics `.pt` artifact. Use the inference runner for a separate
YAML plus raw state dictionary. `la` (layer activation) is a fast smoke method,
not the default consensus configuration; pose defaults to `lgxa`.
BO runs in `run_wisdom.py` after score pretraining and before coverage, not in
these CSV-only scripts. Wheel users replace `uv run python <script>.py` with
`python -m <script>`.

## Docker

Docker uses the locked uv environment and packaged pose definitions.
It installs neither TensorRT nor `torch2trt`, and downloads no model weights.
The test target includes detection and BO dependencies and forces CPU tests;
the locked Torch distribution is CUDA-capable, not a minimal CPU-only wheel.
GPU runs require a compatible host driver and NVIDIA container runtime.

These commands are typed **from the host**. Docker was not built or run during
this update; container availability is not required for local validation.

```shell
cd /shared/storage/cs/scratch/lrr550/package_wisdom/Wisdom
docker build -f Docker/Dockerfile --target test -t wisdom-test .
docker run --rm --network none wisdom-test \
  uv run --offline --no-sync pytest -q tests/test_wisdom_e2e.py
```

No dataset or sibling pose mount is needed for the tests. For manual experiments,
an optional host bind mount is:

```shell
-v /shared/storage/cs/scratch/lrr550/datasets:/datasets:ro
```
