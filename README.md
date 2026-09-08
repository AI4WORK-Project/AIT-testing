# WISDOM Package

WISDOM is a PyTorch library for importance-driven, internal-activation coverage
testing of classification, YOLO detection and pose-estimation (optional) models.

<div align="center">
  <img src="figs/wisdom_overview.png" alt="Wisdom Overview diagram" width="1500"/>
</div>

## Prerequisites

Use the uv-managed `pyproject.toml` / `uv.lock` environment below for current
WISDOM development and testing. `requirements.txt` and `requirements_venv.yaml`
are legacy environment snapshots, not the authoritative package dependencies.

Use conda or pyvenv to build a virtual environment：

```shell
# Legacy requirements (prefer uv sync below)
python -m pip install -r requirements.txt

# (Deprecated) If you are using anaconda or miniconda virtual environment, do:
conda env create -f requirements_venv.yaml
```

### How to get `uv`

```shell
# Install through url
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or
wget -qO- https://astral.sh/uv/install.sh | sh
# Using pip
python -m pip install uv

## Update uv
# self update
uv self update
# with pip
python -m pip install --upgrade uv
```

For more details, check out the [official uv document](https://docs.astral.sh/uv/).

From the repository root, create `.venv` only if it does not already exist:

```shell
uv venv .venv --python 3.12
uv sync --locked

# To activate the virtual environment:
source .venv/bin/activate

# Or you may run the script using `uv run python ...` without activating the virtual environment
uv run python run_wisdom.py --help
```


## Quick validation

From the source checkout, use the existing `.venv` with `uv`, `CUDA_VISIBLE_DEVICES` to activate (or deactivate) GPU:

```shell
cd /shared/storage/cs/scratch/lrr550/package_wisdom/Wisdom
uv sync --locked --group test --extra bo

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
TMPDIR=/tmp uv build --wheel --config-setting=--build-option=--keep-temp --out-dir ../wisdom-wheelhouse

# In the Python environment where you want to use WISDOM:
python -m pip install ../wisdom-wheelhouse/wisdom-0.1.0-py3-none-any.whl
# Optional BoTorch backend (detection is included in the base package):
python -m pip install '../wisdom-wheelhouse/wisdom-0.1.0-py3-none-any.whl[bo]'
python -m run_wisdom --help
```

With `uv`, use `uv pip install` instead of `python -m pip install`. Rebuild the
wheel after changing source. The wheel output above is outside the checkout.
`--keep-temp` avoids a setuptools temporary-directory cleanup failure on this
shared filesystem; it retains intermediate build files.

The repository-root `build/` (temporary package copies) and `wisdom.egg-info/`
(generated setuptools metadata) are disposable, ignored build artifacts, not
source code or runtime dependencies. They can be removed after a build/install
finishes; later builds or editable installs may regenerate them. Start with a
fresh `build/` when changing package contents so stale files cannot enter the
wheel. Keep `.venv/`, `wisdom/`, `pyproject.toml`, `uv.lock` and `setup.py`;
do not confuse root-level `wisdom.egg-info/` with installed environment metadata.
Datasets, research model weights and CIFAR model factories are not included in
the wheel. Dataset helpers and their tests are source-checkout utilities.

The base package includes PyTorch/torchvision, Captum, sklearn, PyYAML and
Ultralytics. BoTorch remains optional; pose needs no TensorRT extra.
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

For the trusted CIFAR-10 ResNet18 supplied in the source checkout (use `python run_wisdom.py ...` if inside an virtual environment):

```shell
uv run python run_wisdom.py \
  --mode wisdom --task classification \
  --weights-path ./models_info/saved_models/resnet18_CIFAR10_whole.pth \
  --checkpoint-format module \
  --build-data-path /path/to/cifar-10-imagefolder/build \
  --validation-data-path /shared/storage/cs/scratch/lrr550/datasets/cifar-10-imagefolder/validation \
  --test-data-path /path/to/cifar-10-imagefolder/test \
  --image-size 32 --normalize custom \
  --normalize-mean 0.4914,0.4822,0.4465 \
  --normalize-std 0.2023,0.1994,0.2010 \
  --wisdom-csv ./saved_files/pre_csv/resnet18_cifar10.csv \
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
  --mode wisdom \
  --task classification \
  --weights-path /path/to/classifier.pth \
  --checkpoint-format state-dict \
  --model-factory my_package.models:make_classifier \
  --build-data-path /path/to/dataset/build \
  --test-data-path /path/to/dataset/test \
  --image-size 32 \
  --normalize none \
  --wisdom-csv /path/to/classifier_wisdom.csv \
  --output-json ./results/classifier_coverage.json \
  --device cpu
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

A trusted standard Ultralytics checkpoint such as `yolo11n.pt` already contains
its model architecture. Use `--checkpoint-format module` without `--model-path`:

```shell
uv run python run_wisdom.py \
  --mode wisdom --task detection \
  --weights-path ./models_info/saved_models/yolo11n.pt \
  --checkpoint-format module \
  --build-data-path /path/to/coco_build.txt \
  --test-data-path /path/to/coco_test.txt \
  --wisdom-csv ./saved_files/pre_csv/wisdom_yolo11n_scores_5k.csv \
  --output-json ./results/yolo11n_coco_coverage.json \
  --imgsz 640 \
  --batch-size 8 \
  --num-workers 4 \
  --selection-mode global \
  --top-m-neurons 10 \
  --cluster-method KMeans \
  --n-clusters 2 \
  --seed 42 \
  --device cpu
```

The two `.txt` inputs list one image path per line (absolute paths, or paths
relative to the list file). For local COCO, build images come from
`/path/to/coco/images/train2017` and test images
from `images/val2017`, which has ground truth; do not use the unlabeled
`test2017` split for detection-quality metrics. An existing nonempty WISDOM CSV
is reused, but clustering still fits on build images. BO is off unless `--bo`
is supplied, and no validation input is needed without BO.

For a quick check, limit each list to a small, disjoint subset (for example,
256 train and 256 validation images). If selecting only images with existing
label files, report that restriction: it is not a full COCO evaluation. This
runner rejects *partial* label presence rather than assuming every missing
label is an empty image. Some COCO label exports omit files for empty/crowd-only
images, so using the full split requires complete, verified YOLO labels,
including empty files where appropriate. Do not create empty labels for
unverified missing annotations.

Ultralytics supplies the Python layer definitions needed to unpickle `.pt`
files; a separate YOLO source clone or YAML is unnecessary for this format.
Only use `module` with trusted weights. WISDOM loads on CPU, selects the saved
EMA model when present, converts FP16 exports to FP32, and then moves to the
requested device. GPU can be used by `--device cuda:0`.

For a raw state dictionary instead, use a local Ultralytics YAML plus matching
weights:

```shell
uv run python run_wisdom.py \
  --mode wisdom \
  --task detection \
  --model-path /path/to/yolo.yaml \
  --weights-path /path/to/yolo_state_dict.pth \
  --checkpoint-format state-dict \
  --build-data-path /path/to/build/images \
  --test-data-path /path/to/test/images \
  --imgsz 640 \
  --selection-mode per-group \
  --num-groups 3 \
  --num-layers 9 \
  --wisdom-csv /path/to/yolo_wisdom.csv \
  --output-json ./results/yolo_coverage.json \
  --device cpu
```

Detection uses RGB inputs scaled to `[0,1]`; classification normalization flags
do not change this path. For metrics, provide matching YOLO `.txt` labels in
`labels/` next to `images/`. Empty label files mean no objects; if every label
file is absent, metrics are unavailable. Partial label presence is an error.
Reported precision/recall/F1 use confidence >= 0.25, class-aware NMS at IoU
0.45 and matching at IoU >= 0.5. Inputs are resized directly to a square, not
Ultralytics letterboxed. These are WISDOM runner metrics, **not** official COCO
mAP or an Ultralytics `val` benchmark. 


### Pose / trt_pose

```shell
uv run python run_wisdom.py \
  --mode wisdom \
  --task pose \
  --weights-path /path/to/random_pose.pth \
  --checkpoint-format state-dict \
  --pose-topology /path/to/human_pose.json \
  --pose-architecture resnet18_baseline_att \
  --build-data-path /path/to/build \
  --test-data-path /path/to/test \
  --image-size 224 \
  --wisdom-csv /path/to/pose_wisdom.csv \
  --output-json ./results/pose_coverage.json \
  --device cpu
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

`--top-m-neurons` means M per scope: globally, per dynamic group, or per
considered layer. The same per-scope budget is respected in consensus fusion.
`--num-groups` chooses the group count; `--num-layers` evenly limits discovered
eligible layers before partitioning. Omit the latter to consider all eligible
layers. Impossible counts are errors; no YOLO-specific layer ranges are needed.

BO is available in WISDOM mode. All controls below have defaults:

```shell
uv run python run_wisdom.py \
  --mode wisdom \
  --task classification \
  --weights-path ./classifier.pth \
  --checkpoint-format state-dict \
  --model-factory my_package.models:make_classifier \
  --build-data-path ./data/build \
  --test-data-path ./data/test \
  --output-json ./results/classifier_bo.json \
  --bo \
  --bo-backend auto \
  --bo-init 3 \
  --bo-iter 3 \
  --bo-candidate-pool-size 32 \
  --bo-cluster-methods KMeans,MiniBatchKMeans,Birch \
  --bo-n-clusters 2,3,4 \
  --device cpu
```

BO maximizes Pearson correlation between coverage and the task metric over
up to five ordered, nested validation subsets. It tunes clustering, not model
weights. Constant metric/coverage series yield objective zero. The chosen
configuration and BO history path appear in terminal output and JSON; without
BO the fixed clustering configuration is printed instead.

### Direct neuron-score pretraining

These scripts generate the consensus CSV only; they do not train model weights
or calculate final coverage. Inputs below are local placeholders as above.
`--top-m` here corresponds to the runner's `--top-m-neurons`.

```shell
uv run python wisdom_classification_train.py \
  --model-path /path/to/classifier.pth \
  --checkpoint-format state-dict \
  --model-factory my_package.models:make_classifier \
  --imagefolder-root /path/to/build \
  --image-size 32 \
  --normalize none \
  --methods lgxa lig lgs \
  --top-m 1 \
  --num-layers 1 \
  --out-csv ./saved_files/pre_csv/classifier_wisdom.csv \
  --device cpu

uv run python wisdom_yolo_train.py \
  --weights ./models_info/saved_models/yolo11n.pt \
  --img-dir /path/to/coco/images/train2017 \
  --imgsz 640 \
  --num-images 4 \
  --methods lgxa lig lgs \
  --top-m 1 \
  --selection-mode per-group \
  --num-groups 2 \
  --num-layers 2 \
  --out-csv ./saved_files/pre_csv/yolo_wisdom.csv \
  --device cpu

uv run python wisdom_pose_train.py \
  --model-path /path/to/random_pose.pth \
  --checkpoint-format state-dict \
  --pose-topology /path/to/human_pose.json \
  --pose-architecture resnet18_baseline_att \
  --img-dir /path/to/data/pose/build \
  --image-size 32 \
  --methods lgxa lig lgs \
  --top-m 1 \
  --num-layers 1 \
  --out-csv ./saved_files/pre_csv/pose_wisdom.csv \
  --device cpu
```

For normal YOLO pretraining, `--weights` points to a **trusted, existing `.pt`**
checkpoint. It contains the model architecture, so no separate model YAML is
required. The script's default already names a `.pt` file (`weights/yolo11n.pt`);
the example above explicitly supplies the checkpoint's location in this checkout.
A model YAML passed to `--weights` constructs an **untrained, randomly initialized
model**, retained for synthetic integration tests and architecture experiments;
it does not load pretrained weights. Use `run_wisdom.py --checkpoint-format
state-dict --model-path ...yaml` for a separate YAML plus raw state dictionary.

The YOLO script's optional `--data` YAML describes the **dataset**, not the model.
When `--img-dir` is supplied, no dataset YAML is read. Neither kind of YAML needs
to be supplied alongside `.pt` + `--img-dir`.

The three examples explicitly use `lgxa lig lgs`. Omitting `--methods` preserves
these task-specific defaults in both the scripts and shared pretraining API:

| Task | Default attribution methods |
| --- | --- |
| Classification | `lrp ldl lig` |
| YOLO detection | `lgxa lig lgs` |
| Pose | `lgxa lig` |

All defaults use at least two distinct methods for consensus voting. An explicit
single method such as `--methods la` remains available for a quick smoke test.
`run_wisdom.py` delegates to the same defaults when it needs to generate a missing
or empty WISDOM CSV; an existing nonempty CSV is reused instead.

BO runs in `run_wisdom.py` after score pretraining and before coverage, not in
these CSV-only scripts. Wheel users replace `uv run python <script>.py` with
`python -m <script>`.

#### Attribution method options

Pass space-separated identifiers to `--methods` (not a comma-separated string).
The current Captum backend registers all of the following:

| Identifier | Captum method |
| --- | --- |
| `lc` | LayerConductance |
| `la` | LayerActivation |
| `ii` | InternalInfluence |
| `lgxa` | LayerGradientXActivation |
| `lgc` | LayerGradCam |
| `ldl` | LayerDeepLift |
| `ldls` | LayerDeepLiftShap |
| `lgs` | LayerGradientShap |
| `lig` | LayerIntegratedGradients |
| `lfa` | LayerFeatureAblation |
| `lrp` | LayerLRP |

Registered does not mean compatible with every model/layer. LRP and DeepLift
depend on Captum's supported operators and module-reuse restrictions;
DeepLiftShap needs multiple baseline examples (the current backend uses the
batch-shaped zero baseline, so a batch of one is unsuitable). GradientShap is
stochastic. LayerGradCam currently aggregates channels into a spatial heatmap,
so prefer `lgxa`, `lig` and `lgs` for channel/neuron ranking. `la` measures
activations without a target and is useful for fast smoke tests. Multiple
gradient-based methods cost more time/memory than a single `la` pass; reduce
batch size, image count or considered layers for a quick check.

## Docker

Docker uses the locked uv environment and packaged pose definitions.
It installs neither TensorRT nor `torch2trt`, and downloads no model weights.
The test target includes detection and BO dependencies and forces CPU tests;
the locked Torch distribution is CUDA-capable, not a minimal CPU-only wheel.
GPU runs require a compatible host driver and NVIDIA container runtime.

```shell
cd Wisdom
docker build -f Docker/Dockerfile --target test -t wisdom .
docker run --rm --network none wisdom \
  uv run --offline --no-sync pytest -q tests/test_wisdom_e2e.py
```
