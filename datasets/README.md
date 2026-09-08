# Datasets preparation

We use the following datasets in our evalution.

- [x] [**CV**] MNIST
- [x] [**CV**] CIFAR10
- [x] [**CV**] ImageNet. Should manually download from either [the official release](https://www.image-net.org/download.php) or [kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)
- [x] [**CV**] COCO2017 images for detection and pose coverage (see below).

## COCO2017 layout

The local dataset root is `/shared/storage/cs/scratch/lrr550/datasets/coco`.
The relevant files are:

```text
coco/
  images/train2017/*.jpg
  images/val2017/*.jpg
  annotations/instances_train2017.json
  annotations/instances_val2017.json
  annotations/person_keypoints_train2017.json
  annotations/person_keypoints_val2017.json
```

Use `train2017` for build/score generation and the separate `val2017` split
for evaluation. For a quick check, use small, disjoint directories of symlinks
to the original images; there is no need to copy or resize the originals.
Person-containing subsets can be selected from the keypoint annotations using
`num_keypoints > 0` and `iscrowd == 0`.

Detection bounding boxes (`instances_*.json` or YOLO box-label TXT files) are
not pose keypoint targets. COCO's `test2017` images do not have public ground
truth suitable for these evaluation examples.

## trt_pose topology and annotation preprocessing

`human_pose.json` defines the NVIDIA human-pose topology:
18 keypoints (the 17 COCO keypoints plus **neck**) and 21 skeleton links.

`preprocess_coco_person.py` is copied from NVIDIA's
[trt_pose/tasks/human_pose](https://github.com/NVIDIA-AI-IOT/trt_pose/tree/master/tasks/human_pose) with its algorithm unchanged. It adds the neck at
the shoulder midpoint, propagates shoulder visibility, and adjusts the
skeleton to match `human_pose.json`.