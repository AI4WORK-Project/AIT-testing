from __future__ import annotations

import importlib
import json
from pathlib import Path

from PIL import Image
import pytest
from torchvision.datasets import ImageFolder


class TinyVisionDataset:
    classes = ["zero", "one"]

    def __init__(self, targets: list[int], *, pixel_offset: int = 0) -> None:
        self.targets = targets
        self.pixel_offset = pixel_offset

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        target = self.targets[index]
        return Image.new(
            "RGB",
            (3, 2),
            color=(self.pixel_offset + index, target, 0),
        ), target


def _converter():
    return importlib.import_module("convert_torchvision_dataset")


def test_export_imagefolder_uses_disjoint_stratified_train_validation_and_test(
    tmp_path: Path,
) -> None:
    """Losing samples, label indices, or the 90/10 split must fail this test."""
    converter = _converter()
    destination = tmp_path / "cifar10-imagefolder"

    result = converter.export_imagefolder(
        dataset_name="cifar10",
        train_dataset=TinyVisionDataset([0] * 10 + [1] * 10),
        test_dataset=TinyVisionDataset([0, 0, 1, 1]),
        destination=destination,
        validation_fraction=0.10,
        seed=42,
    )

    assert result.counts == {"build": 18, "validation": 2, "test": 4}
    assert sorted(path.name for path in (destination / "build").iterdir()) == [
        "000_zero",
        "001_one",
    ]
    assert len(list(destination.glob("build/*/*.png"))) == 18
    assert len(list(destination.glob("validation/*/*.png"))) == 2
    assert len(list(destination.glob("test/*/*.png"))) == 4
    converted_build = ImageFolder(destination / "build")
    converted_validation = ImageFolder(destination / "validation")
    converted_test = ImageFolder(destination / "test")
    expected_mapping = {"000_zero": 0, "001_one": 1}
    assert converted_build.class_to_idx == expected_mapping
    assert converted_validation.class_to_idx == expected_mapping
    assert converted_test.class_to_idx == expected_mapping
    assert converted_validation.targets.count(0) == 1
    assert converted_validation.targets.count(1) == 1
    manifest = json.loads((destination / "conversion.json").read_text())
    assert manifest["dataset"] == "cifar10"
    assert manifest["validation_fraction"] == pytest.approx(0.10)
    assert manifest["seed"] == 42
    assert manifest["counts"] == result.counts


def test_export_membership_is_seeded_and_official_test_pixels_are_preserved(
    tmp_path: Path,
) -> None:
    """Nondeterminism, source mutation, or substituting train data for test must fail."""
    converter = _converter()
    train = TinyVisionDataset([0] * 10 + [1] * 10)
    test = TinyVisionDataset([0, 0, 1, 1], pixel_offset=100)
    source_targets = (list(train.targets), list(test.targets))

    for name in ("first", "second"):
        converter.export_imagefolder(
            dataset_name="cifar10",
            train_dataset=train,
            test_dataset=test,
            destination=tmp_path / name,
            validation_fraction=0.10,
            seed=17,
        )

    for split in ("build", "validation"):
        first_root = tmp_path / "first" / split
        second_root = tmp_path / "second" / split
        first = sorted(path.relative_to(first_root) for path in first_root.glob("*/*.png"))
        second = sorted(
            path.relative_to(second_root) for path in second_root.glob("*/*.png")
        )
        assert first == second

    build_names = {path.name for path in (tmp_path / "first" / "build").glob("*/*.png")}
    validation_names = {
        path.name for path in (tmp_path / "first" / "validation").glob("*/*.png")
    }
    assert build_names.isdisjoint(validation_names)
    assert list(train.targets) == source_targets[0]
    assert list(test.targets) == source_targets[1]
    for index, target in enumerate(test.targets):
        path = (
            tmp_path
            / "first"
            / "test"
            / f"{target:03d}_{TinyVisionDataset.classes[target]}"
            / f"{index:06d}.png"
        )
        with Image.open(path) as exported:
            assert exported.getpixel((0, 0)) == (100 + index, target, 0)


def test_export_imagefolder_refuses_to_mix_with_existing_destination(
    tmp_path: Path,
) -> None:
    """A conversion must never silently overwrite or mix an existing dataset."""
    converter = _converter()
    destination = tmp_path / "mnist-imagefolder"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        converter.export_imagefolder(
            dataset_name="mnist",
            train_dataset=TinyVisionDataset([0, 0, 1, 1]),
            test_dataset=TinyVisionDataset([0, 1]),
            destination=destination,
            validation_fraction=0.10,
            seed=42,
        )

    assert marker.read_text(encoding="utf-8") == "user data"


def test_export_imagefolder_refuses_a_dangling_destination_symlink(
    tmp_path: Path,
) -> None:
    """Resolving a dangling final symlink must not redirect conversion writes."""
    converter = _converter()
    redirected_target = tmp_path / "outside-destination"
    destination = tmp_path / "cifar10-imagefolder"
    destination.symlink_to(redirected_target, target_is_directory=True)
    assert destination.is_symlink()
    assert not destination.exists()

    with pytest.raises(FileExistsError, match="already exists"):
        converter.export_imagefolder(
            dataset_name="cifar10",
            train_dataset=TinyVisionDataset([0] * 10 + [1] * 10),
            test_dataset=TinyVisionDataset([0, 1]),
            destination=destination,
            validation_fraction=0.10,
            seed=42,
        )

    assert destination.is_symlink()
    assert not redirected_target.exists()


@pytest.mark.parametrize("dataset_name", ["cifar10", "cifar100", "mnist"])
def test_torchvision_loader_never_requests_a_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_name: str,
) -> None:
    """Changing the converter to download data would break offline operation."""
    converter = _converter()
    calls: list[tuple[bool, bool]] = []

    class FakeDataset(TinyVisionDataset):
        def __init__(self, *, root: str, train: bool, download: bool) -> None:
            calls.append((train, download))
            super().__init__([0, 0, 1, 1])

    monkeypatch.setitem(converter.DATASET_FACTORIES, dataset_name, FakeDataset)

    train, test = converter.load_torchvision_splits(dataset_name, tmp_path)

    assert len(train) == len(test) == 4
    assert calls == [(True, False), (False, False)]


def test_cli_defaults_output_below_the_supplied_dataset_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    converter = _converter()
    train = TinyVisionDataset([0] * 10 + [1] * 10)
    test = TinyVisionDataset([0, 0, 1, 1])
    monkeypatch.setattr(
        converter,
        "load_torchvision_splits",
        lambda dataset_name, data_root: (train, test),
    )

    status = converter.main(
        ["--dataset", "cifar10", "--data-root", str(tmp_path)]
    )

    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["destination"] == str(
        (tmp_path / "cifar10-imagefolder").resolve()
    )
    assert (tmp_path / "cifar10-imagefolder" / "conversion.json").is_file()


def test_cli_refuses_a_dangling_explicit_output_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI path normalization must not bypass the exporter's symlink guard."""
    converter = _converter()
    train = TinyVisionDataset([0] * 10 + [1] * 10)
    test = TinyVisionDataset([0, 1])
    monkeypatch.setattr(
        converter,
        "load_torchvision_splits",
        lambda dataset_name, data_root: (train, test),
    )
    redirected_target = tmp_path / "outside-destination"
    destination = tmp_path / "explicit-output"
    destination.symlink_to(redirected_target, target_is_directory=True)

    with pytest.raises(FileExistsError, match="already exists"):
        converter.main(
            [
                "--dataset",
                "cifar10",
                "--data-root",
                str(tmp_path),
                "--output-root",
                str(destination),
            ]
        )

    assert destination.is_symlink()
    assert not redirected_target.exists()
