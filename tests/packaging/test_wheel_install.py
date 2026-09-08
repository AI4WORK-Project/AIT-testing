from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile


REPOSITORY = Path(__file__).parents[2]


def _run(*command: str, cwd: Path = REPOSITORY, env=None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_setup_metadata_comes_from_pyproject() -> None:
    project = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())["project"]
    for argument, expected in (("--name", project["name"]), ("--version", project["version"])):
        result = _run(sys.executable, "setup.py", argument)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected
        assert "overwritten" not in result.stderr
        assert "_MissingDynamic" not in result.stderr

    check = _run(sys.executable, "setup.py", "check")
    assert check.returncode == 0, check.stderr
    assert "deprecated" not in check.stderr.lower()


def test_release_metadata_excludes_legacy_coverage() -> None:
    metadata = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())
    assert "legacy-coverage" not in metadata["project"]["optional-dependencies"]
    assert "coverage_methods*" not in metadata["tool"]["setuptools"]["packages"]["find"]["include"]


def test_wheel_contains_runner_and_packaged_pose_models(tmp_path) -> None:
    with tempfile.TemporaryDirectory(prefix="wisdom-wheel-", dir="/tmp") as temp:
        local_root = Path(temp) / "source"
        local_root.mkdir()
        for filename in (
            "README.md",
            "pyproject.toml",
            "setup.py",
            "wisdom_classification_train.py",
            "wisdom_yolo_train.py",
            "wisdom_pose_train.py",
            "run_wisdom.py",
            "convert_torchvision_dataset.py",
        ):
            shutil.copy2(REPOSITORY / filename, local_root / filename)
        for package in ("wisdom",):
            shutil.copytree(
                REPOSITORY / package,
                local_root / package,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        local_models_info = local_root / "models_info"
        local_models_info.mkdir()
        shutil.copy2(REPOSITORY / "models_info" / "__init__.py", local_models_info)
        shutil.copytree(
            REPOSITORY / "models_info" / "trt_pose",
            local_models_info / "trt_pose",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        wheel_dir = Path(temp) / "wheel"
        build = _run(
            "uv",
            "build",
            "--offline",
            "--wheel",
            "--config-setting=--build-option=--keep-temp",
            "--out-dir",
            str(wheel_dir),
            cwd=local_root,
        )
        assert build.returncode == 0, build.stderr
        wheels = list(wheel_dir.glob("wisdom-*.whl"))
        assert len(wheels) == 1
        wheel = wheels[0]

        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            entry_points_names = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
            if entry_points_names:
                assert "wisdom =" not in archive.read(entry_points_names[0]).decode()

        assert any(name.startswith("wisdom/") for name in names)
        assert not any(name.startswith("coverage_methods/") for name in names)
        for module in (
            "wisdom_classification_train.py",
            "wisdom_yolo_train.py",
            "wisdom_pose_train.py",
            "convert_torchvision_dataset.py",
        ):
            assert module in names
        assert "run_wisdom.py" in names
        assert "wisdom/cli.py" not in names
        assert "models_info/__init__.py" in names
        assert "models_info/trt_pose/models/resnet.py" in names
        assert "models_info/trt_pose/LICENSE.md" in names
        assert not any(name.startswith("models_info/models_cv/") for name in names)
        assert not any(
            name.startswith(("tests/", "trt_pose/", "build/", "wisdom_package/", "docs/"))
            for name in names
        )
        assert "run_wisdom_new.py" not in names and "run_wisdom_old.py" not in names
        assert not any(name.endswith((".pt", ".pth")) for name in names)

        target = Path(temp) / "site"
        install = _run(
            "uv",
            "pip",
            "install",
            "--offline",
            "--target",
            str(target),
            "--no-deps",
            str(wheel),
        )
        assert install.returncode == 0, install.stderr
        outside = Path(temp) / "outside"
        outside.mkdir()
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(target)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        imported = _run(
            sys.executable,
            "-c",
            (
                "import pathlib, wisdom, run_wisdom, models_info; "
                "import wisdom_classification_train, wisdom_yolo_train, wisdom_pose_train; "
                f"target=pathlib.Path({str(target)!r}).resolve(); "
                "modules=(wisdom,run_wisdom,models_info,wisdom_classification_train,"
                "wisdom_yolo_train,wisdom_pose_train); "
                "assert all(pathlib.Path(m.__file__).resolve().is_relative_to(target) for m in modules)"
            ),
            cwd=outside,
            env=environment,
        )
        assert imported.returncode == 0, imported.stderr
        help_result = _run(
            sys.executable,
            "-m",
            "run_wisdom",
            "--help",
            cwd=outside,
            env=environment,
        )
        assert help_result.returncode == 0, help_result.stderr
        assert all(task in help_result.stdout for task in ("classification", "detection", "pose"))
        pose_probe = _run(
            sys.executable,
            "-c",
            (
                "import json, pathlib, tempfile; "
                "from wisdom.tasks.pose import build_trt_pose_model; "
                "temp=tempfile.TemporaryDirectory(); root=pathlib.Path(temp.name); topology=root/'pose.json'; "
                "topology.write_text(json.dumps({'keypoints':[str(i) for i in range(18)],"
                "'skeleton':[[i % 18 + 1, (i + 1) % 18 + 1] for i in range(21)]})); "
                "model=build_trt_pose_model(architecture='resnet18_baseline_att', topology_path=topology); "
                "assert len(model.state_dict()) == 172; temp.cleanup()"
            ),
            cwd=outside,
            env=environment,
        )
        assert pose_probe.returncode == 0, pose_probe.stderr
        for module in (
            "wisdom_classification_train",
            "wisdom_yolo_train",
            "wisdom_pose_train",
            "convert_torchvision_dataset",
        ):
            help_result = _run(
                sys.executable,
                "-m",
                module,
                "--help",
                cwd=outside,
                env=environment,
            )
            assert help_result.returncode == 0, help_result.stderr

        # Execute the runtime suite against the installed wheel, not the checkout.
        # Only source-policy/legacy tests and this packaging test are excluded:
        # they inspect the source tree or a feature deliberately not distributed.
        shutil.copytree(
            REPOSITORY / "tests",
            outside / "tests",
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "packaging",
                "test_torch_load_policy.py", "test_legacy_coverage_import.py",
            ),
        )
        runtime = _run(
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "-c", str(local_root / "pyproject.toml"), "--rootdir", str(outside),
            str(outside / "tests"),
            cwd=outside,
            env=environment,
        )
        assert runtime.returncode == 0, runtime.stdout + runtime.stderr
        print("Installed-wheel runtime suite:\n" + runtime.stdout)
