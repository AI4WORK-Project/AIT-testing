from __future__ import annotations


def test_core_library_entry_points_are_exported() -> None:
    from wisdom import (
        ConsensusWisdom,
        WisdomIDC,
        train_wisdom_classification,
        train_wisdom_pose,
        train_wisdom_yolo,
    )

    assert ConsensusWisdom is not None
    assert WisdomIDC is not None
    assert callable(train_wisdom_classification)
    assert callable(train_wisdom_yolo)
    assert callable(train_wisdom_pose)
