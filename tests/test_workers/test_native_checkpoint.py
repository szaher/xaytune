"""The Native managed cursor is a sample position, not a batch index."""

from __future__ import annotations

import pytest

from xaytune.workers.native_checkpoint import ExactIndexedSampler


def test_indexed_sampler_continues_exactly_across_microbatch_change() -> None:
    first = ExactIndexedSampler(12, seed=17)
    order = list(first)
    # Two original batches of two samples were committed at an optimizer step.
    resumed = ExactIndexedSampler(12, seed=17, offset=4)
    assert list(resumed) == order[4:]
    assert len(resumed) == 8
    assert len(set(order[:4]) & set(resumed)) == 0
    assert sorted([*order[:4], *resumed]) == list(range(12))
    resumed.set_epoch(1)
    assert list(resumed) == list(ExactIndexedSampler(12, seed=17, epoch=1))


@pytest.mark.parametrize("size,epoch,offset", [(0, 0, 0), (4, -1, 0), (4, 0, 5)])
def test_indexed_sampler_rejects_invalid_position(size: int, epoch: int, offset: int) -> None:
    with pytest.raises(ValueError, match="invalid exact indexed sampler position"):
        ExactIndexedSampler(size, 17, epoch=epoch, offset=offset)
