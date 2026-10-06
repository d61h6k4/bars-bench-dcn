import numpy as np
import pytest
import torch

from bars_dcn.batches import RowGather, ShuffledBatches, make_loader
from bars_dcn.estimator import DCNClassifier


def _data(n=103):
    x = np.arange(n * 3, dtype=np.int32).reshape(n, 3)
    return x, np.arange(n, dtype=np.float32)


def _epoch_rows(loader):
    return [row for index, _ in loader for row in (index[:, 0] // 3).tolist()]


def test_order_is_the_seeded_randperm_of_the_previous_loop():
    x, y = _data()
    generator = torch.Generator().manual_seed(7)
    reference = torch.Generator().manual_seed(7)
    loader = make_loader(x, y, 16, generator)
    for _ in range(3):  # a fresh permutation per epoch, drawn in the same sequence
        assert _epoch_rows(loader) == torch.randperm(103, generator=reference).tolist()


@pytest.mark.parametrize("workers", [0, 2])
def test_batches_hold_the_matching_rows_and_labels(workers):
    x, y = _data()
    loader = make_loader(x, y, 16, torch.Generator().manual_seed(0), num_workers=workers)
    batches = list(loader)
    assert [len(b[0]) for b in batches] == [16] * 6 + [7]
    for index, labels in batches:
        assert index.dtype == torch.int32
        assert labels.dtype == torch.float32
        assert (index[:, 0] // 3).tolist() == labels.long().tolist()  # row i has label i


def test_order_does_not_depend_on_the_number_of_workers():
    x, y = _data()
    orders = [
        _epoch_rows(make_loader(x, y, 16, torch.Generator().manual_seed(3), num_workers=w))
        for w in (0, 2)
    ]
    assert orders[0] == orders[1]


def test_a_trailing_single_row_batch_is_dropped():
    sampler = ShuffledBatches(33, 16, torch.Generator().manual_seed(0))
    batches = list(sampler)
    assert [len(b) for b in batches] == [16, 16]
    assert len(sampler) == 2
    assert len(ShuffledBatches(34, 16, torch.Generator())) == 3  # a 2-row tail is kept


def test_workers_share_the_data_instead_of_copying_it():
    x, y = _data()
    assert not RowGather(x, y).index.is_shared()
    assert RowGather(x, y, share_memory=True).index.is_shared()


def test_training_results_do_not_depend_on_num_workers():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, size=(200, 3))
    y = (x[:, 0] > 2).astype(int)

    def fit(workers):
        return DCNClassifier(
            embedding_dim=2, parallel_hidden_units=[4], batch_size=64, max_epochs=2,
            device="cpu", num_workers=workers, random_state=0,
        ).fit(x, y, eval_set=(x, y))  # fmt: skip

    assert fit(0).history_ == fit(2).history_
