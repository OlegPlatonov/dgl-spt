import sys
import torch
import pytest

sys.path.append("../")
sys.path.append("./")

from dataset import SpatioTemporalSampler


@pytest.mark.parametrize("size, batch_size, seed", [(10, 2, 42), (12, 3, 123)])
def test_spatio_temporal_sampler_no_shuffle(size, batch_size, seed):
    """
    Test that when shuffle=False, the sampler yields ascending indices [0, 1, 2, ..., size-1]
    and that it is reproducible for the same seed.
    """
    sampler = SpatioTemporalSampler(
        size=size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        number_of_batches_to_skip=0
    )
    indices = list(iter(sampler))

    assert indices == list(range(size))

    sampler2 = SpatioTemporalSampler(
        size=size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        number_of_batches_to_skip=0
    )
    indices2 = list(iter(sampler2))
    assert indices == indices2


@pytest.mark.parametrize("size, batch_size, seed", [(10, 2, 42), (12, 3, 123)])
def test_spatio_temporal_sampler_shuffle(size, batch_size, seed):
    """
    Test that when shuffle=True, the sampler yields a shuffled permutation of the indices [0..size-1].
    Also test reproducibility when using the same seed.
    """
    sampler = SpatioTemporalSampler(
        size=size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        number_of_batches_to_skip=0
    )
    indices = list(iter(sampler))
    # We expect indices to be some permutation of 0..(size-1)
    assert sorted(indices) == list(range(size))

    # Check reproducibility: creating a second sampler with the same seed and shuffle=True
    # should result in the exact same random permutation.
    sampler2 = SpatioTemporalSampler(
        size=size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        number_of_batches_to_skip=0
    )
    indices2 = list(iter(sampler2))
    assert indices == indices2


@pytest.mark.parametrize("size, batch_size, seed, skip", [
    (10, 2, 42, 1),  # skip first batch (2 items)
    (10, 2, 42, 2),  # skip first 2 batches (4 items)
    (12, 3, 42, 2),  # skip first 2 batches (6 items)
])
def test_spatio_temporal_sampler_skip_batches_no_shuffle(size, batch_size, seed, skip):
    """
    Test that when shuffle=False and we skip `number_of_batches_to_skip` batches,
    the sampler yields the correct slice of indices.
    """
    sampler = SpatioTemporalSampler(
        size=size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        number_of_batches_to_skip=skip
    )
    indices = list(iter(sampler))

    expected_skipped_amount = skip * batch_size
    expected = list(range(size))[expected_skipped_amount:]
    assert indices == expected


@pytest.mark.parametrize("size, batch_size, seed, skip", [
    (10, 2, 42, 1),
    (10, 2, 42, 2),
    (12, 3, 42, 2),
])
def test_spatio_temporal_sampler_skip_batches_shuffle(size, batch_size, seed, skip):
    """
    Test that when shuffle=True and we skip `number_of_batches_to_skip` batches,
    the sampler yields the correct slice of a shuffled sequence.
    We also verify reproducibility with the same seed and skip.
    """
    sampler = SpatioTemporalSampler(
        size=size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        number_of_batches_to_skip=skip
    )
    indices = list(iter(sampler))
    # indices should be a permutation of 0..(size-1) with the first skip*batch_size items removed

    # Create the reference permutation (full shuffle) so we can see what we expect after skipping
    # internal function to mimic what the sampler does:
    generator = torch.Generator().manual_seed(seed)
    full_indices = torch.randperm(n=size, generator=generator).tolist()
    expected_skipped_amount = skip * batch_size
    expected = full_indices[expected_skipped_amount:]
    assert indices == expected

    # Check reproducibility with a second sampler having the same parameters
    sampler2 = SpatioTemporalSampler(
        size=size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        number_of_batches_to_skip=skip
    )
    indices2 = list(iter(sampler2))
    assert indices2 == indices
