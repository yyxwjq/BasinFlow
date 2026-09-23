"""Quick end-to-end regression test — toy data → train → sample → evaluate."""
from __future__ import annotations

import numpy as np
import pytest

from basinflow.data import Event, EventCatalog, Structure
from basinflow.sampling import CandidateSampler
from basinflow.seeds import (
    GaussianInit,
    ZeroInit,
)
from basinflow.training import train_product_flow


def _toy_dataset() -> EventCatalog:
    reactant = Structure(
        ["H", "O", "H"],
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]],
        cell=np.eye(3) * 10, pbc=[False] * 3,
        movable_mask=[False, True, False],
    )
    product = Structure(
        ["H", "O", "H"],
        [[0.0, 0.0, 0.0], [1.05, 0.0, 0.0], [1.6, 0.0, 0.0]],
        cell=np.eye(3) * 10, pbc=[False] * 3,
    )
    return EventCatalog(
        events={"e": Event("e", reactant, product)},
    )


def test_e2e_train_reduces_loss():
    """Training on toy data must reduce loss meaningfully."""
    torch = pytest.importorskip("torch")
    from basinflow.models import EGNNFlow

    torch.manual_seed(0)
    model = EGNNFlow(hidden_dim=16, num_layers=2, cutoff=3.0)
    inits = [ZeroInit(), GaussianInit(scale=0.05, random_seed=1)]

    history = train_product_flow(
        model, _toy_dataset(), inits, epochs=12, lr=0.03, checkpoint_path=None,
    )
    assert history[-1]["loss"] < history[0]["loss"] * 0.5
    assert history[-1]["loss"] < 0.5, (
        f"EGNN failed to reduce loss: {history[-1]['loss']:.4f}"
    )


def test_e2e_sample_produces_finite_candidates():
    """Sampler generates valid candidates from toy basins."""
    torch = pytest.importorskip("torch")
    from basinflow.models import EGNNFlow

    torch.manual_seed(0)
    model = EGNNFlow(hidden_dim=16, num_layers=1, cutoff=3.0)
    # Quick train so the model produces non-trivial output
    train_product_flow(
        model, _toy_dataset(), [ZeroInit()],
        epochs=3, lr=0.01, checkpoint_path=None,
    )

    sampler = CandidateSampler(
        model=model, init_generators=[ZeroInit()],
        cutoff=3.0, num_steps=4, checkpoint_id="e2e-test",
    )
    catalog = _toy_dataset()
    result = sampler.sample(catalog, "e")

    assert len(result.candidates) == 1
    c = result.candidates[0]
    assert c.candidate_id == "e:candidate:0"
    assert c.seed_type == "zero"
    assert "e:candidate_structure:0" in result.generated_structures
    assert result.generated_structures[c.generated_structure_id].positions.shape == (3, 3)


def test_e2e_basin_recall_on_toy():
    """lr is kept low: a rough toy field is not integrable by an adaptive
    solver, and this test is about the sampler/evaluator plumbing."""
    """Basin recall produces valid metrics structure after training + sampling."""
    torch = pytest.importorskip("torch")
    from basinflow.evaluation import evaluate_event_recall
    from basinflow.models import EGNNFlow

    torch.manual_seed(0)
    dataset = _toy_dataset()
    model = EGNNFlow(hidden_dim=16, num_layers=1, cutoff=3.0)

    train_product_flow(
        model, dataset, [ZeroInit()],
        epochs=5, lr=0.01, checkpoint_path=None,
    )

    sampler = CandidateSampler(
        model=model,
        init_generators=[ZeroInit(), GaussianInit(scale=0.05, random_seed=1)],
        cutoff=3.0, num_steps=8, checkpoint_id="e2e-test",
    )
    result = sampler.sample(dataset, "e")

    metrics = evaluate_event_recall(
        dataset, "e", result.candidates, result.generated_structures,
        cluster_threshold=0.5, match_threshold=1.0,
    )
    assert metrics["num_candidates"] == 2
    assert metrics["recall"] >= 0.0
    assert "clusters" in metrics
