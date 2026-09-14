def test_stage3_public_names_are_canonical():
    from basinflow.seeds import GaussianInit, ProductInit, ZeroInit
    from basinflow.models import EGNNFlow, FlowLossWeights, flow_loss

    assert EGNNFlow.__name__ == "EGNNFlow"
    assert FlowLossWeights.__name__ == "FlowLossWeights"
    assert flow_loss.__name__ == "flow_loss"
    assert ZeroInit.__name__ == "ZeroInit"
    assert GaussianInit.__name__ == "GaussianInit"
    assert ProductInit.__name__ == "ProductInit"


def test_stage3_legacy_aliases_are_removed():
    import importlib

    import basinflow.models as models
    import basinflow.seeds as seeds

    assert not hasattr(models, "ProductEventFlowLossWeights")
    assert not hasattr(models, "SeedConditionedEGNNProductFlow")
    assert not hasattr(models, "product_event_flow_loss")
    assert not hasattr(seeds, "GaussianMovableSeedGenerator")
    assert not hasattr(seeds, "ProductDisplacementSeedGenerator")
    assert not hasattr(seeds, "ZeroSeedGenerator")
    assert not hasattr(seeds, "event_seed_from_pairwise_item")

    try:
        importlib.import_module("basinflow.inits")
    except ModuleNotFoundError:
        pass
    else:  # pragma: no cover - kept explicit for module-naming regression
        raise AssertionError("basinflow.inits must stay removed; seed code lives in basinflow.seeds")


def test_dataset_public_names_are_canonical():
    import importlib

    import basinflow.data as data

    assert hasattr(data, "EventFlowDataset")
    assert hasattr(data, "BasinDataset")
    assert hasattr(data, "EventCatalog")
    assert not hasattr(data, "EventDataset")
    assert not hasattr(data, "ProductFlowDataset")
    assert not hasattr(data, "BasinProposalDataset")

    try:
        importlib.import_module("basinflow.data.dataset")
    except ModuleNotFoundError:
        pass
    else:  # pragma: no cover - explicit delivery API regression
        raise AssertionError("basinflow.data.dataset must be removed")
