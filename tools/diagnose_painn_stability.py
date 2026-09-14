from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
import sys

import torch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basinflow.data import BasinSplit, EventCatalog, EventFlowDataset
from basinflow.seeds import GaussianInit
from basinflow.models import flow_loss
from basinflow.models.factory import build_model, model_spec
from basinflow.models.painn.modules import prepare_flow_input


def tensor_stats(value):
    return {
        "abs_max": float(value.detach().abs().max()) if value.numel() else 0.0,
        "finite": bool(torch.isfinite(value).all()),
    }


def geometry_stats(batch, cutoff):
    prepared = prepare_flow_input(batch, cutoff)
    receiver, sender = prepared["edge_index"]
    reference_vectors = batch.reactant_pos[sender] - batch.reactant_pos[receiver] + prepared["shifts"]
    current_vectors = batch.pos[sender] - batch.pos[receiver] + prepared["shifts"]
    reference_lengths = torch.linalg.vector_norm(reference_vectors, dim=1)
    current_lengths = torch.linalg.vector_norm(current_vectors, dim=1)
    degree = torch.bincount(receiver, minlength=batch.num_nodes)
    results = []
    for graph_index, event_id in enumerate(batch.event_id):
        nodes = batch.batch == graph_index
        edges = batch.batch[receiver] == graph_index
        displacement = batch.target_pos[nodes] - batch.reactant_pos[nodes]
        results.append({
            "event_id": event_id,
            "flow_time": float(batch.flow_time[graph_index]),
            "n_atoms": int(nodes.sum()),
            "n_movable": int(batch.movable_mask[nodes].sum()),
            "reference_min_edge": float(reference_lengths[edges].min()) if edges.any() else None,
            "current_min_edge": float(current_lengths[edges].min()) if edges.any() else None,
            "max_target_displacement": float(torch.linalg.vector_norm(displacement, dim=1).max()),
            "target_velocity": tensor_stats(batch.target_velocity[nodes]),
            "seed_displacement": tensor_stats(batch.seed_displacement[nodes]),
            "max_degree": int(degree[nodes].max()),
            "mean_degree": float(degree[nodes].float().mean()),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/0910/official_pt.ini")
    parser.add_argument("--output-dir", default="/tmp/painn_pt_debug")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--stability-mode", choices=["none", "scaled"], default="none")
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["runtime"].getint("cpu_threads"))
    if args.replay is not None:
        saved = torch.load(args.replay, map_location="cpu", weights_only=False)
        model_config = {**saved["model_config"], "stability_mode": args.stability_mode}
        model = build_model(saved["model_backend"], model_config)
        model.load_state_dict(saved["model_state_dict"])
        batch = saved["batch"]
        output = model(batch)
        loss, _ = flow_loss(output, batch)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        result = {
            "stability_mode": args.stability_mode,
            "loss": float(loss.detach()),
            "max_velocity": float(output["velocity"].detach().abs().max()),
            "all_gradients_finite": all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
            "gradient_norm_float64": float(torch.stack([gradient.double().norm() for gradient in gradients]).norm()),
        }
        try:
            result["gradient_norm_float32"] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True))
        except RuntimeError as error:
            result["gradient_norm_error"] = str(error)
        print(json.dumps(result, indent=2), flush=True)
        (output_dir / f"replay_{args.stability_mode}.json").write_text(json.dumps(result, indent=2))
        return
    training_seed = config["split"].getint("seed")
    torch.manual_seed(training_seed)
    pbc = [value.strip().lower() == "true" for value in config["data"]["pbc_override"].split(",")]
    catalog = EventCatalog.from_eon_directory(config["data"]["events_dir"], pbc_override=pbc)
    split = BasinSplit.load(config["split"]["manifest"])
    train_catalog = split.select(catalog, "train")
    backend, model_config = model_spec(config["model"])
    model_config["stability_mode"] = args.stability_mode
    model = build_model(backend, model_config)
    initializers = [GaussianInit(scale=config["init"].getfloat("gaussian_scale"), random_seed=training_seed)]
    dataset = EventFlowDataset(train_catalog, initializers, seed=args.dataset_seed, active_threshold=config["data"].getfloat("active_threshold"))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"].getfloat("lr"))
    active_batch = None
    stage_stats = {}

    def capture(name):
        def hook(module, inputs, output):
            values = output if isinstance(output, tuple) else (output,)
            stage_stats[name] = [
                [tensor_stats(value[active_batch.batch == graph_index]) for value in values]
                for graph_index in range(active_batch.num_graphs)
            ]
        return hook

    for name, module in model.named_modules():
        if name.startswith("backbone.messages.") or name.startswith("backbone.updates."):
            if name.count(".") == 2:
                module.register_forward_hook(capture(name))
        elif name == "backbone.output":
            module.register_forward_hook(capture(name))

    completed = 0
    model.train()
    with (output_dir / "trace.jsonl").open("w") as trace:
        for epoch in range(1, config["training"].getint("epochs") + 1):
            dataset.set_epoch(epoch)
            loader = DataLoader(dataset, batch_size=config["training"].getint("batch_size"), shuffle=True)
            for batch in loader:
                active_batch = batch
                stage_stats = {}
                optimizer.zero_grad()
                output = model(batch)
                loss, _ = flow_loss(output, batch)
                step = completed + 1
                row = {"epoch": epoch, "step": step, "loss": float(loss.detach()), "event_ids": batch.event_id, "stages": stage_stats}
                failed = not bool(torch.isfinite(loss))
                if failed or loss.detach() > 100:
                    row["geometry"] = geometry_stats(batch, model.cutoff)
                    torch.save({
                        "model_backend": backend,
                        "model_config": model_config,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "batch": batch,
                        "torch_rng_state": torch.get_rng_state(),
                        "diagnostics": row,
                        "dataset_seed": args.dataset_seed,
                    }, output_dir / f"pre_step_{step}.pt")
                    (output_dir / f"pre_step_{step}.json").write_text(json.dumps(row, indent=2))
                    print(json.dumps({key: value for key, value in row.items() if key != "stages"}), flush=True)
                trace.write(json.dumps(row) + "\n")
                trace.flush()
                if failed:
                    print(f"NONFINITE at step {step}; saved {output_dir / f'pre_step_{step}.pt'}", flush=True)
                    return
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"].getfloat("max_grad_norm"), error_if_nonfinite=True)
                optimizer.step()
                completed += 1
                if step <= 5 or step % 10 == 0:
                    print(f"step={step} loss={float(loss.detach()):.8f}", flush=True)
                if completed >= args.max_steps:
                    print(f"Completed {completed} finite steps", flush=True)
                    return


if __name__ == "__main__":
    main()
