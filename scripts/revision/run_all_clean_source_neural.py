"""Strict target-specific clean-source rerun for MTL, MAML, and FEFA.

MTL and FEFA share each target-specific pretrained encoder. Ensemble weights
are selected by three-fold scaffold-grouped inner validation; each inner-fold
encoder is retrained after excluding both the outer target-test identities and
the inner target-validation identities from every source endpoint.
"""

from __future__ import annotations

import argparse
import copy
import gc
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

from clean_source_common import (
    CLEAN_RESULTS,
    ENDPOINTS,
    add_common_fields,
    baseline_probability,
    clean_train_indices,
    evaluate,
    load_inputs,
    quick_lgb_predict,
    read_json,
    removal_audit,
    result_key,
    smiles_array,
    tune_lgb_model,
    write_json_atomic,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiTaskNetV3(nn.Module):
    def __init__(self, in_dim=2048, hid=256, n_endpoints=13):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, hid),
            nn.ReLU(),
            nn.BatchNorm1d(hid),
            nn.Dropout(0.2),
        )
        self.heads = nn.ModuleList([nn.Linear(hid, 1) for _ in range(n_endpoints)])

    def forward(self, x):
        hidden = self.encoder(x)
        return hidden, [head(hidden).squeeze(-1) for head in self.heads]


class MAMLModel(nn.Module):
    def __init__(self, in_dim=2048, hidden_1=256, hidden_2=128):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(hidden_1, in_dim) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(hidden_1))
        self.w2 = nn.Parameter(torch.randn(hidden_2, hidden_1) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(hidden_2))
        self.w3 = nn.Parameter(torch.randn(1, hidden_2) * 0.01)
        self.b3 = nn.Parameter(torch.zeros(1))

    def fast_forward(self, x, params=None):
        if params is None:
            params = (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)
        w1, b1, w2, b2, w3, b3 = params
        hidden = torch.relu(torch.nn.functional.linear(x, w1, b1))
        hidden = torch.relu(torch.nn.functional.linear(hidden, w2, b2))
        return torch.nn.functional.linear(hidden, w3, b3).squeeze(-1)


def scaffold_groups(smiles):
    groups = []
    for smile in np.asarray(smiles, dtype=str):
        molecule = Chem.MolFromSmiles(smile)
        if molecule is None:
            groups.append(smile)
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        groups.append(scaffold if scaffold else smile)
    return np.asarray(groups, dtype=str)


def multitask_pool(datasets, splits, target, rep, excluded_extra=None):
    excluded = set(excluded_extra or [])
    records = {}
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        indices = clean_train_indices(
            datasets,
            splits,
            endpoint,
            target,
            rep,
            excluded_extra=excluded,
        )
        endpoint_smiles = smiles_array(datasets, endpoint)
        for index in indices:
            smile = endpoint_smiles[index]
            if smile not in records:
                records[smile] = {
                    "fingerprint": datasets[endpoint]["X"][index],
                    "labels": {},
                }
            records[smile]["labels"][endpoint_index] = datasets[endpoint]["y"][index]
    if set(records) & excluded:
        raise AssertionError("Inner validation identity remains in multitask pool")
    fingerprints = np.asarray(
        [record["fingerprint"] for record in records.values()], dtype=np.float32
    )
    labels = np.asarray(
        [
            [record["labels"].get(index, np.nan) for index in range(len(ENDPOINTS))]
            for record in records.values()
        ],
        dtype=np.float32,
    )
    return fingerprints, labels, list(records)


def train_encoder(fingerprints, labels, seed, max_epochs=200, patience=15):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MultiTaskNetV3().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    x_tensor = torch.as_tensor(fingerprints, dtype=torch.float32, device=DEVICE)
    label_tensor = torch.as_tensor(labels, dtype=torch.float32, device=DEVICE)
    mask_tensor = ~torch.isnan(label_tensor)
    label_tensor = torch.nan_to_num(label_tensor, 0.0)
    permutation = np.random.permutation(len(fingerprints))
    n_validation = max(1, int(0.15 * len(fingerprints)))
    validation_index = permutation[:n_validation]
    training_index = permutation[n_validation:]
    x_train = x_tensor[training_index]
    y_train = label_tensor[training_index]
    mask_train = mask_tensor[training_index]
    x_validation = x_tensor[validation_index]
    y_validation = label_tensor[validation_index]
    mask_validation = mask_tensor[validation_index]
    loader = DataLoader(
        TensorDataset(x_train, y_train, mask_train.float()),
        batch_size=256,
        shuffle=True,
        drop_last=len(training_index) > 256,
    )
    best_loss = float("inf")
    best_state = None
    stale = 0
    epochs_run = 0
    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        for batch_x, batch_y, batch_mask in loader:
            _, outputs = model(batch_x)
            loss = torch.tensor(0.0, device=DEVICE)
            tasks = 0
            for endpoint_index in range(len(ENDPOINTS)):
                present = batch_mask[:, endpoint_index].bool()
                if present.sum() > 0:
                    loss = loss + criterion(
                        outputs[endpoint_index][present], batch_y[:, endpoint_index][present]
                    ).mean()
                    tasks += 1
            if tasks:
                loss = loss / tasks
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            _, validation_outputs = model(x_validation)
            validation_loss = 0.0
            tasks = 0
            for endpoint_index in range(len(ENDPOINTS)):
                present = mask_validation[:, endpoint_index]
                if present.sum() > 0:
                    validation_loss += criterion(
                        validation_outputs[endpoint_index][present],
                        y_validation[:, endpoint_index][present],
                    ).mean().item()
                    tasks += 1
            validation_loss = validation_loss / tasks if tasks else float("inf")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, epochs_run


def fine_tune_mtl_predict(model, endpoint_index, x_train, y_train, x_predict, epochs, seed):
    torch.manual_seed(seed)
    fine_tuned = copy.deepcopy(model).to(DEVICE)
    optimizer = optim.Adam(fine_tuned.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_x = torch.as_tensor(x_train, dtype=torch.float32, device=DEVICE)
    train_y = torch.as_tensor(y_train, dtype=torch.float32, device=DEVICE)
    fine_tuned.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(train_x), device=DEVICE)
        for start in range(0, len(train_x), 256):
            batch = permutation[start : start + 256]
            if len(batch) < 2:
                continue
            _, outputs = fine_tuned(train_x[batch])
            loss = criterion(outputs[endpoint_index], train_y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    fine_tuned.eval()
    predictions = []
    with torch.no_grad():
        predict_x = torch.as_tensor(x_predict, dtype=torch.float32, device=DEVICE)
        for start in range(0, len(predict_x), 512):
            _, outputs = fine_tuned(predict_x[start : start + 512])
            predictions.append(torch.sigmoid(outputs[endpoint_index]).cpu().numpy())
    del fine_tuned
    return np.concatenate(predictions)


def embeddings(model, x):
    model.eval()
    outputs = []
    with torch.no_grad():
        tensor = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
        for start in range(0, len(tensor), 512):
            outputs.append(model.encoder(tensor[start : start + 512]).cpu().numpy())
    return np.concatenate(outputs)


def select_lambda(y, baseline_oof, transfer_oof):
    best_lambda = 0.5
    best_auc = -np.inf
    for weight in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        auc = roc_auc_score(y, weight * baseline_oof + (1 - weight) * transfer_oof)
        if auc > best_auc:
            best_auc = auc
            best_lambda = weight
    return float(best_lambda), float(best_auc)


def run_mtl_fefa(
    datasets,
    splits,
    baseline,
    target,
    rep,
    encoder_epochs,
    encoder_patience,
    fine_tune_epochs,
    n_inner_folds,
    fefa_max_evals,
):
    seed = 101 + rep
    endpoint_index = ENDPOINTS.index(target)
    train_index, test_index = splits[target][rep]
    x_target = datasets[target]["X"].astype(np.float32)
    y_target = datasets[target]["y"]
    target_smiles = smiles_array(datasets, target)
    x_train = x_target[train_index]
    y_train = y_target[train_index]
    x_test = x_target[test_index]
    y_test = y_target[test_index]
    pool_x, pool_y, pool_smiles = multitask_pool(datasets, splits, target, rep)
    outer_encoder, outer_epochs = train_encoder(
        pool_x,
        pool_y,
        seed,
        max_epochs=encoder_epochs,
        patience=encoder_patience,
    )
    mtl_probability = fine_tune_mtl_predict(
        outer_encoder,
        endpoint_index,
        x_train,
        y_train,
        x_test,
        fine_tune_epochs,
        seed,
    )
    outer_train_embeddings = embeddings(outer_encoder, x_train)
    outer_test_embeddings = embeddings(outer_encoder, x_test)
    x_train_augmented = np.hstack([x_train, outer_train_embeddings])
    x_test_augmented = np.hstack([x_test, outer_test_embeddings])
    fefa_model, fefa_params = tune_lgb_model(
        x_train_augmented,
        y_train,
        seed,
        max_evals=fefa_max_evals,
        n_folds=3,
        groups=scaffold_groups(target_smiles[train_index]),
    )
    fefa_probability = fefa_model.predict_proba(x_test_augmented)[:, 1]

    splitter = StratifiedGroupKFold(
        n_splits=n_inner_folds, shuffle=True, random_state=seed
    )
    inner_groups = scaffold_groups(target_smiles[train_index])
    oof_baseline = np.zeros(len(train_index), dtype=float)
    oof_mtl = np.zeros(len(train_index), dtype=float)
    oof_fefa = np.zeros(len(train_index), dtype=float)
    inner_epoch_counts = []
    for fold, (inner_train_position, inner_valid_position) in enumerate(
        splitter.split(x_train, y_train, inner_groups)
    ):
        inner_validation_smiles = set(
            target_smiles[train_index][inner_valid_position].tolist()
        )
        inner_pool_x, inner_pool_y, _ = multitask_pool(
            datasets,
            splits,
            target,
            rep,
            excluded_extra=inner_validation_smiles,
        )
        inner_encoder, inner_epochs = train_encoder(
            inner_pool_x,
            inner_pool_y,
            seed + 1000 + fold,
            max_epochs=encoder_epochs,
            patience=encoder_patience,
        )
        inner_epoch_counts.append(inner_epochs)
        inner_train_x = x_train[inner_train_position]
        inner_train_y = y_train[inner_train_position]
        inner_valid_x = x_train[inner_valid_position]
        oof_baseline[inner_valid_position] = quick_lgb_predict(
            inner_train_x,
            inner_train_y,
            inner_valid_x,
            seed + fold,
        )
        oof_mtl[inner_valid_position] = fine_tune_mtl_predict(
            inner_encoder,
            endpoint_index,
            inner_train_x,
            inner_train_y,
            inner_valid_x,
            fine_tune_epochs,
            seed + fold,
        )
        inner_train_embeddings = embeddings(inner_encoder, inner_train_x)
        inner_valid_embeddings = embeddings(inner_encoder, inner_valid_x)
        oof_fefa[inner_valid_position] = quick_lgb_predict(
            np.hstack([inner_train_x, inner_train_embeddings]),
            inner_train_y,
            np.hstack([inner_valid_x, inner_valid_embeddings]),
            seed + fold,
        )
        del inner_encoder
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    mtl_lambda, mtl_inner_auc = select_lambda(y_train, oof_baseline, oof_mtl)
    fefa_lambda, fefa_inner_auc = select_lambda(y_train, oof_baseline, oof_fefa)
    baseline_test_probability = baseline_probability(baseline, target, rep)
    mtl_ensemble = mtl_lambda * baseline_test_probability + (1 - mtl_lambda) * mtl_probability
    fefa_ensemble = fefa_lambda * baseline_test_probability + (1 - fefa_lambda) * fefa_probability
    removed = removal_audit(datasets, splits, target, rep)
    mtl_metrics = evaluate(y_test, mtl_ensemble)
    mtl_metrics.update(
        {
            "lambda": mtl_lambda,
            "inner_selection_AUC": mtl_inner_auc,
            "outer_pool_size": len(pool_smiles),
            "outer_encoder_epochs": outer_epochs,
            "inner_encoder_epochs": inner_epoch_counts,
        }
    )
    fefa_metrics = evaluate(y_test, fefa_ensemble)
    fefa_metrics.update(
        {
            "lambda": fefa_lambda,
            "inner_selection_AUC": fefa_inner_auc,
            "outer_pool_size": len(pool_smiles),
            "outer_encoder_epochs": outer_epochs,
            "inner_encoder_epochs": inner_epoch_counts,
            "best_params": fefa_params,
        }
    )
    mtl_metrics = add_common_fields(
        mtl_metrics, baseline, target, rep, removed, "MTL_clean", seed
    )
    fefa_metrics = add_common_fields(
        fefa_metrics, baseline, target, rep, removed, "FEFA_clean", seed
    )
    del outer_encoder
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return mtl_metrics, fefa_metrics


def run_maml(datasets, splits, baseline, target, rep, meta_epochs, fine_tune_epochs):
    seed = 101 + rep
    torch.manual_seed(seed)
    np.random.seed(seed)
    pools = {}
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        indices = clean_train_indices(datasets, splits, endpoint, target, rep)
        pools[endpoint_index] = (
            torch.as_tensor(
                datasets[endpoint]["X"][indices], dtype=torch.float32, device=DEVICE
            ),
            torch.as_tensor(
                datasets[endpoint]["y"][indices], dtype=torch.float32, device=DEVICE
            ),
        )
    model = MAMLModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    inner_steps = 5
    inner_learning_rate = 0.01
    support_size = 64
    for _meta_epoch in range(meta_epochs):
        task_ids = np.random.choice(len(ENDPOINTS), 5, replace=False)
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=DEVICE)
        used_tasks = 0
        for task_id in task_ids:
            pool_x, pool_y = pools[task_id]
            if len(pool_x) < support_size * 2:
                continue
            permutation = torch.randperm(len(pool_x), device=DEVICE)[: support_size * 2]
            support_x = pool_x[permutation[:support_size]]
            support_y = pool_y[permutation[:support_size]]
            query_x = pool_x[permutation[support_size:]]
            query_y = pool_y[permutation[support_size:]]
            params = [model.w1, model.b1, model.w2, model.b2, model.w3, model.b3]
            for _ in range(inner_steps):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model.fast_forward(support_x, params), support_y
                )
                gradients = torch.autograd.grad(loss, params, create_graph=True)
                params = [
                    parameter - inner_learning_rate * gradient
                    for parameter, gradient in zip(params, gradients)
                ]
            total_loss = total_loss + torch.nn.functional.binary_cross_entropy_with_logits(
                model.fast_forward(query_x, params), query_y
            )
            used_tasks += 1
        if used_tasks:
            total_loss = total_loss / used_tasks
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    train_index, test_index = splits[target][rep]
    x_train = torch.as_tensor(
        datasets[target]["X"][train_index], dtype=torch.float32, device=DEVICE
    )
    y_train = torch.as_tensor(
        datasets[target]["y"][train_index], dtype=torch.float32, device=DEVICE
    )
    x_test = torch.as_tensor(
        datasets[target]["X"][test_index], dtype=torch.float32, device=DEVICE
    )
    fine_tuned = copy.deepcopy(model)
    optimizer = optim.Adam(fine_tuned.parameters(), lr=1e-3)
    fine_tuned.train()
    for _ in range(fine_tune_epochs):
        permutation = torch.randperm(len(x_train), device=DEVICE)
        for start in range(0, len(x_train), 256):
            batch = permutation[start : start + 256]
            logits = fine_tuned.fast_forward(x_train[batch])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y_train[batch]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    fine_tuned.eval()
    with torch.no_grad():
        maml_probability = torch.sigmoid(fine_tuned.fast_forward(x_test)).cpu().numpy()
    baseline_test_probability = baseline_probability(baseline, target, rep)
    probability = 0.6 * baseline_test_probability + 0.4 * maml_probability
    removed = removal_audit(datasets, splits, target, rep)
    metrics = evaluate(datasets[target]["y"][test_index], probability)
    metrics["lambda"] = 0.6
    metrics["meta_epochs"] = int(meta_epochs)
    metrics = add_common_fields(
        metrics, baseline, target, rep, removed, "MAML_clean", seed
    )
    del model, fine_tuned, pools
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["mtl_fefa", "maml", "all"], default="all")
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-end", type=int, default=10)
    parser.add_argument("--target", choices=ENDPOINTS)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"device={DEVICE}", flush=True)
    datasets, splits, baseline = load_inputs()
    suffix = "_smoke" if args.smoke else ""
    mtl_path = CLEAN_RESULTS / f"mtl_clean_results{suffix}.json"
    fefa_path = CLEAN_RESULTS / f"fefa_clean_results{suffix}.json"
    maml_path = CLEAN_RESULTS / f"maml_clean_results{suffix}.json"
    mtl_payload = read_json(mtl_path)
    fefa_payload = read_json(fefa_path)
    maml_payload = read_json(maml_path)
    targets = [args.target] if args.target else ENDPOINTS
    if args.smoke:
        encoder_epochs, patience, fine_tune_epochs = 4, 2, 2
        n_inner_folds, fefa_max_evals, meta_epochs = 2, 1, 3
    else:
        encoder_epochs, patience, fine_tune_epochs = 200, 15, 30
        n_inner_folds, fefa_max_evals, meta_epochs = 3, 15, 500
    for rep in range(args.rep_start, min(args.rep_end, 10)):
        for target in targets:
            key = result_key(target, rep)
            if args.method in {"mtl_fefa", "all"} and (
                key not in mtl_payload or key not in fefa_payload
            ):
                started = time.time()
                mtl_result, fefa_result = run_mtl_fefa(
                    datasets,
                    splits,
                    baseline,
                    target,
                    rep,
                    encoder_epochs,
                    patience,
                    fine_tune_epochs,
                    n_inner_folds,
                    fefa_max_evals,
                )
                mtl_payload[key] = mtl_result
                fefa_payload[key] = fefa_result
                write_json_atomic(mtl_payload, mtl_path)
                write_json_atomic(fefa_payload, fefa_path)
                print(
                    f"mtl_fefa rep={rep} target={target} "
                    f"MTL={mtl_result['delta_AUC']:+.4f} "
                    f"FEFA={fefa_result['delta_AUC']:+.4f} "
                    f"removed={mtl_result['source_rows_removed']} "
                    f"seconds={time.time()-started:.1f}",
                    flush=True,
                )
            if args.method in {"maml", "all"} and key not in maml_payload:
                started = time.time()
                result = run_maml(
                    datasets,
                    splits,
                    baseline,
                    target,
                    rep,
                    meta_epochs,
                    2 if args.smoke else 100,
                )
                maml_payload[key] = result
                write_json_atomic(maml_payload, maml_path)
                print(
                    f"maml rep={rep} target={target} delta={result['delta_AUC']:+.4f} "
                    f"removed={result['source_rows_removed']} "
                    f"seconds={time.time()-started:.1f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
