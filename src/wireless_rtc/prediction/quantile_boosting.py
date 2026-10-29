"""Offline quantile boosting, validation selection, and separate held-out evaluation."""
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform
import statistics
import time

from wireless_rtc.analysis.formatting import ReportWriter
from wireless_rtc.client.schedule import delay_steps
from wireless_rtc.prediction.dataset import partition_records, read_records
from wireless_rtc.prediction.features import build_examples, feature_names, feature_values
from wireless_rtc.prediction.rolling import quantile


SCHEMA_VERSION = 1


def learning_dependencies():
    try:
        import numpy as np
        import sklearn
        from sklearn.ensemble import HistGradientBoostingRegressor
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise RuntimeError('Install prediction dependencies: uv pip install -e ".[prediction]"') from exc
    return np, sklearn, HistGradientBoostingRegressor, threadpool_limits


def validate_config(config):
    if type(config.get("seed")) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(config.get("history")) is not int or config["history"] < 2:
        raise ValueError("history must be an integer >= 2")
    qs = config.get("quantiles", [])
    if not qs or any(type(q) not in (int,float) or not math.isfinite(q) or not 0<q<1 for q in qs):
        raise ValueError("quantiles must be finite values in (0,1)")
    if qs != sorted(set(qs)):
        raise ValueError("quantiles must be unique and increasing")
    variants = config.get("feature_sets", [])
    if not variants or len(variants) != len(set(variants)) or any(v not in {"rtt", "rtt_snr"} for v in variants):
        raise ValueError("feature_sets must be unique rtt and/or rtt_snr")
    if config.get("split_mode") not in {"chronological", "session"}:
        raise ValueError("Unknown split_mode")
    a, b = config.get("train_fraction", 0), config.get("validation_fraction", 0)
    if not (0<a<1 and 0<b<1 and a+b<1):
        raise ValueError("Positive train and validation fractions must sum to less than one")
    for field in ("control_ms", "max_gap_ms"):
        if not math.isfinite(config[field]) or config[field] <= 0:
            raise ValueError(f"{field} must be finite and positive")
    allowed = {"max_iter", "max_leaf_nodes", "min_samples_leaf", "learning_rate", "l2_regularization", "max_depth"}
    grid = config.get("parameter_grid")
    if not isinstance(grid, list) or not grid or any(not isinstance(p,dict) or set(p)-allowed for p in grid):
        raise ValueError("parameter_grid must contain supported estimator parameter dictionaries")


def pinball(actual, predicted, q):
    error = actual-predicted
    return max(q*error, (q-1)*error)


def score(actual, predicted, q, control_ms):
    if not actual or len(actual) != len(predicted):
        raise ValueError("Metrics require equal, nonempty actual and predicted sequences")
    mean = statistics.mean
    true_steps = [delay_steps(y, control_ms) for y in actual]
    pred_steps = [delay_steps(y, control_ms) for y in predicted]
    return dict(samples=len(actual), mae_ms=mean(abs(y-p) for y,p in zip(actual,predicted)),
                bias_ms=mean(p-y for y,p in zip(actual,predicted)),
                pinball_loss_ms=mean(pinball(y,p,q) for y,p in zip(actual,predicted)),
                coverage=mean(y<=p for y,p in zip(actual,predicted)),
                exceedance_rate=mean(y>p for y,p in zip(actual,predicted)),
                step_underprediction_rate=mean(y>p for y,p in zip(true_steps,pred_steps)),
                step_overprediction_rate=mean(p>y for y,p in zip(true_steps,pred_steps)),
                mean_excess_steps=mean(max(0,p-y) for y,p in zip(true_steps,pred_steps)),
                mean_prefix_steps=mean(pred_steps))


def rearrange(predictions):
    np, _, _, _ = learning_dependencies()
    result = np.asarray(predictions, dtype=float).T
    if not np.isfinite(result).all():
        raise ValueError("Estimator produced nonfinite predictions")
    # Deterministic non-crossing quantiles, applied identically during selection and inference.
    return np.sort(np.maximum(result, 0), axis=1)


@dataclass
class QuantileBundle:
    config: dict
    estimators: dict
    used_record_ids: set
    sklearn_version: str
    schema_version: int = SCHEMA_VERSION

    def predict(self, history, feature_set="rtt_snr"):
        """Predict from exactly `history` completed samples; does not update state."""
        if len(history) != self.config["history"]:
            raise ValueError("Incorrect history length")
        for a,b in zip(history, history[1:]):
            if a.session != b.session or not 0 < b.timestamp_ms-a.timestamp_ms <= self.config["max_gap_ms"]:
                raise ValueError("History must be one ordered session with no long gaps")
        np, _, _, limit = learning_dependencies()
        x = np.asarray([feature_values(history, feature_set)], dtype=float)
        with limit(limits=1):
            prediction = rearrange([m.predict(x) for m in self.estimators[feature_set]])[0]
        return dict(zip(self.config["quantiles"], map(float,prediction)))


def load_bundle(path):
    # Pickle is only for locally trained, trusted artifacts; never load downloaded models.
    with Path(path).open("rb") as stream:
        bundle = pickle.load(stream)
    _, sklearn, _, _ = learning_dependencies()
    if not isinstance(bundle, QuantileBundle) or bundle.schema_version != SCHEMA_VERSION:
        raise ValueError("Unsupported quantile model artifact")
    if sklearn.__version__ != bundle.sklearn_version:
        raise ValueError(f"Use scikit-learn {bundle.sklearn_version} to load this model")
    return bundle


def report_rows(examples, predictions, name, quantiles, control_ms):
    actual = [e["target"].rtt_ms for e in examples]
    return [dict(model=name, quantile=q, **score(actual,list(map(float,predictions[:,i])),q,control_ms))
            for i,q in enumerate(quantiles)]


def baseline_predictions(examples, quantiles):
    np, _, _, _ = learning_dependencies()
    result = {}
    for name in ("last_value", "rolling_median", "recent_max", "rolling_quantile"):
        rows = []
        for example in examples:
            values = [r.rtt_ms for r in example["history"]]
            rows.append([values[-1] if name == "last_value" else statistics.median(values) if name == "rolling_median"
                         else max(values) if name == "recent_max" else quantile(values,q) for q in quantiles])
        result[name] = np.asarray(rows)
    return result


def save_reports(output, examples, predictions, config):
    rows = [row for name,preds in predictions.items() for row in
            report_rows(examples,preds,name,config["quantiles"],config["control_ms"])]
    with (output/"metrics.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = ReportWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/"metrics.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    with (output/"predictions.csv").open("w",newline="",encoding="utf-8") as stream:
        fields = ["model", "quantile", "session_id", "history_end_ms", "target_timestamp_ms", "forecast_horizon_ms",
                  "actual_ms", "predicted_ms"]
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for name,preds in predictions.items():
            for i,example in enumerate(examples):
                target, end = example["target"], example["history"][-1].timestamp_ms
                for j,q in enumerate(config["quantiles"]):
                    writer.writerow(dict(model=name,quantile=q,session_id=target.session,history_end_ms=end,
                        target_timestamp_ms=target.timestamp_ms,forecast_horizon_ms=target.timestamp_ms-end,
                        actual_ms=target.rtt_ms,predicted_ms=float(preds[i,j])))
    return rows


def train_quantiles(paths, config, output):
    """Tune only on validation pinball loss; never score or select against test data."""
    validate_config(config)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    np, sklearn, Estimator, limit = learning_dependencies()
    sessions, provenance = read_records(paths)
    sessions = partition_records(sessions, config)
    datasets, audits = {}, {}
    for partition in ("train", "validation"):
        datasets[partition], audits[partition] = build_examples(
            sessions,config["history"],config["max_gap_ms"],partition)
        if len(datasets[partition]) < 10:
            raise ValueError(f"Need at least ten {partition} examples after warmup, errors, gaps and partition resets")
    train, validation = datasets["train"], datasets["validation"]
    y = np.asarray([e["target"].rtt_ms for e in train])
    actual = [e["target"].rtt_ms for e in validation]
    models, selection, choices = {}, [], {}
    validation_predictions = baseline_predictions(validation,config["quantiles"])
    started = time.perf_counter()
    for variant in config["feature_sets"]:
        if variant == "rtt_snr":
            for partition in ("train", "validation"):
                if not any(r.snr_db is not None for e in datasets[partition] for r in e["history"]):
                    raise ValueError(f"No historical SNR in {partition}; use feature_sets=[\"rtt\"] for an RTT-only experiment")
        x = np.asarray([feature_values(e["history"],variant) for e in train], dtype=float)
        xv = np.asarray([feature_values(e["history"],variant) for e in validation], dtype=float)
        best_loss = math.inf
        for index, params in enumerate(config["parameter_grid"]):
            print(f"Fitting {variant}: candidate {index+1}/{len(config['parameter_grid'])}",flush=True)
            candidate = []
            with limit(limits=1):
                for q in config["quantiles"]:
                    estimator = Estimator(loss="quantile",quantile=q,early_stopping=False,
                                          random_state=config["seed"],**params)
                    estimator.fit(x,y)
                    candidate.append(estimator)
                preds = rearrange([m.predict(xv) for m in candidate])
            loss = statistics.mean(pinball(a,float(preds[i,j]),q) for i,a in enumerate(actual)
                                   for j,q in enumerate(config["quantiles"]))
            selection.append(dict(feature_set=variant,candidate=index,validation_mean_pinball_ms=loss,parameters=params))
            if loss < best_loss:
                best_loss = loss
                models[variant] = candidate
                choices[variant] = index
                validation_predictions[f"qgb_{variant}"] = preds
    used = {r.identity for rows in sessions.values() for r in rows if r.split in {"train","validation"}}
    bundle = QuantileBundle(config,models,used,sklearn.__version__)
    output.mkdir(parents=True,exist_ok=False)
    with (output/"model.pkl").open("wb") as stream:
        pickle.dump(bundle,stream)
    save_reports(output,validation,validation_predictions,config)
    manifest = dict(schema_version=SCHEMA_VERSION,kind="quantile_gradient_boosting",evaluation_partition="validation",
                    implementation="sklearn.HistGradientBoostingRegressor",sklearn=sklearn.__version__,
                    numpy=np.__version__,python=platform.python_version(),config=config,inputs=provenance,
                    selected_candidates=choices,candidates=selection,audits=audits,
                    feature_names={v:feature_names(config["history"],v) for v in config["feature_sets"]},
                    training_seconds=time.perf_counter()-started,
                    target="next successful measured RTT; failures reset history and are reported separately",
                    features="strictly earlier completed records; current target SNR is excluded",
                    test_policy="No test scores calculated during model selection; no refit on validation",
                    quantile_postprocessing="clip to nonnegative and sort quantiles in increasing order")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return output


def evaluate_quantiles(model, paths, output, partition="test"):
    if partition not in {"test", "all"}:
        raise ValueError("Evaluation partition must be test or all")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    bundle = load_bundle(model)
    np, _, _, limit = learning_dependencies()
    config = bundle.config
    sessions, provenance = read_records(paths)
    if partition == "test":
        sessions = partition_records(sessions,config)
    examples, audit = build_examples(sessions,config["history"],config["max_gap_ms"],
                                     None if partition == "all" else "test")
    if not examples:
        raise ValueError("No test examples after warmup; use --partition all for entirely new unpartitioned traces")
    footprint = {r.identity for rows in sessions.values() for r in rows if partition=="all" or r.split=="test"}
    if footprint & bundle.used_record_ids:
        raise ValueError("Test data overlap training/validation observations")
    predictions = baseline_predictions(examples,config["quantiles"])
    for variant, estimators in bundle.estimators.items():
        x = np.asarray([feature_values(e["history"],variant) for e in examples],dtype=float)
        with limit(limits=1):
            predictions[f"qgb_{variant}"] = rearrange([m.predict(x) for m in estimators])
    output.mkdir(parents=True,exist_ok=False)
    save_reports(output,examples,predictions,config)
    manifest = dict(kind="frozen_quantile_model_test",inputs=provenance,partition=partition,audit=audit,
                    model_path=str(Path(model).resolve()),model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),
                    snr_observed_history_fraction=statistics.mean(r.snr_db is not None for e in examples for r in e["history"]),
                    update_policy="Weights frozen; features may use completed earlier RTTs within the held-out stream")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return output
