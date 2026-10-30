"""Explicit, immutable selection of input sessions for report procedures."""
import hashlib
import json
from pathlib import Path


def read_config(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return path, document


def data_path(config_path, document, value):
    """Data paths resolve from base_dir, itself relative to the config file."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Input paths must be nonempty strings")
    base_dir = document.get("base_dir", ".")
    if not isinstance(base_dir, str):
        raise ValueError("base_dir must be a directory path string")
    base = Path(config_path).parent / base_dir
    return (base / value).resolve()


def load_run_selection(config_path, settings=None):
    path, document = read_config(config_path)
    if document.get("schema_version") != 1:
        raise ValueError(f"{path}: require schema_version=1")
    if "runs_config" in document:
        if "runs" in document:
            raise ValueError("Choose runs or runs_config, not both")
        reference = document["runs_config"]
        if not isinstance(reference, str) or not reference:
            raise ValueError("runs_config must name a selection configuration")
        reference_path = (path.parent / reference).resolve()
        if reference_path == path:
            raise ValueError("A selection config cannot refer to itself")
        # Exactly one reference level, so accidental cycles cannot recurse.
        _, referenced = read_config(reference_path)
        if "runs_config" in referenced:
            raise ValueError("runs_config must refer directly to a file containing runs")
        selection = load_run_selection(reference_path, settings)
        selection["procedure_config"] = str(path)
        selection["procedure_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return selection
    groups = document.get("runs")
    if document.get("schema_version") != 1 or not isinstance(groups, dict) or not groups:
        raise ValueError(f"{path}: require schema_version=1 and an explicit runs object")
    selected = list(groups) if settings is None else list(settings)
    if not selected or len(set(selected)) != len(selected) or any(s not in groups for s in selected):
        raise ValueError("Requested settings must be unique groups present in the run-selection config")
    entries, seen = [], set()
    for setting in selected:
        if not isinstance(setting, str) or not setting:
            raise ValueError("Each run group needs a nonempty setting name")
        paths = groups[setting]
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"{setting}: list at least one run directory")
        for value in paths:
            directory = data_path(path, document, value)
            if directory in seen:
                raise ValueError(f"Duplicate run in selection (possibly in two settings): {directory}")
            manifest = directory / "manifest.json"
            if not directory.is_dir() or not manifest.is_file():
                raise ValueError(f"Selected run is missing: {directory}")
            seen.add(directory)
            entries.append(dict(setting=setting, path=str(directory),
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                manifest=json.loads(manifest.read_text(encoding="utf-8-sig"))))
    return dict(config_path=str(path), config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                config=document, settings=selected, entries=entries)


def selection_audit(selection):
    """Snapshot configured and resolved paths; original manifests are unchanged."""
    return {**{key:value for key,value in selection.items() if key != "entries"},
            "entries": [dict(**{key:value for key,value in row.items() if key != "manifest"},
                saved_setting=row["manifest"].get("setting"), saved_split=row["manifest"].get("split"),
                saved_status=row["manifest"].get("status")) for row in selection["entries"]]}
