"""Compare RTT-only predictors on a CSV segment using its measured source for history."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from wireless_rtc.analysis.formatting import ReportWriter
from wireless_rtc.prediction.dataset import read_records
from wireless_rtc.prediction.quantile_boosting import load_bundle, score
from wireless_rtc.run_paths import add_output_arguments, resolve_output


def rtt_key(record):
    """Match source RTT measurements independently of optional radio telemetry."""
    return record.session, record.timestamp_ms, record.rtt_ms, record.status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment",type=Path,help="Optional subset to score; defaults to the full source dataset")
    parser.add_argument("--source",type=Path,default=Path("data/commect/measured.csv"),
                        help="Measured dataset providing RTT history (default: data/commect/measured.csv)")
    parser.add_argument("--model",type=Path,help="Optional trusted, frozen QGB model.pkl; evaluates qgb_rtt only")
    parser.add_argument("--history",type=int,default=8,help="Completed RTT window for rolling baselines (default: 8)")
    parser.add_argument("--control-ms",type=float,default=20)
    add_output_arguments(parser)
    parser.set_defaults(command="segment-comparison")
    args = parser.parse_args()
    args.segment = args.segment or args.source
    if args.history<2:
        parser.error("history must be at least two")
    if args.model is not None and not args.model.is_file():
        parser.error(f"Saved QGB model not found: {args.model}. Supply an existing model.pkl; evaluation does not train one.")
    selected,_ = read_records([args.segment])
    targets = [r for rows in selected.values() for r in rows]
    target_ids = {rtt_key(r) for r in targets}
    sessions,_ = read_records([args.source])
    source_ids = {rtt_key(r) for rows in sessions.values() for r in rows}
    if not target_ids.issubset(source_ids):
        raise ValueError("Segment records must match the measured source")
    start = min(r.timestamp_ms for r in targets)
    bundle = load_bundle(args.model) if args.model else None
    if bundle and bundle.config["history"]!=args.history:
        raise ValueError("Use the same history length as the saved QGB model")
    max_gap = bundle.config["max_gap_ms"] if bundle else 10000
    quantiles = bundle.config["quantiles"] if bundle else [.5,.8,.95]
    predictions = []
    skipped_warmup = 0
    scored_ids = set()
    for rows in sessions.values():
        for i,target in enumerate(rows):
            if rtt_key(target) not in target_ids:
                continue
            history = rows[max(0,i-args.history):i]
            if len(history)!=args.history:
                skipped_warmup += 1
                continue
            if bundle and any(r.identity in bundle.used_record_ids for r in history+[target]):
                raise ValueError("Segment history overlaps the model's training/validation data")
            if any(r.status!="ok" or r.rtt_ms is None for r in history+[target]):
                raise ValueError("Insufficient valid context; benchmark does not silently drop targets")
            if any(b.timestamp_ms-a.timestamp_ms>max_gap for a,b in zip(history+[target],(history+[target])[1:])):
                raise ValueError("A long measurement gap crosses a prediction window")
            values = [r.rtt_ms for r in history]
            simple = dict(last_value=values[-1],rolling_mean=statistics.mean(values),
                          rolling_median=statistics.median(values),recent_max=max(values))
            learned = bundle.predict(history,"rtt") if bundle else {}
            scored_ids.add(rtt_key(target))
            for q in quantiles:
                methods = dict(simple)
                if bundle:
                    methods["qgb_rtt"] = learned[q]
                for name,value in methods.items():
                    predictions.append(dict(model=name,quantile=q,session_id=target.session,
                        timestamp_ms=target.timestamp_ms,elapsed_s=(target.timestamp_ms-start)/1000,
                        actual_ms=target.rtt_ms,predicted_ms=value))
    if not predictions:
        raise ValueError("No targets remain after per-session history warmup")
    metrics = []
    methods = list(dict.fromkeys(r["model"] for r in predictions))
    for method in methods:
        for q in quantiles:
            selected = [r for r in predictions if r["model"]==method and r["quantile"]==q]
            metrics.append(dict(model=method,quantile=q,**score([r["actual_ms"] for r in selected],
                                 [r["predicted_ms"] for r in selected],q,args.control_ms)))
    output = resolve_output(args)
    output.mkdir(parents=True,exist_ok=False)
    for filename,rows,writer_class in [("predictions.csv",predictions,csv.DictWriter),("metrics.csv",metrics,ReportWriter)]:
        with (output/filename).open("w",newline="",encoding="utf-8") as stream:
            writer = writer_class(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    (output/"manifest.json").write_text(json.dumps(dict(segment=str(args.segment.resolve()),
        segment_sha256=hashlib.sha256(args.segment.read_bytes()).hexdigest(),source=str(args.source.resolve()),
        requested_samples=len(targets),samples=len(scored_ids),skipped_warmup_samples=skipped_warmup,history=args.history,
        control_ms=args.control_ms,model=str(args.model.resolve()) if bundle else None,
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest() if bundle else None,
        policy="RTT-only, completed past history, no fitting; initial per-session history warmup excluded and counted"),indent=2),encoding="utf-8")
    print(f"Scored {len(scored_ids)} measurements; skipped {skipped_warmup} per-session warmup rows.")
    print(f"Comparison saved: {output.resolve()}")


if __name__=="__main__":
    main()
