"""Per-task reporting and equal-weight task averages for each control condition."""
import csv
import json
from wireless_rtc.analysis.latency import LATENCY_METRICS, combine_latency


def aggregate_tasks(rows, tasks, conditions):
    if not tasks or len(tasks) != len(set(tasks)) or not conditions:
        raise ValueError("Summary requires unique selected tasks and nonempty conditions")
    indexed = {}
    for row in rows:
        key = (row["task"],row["method"],row["predictor"])
        if row["task"] not in tasks or (row["method"],row["predictor"]) not in conditions or key in indexed:
            raise ValueError("Unknown or duplicate task/condition summary")
        indexed[key] = row
    overall = []
    for method,predictor in conditions:
        selected = [indexed.get((task,method,predictor)) for task in tasks]
        if any(row is None for row in selected):
            raise ValueError("Cannot average an incomplete task sweep")
        scores = {row["task"]:row["solve_rate"] for row in selected}
        overall.append(dict(method=method,predictor=predictor,predictor_basis=selected[0]["predictor_basis"],
                            tasks=len(tasks),episodes=sum(row["episodes"] for row in selected),
                            average_solve_rate=sum(scores.values())/len(tasks),task_solve_rates=scores,
                            buffer_exhaustion_steps=sum(row["buffer_exhaustion_steps"] for row in selected),
                            deadline_overrun_steps=sum(row["deadline_overrun_steps"] for row in selected),
                            **combine_latency(selected)))
    return dict(schema_version=2,aggregation="equal_weight_mean_of_selected_task_solve_rates",
                latency_aggregation="successful_request_weighted",
                selected_tasks=list(tasks),per_task=rows,overall=overall)


def write_task_summary(output, summary):
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (output/"task_summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=list(summary["per_task"][0]))
        writer.writeheader()
        writer.writerows(summary["per_task"])
    with (output/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        latency_fields = [field for name, (_, average) in LATENCY_METRICS.items()
                          for field in (average, f"{name}_samples")]
        fields = ["method","predictor","average_solve_rate",*latency_fields,*summary["selected_tasks"]]
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in summary["overall"]:
            writer.writerow(dict(method=row["method"],predictor=row["predictor"],
                                 average_solve_rate=row["average_solve_rate"],
                                 **{field:row[field] for field in latency_fields},**row["task_solve_rates"]))
