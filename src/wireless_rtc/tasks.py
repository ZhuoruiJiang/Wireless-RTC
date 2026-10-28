"""Named benchmark tasks and their matching upstream checkpoints."""
from pathlib import Path


TASKS = {name:f"worlds/l/{name}.json" for name in (
    "grasp_easy", "cartpole_thrust", "catcher_v3", "hard_lunar_lander", "mjc_walker")}


def selected_tasks(config):
    if "tasks" in config:
        names = config["tasks"]
        if not isinstance(names,list) or not names:
            raise ValueError("tasks must be a nonempty list of task names")
    else:
        # Existing single-level configurations remain runnable.
        level = config.get("level",TASKS["grasp_easy"])
        names = [next((name for name,path in TASKS.items() if path == level),None)]
    if any(not isinstance(name,str) or name not in TASKS for name in names):
        raise ValueError(f"Unknown task; choose from {', '.join(TASKS)}")
    if len(names) != len(set(names)):
        raise ValueError("Selected tasks must be unique")
    return [dict(name=name,level=TASKS[name]) for name in names]


def task_checkpoint(directory, task):
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    return Path(directory)/f"worlds_l_{task}.pkl"
