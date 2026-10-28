"""Server-owned task policy selection; clients never choose checkpoint paths."""
from pathlib import Path

from wireless_rtc.control_modes import CONTROL_MODES, PROTOCOL_VERSION
from wireless_rtc.tasks import TASKS, task_checkpoint


class PolicyRegistry:
    def __init__(self, checkpoint_dir, factory=None):
        self.directory = Path(checkpoint_dir)
        if not self.directory.is_dir():
            raise ValueError(f"Checkpoint directory does not exist: {self.directory}")
        if factory is None:
            from wireless_rtc.simulation.kinetix import KinetixPolicy
            factory = KinetixPolicy
        self.factory = factory
        self._policies = {}
        self._active = None

    def available_tasks(self):
        return [name for name in TASKS if task_checkpoint(self.directory,name).is_file()]

    def describe(self):
        result = dict(protocol_version=PROTOCOL_VERSION,control_modes=CONTROL_MODES.copy(),
                      available_tasks=self.available_tasks(),active_task=self._active)
        if self._active is not None:
            result.update(self._policies[self._active].describe(),task=self._active)
        result["available_tasks"] = self.available_tasks()
        return result

    def select_task(self, task):
        checkpoint = task_checkpoint(self.directory,task)
        if not checkpoint.is_file():
            raise ValueError(f"Missing checkpoint for {task}: {checkpoint}")
        if task not in self._policies:
            try:
                backend = self.factory(str(checkpoint),TASKS[task])
            except Exception as exc:
                raise ValueError(f"Could not load policy for {task}: {exc}") from exc
            if backend.describe().get("level") != TASKS[task]:
                raise ValueError(f"Policy level mismatch for {task}")
            self._policies[task] = backend
        self._active = task
        return self.describe()

    def _policy(self, request):
        if self._active is None or request.get("task") != self._active:
            raise ValueError("Request task does not match selected policy; select_task first")
        return self._policies[self._active]

    def initialize(self, request):
        return dict(self._policy(request).initialize(request),task=self._active)

    def infer(self, request):
        return dict(self._policy(request).infer(request),task=self._active)
