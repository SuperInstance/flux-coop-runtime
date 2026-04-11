"""
Git-Based Message Transport Layer.

Sends and receives cooperative execution messages through the fleet's
existing message-in-a-bottle infrastructure (git commits + pushes).
"""

import json
import os
import time
import subprocess
from typing import Optional
from pathlib import Path
from src.cooperative_types import CooperativeTask, CooperativeResponse


class GitTransportError(Exception):
    """Raised when git transport operations fail."""
    pass


class GitTransport:
    """
    Transport layer using git for cooperative message passing.
    
    Messages are written as JSON files in the target repo's
    message-in-a-bottle directory and pushed via git.
    """

    def __init__(self, local_repo_path: str, github_token: str = "",
                 agent_name: str = "Quill"):
        self.local_repo_path = Path(local_repo_path)
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN", "")
        self.agent_name = agent_name
        if not self.local_repo_path.exists():
            raise GitTransportError(f"Local repo not found: {local_repo_path}")
        if not self.github_token:
            raise GitTransportError("GitHub token required.")

    @property
    def bottle_dir(self) -> Path:
        return self.local_repo_path / "message-in-a-bottle" / "for-fleet"

    def _ensure_dirs(self) -> None:
        self.bottle_dir.mkdir(parents=True, exist_ok=True)

    def _git_add_commit_push(self, message: str) -> None:
        try:
            subprocess.run(["git", "add", "-A"], cwd=str(self.local_repo_path),
                           capture_output=True, check=True, timeout=30)
            subprocess.run(["git", "commit", "-m", message], cwd=str(self.local_repo_path),
                           capture_output=True, check=True, timeout=30)
            subprocess.run(["git", "push", "origin", "main"], cwd=str(self.local_repo_path),
                           capture_output=True, check=True, timeout=60)
        except subprocess.TimeoutExpired:
            raise GitTransportError("Git operation timed out")
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else "unknown"
            raise GitTransportError(f"Git failed: {stderr}")

    def _git_pull(self) -> None:
        try:
            subprocess.run(["git", "pull", "origin", "main"], cwd=str(self.local_repo_path),
                           capture_output=True, check=True, timeout=30)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass

    def send_task(self, task: CooperativeTask) -> str:
        """Send a cooperative task to the target agent's bottle directory."""
        self._ensure_dirs()
        agent_dir = self.bottle_dir / task.target_agent
        agent_dir.mkdir(parents=True, exist_ok=True)
        filename = f"task-{task.task_id}.json"
        filepath = agent_dir / filename
        filepath.write_text(json.dumps(task.to_json(), indent=2) + "\n", encoding='utf-8')
        self._git_add_commit_push(f"coop(send): task to {task.target_agent} [{task.task_id[:12]}]")
        return str(filepath)

    def send_response(self, response: CooperativeResponse) -> str:
        """Send a cooperative response back to the requesting agent."""
        self._ensure_dirs()
        agent_dir = self.bottle_dir / response.target_agent
        agent_dir.mkdir(parents=True, exist_ok=True)
        filename = f"response-{response.task_id}.json"
        filepath = agent_dir / filename
        filepath.write_text(json.dumps(response.to_json(), indent=2) + "\n", encoding='utf-8')
        self._git_add_commit_push(f"coop(respond): {response.status} for {response.task_id[:12]}")
        return str(filepath)

    def check_for_responses(self, task_id: str) -> Optional[CooperativeResponse]:
        """Pull from remote and scan for matching response files."""
        self._git_pull()
        if not self.bottle_dir.exists():
            return None
        for agent_dir in self.bottle_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            for f in agent_dir.glob(f"response-{task_id}.json"):
                try:
                    data = json.loads(f.read_text(encoding='utf-8'))
                    return CooperativeResponse.from_json(data)
                except (json.JSONDecodeError, KeyError):
                    continue
        return None

    def check_for_tasks(self) -> list:
        """Check for incoming tasks addressed to this agent."""
        self._git_pull()
        tasks = []
        agent_dir = self.bottle_dir / self.agent_name
        if not agent_dir.exists():
            return tasks
        for f in agent_dir.glob("task-*.json"):
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                tasks.append(CooperativeTask.from_json(data))
            except (json.JSONDecodeError, KeyError):
                continue
        return tasks

    def poll_for_response(self, task_id: str, timeout_ms: int = 30000,
                          poll_interval_ms: int = 5000) -> Optional[CooperativeResponse]:
        """Poll for a response, blocking until timeout."""
        start = time.time()
        timeout_sec = timeout_ms / 1000.0
        poll_sec = poll_interval_ms / 1000.0
        while (time.time() - start) < timeout_sec:
            response = self.check_for_responses(task_id)
            if response is not None:
                return response
            time.sleep(poll_sec)
        return None
