"""Tests for git-based message transport."""

import json
import os
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch
from src.transport.git_transport import GitTransport, GitTransportError
from src.cooperative_types import CooperativeTask, CooperativeResponse


@pytest.fixture
def mock_repo(tmp_path):
    repo = tmp_path / "test-repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.email", "quill@test.com"],
                   cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Quill"],
                   cwd=str(repo), capture_output=True)
    (repo / "README.md").write_text("test\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)
    return repo


class TestGitTransportInit:
    def test_init_valid_repo(self, mock_repo):
        t = GitTransport(str(mock_repo), "fake-token", "TestAgent")
        assert t.agent_name == "TestAgent"

    def test_init_missing_repo(self):
        with pytest.raises(GitTransportError, match="not found"):
            GitTransport("/nonexistent", "token")

    def test_init_no_token(self, mock_repo):
        os.environ.pop("GITHUB_TOKEN", None)
        with pytest.raises(GitTransportError, match="token"):
            GitTransport(str(mock_repo))


class TestSendTask:
    def test_creates_json_file(self, mock_repo):
        t = GitTransport(str(mock_repo), "fake-token", "Quill")
        task = CooperativeTask.create("Quill", "Oracle1", "ping", {})
        with patch.object(t, '_git_add_commit_push'):
            path = t.send_task(task)
        assert os.path.exists(path)
        data = json.loads(Path(path).read_text())
        assert data["task_id"] == task.task_id

    def test_creates_agent_directory(self, mock_repo):
        t = GitTransport(str(mock_repo), "fake-token", "Quill")
        task = CooperativeTask.create("Quill", "NewAgent", "ping", {})
        with patch.object(t, '_git_add_commit_push'):
            t.send_task(task)
        assert (t.bottle_dir / "NewAgent").exists()


class TestSendResponse:
    def test_creates_response_file(self, mock_repo):
        t = GitTransport(str(mock_repo), "fake-token", "Quill")
        resp = CooperativeResponse.success("t-1", "Quill", "Oracle1", {"v": 42})
        with patch.object(t, '_git_add_commit_push'):
            path = t.send_response(resp)
        assert os.path.exists(path)
        data = json.loads(Path(path).read_text())
        assert data["status"] == "success"


class TestCheckForResponses:
    def test_finds_response(self, mock_repo):
        t = GitTransport(str(mock_repo), "fake-token", "Quill")
        d = t.bottle_dir / "Oracle1"
        d.mkdir(parents=True)
        (d / "response-test-xyz.json").write_text(json.dumps({
            "version": "1.0", "task_id": "test-xyz", "source_agent": "Oracle1",
            "target_agent": "Quill", "status": "success", "result": {"ok": True},
            "execution_time_ms": 5, "responded_at": "2026-04-12T12:00:00Z",
        }))
        with patch.object(t, '_git_pull'):
            resp = t.check_for_responses("test-xyz")
        assert resp is not None
        assert resp.status == "success"

    def test_returns_none_when_missing(self, mock_repo):
        t = GitTransport(str(mock_repo), "fake-token", "Quill")
        with patch.object(t, '_git_pull'):
            assert t.check_for_responses("nope") is None


class TestCheckForTasks:
    def test_finds_agent_tasks(self, mock_repo):
        t = GitTransport(str(mock_repo), "fake-token", "Quill")
        d = t.bottle_dir / "Quill"
        d.mkdir(parents=True)
        (d / "task-incoming-001.json").write_text(json.dumps({
            "version": "1.0", "task_id": "incoming-001", "source_agent": "Oracle1",
            "target_agent": "Quill", "request_type": "ping", "payload": {},
            "created_at": "2026-04-12T12:00:00Z", "expires_at": "2026-04-12T13:00:00Z",
        }))
        with patch.object(t, '_git_pull'):
            tasks = t.check_for_tasks()
        assert len(tasks) == 1
        assert tasks[0].task_id == "incoming-001"

    def test_ignores_other_agent_tasks(self, mock_repo):
        t = GitTransport(str(mock_repo), "fake-token", "Quill")
        d = t.bottle_dir / "Oracle1"
        d.mkdir(parents=True)
        (d / "task-not-for-quill.json").write_text(json.dumps({
            "version": "1.0", "task_id": "x", "source_agent": "A",
            "target_agent": "Oracle1", "request_type": "ping", "payload": {},
        }))
        with patch.object(t, '_git_pull'):
            assert t.check_for_tasks() == []
