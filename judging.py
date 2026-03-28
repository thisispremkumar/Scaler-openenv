from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import request


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
OPENENV_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "openenv.exe"
DOCKER_IMAGE = "support-triage-env:submission"
INFERENCE_FILE = PROJECT_ROOT / "inference.py"


class CmdError(RuntimeError):
    pass


def run_cmd(command: List[str], cwd: Path | None = None, timeout: int = 300) -> Dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=str(cwd or PROJECT_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
    )
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }


def _http_get_json(url: str, timeout: int = 10) -> Tuple[bool, Dict[str, Any] | None, str]:
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            payload = json.loads(body) if body else None
            return resp.status == 200, payload, ""
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)


def _http_post_json(url: str, payload: Dict[str, Any], timeout: int = 10) -> Tuple[bool, Dict[str, Any] | None, str]:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, method="POST", data=data)
        req.add_header("Content-Type", "application/json")
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body) if body else None
            return resp.status == 200, parsed, ""
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)


@contextlib.contextmanager
def local_server(base_url: str = "http://127.0.0.1:8000"):
    command = [
        str(VENV_PYTHON),
        "-m",
        "uvicorn",
        "server.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]

    proc = subprocess.Popen(  # noqa: S603
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    started = False
    try:
        for _ in range(20):
            ok, _, _ = _http_get_json(f"{base_url}/health")
            if ok:
                started = True
                break
            time.sleep(0.5)
        if not started:
            raise CmdError("Local OpenEnv server failed to start")
        yield
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def check_required_env() -> Dict[str, Any]:
    required = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    present = {name: bool(os.getenv(name)) for name in required}
    passed = all(present.values())
    return {
        "name": "required_env",
        "passed": passed,
        "required": required,
        "present": present,
    }


def check_hf_space() -> Dict[str, Any]:
    space_url = os.getenv("HF_SPACE_URL")
    if not space_url:
        return {
            "name": "hf_space_deploy",
            "passed": False,
            "blocked": True,
            "reason": "HF_SPACE_URL is not set",
        }

    base = space_url.rstrip("/")
    ok_health, health_payload, health_err = _http_get_json(f"{base}/health")
    ok_reset, reset_payload, reset_err = _http_post_json(f"{base}/reset", {})

    return {
        "name": "hf_space_deploy",
        "passed": bool(ok_health and ok_reset),
        "space_url": space_url,
        "health_ok": ok_health,
        "reset_ok": ok_reset,
        "health_payload": health_payload,
        "reset_payload": reset_payload,
        "health_error": health_err,
        "reset_error": reset_err,
    }


def check_openenv_spec() -> Dict[str, Any]:
    validate = run_cmd([str(OPENENV_EXE), "validate"])

    state_check_code = (
        "from server.my_real_world_env_environment import MyRealWorldEnvironment\n"
        "from models import SupportTriageAction\n"
        "env=MyRealWorldEnvironment()\n"
        "obs=env.reset()\n"
        "_ = env.step(SupportTriageAction(action_type='noop'))\n"
        "st=env.state\n"
        "ok=bool(hasattr(st,'episode_id') and hasattr(st,'step_count'))\n"
        "print({'state_ok':ok,'task_id':obs.task_id})\n"
    )
    state_probe = run_cmd([str(VENV_PYTHON), "-c", state_check_code])

    state_ok = False
    state_details: Dict[str, Any] = {}
    if state_probe["returncode"] == 0:
        try:
            state_details = ast.literal_eval(state_probe["stdout"])
            state_ok = bool(state_details.get("state_ok"))
        except Exception:  # noqa: BLE001
            state_ok = False

    endpoint_checks: Dict[str, Any] = {
        "reset_ok": False,
        "step_ok": False,
        "state_ok": False,
        "errors": [],
    }

    try:
        with local_server():
            reset_ok, reset_payload, reset_err = _http_post_json("http://127.0.0.1:8000/reset", {})
            endpoint_checks["reset_ok"] = reset_ok
            if not reset_ok:
                endpoint_checks["errors"].append(reset_err)

            step_ok, step_payload, step_err = _http_post_json(
                "http://127.0.0.1:8000/step",
                {"action": {"action_type": "noop"}},
            )
            endpoint_checks["step_ok"] = step_ok
            if not step_ok:
                endpoint_checks["errors"].append(step_err)

            state_ok_http, state_payload, state_err = _http_get_json("http://127.0.0.1:8000/state")
            endpoint_checks["state_ok"] = state_ok_http
            if not state_ok_http:
                endpoint_checks["errors"].append(state_err)

            endpoint_checks["reset_payload"] = reset_payload
            endpoint_checks["step_payload"] = step_payload
            endpoint_checks["state_payload"] = state_payload
    except Exception as exc:  # noqa: BLE001
        endpoint_checks["errors"].append(str(exc))

    return {
        "name": "openenv_spec",
        "passed": (
            validate["returncode"] == 0
            and state_ok
            and endpoint_checks["reset_ok"]
            and endpoint_checks["step_ok"]
            and endpoint_checks["state_ok"]
        ),
        "openenv_validate": validate,
        "state_probe": state_probe,
        "state_details": state_details,
        "endpoint_checks": endpoint_checks,
    }


def check_docker_build() -> Dict[str, Any]:
    docker_info = run_cmd(["docker", "info", "--format", "{{.ServerVersion}}"])
    if docker_info["returncode"] != 0:
        return {
            "name": "docker_build",
            "passed": False,
            "blocked": True,
            "reason": docker_info["stderr"] or docker_info["stdout"],
        }

    build = run_cmd(
        ["docker", "build", "-t", DOCKER_IMAGE, "-f", "server/Dockerfile", "."],
        timeout=1800,
    )
    return {
        "name": "docker_build",
        "passed": build["returncode"] == 0,
        "docker_info": docker_info,
        "build": build,
    }


def check_tasks_graders() -> Dict[str, Any]:
    code = (
        "from copy import deepcopy\n"
        "from server.tasks import TASKS\n"
        "rows=[]\n"
        "for task in TASKS:\n"
        "  m={t['ticket_id']:deepcopy(t) for t in task.initial_tickets}\n"
        "  a=task.grader(m)\n"
        "  b=task.grader(m)\n"
        "  rows.append({'task_id':task.task_id,'difficulty':task.difficulty,'score':a,'bounded':0.0<=a<=1.0,'deterministic':a==b})\n"
        "print({'task_count':len(TASKS),'rows':rows})\n"
    )
    probe = run_cmd([str(VENV_PYTHON), "-c", code])
    if probe["returncode"] != 0:
        return {
            "name": "tasks_graders",
            "passed": False,
            "probe": probe,
        }

    parsed = ast.literal_eval(probe["stdout"])
    rows = parsed["rows"]
    passed = parsed["task_count"] >= 3 and all(
        row["bounded"] and row["deterministic"] for row in rows
    )
    return {
        "name": "tasks_graders",
        "passed": passed,
        "details": parsed,
    }


def _run_inference_once(seed: int, model_name: str) -> Dict[str, Any]:
    env = os.environ.copy()
    env["MODEL_NAME"] = model_name
    env["INFERENCE_SEED"] = str(seed)

    command = [str(VENV_PYTHON), str(INFERENCE_FILE)]
    started = time.time()
    proc = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT.parent),
        env=env,
        text=True,
        capture_output=True,
        timeout=1200,
    )
    runtime = time.time() - started

    average = None
    for line in proc.stdout.splitlines():
        if line.strip().startswith("average="):
            try:
                average = float(line.split("=", 1)[1])
            except Exception:  # noqa: BLE001
                average = None

    return {
        "command": command,
        "returncode": proc.returncode,
        "runtime_seconds": runtime,
        "average": average,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def check_inference_repro() -> Dict[str, Any]:
    env_check = check_required_env()
    if not env_check["passed"]:
        return {
            "name": "baseline_repro",
            "passed": False,
            "blocked": True,
            "reason": "Required inference env vars missing",
            "env": env_check,
        }

    try:
        with local_server():
            run_a = _run_inference_once(seed=7, model_name=os.getenv("MODEL_NAME", ""))
            run_b = _run_inference_once(seed=7, model_name=os.getenv("MODEL_NAME", ""))
    except Exception as exc:  # noqa: BLE001
        return {
            "name": "baseline_repro",
            "passed": False,
            "blocked": True,
            "reason": f"Local server start failed: {exc}",
        }

    passed = (
        run_a["returncode"] == 0
        and run_b["returncode"] == 0
        and run_a["average"] is not None
        and run_a["average"] == run_b["average"]
        and run_a["runtime_seconds"] < 20 * 60
        and run_b["runtime_seconds"] < 20 * 60
    )

    return {
        "name": "baseline_repro",
        "passed": passed,
        "run_a": run_a,
        "run_b": run_b,
    }


def run_phase1() -> Dict[str, Any]:
    results = {
        "required_env": check_required_env(),
        "hf_space_deploy": check_hf_space(),
        "openenv_spec": check_openenv_spec(),
        "docker_build": check_docker_build(),
        "tasks_graders": check_tasks_graders(),
        "baseline_repro": check_inference_repro(),
    }

    all_pass = all(result.get("passed", False) for result in results.values())
    return {"passed": all_pass, "checks": results}


def run_phase2(seeds: List[int], baseline_model: str, open_llm_model: str) -> Dict[str, Any]:
    env_check = check_required_env()
    if not env_check["passed"]:
        return {
            "passed": False,
            "blocked": True,
            "reason": "Required inference env vars missing",
            "env": env_check,
        }

    try:
        with local_server():
            baseline_runs = [_run_inference_once(seed=s, model_name=baseline_model) for s in seeds]
            open_runs = [_run_inference_once(seed=s, model_name=open_llm_model) for s in seeds]
    except Exception as exc:  # noqa: BLE001
        return {
            "passed": False,
            "blocked": True,
            "reason": f"Local server start failed: {exc}",
        }

    def _stats(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        avgs = [r["average"] for r in runs if isinstance(r.get("average"), float)]
        if not avgs:
            return {"count": 0, "mean": None, "stdev": None}
        return {
            "count": len(avgs),
            "mean": sum(avgs) / len(avgs),
            "stdev": statistics.pstdev(avgs) if len(avgs) > 1 else 0.0,
        }

    all_returned = all(r["returncode"] == 0 for r in baseline_runs + open_runs)
    baseline_stats = _stats(baseline_runs)
    open_stats = _stats(open_runs)

    return {
        "passed": all_returned,
        "baseline_model": baseline_model,
        "open_llm_model": open_llm_model,
        "seeds": seeds,
        "baseline_runs": baseline_runs,
        "open_runs": open_runs,
        "baseline_stats": baseline_stats,
        "open_stats": open_stats,
        "variance_check": {
            "baseline_stdev": baseline_stats["stdev"],
            "open_stdev": open_stats["stdev"],
        },
    }


def run_phase3_stub() -> Dict[str, Any]:
    return {
        "status": "pending_manual_review",
        "reviewers": ["Meta engineer", "Hugging Face engineer"],
        "checkpoints": [
            "Real-world utility",
            "Creativity and novelty",
            "Exploit checks",
            "Plagiarism check",
        ],
    }


def disqualification(phase1: Dict[str, Any]) -> Dict[str, Any]:
    checks = phase1["checks"]
    return {
        "environment_does_not_deploy_or_respond": not checks["hf_space_deploy"].get("passed", False),
        "graders_always_same_score": not checks["tasks_graders"].get("passed", False),
        "missing_inference_script": not INFERENCE_FILE.exists(),
        "plagiarism_manual_required": True,
    }


def write_reports(report: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "pre_submission_report.json"
    md_path = out_dir / "PRE_SUBMISSION_REPORT.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    dq = report["disqualification"]
    lines = [
        "# Pre-Submission Validation Report",
        "",
        f"- Timestamp: {report['timestamp']}",
        f"- Overall status: {report['overall_status']}",
        "",
        "## Phase 1",
        f"- Required env vars set: {report['phase1']['checks']['required_env']['passed']}",
        f"- HF Space deploy ping/reset: {report['phase1']['checks']['hf_space_deploy']['passed']}",
        f"- OpenEnv spec compliance: {report['phase1']['checks']['openenv_spec']['passed']}",
        f"- Docker build: {report['phase1']['checks']['docker_build']['passed']}",
        f"- Baseline reproduces via inference.py: {report['phase1']['checks']['baseline_repro']['passed']}",
        f"- 3+ tasks with graders: {report['phase1']['checks']['tasks_graders']['passed']}",
        "",
        "## Phase 2",
        f"- Agentic evaluation passed: {report['phase2'].get('passed', False)}",
        "",
        "## Phase 3",
        "- Human review required by Meta and Hugging Face engineers.",
        "",
        "## Disqualification",
        f"- Environment does not deploy/respond: {dq['environment_does_not_deploy_or_respond']}",
        f"- Graders always return same score: {dq['graders_always_same_score']}",
        f"- Missing inference.py: {dq['missing_inference_script']}",
        f"- Plagiarism manual check required: {dq['plagiarism_manual_required']}",
    ]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete judging pipeline (phase1+phase2+phase3 stub).")
    parser.add_argument("--seeds", default="7,8,9")
    parser.add_argument("--baseline-model", default=os.getenv("MODEL_NAME", ""))
    parser.add_argument(
        "--open-llm-model",
        default=os.getenv("OPEN_LLM_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct"),
    )
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "artifacts" / "submission"))
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    phase1 = run_phase1()
    phase2 = run_phase2(seeds=seeds, baseline_model=args.baseline_model, open_llm_model=args.open_llm_model)
    phase3 = run_phase3_stub()

    dq = disqualification(phase1)
    hard_dq = dq["environment_does_not_deploy_or_respond"] or dq["graders_always_same_score"] or dq["missing_inference_script"]

    if hard_dq:
        status = "DISQUALIFIED"
    elif not phase1["passed"]:
        status = "PHASE1_FAILED"
    elif not phase2.get("passed", False):
        status = "PHASE2_FAILED"
    else:
        status = "PASS_PENDING_HUMAN_REVIEW"

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": status,
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3,
        "disqualification": dq,
    }

    out_dir = Path(args.out_dir)
    write_reports(report, out_dir)

    print(
        json.dumps(
            {
                "overall_status": status,
                "report_json": str((out_dir / "pre_submission_report.json").resolve()),
                "report_md": str((out_dir / "PRE_SUBMISSION_REPORT.md").resolve()),
            },
            indent=2,
        )
    )

    if status in {"DISQUALIFIED", "PHASE1_FAILED", "PHASE2_FAILED"}:
        sys.exit(1)


if __name__ == "__main__":
    main()
