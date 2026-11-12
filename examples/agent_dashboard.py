"""Live terminal views of MechCAL agent runs (read-only; reads events.jsonl, result.json, grade.json).

    python examples/agent_dashboard.py board outputs/agent/dev2    # one row per case
    python examples/agent_dashboard.py feed  outputs/agent/dev2    # event stream across cases
    python examples/agent_dashboard.py procs outputs/agent/dev2    # compute processes per case

``examples/agent_tmux.sh <run root>`` opens all three in one tmux session.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

SHORT = {"ICT_TICT_CT": "ICT", "RADIATIVE_RATE_STATE_BALANCE": "RAD", "TRIPLET_METAL_ENERGY_TRANSFER": "TRIP",
         "RIM_RIR_RIV": "RIM", "PACKING_HOST_MATRIX_CONFINEMENT": "PACK", "AGGREGATE_EXCITON_EXCIMER": "AGG",
         "ESIPT_PT": "ESIPT", "PET_ET": "PET", "HOST_GUEST_INTERACTION": "HG", "RACI_CI_ACCESS": "RACI",
         "SOKR_ANTI_KASHA": "SOKR"}
STATUS_STYLE = {"done": "green", "done_with_warnings": "yellow", "blocked_technical": "red", "needs_approval": "magenta",
                "not_applicable": "bright_black", "cost_check": "cyan", "rejected": "red"}
WALL = 3600.0


def events(path: Path) -> list[dict]:
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return out


def alive_runs() -> dict[str, int]:
    """Output dirs of running agent CLIs -> pid."""
    found = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        args = [a.decode(errors="replace") for a in argv]
        if "mechcal.agent.cli" in args and "--out" in args:
            found[str(Path(args[args.index("--out") + 1]).resolve())] = int(pid)
    return found


def fmt_secs(s: float) -> str:
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def labels(xs) -> str:
    return " ".join(SHORT.get(x, x) for x in xs or [])


def case_state(d: Path, alive: dict) -> dict:
    ev = events(d / "events.jsonl")
    st = {"case": d.name, "ev": ev}
    if not ev:
        st["status"] = "starting"
        return st
    start = ev[0]["t"]
    fin = next((e for e in ev if e.get("role") == "finish"), None)
    running = str(d.resolve()) in alive
    st["status"] = "finished" if fin else ("running" if running else "stopped")
    st["elapsed"] = (fin["t"] if fin else time.time()) - start
    usage = {"planner": [0, 0], "worker": [0, 0], "critic": [0, 0]}
    for e in ev:
        if e.get("role") in usage and isinstance(e.get("usage"), dict):
            usage[e["role"]][0] += 1
            usage[e["role"]][1] += int(e["usage"].get("total_tokens") or 0)
    st["usage"] = usage
    attempts = [e for e in ev if e.get("role") == "attempt"]
    st["ok"] = sum(a.get("outcome") == "success" for a in attempts)
    st["fail"] = len(attempts) - st["ok"]
    st["wall"] = sum(float(a.get("wall_seconds") or 0) for a in attempts)
    st["tasks"] = [e for e in ev if e.get("role") == "task_created"]
    st["intercepts"] = sum(e.get("role") == "interception" for e in ev)
    revs = [e for e in ev if e.get("role") == "task_revised"]
    st["revisions"], st["approvals"] = len(revs), sum(bool(e.get("approved")) for e in revs)
    st["cost_checks"] = sum(e.get("role") == "cost_check" for e in ev)
    reports = [e for e in ev if e.get("role") == "worker_report"]
    st["last_report"] = reports[-1] if reports else None
    acts = [e for e in ev if e.get("role") == "action"]
    st["top3"] = acts[-1].get("board_top3") if acts else []
    st["step"] = acts[-1].get("step") if acts else 0
    st["activity"] = activity(ev)
    if (d / "result.json").is_file():
        try:
            r = json.loads((d / "result.json").read_text())
            st["wall"] = r["tool_cost"]["tool_wall_seconds"]
            st["top3"] = [p["label"] for p in r["submission"]["mechanism_predictions"]]
        except (ValueError, KeyError):
            pass
    if (d / "grade.json").is_file():
        try:
            st["grade"] = json.loads((d / "grade.json").read_text())
        except ValueError:
            pass
    return st


def activity(ev: list[dict]) -> Text:
    last = ev[-1]
    role = last.get("role")
    if role == "finish":
        return Text(f"submitted ({last.get('stop_reason')})", style="green")
    if role == "worker":
        a = (last.get("response") or {}).get("action") or {}
        if a.get("type") in {"recipe", "tool"}:
            return Text(f"⚙ running {a.get('type')} {str(a.get('name', '')).split('.')[-1]}", style="bold cyan")
        if a.get("type") == "read":
            return Text("student reading a value", style="cyan")
        return Text("student writing report", style="cyan")
    if role == "planner":
        a = (last.get("response") or {}).get("action") or {}
        if a.get("type") == "delegate":
            return Text(f"assigning: {(a.get('task') or '')[:50]}", style="blue")
        if a.get("type") == "tool":
            return Text(f"⚙ planner tool {str(a.get('name', '')).split('.')[-1]}", style="bold blue")
        return Text(f"professor: {a.get('type', 'thinking')}", style="blue")
    if role == "critic":
        return Text("critic reviewing", style="magenta")
    if role in {"task_created", "task_revised"}:
        return Text(f"student starting {last.get('task_id')}", style="cyan")
    if role == "attempt":
        return Text(f"student: {last.get('name', '').split('.')[-1]} -> {last.get('outcome')}", style="cyan")
    if role == "worker_report":
        return Text("professor reading the report", style="blue")
    return Text("professor thinking", style="blue")


def board(root: Path) -> Group:
    alive = alive_runs()
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists() and "crashed" not in p.name)
    states = [case_state(d, alive) for d in dirs]
    t = Table(expand=True, header_style="bold", pad_edge=False, show_lines=False)
    for col, kw in (("case", {}), ("status", {}), ("time", {"justify": "right"}), ("now", {"ratio": 3}),
                    ("tasks", {"justify": "right"}), ("att ✓/✗", {"justify": "right"}), ("int/rev/appr", {"justify": "right"}),
                    ("tool time", {"ratio": 2}), ("tokens P/W/C (k)", {"justify": "right"}), ("top-3", {"ratio": 2}),
                    ("score", {"ratio": 2})):
        t.add_column(col, **kw)
    total_tokens = 0
    for s in states:
        if "elapsed" not in s:
            t.add_row(s["case"], s["status"], "", "", "", "", "", "", "", "", "")
            continue
        status = {"finished": "[green]✔ done[/]", "running": "[cyan]● run[/]", "stopped": "[red]✖ stopped[/]"}[s["status"]]
        u = s["usage"]
        total_tokens += sum(v[1] for v in u.values())
        bar = ProgressBar(total=WALL, completed=min(s["wall"], WALL), width=14,
                          complete_style="red" if s["wall"] > 0.8 * WALL else "cyan")
        wall_cell = Group(bar, Text(f"{fmt_secs(s['wall'])} / {fmt_secs(WALL)}", style="dim"))
        last = s["last_report"]
        now = Group(s["activity"], Text(f"{last['task_id']}: {last['status']}", style=STATUS_STYLE.get(last["status"], "white"))
                    if last else Text(""))
        grade = s.get("grade")
        if grade:
            m = grade["metrics"]
            score = Text.assemble(("nDCG ", "dim"), (f"{m['ndcg_at_3']:.2f}", "bold"), ("  ES ", "dim"), f"{m['es_ndcg_at_3']:.2f}",
                                  ("\nref ", "dim"), labels(r[0] for r in grade["reference"]))
        else:
            score = Text("grading…" if s["status"] == "finished" else "", style="dim")
        t.add_row(s["case"], status, fmt_secs(s["elapsed"]), now, str(len(s["tasks"])), f"{s['ok']}/{s['fail']}",
                  f"{s['intercepts']}/{s['revisions']}/{s['approvals']}", wall_cell,
                  f"{u['planner'][1] // 1000}/{u['worker'][1] // 1000}/{u['critic'][1] // 1000}",
                  labels(s["top3"]), score)
    load = os.getloadavg()[0]
    mem = meminfo()
    graded = [s["grade"]["metrics"] for s in states if s.get("grade")]
    summary = Text.assemble(
        ("MechCAL agent runs  ", "bold"), (str(root), "cyan"), "   ",
        (f"{sum(s['status'] == 'finished' for s in states)}/{len(states)} finished", "green"), "   ",
        (f"load {load:.0f}/{os.cpu_count()}", "yellow"), "   ", (f"mem {mem}", "yellow"), "   ",
        (f"tokens {total_tokens / 1e6:.2f} M", "magenta"), "   ",
        *((("mean nDCG ", "dim"), (f"{sum(m['ndcg_at_3'] for m in graded) / len(graded):.3f}", "bold green"),
           ("  ES-nDCG ", "dim"), (f"{sum(m['es_ndcg_at_3'] for m in graded) / len(graded):.3f}", "bold green"))
          if graded else ()), "   ", (time.strftime("%H:%M:%S"), "dim"))
    legend = Text("int/rev/appr = interceptions / professor revisions / approvals;  att = student attempts;  "
                  "tool time = recorded calculation time of the 3600 s budget", style="dim")
    return Group(Panel(summary, border_style="blue"), t, legend)


def meminfo() -> str:
    try:
        info = dict(re.findall(r"(\w+):\s+(\d+)", Path("/proc/meminfo").read_text()))
        total, avail = int(info["MemTotal"]) / 1e6, int(info["MemAvailable"]) / 1e6
        return f"{total - avail:.0f}/{total:.0f} GB"
    except (OSError, KeyError):
        return "?"


def feed_line(case: str, e: dict) -> Text | None:
    r = e.get("role")
    ts = time.strftime("%H:%M:%S", time.localtime(e.get("t", time.time())))
    head = Text.assemble((ts + " ", "dim"), (f"{case:>5} ", "bold"))
    if r == "task_created":
        what = e.get("recipe") or ",".join(t.split(".")[-1] for t in e.get("tools") or [])
        fixed = ", ".join(f"{k}={v}" for k, v in (e.get("fixed_args") or {}).items() if k != "fragments")[:70]
        return head + Text.assemble(("ASSIGN  ", "bold blue"), f"{e['task_id']} {what}  ", (fixed, "dim"))
    if r == "attempt":
        ok = e.get("outcome") == "success"
        warn = ",".join(w["code"] for w in e.get("warnings", []))
        changed = ",".join((e.get("changed") or {}).keys())
        return head + Text.assemble(("ATTEMPT ", "bold green" if ok else "bold red"),
                                    f"{e['task_id']} {e['name'].split('.')[-1]} → {e.get('outcome')}",
                                    (f" {e.get('failure_code')}" if not ok and e.get("failure_code") else "", "red"),
                                    (f" {round(e.get('wall_seconds', 0))}s", "dim"),
                                    (f"  changed: {changed}" if changed else "", "yellow"), (f"  ⚠ {warn}" if warn else "", "yellow"))
    if r == "interception":
        return head + Text.assemble(("BLOCKED ", "bold magenta"), f"{e['task_id']} {e.get('code')}  ", (e.get("message", "")[:80], "dim"))
    if r == "worker_report":
        style = STATUS_STYLE.get(e.get("status"), "white")
        q = (e.get("question_for_planner") or "")[:70]
        return head + Text.assemble(("REPORT  ", f"bold {style}"), f"{e['task_id']} {e.get('status')} ({len(e.get('claims', []))} findings)",
                                    (f"  ? {q}" if q else "", "dim"),
                                    (f"  corrected: {e['system_corrections']}" if e.get("system_corrections") else "", "red"))
    if r == "task_revised":
        return head + Text.assemble(("REVISE  ", "bold yellow"), f"{e['task_id']} {'✔ approved ' if e.get('approved') else ''}",
                                    (e.get("instruction", "")[:80], "dim"))
    if r == "cost_check":
        est = e.get("estimate") or {}
        return head + Text.assemble(("COST    ", "bold cyan"), f"~{est.get('seconds', est.get('at_least_seconds'))} s of {e.get('left')} s left: ",
                                    ((e.get("objective") or "")[:60], "dim"))
    if r == "action" and e["action"]["type"] in {"tool", "read", "submit"}:
        return head + Text.assemble(("PLANNER ", "bold blue"), e.get("summary", "")[:90])
    if r == "critic_summary":
        props = [f"{SHORT.get(p.get('label'), p.get('label'))}:{p.get('kind', '')[:5]}" for p in e.get("proposals", [])]
        return head + Text.assemble(("CRITIC  ", "bold magenta"), " ".join(props) or "no proposals")
    if r == "finish":
        return head + Text.assemble(("FINISH  ", "bold green"), e.get("stop_reason", ""))
    if r in {"planner", "worker", "critic"} and e.get("valid") is False:
        return head + Text.assemble(("INVALID ", "bold red"), f"{r} output repaired: ", ((e.get("error") or "")[:70], "dim"))
    return None


def feed(root: Path, console: Console) -> None:
    offsets: dict[Path, int] = {}
    first = True
    while True:
        for path in sorted(root.glob("*/events.jsonl")):
            if "crashed" in str(path):
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            start = offsets.get(path, max(0, len(lines) - 6) if first else 0)
            for line in lines[start:]:
                try:
                    out = feed_line(path.parent.name, json.loads(line))
                except (ValueError, KeyError):
                    continue
                if out is not None:
                    console.print(out, overflow="ellipsis", no_wrap=True)
            offsets[path] = len(lines)
        first = False
        time.sleep(1.5)


HEAVY = re.compile(r"amesp|pyscf|xtb|crest|dftb|Multiwfn|pysoc|m2m", re.IGNORECASE)


_SEEN: dict[int, tuple[float, float, float]] = {}   # pid -> (first seen, last cpu seconds, last sample time)


def procs(root: Path) -> Table:
    """QC processes per case. CPU % is measured between refreshes (container clocks make ps/stat ages unreliable)."""
    t = Table(expand=True, header_style="bold", title=f"compute processes  {time.strftime('%H:%M:%S')}")
    for col in ("case", "program", "cpu %", "threads", "mem GB", "seen for"):
        t.add_column(col, justify="left" if col in {"case", "program"} else "right")
    ticks, page = os.sysconf("SC_CLK_TCK"), os.sysconf("SC_PAGE_SIZE")
    now, rows, live = time.time(), [], set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            args = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            if not HEAVY.search(args) or "agent_dashboard" in args or args.startswith("/usr/bin/time"):
                continue
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            continue
        n = int(pid)
        live.add(n)
        cpu_s = (int(stat[11]) + int(stat[12])) / ticks
        first, last_cpu, last_t = _SEEN.get(n, (now, cpu_s, now))
        cpu = 100 * (cpu_s - last_cpu) / (now - last_t) if now > last_t else 0.0
        _SEEN[n] = (first, cpu_s, now)
        where = cwd + " " + args
        case = next((p.name for p in root.iterdir() if p.is_dir() and f"/{p.name}/" in where + "/"), "?")
        program = Path(args.split()[0]).name
        if "pyscf_states_worker" in args:
            program = "PySCF spin states"
        elif "python" in program:
            program = next((Path(a).stem for a in args.split()[1:] if a.endswith(".py")), program)
        rows.append((case, program, cpu, int(stat[17]), int(stat[21]) * page / 1e9, now - first))
    for gone in set(_SEEN) - live:
        _SEEN.pop(gone)
    for case, program, cpu, threads, mem, age in sorted(rows):
        style = "bold green" if cpu > 150 else ("green" if cpu > 50 else "dim")
        t.add_row(case, program, Text(f"{cpu:.0f}", style=style), str(threads), f"{mem:.1f}", fmt_secs(age))
    if not rows:
        t.add_row("", "[dim]no QC process right now[/]", "", "", "", "")
    return t


def main() -> None:
    mode, root = sys.argv[1], Path(sys.argv[2] if len(sys.argv) > 2 else "outputs/agent/dev2")
    console = Console()
    if mode == "feed":
        console.rule(f"[bold]event stream[/]  {root}")
        feed(root, console)
        return
    render = board if mode == "board" else procs
    with Live(render(root), console=console, refresh_per_second=1, screen=True) as live:
        while True:
            time.sleep(2)
            live.update(render(root))


if __name__ == "__main__":
    main()
