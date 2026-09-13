from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main() -> int:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    results_dir = project_root / "results" / "submission_snapshot_latest"
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "name": "unified_lofo",
            "script": project_root / "scripts" / "paper_experiments" / "run_unified_lofo_metrics.py",
            "description": "Unified engineering LOFO table with significance checks.",
        },
        {
            "name": "family_level",
            "script": project_root / "scripts" / "paper_experiments" / "run_family_level_stats.py",
            "description": "Family-level macro aggregation and robustness statistics.",
        },
        {
            "name": "k_ablation",
            "script": project_root / "scripts" / "paper_experiments" / "run_K_ablation.py",
            "description": "Trajectory-length K ablation under the current LOFO protocol.",
        },
        {
            "name": "conformal_routing",
            "script": project_root / "scripts" / "paper_experiments" / "run_conformal_risk_routing.py",
            "description": "Conservative / conformal routing comparison under the current setup.",
        },
        {
            "name": "final_ablations",
            "script": project_root / "scripts" / "paper_experiments" / "run_final_ablations.py",
            "description": "Alpha sensitivity and final ablation table.",
        },
        {
            "name": "comprehensive_baselines",
            "script": project_root / "scripts" / "paper_experiments" / "run_comprehensive_baselines.py",
            "description": "Orthogonal baselines and tau frontier analysis.",
        },
    ]

    manifest: list[dict[str, object]] = []

    for exp in experiments:
        script = exp["script"]
        output_file = raw_dir / f"{exp['name']}.txt"
        command = [sys.executable, str(script)]

        completed = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        combined_output = completed.stdout
        if completed.stderr.strip():
            combined_output += "\n[stderr]\n" + completed.stderr
        output_file.write_text(combined_output, encoding="utf-8")

        manifest.append(
            {
                "name": exp["name"],
                "script": str(script.relative_to(project_root)),
                "description": exp["description"],
                "command": command,
                "returncode": completed.returncode,
                "output_file": str(output_file.relative_to(project_root)),
            }
        )

    timestamp = datetime.now().isoformat(timespec="seconds")
    manifest_path = results_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": timestamp,
                "project_root": str(project_root),
                "experiments": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    readme_lines = [
        "# Submission Snapshot",
        "",
        f"Generated at: `{timestamp}`",
        "",
        "This directory stores a reproducible snapshot of the current low-cost paper-facing result scripts.",
        "",
        "## Included Runs",
        "",
    ]

    for item in manifest:
        status = "OK" if item["returncode"] == 0 else f"FAILED ({item['returncode']})"
        readme_lines.extend(
            [
                f"### {item['name']}",
                "",
                f"- Status: `{status}`",
                f"- Script: `{item['script']}`",
                f"- Output: `{item['output_file']}`",
                f"- Description: {item['description']}",
                "",
            ]
        )

    readme_lines.extend(
        [
            "## Notes",
            "",
            "- These outputs are intended to freeze the current paper-facing evidence before any further protocol cleanup.",
            "- The files under `raw/` are the authoritative captured stdout/stderr traces for this snapshot.",
            "- `manifest.json` records the exact commands and return codes.",
            "",
        ]
    )

    (results_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
