"""実験の記録をファイルに保存する ``ExperimentRepository`` の実装。

runs_dir/<run_id>/ に次のファイルを作る。

- config.json: 設定 (実験・環境・エージェント) と実行環境の情報 (コミット、Python のバージョンなど)
- episodes.csv: エピソードごとの結果
- evaluations.csv: 途中評価の結果
- agent.json: 学習済みエージェント
- summary.json: 最終評価と終了時刻
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from lights_out.domain.experiment import EpisodeRecord, EvaluationRecord, ExperimentRepository
from lights_out.infrastructure.provenance import collect_provenance


class FileExperimentRepository(ExperimentRepository):
    def __init__(self, runs_dir: str | Path = "runs") -> None:
        self._runs_dir = Path(runs_dir)

    def begin(self, name: str, config: dict[str, Any]) -> str:
        started_at = datetime.now().astimezone()
        run_dir = self._create_run_dir(f"{started_at:%Y%m%d-%H%M%S}_{name}")
        self._write_json(
            run_dir / "config.json",
            {
                "run_id": run_dir.name,
                "started_at": started_at.isoformat(timespec="seconds"),
                **config,
                "provenance": collect_provenance(),
            },
        )
        return run_dir.name

    def finish(
        self,
        run_id: str,
        *,
        episodes: Sequence[EpisodeRecord],
        evaluations: Sequence[EvaluationRecord],
        agent_snapshot: dict[str, Any],
    ) -> None:
        run_dir = self._runs_dir / run_id
        self._write_episodes(run_dir / "episodes.csv", episodes)
        self._write_evaluations(run_dir / "evaluations.csv", evaluations)
        self._write_json(run_dir / "agent.json", agent_snapshot)
        self._write_json(
            run_dir / "summary.json",
            {
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "n_episodes": len(episodes),
                "final_evaluation": _evaluation_row(evaluations[-1]) if evaluations else None,
            },
        )

    def _create_run_dir(self, run_id: str) -> Path:
        """同じ秒に始めた run と衝突したら、末尾に連番を付ける。"""
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("", *(f"-{n}" for n in range(2, 1000))):
            run_dir = self._runs_dir / f"{run_id}{suffix}"
            try:
                run_dir.mkdir()
            except FileExistsError:
                continue
            return run_dir
        raise RuntimeError(f"run ディレクトリを作れませんでした: {run_id}")

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _write_episodes(path: Path, episodes: Sequence[EpisodeRecord]) -> None:
        metric_names = list(dict.fromkeys(name for e in episodes for name in e.metrics))
        _write_csv(
            path,
            ["episode", "total_reward", "steps", "solved", *metric_names],
            (
                {
                    "episode": e.episode,
                    "total_reward": e.total_reward,
                    "steps": e.steps,
                    "solved": int(e.solved),
                    **e.metrics,
                }
                for e in episodes
            ),
        )

    @staticmethod
    def _write_evaluations(path: Path, evaluations: Sequence[EvaluationRecord]) -> None:
        _write_csv(
            path,
            ["episode", "n_episodes", "success_rate", "mean_reward", "mean_steps"],
            (_evaluation_row(e) for e in evaluations),
        )


def _evaluation_row(evaluation: EvaluationRecord) -> dict[str, Any]:
    summary = evaluation.summary
    return {
        "episode": evaluation.episode,
        "n_episodes": summary.n_episodes,
        "success_rate": summary.success_rate,
        "mean_reward": summary.mean_reward,
        "mean_steps": summary.mean_steps,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: Any) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
