"""Reproduce the 500-game humanexp comparison; run from the repository root."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

# Set before importing NumPy or the controller modules.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

from powergrid.ai import BaseAiController, build_ai_controller, derive_final_standings
from powergrid.ai.nn_rl_based.controller import DEFAULT_CHECKPOINT_PATH, DISPLAY_LABEL, NnRlBasedAiController
from powergrid.model import GameConfig, SeatConfig, legal_region_sets
from powergrid.session import GameSession

NEW = "ai_humanexp_heuristics_v1"
OPPONENTS = (
    "ai_deterministic", "ai_deterministic_efficiency",
    "ai_deterministic_expansion", "ai_deterministic_reserve", "ai_nn_rl_based_v1",
)
LABELS = {
    "opening_auction_start": "首轮起拍／选厂",
    "opening_auction_bid": "首轮跟价／放弃",
    "later_auction_start": "后续轮起拍／选厂",
    "later_auction_bid": "后续轮跟价／放弃",
    "resource_plan": "资源购买：首次规划",
    "resource_cached": "资源购买：后续缓存决策",
    "build_houses": "建城",
    "bureaucracy": "发电",
    "discard_power_plant": "弃置电厂",
    "discard_hybrid_resources": "弃置混合燃料",
}
FIELDS = ["opponent", "game_index", "seed", "composition", "player_id", "round",
          "phase", "decision_type", "category", "resource", "intent_type", "wall_ns", "cpu_ns"]


class TimedController(BaseAiController):
    controller = NEW

    def __init__(self, inner, rows):
        self.inner = inner
        self.rows = rows

    def choose_intent(self, request, snapshot):
        state = snapshot.state
        category = request.decision_type
        if category in ("auction_start", "auction_bid"):
            category = ("opening_" if state.round_number == 1 else "later_") + category
        elif category == "buy_resources":
            key = (self.inner._game_key, state.round_number, request.player_id)
            category = "resource_cached" if self.inner._resource_key == key else "resource_plan"
        cpu_start = time.process_time_ns()
        wall_start = time.perf_counter_ns()
        intent = self.inner.choose_intent(request, snapshot)
        wall_ns = time.perf_counter_ns() - wall_start
        cpu_ns = time.process_time_ns() - cpu_start
        self.rows.append({
            "player_id": request.player_id, "round": state.round_number,
            "phase": request.phase, "decision_type": request.decision_type,
            "category": category, "resource": request.metadata.get("resource", ""),
            "intent_type": intent.intent_type, "wall_ns": wall_ns, "cpu_ns": cpu_ns,
        })
        if len(self.rows) > 30000:
            raise RuntimeError("Game exceeded 30000 humanexp decisions")
        return intent


def source_hashes(root):
    paths = sorted((root / "src/powergrid").rglob("*.py"))
    paths += sorted((root / "src/powergrid/data").rglob("*.json"))
    paths += [DEFAULT_CHECKPOINT_PATH, Path(__file__).resolve()]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def timing_stats(rows):
    if not rows:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "max_ms": None}
    wall = np.array([r["wall_ns"] for r in rows], dtype=np.float64) / 1e6
    return {
        "count": len(rows), "total_ms": float(wall.sum()), "mean_ms": float(wall.mean()),
        "median_ms": float(np.median(wall)), "p95_ms": float(np.percentile(wall, 95)),
        "max_ms": float(wall.max()), "mean_cpu_ms": sum(r["cpu_ns"] for r in rows) / len(rows) / 1e6,
    }


def summarize_games(games):
    count = len(games)
    seats = [s for g in games for s in g["standings"] if s["controller_name"] == NEW]
    return {
        "games": count,
        "outright_controller_wins": sum(g["new_win_credit"] == 1 for g in games),
        "shared_controller_wins": sum(0 < g["new_win_credit"] < 1 for g in games),
        "losses": sum(g["new_win_credit"] == 0 for g in games),
        "new_win_credit": sum(g["new_win_credit"] for g in games),
        "new_win_rate": sum(g["new_win_credit"] for g in games) / count,
        "opponent_win_rate": sum(1 - g["new_win_credit"] for g in games) / count,
        "pairwise_wins": sum(g["pairwise_wins"] for g in games),
        "pairwise_draws": sum(g["pairwise_draws"] for g in games),
        "pairwise_losses": sum(g["pairwise_losses"] for g in games),
        "pairwise_score": sum(g["pairwise_score"] for g in games) / count,
        "new_seat_appearances": len(seats),
        "new_seat_win_rate": sum(sum(s["player_id"] in g["winner_ids"] for s in g["standings"]
            if s["controller_name"] == NEW) for g in games) / len(seats),
        "new_mean_place": sum(s["place"] for s in seats) / len(seats),
        "new_mean_powered_cities": sum(s["powered_cities"] for s in seats) / len(seats),
        "mean_game_seconds": sum(g["elapsed_seconds"] for g in games) / count,
    }


def summarize_opponent(games, rows):
    result = summarize_games(games)
    result["compositions"] = {composition: summarize_games([g for g in games if g["composition"] == composition])
                              for composition in ("HOO", "HHO")}
    clusters = defaultdict(list)
    for game in games:
        clusters[game["seed"]].append(game)
    rng = np.random.default_rng(20260922)
    sample_indices = rng.integers(0, len(clusters), size=(5000, len(clusters)))
    for metric, field in (("pairwise_score", "pairwise_score"), ("new_win_rate", "new_win_credit")):
        values = np.array([sum(g[field] for g in group) / len(group) for group in clusters.values()])
        result[metric + "_95_ci_seed_bootstrap"] = np.quantile(values[sample_indices].mean(axis=1), [0.025, 0.975]).tolist()
    result["timing_by_category"] = {key: timing_stats([r for r in rows if r["category"] == key]) for key in LABELS}
    return result


def write_report(output, report):
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    config = report["configuration"]
    lines = ["# 经验启发式 v1 实战评测", "",
        f"实际完成 {report['completed_games']} 局，失败 {report['failed_games']} 局。", "",
        "德国三人局；对每个版本运行 100 局：一席新 AI + 两席对手（HOO）50 局，两席新 AI + 一席对手（HHO）50 局。每席独立控制器，不共享策略记忆。",
        "同一对种子使用相反人数配置，少数一方的座位轮换；每个版本合计各占 150 个席位，在每个座位各占 50 次。",
        f"种子 {config['seed_start']}–{config['seed_start'] + config['seed_pairs'] - 1}，依次循环全部 {config['legal_region_sets']} 组合法区域；各对手使用相同赛程。",
        "胜率指冠军所属 AI 版本的份额；跨版本并列冠军按获胜席位分摊。HOO/HHO 单独列出。同等实力时，均衡赛程的期望版本胜率为 50%。",
        "名次对比得分是每局新 AI 与对手的两组跨版本席位比较，胜=1、平=0.5、负=0；它不是夺冠率。", "",
        "| 对手 | 完成局数 | 新 AI 夺冠份额 | 对手夺冠份额 | 新 AI 胜率 | HOO 胜率 | HHO 胜率 | 名次对比得分 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for opponent, result in report["opponents"].items():
        lines.append(f"| {opponent} | {result['games']} | {result['new_win_credit']:g} | {result['games']-result['new_win_credit']:g} | {result['new_win_rate']:.1%} | {result['compositions']['HOO']['new_win_rate']:.1%} | {result['compositions']['HHO']['new_win_rate']:.1%} | {result['pairwise_score']:.1%} |")
    lines += ["", "## 新 AI 决策耗时", "",
        "在当前机器上单进程顺序运行，数值为 choose_intent 的实际墙钟耗时，包含 AI 自身分析日志写入；不含快照构造、游戏规则执行、界面渲染及控制器初始化。算术平均按调用次数加权。",
        "资源首次规划包含最优发电组合、建城预留与额外囤货计算；同轮其余可购买资源复用该规划。未发生的决策标为无样本。", "",
        "| 决策 | 调用次数 | 平均 ms | 中位数 ms | P95 ms | 最大 ms |",
        "|---|---:|---:|---:|---:|---:|"]
    for key, stats in report["timing_by_category"].items():
        if stats["count"]:
            lines.append(f"| {LABELS[key]} | {stats['count']} | {stats['mean_ms']:.4f} | {stats['median_ms']:.4f} | {stats['p95_ms']:.4f} | {stats['max_ms']:.4f} |")
        else:
            lines.append(f"| {LABELS[key]} | 0 | 无样本 | — | — | — |")
    lines += ["", f"总决策次数：{report['timing_overall']['count']}；每席新 AI 每局平均累计决策耗时：{report['new_seat_game_mean_decision_ms']:.3f} ms。",
        f"完整评测耗时：{report['elapsed_seconds']:.2f} 秒。机器：{report['environment']['platform']}，{report['environment']['cpu']}；Python {platform.python_version()}，NumPy {np.__version__}。",
        f"RL 使用仓库自带 {DISPLAY_LABEL}；SHA-256：`{report['source_hashes'][str(DEFAULT_CHECKPOINT_PATH.relative_to(Path.cwd()))]}`。",
        f"Git HEAD：`{report['environment']['git_head']}`。源码和规则文件哈希在 report.json 中；结束时复核未变化。", "",
        "## 复现与原始数据", "", "从仓库根目录运行；输出目录必须为空，避免覆盖已有结果：", "", "```bash",
        "PYTHONPATH=src .venv/bin/python artifacts/ai_ratings/humanexp_v1_20260922/run_benchmark.py --output /tmp/powergrid-humanexp-rerun", "```", "",
        "- games.jsonl：逐局种子、座位、区域、最终名次、冠军、双方比较分及游戏耗时。",
        "- decisions.csv：新 AI 全部决策的类别、阶段、轮次、结果类型及墙钟/CPU 纳秒耗时。",
        "- report.json：聚合结果、按对手分类的耗时、以种子对为单位的 bootstrap 95% 区间、运行环境和源码哈希。",
        "- run_benchmark.py：本次实际执行的脚本。", "",
        "100 局提供本赛程上的实测结果；只覆盖德国三人局，不直接外推其他人数或地图。", ""]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-pairs", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=920001)
    args = parser.parse_args()
    if args.seed_pairs <= 0:
        parser.error("--seed-pairs must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("games.jsonl", "decisions.csv", "report.json", "error.json"):
        if (output / name).exists():
            parser.error(f"refusing to overwrite {output / name}")
    root = Path.cwd()
    hashes = source_hashes(root)
    region_sets = legal_region_sets("germany", 3)
    started = time.perf_counter()
    report = {
        "format": "powergrid.humanexp_benchmark.v1", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {"new_controller": NEW, "opponents": OPPONENTS, "map": "germany", "players": 3,
            "seed_start": args.seed_start, "seed_pairs": args.seed_pairs, "games_per_opponent": args.seed_pairs * 2,
            "legal_region_sets": len(region_sets), "region_schedule": region_sets,
            "timing": "perf_counter_ns around choose_intent including normal AI logging; no rendering; serial",
            "rl_checkpoint": str(DEFAULT_CHECKPOINT_PATH), "rl_label": DISPLAY_LABEL},
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status_before": subprocess.check_output(["git", "status", "--short"], text=True),
            "thread_limits": {name: os.environ[name] for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
        "source_hashes": hashes, "completed_games": 0, "failed_games": 0, "opponents": {},
    }
    (output / "configuration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    all_rows = []
    with (output / "games.jsonl").open("w") as game_file, (output / "decisions.csv").open("w", newline="") as decision_file:
        writer = csv.DictWriter(decision_file, fieldnames=FIELDS)
        writer.writeheader()
        for opponent in OPPONENTS:
            games, opponent_rows = [], []
            for offset in range(args.seed_pairs):
                for new_count in (1, 2):
                    seed = args.seed_start + offset
                    minority_seat = offset % 3
                    lineup = tuple((NEW if index == minority_seat else opponent) if new_count == 1
                                   else (opponent if index == minority_seat else NEW) for index in range(3))
                    composition = "HOO" if new_count == 1 else "HHO"
                    rows = []
                    config = GameConfig(map_id="germany", seed=seed, selected_regions=region_sets[offset % len(region_sets)],
                        players=tuple(SeatConfig(player_id=f"p{i+1}", name=f"Player {i+1}", controller=name) for i, name in enumerate(lineup)))
                    agents = {}
                    for index, name in enumerate(lineup):
                        inner = NnRlBasedAiController(checkpoint_path=DEFAULT_CHECKPOINT_PATH) if name == "ai_nn_rl_based_v1" else build_ai_controller(name)
                        agents[f"p{index+1}"] = TimedController(inner, rows) if name == NEW else inner
                    game_started = time.perf_counter()
                    session = GameSession.new_game(config, seat_agents=agents)
                    try:
                        snapshot = session.advance_until_blocked()
                        if snapshot.winner_result is None:
                            raise RuntimeError(str(snapshot.event_log[-1] if snapshot.event_log else "No winner"))
                        if any(event.level == "error" for event in snapshot.event_log):
                            raise RuntimeError("Game contained a session error")
                    except Exception:
                        session.dump_game_log(output / "failed_game_log.json")
                        (output / "error.json").write_text(json.dumps({"opponent": opponent, "seed": seed,
                            "lineup": lineup, "traceback": traceback.format_exc()}, indent=2))
                        raise
                    elapsed = time.perf_counter() - game_started
                    standings = [s.to_dict() for s in derive_final_standings(snapshot.state, snapshot.winner_result)]
                    winner_ids = snapshot.winner_result.winner_ids
                    winners_new = sum(s["controller_name"] == NEW and s["player_id"] in winner_ids for s in standings)
                    wins = draws = losses = 0
                    for a in (s for s in standings if s["controller_name"] == NEW):
                        for b in (s for s in standings if s["controller_name"] != NEW):
                            ka = tuple(a[k] for k in ("powered_cities", "money", "connected_cities"))
                            kb = tuple(b[k] for k in ("powered_cities", "money", "connected_cities"))
                            wins += ka > kb
                            draws += ka == kb
                            losses += ka < kb
                    assert wins + draws + losses == 2
                    metadata = {"opponent": opponent, "game_index": len(games) + 1, "seed": seed, "composition": composition}
                    game = {**metadata, "lineup": lineup, "selected_regions": config.selected_regions,
                        "winner_ids": winner_ids, "new_win_credit": winners_new / len(winner_ids),
                        "standings": standings, "rounds": snapshot.state.round_number,
                        "pairwise_wins": wins, "pairwise_draws": draws, "pairwise_losses": losses,
                        "pairwise_score": (wins + draws / 2) / 2, "elapsed_seconds": elapsed,
                        "humanexp_decisions": len(rows), "humanexp_decision_ns": sum(r["wall_ns"] for r in rows)}
                    games.append(game)
                    for row in rows:
                        row.update(metadata)
                    writer.writerows(rows)
                    game_file.write(json.dumps(game, ensure_ascii=False) + "\n")
                    game_file.flush()
                    decision_file.flush()
                    opponent_rows.extend(rows)
                    report["completed_games"] += 1
                    if len(games) % 10 == 0:
                        print(json.dumps({"opponent": opponent, "completed": len(games), "target": args.seed_pairs * 2,
                            "total_completed": report["completed_games"], "elapsed_seconds": round(time.perf_counter()-started, 1)}, ensure_ascii=False), flush=True)
            report["opponents"][opponent] = summarize_opponent(games, opponent_rows)
            all_rows.extend(opponent_rows)
            print("RESULT " + json.dumps({"opponent": opponent, **summarize_games(games)}, ensure_ascii=False), flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["source_unchanged_during_run"] = source_hashes(root) == hashes
    assert report["source_unchanged_during_run"], "Source changed during benchmark"
    report["timing_by_category"] = {key: timing_stats([r for r in all_rows if r["category"] == key]) for key in LABELS}
    report["timing_by_request_type"] = {key: timing_stats([r for r in all_rows if r["decision_type"] == key]) for key in sorted({r["decision_type"] for r in all_rows})}
    report["timing_overall"] = timing_stats(all_rows)
    report["new_seat_game_mean_decision_ms"] = report["timing_overall"]["total_ms"] / sum(r["new_seat_appearances"] for r in report["opponents"].values())
    assert sum(s["count"] for s in report["timing_by_category"].values()) == len(all_rows)
    assert report["completed_games"] == len(OPPONENTS) * args.seed_pairs * 2
    write_report(output, report)
    print(f"DONE {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
