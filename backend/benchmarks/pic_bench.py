"""PIC 位相別プロファイル計測ベンチマーク (prompts/75)。

scripts/bench_pic.py (prompts/50、サブサイクル/チャンク並列の効果測定) とは別に、
PicSimulation.timing (prompts/75 で追加) の位相別内訳を報告するためのスクリプト。
次フェーズ (Numba 化②・粒子マージ③) の効果を「同条件比較」できるよう、代表ケース
(既定: 2万マクロ粒子・MCCあり (電離)・500ステップ) の設定を固定してリポジトリに残す。

使い方:
    python backend/benchmarks/pic_bench.py                       # 既定 (2万マクロ, 500ステップ, MCCあり)
    python backend/benchmarks/pic_bench.py --n-macro 50000 --steps 200
    python backend/benchmarks/pic_bench.py --no-mcc              # MCC 無効で比較
    python backend/benchmarks/pic_bench.py --sub 10 --threads 8  # サブサイクル/チャンク並列との組合せ確認
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from es_sim.pic import PicSimulation  # noqa: E402
from es_sim.schema import Project  # noqa: E402

# 表示順・日本語ラベル (PicPanel の表示と揃える)
PHASE_LABELS = [
    ("solve", "場ソルブ"),
    ("gather_push", "粒子押し出し"),
    ("walk", "walk探索"),
    ("deposit", "電荷デポジット"),
    ("mcc", "MCC衝突"),
    ("other", "その他"),
]


def build_project(n_macro: int, steps: int, sub: int, threads: int, use_mcc: bool) -> Project:
    """CCP 類似ケース (RF 平行平板 + 初期プラズマ、任意で MCC 電離あり)。"""
    pic: dict = {
        "initial_plasma": {
            "density": 1e15, "te_ev": 3.0, "ti_ev": 0.03,
            "ion_mass_amu": 40.0, "seed": 7,
        },
        "n_macro": n_macro,
        "dt": None,  # 既定 (ωpe·dt = 0.1)
        "n_steps": steps,
        "frame_every": 10**9,  # フレーム構築コストを排し、ステップ本体だけを計測する
        "ion_subcycle": sub,
        "threads": threads,
    }
    if use_mcc:
        # 断面積は解析用の合成値 (実ガス物性ではない。衝突頻度をベンチ向けに確保する目的)
        pic["mcc"] = {
            "gas": {"name": "Ar", "pressure_pa": 50.0, "temperature_k": 300.0},
            "electron_processes": [
                {
                    "kind": "ionization",
                    "label": "synthetic ionization",
                    "threshold_ev": 15.76,
                    "energy_ev": [15.76, 15.76 + 1e-6, 1000.0],
                    "sigma_m2": [0.0, 1.0e-20, 1.0e-20],
                },
                {
                    "kind": "elastic",
                    "label": "synthetic elastic",
                    "threshold_ev": 0.0,
                    "energy_ev": [0.0, 1000.0],
                    "sigma_m2": [1.0e-19, 1.0e-19],
                },
            ],
            "ion_processes": [],
            "seed": 11,
        }
    return Project.model_validate(
        {
            "geometry": {
                "domain": {"polygon": [[0, 0], [0.02, 0], [0.02, 0.01], [0, 0.01]]},
                "boundaries": [
                    {
                        "edges": [3], "type": "dirichlet", "voltage": 0.0,
                        "voltage_rf": {"amplitude": 200.0, "freq_hz": 13.56e6},
                    },
                    {"edges": [1], "type": "dirichlet", "voltage": 0.0},
                ],
            },
            "mesh": {"size": 8e-4},
            "pic": pic,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-macro", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--sub", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no-mcc", action="store_true", help="MCC を無効にして比較する")
    ap.add_argument("--warmup", type=int, default=5, help="計測から除外するウォームアップステップ数")
    args = ap.parse_args()
    use_mcc = not args.no_mcc

    sim = PicSimulation(
        build_project(args.n_macro, args.steps, args.sub, args.threads, use_mcc)
    )
    # ウォームアップ (キャッシュ・スレッドプール初期化) 分は timing/wall のどちらからも除外する
    for _ in range(args.warmup):
        sim.step()
    sim.timing = {k: 0.0 for k in sim.timing}
    n0 = len(sim.species["electron"].x) + len(sim.species["ion"].x)

    t0 = time.perf_counter()
    for _ in range(args.steps):
        sim.step()
    wall = time.perf_counter() - t0

    total = sum(sim.timing.values())
    print(
        f"n_macro={args.n_macro} particles≈{n0} steps={args.steps} "
        f"sub={args.sub} threads={args.threads} mcc={'on' if use_mcc else 'off'}"
    )
    print(f"wall = {wall:.3f} s ({wall / args.steps * 1e3:.3f} ms/step)")
    print(f"{'phase':14s} {'label':12s} {'sec':>10s} {'%':>7s} {'ms/step':>10s}")
    for key, label in sorted(PHASE_LABELS, key=lambda kv: -sim.timing[kv[0]]):
        sec = sim.timing[key]
        pct = 100.0 * sec / total if total > 0 else 0.0
        ms_step = sec / args.steps * 1e3
        print(f"{key:14s} {label:12s} {sec:10.3f} {pct:6.1f}% {ms_step:10.4f}")
    print(f"{'total':14s} {'合計':12s} {total:10.3f} {100.0:6.1f}% {total / args.steps * 1e3:10.4f}")


if __name__ == "__main__":
    main()
