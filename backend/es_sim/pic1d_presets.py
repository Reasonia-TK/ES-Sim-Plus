"""1D PIC/MCC のベンチマークプリセット (prompts/91)。GET /pic1d/presets で返す。

ユーザーがワンクリックで CCP (容量結合プラズマ) の標準ベンチマーク条件を
再現できるようにする。

- "edupic_ar": eduPIC (Donkó et al., PSST 30, 095017 (2021), GPL) の Ar 解析式
  断面積 (出典: Phelps & Petrovic, PSST 8, R21 (1999) [電子]; Phelps,
  J. Appl. Phys. 76, 747 (1994) [イオン]) をエネルギーグリッド上でサンプルして
  XsProcess テーブル化し、断面積データの外部インポートなしで自己完結的に動く。
- "turner_he_case1": Turner et al., Phys. Plasmas 20, 013507 (2013) の He
  ベンチマーク Case 1 相当の放電条件のみを設定する。He 断面積は著作権のある
  データセット (Biagi/Phelps) のため本リポジトリには同梱せず、
  electron_processes/ion_processes は空のまま返す (LXCat からのインポートが必要)。
"""

from __future__ import annotations

import numpy as np

from .mcc import KB
from .particles import MP, ME
from .schema import VoltageWaveform, XsProcess

# eduPIC 解析式断面積のエネルギーグリッド点数 (geomspace)。数十〜百点程度の
# LXCat 実テーブルに対して MCC 側の np.interp コストは無視できる規模なので、
# 閾値近傍の立ち上がりを十分な解像度で捉えられるよう余裕を持った点数にする
_XS_GRID_N = 400
_XS_GRID_MAX_EV = 1.0e3
_EXC_THRESHOLD_EV = 11.5
_ION_THRESHOLD_EV = 15.8

# RF 電圧波形 (VoltageWaveform) をサンプルする1周期あたりの点数。線形補間
# (np.interp) でサイン波を近似するため、粗すぎると波形が三角波に近づいてしまう。
# 360点 (1度刻み) なら角度2次の補間誤差は無視できる小ささになる
_SINE_SAMPLES = 360


def _sine_waveform(amplitude_v: float, freq_hz: float) -> VoltageWaveform:
    """振幅 amplitude_v・周波数 freq_hz の正弦波を VoltageWaveform としてサンプルする。

    Pic1dElectrode は voltage_rf (sin 成分) を持たないため、RF 駆動は CSV インポート
    波形と同じ機構 (pic.py の _eval_waveform、線形補間) で表現する。
    """
    n = _SINE_SAMPLES
    phase = (np.arange(n) / n).tolist()
    v = (amplitude_v * np.sin(2.0 * np.pi * np.arange(n) / n)).tolist()
    return VoltageWaveform(freq_hz=freq_hz, phase=phase, v=v)


def edupic_ar_processes() -> tuple[list[XsProcess], list[XsProcess]]:
    """eduPIC の Ar 解析式断面積 (電子: elastic/excitation/ionization、
    イオン: isotropic/backscat) を XsProcess テーブルとして返す。

    エネルギーグリッドは geomspace(1e-3, 1e3, 400) を基本とし、電子側は
    0 eV (先頭) と励起/電離の閾値ちょうどの点を挿入する (閾値直上の立ち上がりが
    線形補間で鈍らないようにするため)。イオン側は E^-0.5 特異点を避けるため
    0 eV を含めない。
    """
    base = np.geomspace(1.0e-3, _XS_GRID_MAX_EV, _XS_GRID_N)
    e_grid = np.unique(np.concatenate([[0.0], base, [_EXC_THRESHOLD_EV, _ION_THRESHOLD_EV]]))
    en = e_grid

    # ---- 電子: 弾性 (Phelps & Petrovic 1999) ------------------------------
    term1 = 6.0 / (1.0 + (en / 0.1) + (en / 0.6) ** 2) ** 3.3
    term2 = (
        1.1 * en**1.4 / (1.0 + (en / 15.0) ** 1.2)
        / np.sqrt(1.0 + (en / 5.5) ** 2.5 + (en / 60.0) ** 4.1)
    )
    sigma_elastic = 1.0e-20 * (
        np.abs(term1 - term2) + 0.05 / (1.0 + en / 10.0) ** 2 + 0.01 * en**3 / (1.0 + (en / 12.0) ** 6)
    )

    # ---- 電子: 励起 (閾値 11.5 eV) ------------------------------------------
    exc_mask = en > _EXC_THRESHOLD_EV
    de_exc = np.clip(en - _EXC_THRESHOLD_EV, 0.0, None)
    sigma_exc_full = 1.0e-20 * (
        0.034 * de_exc**1.1 * (1.0 + (en / 15.0) ** 2.8) / (1.0 + (en / 23.0) ** 5.5)
        + 0.023 * de_exc / (1.0 + en / 80.0) ** 1.9
    )
    sigma_excitation = np.where(exc_mask, sigma_exc_full, 0.0)

    # ---- 電子: 電離 (閾値 15.8 eV) ------------------------------------------
    ion_mask = en > _ION_THRESHOLD_EV
    de_ion = np.clip(en - _ION_THRESHOLD_EV, 0.0, None)
    sigma_ion_full = 1.0e-20 * (
        970.0 * de_ion / (70.0 + en) ** 2 + 0.06 * de_ion**2 * np.exp(-en / 9.0)
    )
    sigma_ionization = np.where(ion_mask, sigma_ion_full, 0.0)

    ar_mass_amu = 39.948
    mass_ratio = ME / (ar_mass_amu * MP)  # 2D/1D 共通の amu→kg 換算規約 (ion_mass_amu*MP)

    electron_processes = [
        XsProcess(
            kind="elastic", label="eduPIC Ar elastic (Phelps & Petrovic 1999, analytic)",
            threshold_ev=0.0, mass_ratio=mass_ratio,
            energy_ev=e_grid.tolist(), sigma_m2=sigma_elastic.tolist(),
        ),
        XsProcess(
            kind="excitation", label="eduPIC Ar excitation (Phelps & Petrovic 1999, analytic)",
            threshold_ev=_EXC_THRESHOLD_EV, mass_ratio=0.0,
            energy_ev=e_grid.tolist(), sigma_m2=sigma_excitation.tolist(),
        ),
        XsProcess(
            kind="ionization", label="eduPIC Ar ionization (Phelps & Petrovic 1999, analytic)",
            threshold_ev=_ION_THRESHOLD_EV, mass_ratio=0.0,
            energy_ev=e_grid.tolist(), sigma_m2=sigma_ionization.tolist(),
        ),
    ]

    # ---- イオン: 等方・後方散乱 (Phelps 1994、実験室系エネルギー) -----------------
    i_grid = np.geomspace(1.0e-3, _XS_GRID_MAX_EV, _XS_GRID_N)
    e_lab = i_grid
    q_iso = 2.0e-19 * e_lab**-0.5 / (1.0 + e_lab) + 3.0e-19 * e_lab / (1.0 + e_lab / 3.0) ** 2
    q_mom = 1.15e-18 * e_lab**-0.1 * (1.0 + 0.015 / e_lab) ** 0.6
    q_back = np.maximum((q_mom - q_iso) / 2.0, 0.0)
    q_iso = np.maximum(q_iso, 0.0)

    ion_processes = [
        XsProcess(
            kind="isotropic", label="eduPIC Ar+/Ar isotropic (Phelps 1994, analytic)",
            threshold_ev=0.0, mass_ratio=0.0,
            energy_ev=i_grid.tolist(), sigma_m2=q_iso.tolist(),
        ),
        XsProcess(
            kind="backscat", label="eduPIC Ar+/Ar backscatter (Phelps 1994, analytic)",
            threshold_ev=0.0, mass_ratio=0.0,
            energy_ev=i_grid.tolist(), sigma_m2=q_back.tolist(),
        ),
    ]
    return electron_processes, ion_processes


def get_presets() -> dict:
    """GET /pic1d/presets が返す辞書。プリセット名 → {label, description, pic1d, ...}。"""
    e_procs, i_procs = edupic_ar_processes()
    edupic_left_wf = _sine_waveform(250.0, 13.56e6)

    edupic_ar: dict = {
        "label": "eduPIC Ar base case (解析式断面積)",
        "description": (
            "eduPIC (Donkó et al., PSST 30, 095017 (2021)) の Ar 解析式断面積 "
            "(Phelps & Petrovic, PSST 8, R21 (1999) [電子]; Phelps, J. Appl. Phys. 76, "
            "747 (1994) [イオン]) を使った自己完結型のベンチマーク条件。片側 RF "
            "(左電極 13.56 MHz, 250 V)・対向接地、Ar 10 Pa・350 K。セル数・マクロ粒子数・"
            "ステップ数はアプリでの実用速度に合わせて調整した値 (eduPIC 論文の base case 相当)。"
        ),
        "pic1d": {
            "gap_m": 0.025,
            "n_cells": 256,
            "left": {"v_dc": 0.0, "waveforms": [edupic_left_wf.model_dump()], "see_gamma": 0.0},
            "right": {"v_dc": 0.0, "waveforms": [], "see_gamma": 0.0},
            "init_density_m3": 1.0e15,
            "init_te_ev": 2.0,
            "init_ti_ev": 0.03,
            "ion_mass_amu": 39.948,
            "n_macro": 30000,
            "dt": 1.0 / (400.0 * 13.56e6),
            "n_steps": 20000,
            "frame_every": 100,
            "avg_steps": None,
            "phase_bins": 40,
            "mcc": {
                "gas": {"name": "Ar", "pressure_pa": 10.0, "temperature_k": 350.0},
                "electron_processes": [p.model_dump() for p in e_procs],
                "ion_processes": [p.model_dump() for p in i_procs],
                "seed": 0,
                "ionization_split": "half",
                "ion_energy_frame": "lab",
                "use_dsmc_gas": False,
            },
            "see_energy_ev": 2.0,
            "eedf_regions": [],
            "seed": 0,
        },
    }

    n_he = 9.64e20
    pressure_he_pa = n_he * KB * 300.0
    ion_mass_amu_he = 6.67e-27 / 1.66054e-27
    turner_left_wf = _sine_waveform(450.0, 13.56e6)

    turner_he_case1: dict = {
        "label": "Turner et al. (2013) He Case 1 (断面積は要インポート)",
        "description": (
            "Turner et al., Phys. Plasmas 20, 013507 (2013) の He CCP ベンチマーク "
            "Case 1 相当の放電条件。gap 6.7 cm、n=9.64e20 m^-3・300 K "
            "(pressure_pa = n·kB·T)、13.56 MHz 450 V (左電極駆動・右接地)。放電が定常に"
            "達すれば初期密度・初期温度への依存は消えるため、初期値は妥当な目安値としている。"
            "electron_processes/ion_processes は空のまま返すので、LXCat から He "
            "(Biagi/Phelps) 断面積をインポートして設定すること (断面積データは同梱しない)。"
        ),
        "pic1d": {
            "gap_m": 0.067,
            "n_cells": 128,
            "left": {"v_dc": 0.0, "waveforms": [turner_left_wf.model_dump()], "see_gamma": 0.0},
            "right": {"v_dc": 0.0, "waveforms": [], "see_gamma": 0.0},
            "init_density_m3": 2.56e14,
            "init_te_ev": 3.0,
            "init_ti_ev": 0.03,
            "ion_mass_amu": ion_mass_amu_he,
            "n_macro": 20000,
            "dt": 1.0 / (400.0 * 13.56e6),
            "n_steps": 20000,
            "frame_every": 100,
            "avg_steps": None,
            "phase_bins": 40,
            "mcc": {
                "gas": {"name": "He", "pressure_pa": pressure_he_pa, "temperature_k": 300.0},
                "electron_processes": [],
                "ion_processes": [],
                "seed": 0,
                "ionization_split": "half",
                "ion_energy_frame": "com",
                "use_dsmc_gas": False,
            },
            "see_energy_ev": 2.0,
            "eedf_regions": [],
            "seed": 0,
        },
        "note": "He 断面積 (Biagi/Phelps) を LXCat からインポートして設定してください",
    }

    return {"edupic_ar": edupic_ar, "turner_he_case1": turner_he_case1}
