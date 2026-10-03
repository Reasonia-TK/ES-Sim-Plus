"""軸対称 (rz) モードのテスト (prompts/39)。x = z (軸方向)、y = r (径方向)。

1. 円筒コンデンサ: V(r) = V1·ln(b/r)/ln(b/a) と一致 (<1e-2)。xy では線形 (対比)
2. エネルギー/容量: W = ½CV1²、C = 2πεL/ln(b/a) と数%以内
3. 軸を含む解: 有限で軸上 ∂V/∂r ≈ 0
4. 粒子: Ez 加速の解析解一致 (vθ=0)、L 保存 + エネルギー保存 (vθ≠0)、軸交差の鏡映
4b. 軸のすぐ近く (1 µm) の熱電子 (回転法、rz と rz_x0): 無電場では 3D 直線運動を丸め誤差でなぞり、
   一様 Ez で電極に当たる粒子の衝突時のエネルギー・半径・角度が解析解と一致
5. バリデーション: y<0 domain・軸への Dirichlet・pic 実行のエラー
"""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from es_sim.fem import EPS0, solve
from es_sim.meshing import generate_mesh
from es_sim.particles import ME, QE, _init_particles, trace
from es_sim.pic import PicSimulation
from es_sim.schema import Emitter, Project

A, B = 0.02, 0.05   # 内径 a・外径 b [m]
LZ = 0.03           # 軸方向長さ [m]
V1 = 100.0


def _cyl_project(coord: str, mesh_size: float = 0.002) -> Project:
    """円筒コンデンサ: domain 矩形 [0,L]×[a,b]、下辺 r=a に V1・上辺 r=b に 0V。"""
    return Project.model_validate(
        {
            "coord": coord,
            "geometry": {
                "domain": {"polygon": [[0, A], [LZ, A], [LZ, B], [0, B]]},
                "boundaries": [
                    {"edges": [0], "voltage": V1},   # 下辺 r=a
                    {"edges": [2], "voltage": 0.0},  # 上辺 r=b (左右は Neumann)
                ],
            },
            "mesh": {"size": mesh_size},
        }
    )


# ---- 1. 円筒コンデンサ (対数分布) と xy 対比 ---------------------------------------


def test_cylindrical_capacitor_log_profile():
    project = _cyl_project("rz")
    mesh = generate_mesh(project)
    sol = solve(project, mesh)

    r = mesh.nodes[:, 1]
    v_exact = V1 * np.log(B / r) / np.log(B / A)
    err = np.max(np.abs(sol.v - v_exact)) / V1
    assert err < 1e-2

    # メッシュ細分で誤差が改善する (収束性)
    fine = _cyl_project("rz", mesh_size=0.001)
    mesh_f = generate_mesh(fine)
    sol_f = solve(fine, mesh_f)
    err_f = np.max(np.abs(sol_f.v - V1 * np.log(B / mesh_f.nodes[:, 1]) / np.log(B / A))) / V1
    assert err_f < err

    # 対比: 平面 (xy) モードでは線形になる (= r 重みが効いている証拠)
    project_xy = _cyl_project("xy")
    mesh_xy = generate_mesh(project_xy)
    sol_xy = solve(project_xy, mesh_xy)
    v_lin = V1 * (B - mesh_xy.nodes[:, 1]) / (B - A)
    assert np.max(np.abs(sol_xy.v - v_lin)) < 1e-8 * V1


# ---- 2. エネルギー / 容量 -----------------------------------------------------------


def test_cylindrical_capacitor_energy():
    """W = ½·C·V1²、C = 2πεL/ln(b/a) [F] と数%以内 (rz のエネルギーは [J])。"""
    project = _cyl_project("rz")
    mesh = generate_mesh(project)
    sol = solve(project, mesh)

    c_exact = 2.0 * math.pi * EPS0 * LZ / math.log(B / A)
    w_exact = 0.5 * c_exact * V1**2
    assert sol.energy == pytest.approx(w_exact, rel=0.03)


# ---- 3. 軸 (r=0) を含む解 ----------------------------------------------------------


def _disk_project(particles: dict | None = None, pic: dict | None = None) -> Project:
    """軸対称の平行平板 (円板コンデンサ中心部): domain [0,L]×[0,R]、左右 Dirichlet。"""
    data = {
        "coord": "rz",
        "geometry": {
            "domain": {"polygon": [[0, 0], [LZ, 0], [LZ, 0.02], [0, 0.02]]},
            "boundaries": [
                {"edges": [3], "voltage": 0.0},   # 左 (z=0)
                {"edges": [1], "voltage": V1},    # 右 (z=L)。下辺 (軸) は自然境界
            ],
        },
        "mesh": {"size": 0.0015},
    }
    if particles is not None:
        data["particles"] = particles
    if pic is not None:
        data["pic"] = pic
    return Project.model_validate(data)


def test_axis_included_solution_finite_and_flat():
    project = _disk_project()
    mesh = generate_mesh(project)
    sol = solve(project, mesh)

    assert np.all(np.isfinite(sol.v))
    # 解は V ≈ V1·z/L (r に依存しない)
    v_exact = V1 * mesh.nodes[:, 0] / LZ
    assert np.max(np.abs(sol.v - v_exact)) < 1e-2 * V1
    # 軸上 (r=0) 付近の要素で ∂V/∂r ≈ 0 (|Er| << V1/L)
    r_cent = mesh.nodes[mesh.triangles][:, :, 1].mean(axis=1)
    near_axis = r_cent < 0.002
    assert np.any(near_axis)
    assert np.max(np.abs(sol.e_field[near_axis, 1])) < 0.02 * V1 / LZ


# ---- 4. 粒子軌道 (rz) --------------------------------------------------------------


def test_rz_acceleration_matches_analytic():
    """一様 Ez 場での加速 (vθ=0)。エネルギー・飛行時間が解析解と一致する。"""
    z0 = 1e-4
    project = _disk_project(
        particles={
            "species": {"preset": "electron"},
            "emitter": {"kind": "point", "p1": [z0, 0.01], "n": 1, "energy_ev": 0.0},
            "dt": None,
            "n_steps": 3000,
            "save_every": 10,
        }
    )
    mesh = generate_mesh(project)
    sol = solve(project, mesh)
    result = trace(project, mesh, sol)

    assert result.absorbed[0]
    d = LZ - z0
    e_field = V1 / LZ
    energy_exact = e_field * d
    t_exact = math.sqrt(2.0 * d * ME / (QE * e_field))
    assert result.final_energy_ev[0] == pytest.approx(energy_exact, rel=0.02)
    assert result.tof[0] == pytest.approx(t_exact, rel=0.02)
    # vθ=0 なので面内運動: r は不変
    traj = result.trajectories[0]
    assert np.max(np.abs(traj[:, 1] - 0.01)) < 1e-6


def test_rz_angular_momentum_and_energy_conservation():
    """無電場・vθ≠0 (maxwell の第3成分) の自由粒子: r(t) が角運動量保存の解析解
    r(t) = √((r0+vr·t)² + (vθ·t)²) と一致し、全エネルギーが保存する。"""
    n, seed, kt = 8, 42, 0.01
    z0, r0 = 0.015, 0.035
    dt, n_steps = 1e-10, 100
    emitter = {
        "kind": "point", "p1": [z0, r0], "n": n,
        "energy_ev": 0.0, "energy_dist": "maxwell",
        "temperature_ev": kt, "seed": seed,
    }
    project = Project.model_validate(
        {
            "coord": "rz",
            "geometry": {
                "domain": {"polygon": [[0, A], [LZ, A], [LZ, B], [0, B]]},
                "boundaries": [{"edges": [0, 1, 2, 3], "voltage": 0.0}],  # 無電場
            },
            "mesh": {"size": 0.002},
            "particles": {
                "species": {"preset": "electron"},
                "emitter": emitter,
                "dt": dt,
                "n_steps": n_steps,
                "save_every": n_steps,
            },
        }
    )
    mesh = generate_mesh(project)
    sol = solve(project, mesh)
    result = trace(project, mesh, sol)
    assert not np.any(result.absorbed)

    # 初期速度 (vz, vr, vθ) を同じ seed で再現して解析解と比較
    _, v0 = _init_particles(Emitter.model_validate(emitter), ME, vtheta=True)
    assert v0.shape == (n, 3) and np.any(v0[:, 2] != 0.0)

    t = n_steps * dt
    r_exact = np.sqrt((r0 + v0[:, 1] * t) ** 2 + (v0[:, 2] * t) ** 2)
    z_exact = z0 + v0[:, 0] * t
    final = result.trajectories[:, -1, :]
    # L = r·vθ の保存を含む径方向運動 (回転法の 3D 直線移動) の検証
    assert np.allclose(final[:, 1], r_exact, rtol=1e-6)
    assert np.allclose(final[:, 0], z_exact, rtol=1e-6)

    # 全エネルギー保存 (無電場なので運動エネルギー一定)
    e0_ev = 0.5 * ME * np.sum(v0**2, axis=1) / QE
    assert np.allclose(result.final_energy_ev, e0_ev, rtol=1e-3)


def test_rz_axis_crossing_mirrors():
    """L=0 の粒子が軸へ向かい、r → −r, vr → −vr の鏡映で通過する。"""
    z0, r0 = 0.015, 0.01
    energy_ev = 10.0
    dt, n_steps = 2e-11, 600
    project = _disk_project(
        particles={
            "species": {"preset": "electron"},
            "emitter": {
                "kind": "point", "p1": [z0, r0], "n": 1,
                "energy_ev": energy_ev, "direction_deg": -90.0,  # −r 方向 (軸へ)
            },
            "dt": dt,
            "n_steps": n_steps,
            "save_every": 5,
        }
    )
    # 無電場にする (左右とも 0V)
    project = project.model_copy(deep=True)
    project.geometry.boundaries[1].voltage = 0.0
    mesh = generate_mesh(project)
    sol = solve(project, mesh)
    result = trace(project, mesh, sol)

    assert not result.absorbed[0]  # 軸 (下辺) では吸収されず鏡映される
    traj = result.trajectories[0]
    v = math.sqrt(2.0 * energy_ev * QE / ME)
    t = n_steps * dt
    assert v * t > r0  # 軸を必ず横切る設定
    # r は常に非負で、鏡映後の解析解 |r0 − v·t| に一致する
    assert np.min(traj[:, 1]) >= 0.0
    assert traj[-1, 1] == pytest.approx(abs(r0 - v * t), rel=1e-3)
    # 反射後は +r 方向へ直進 (z は不変)
    assert result.final_angle_deg[0] == pytest.approx(90.0, abs=1.0)
    assert traj[-1, 0] == pytest.approx(z0, abs=1e-6)


# ---- 4b. 軸のすぐ近くの熱電子 (回転法) ------------------------------------------------


def _near_axis_project(
    coord: str, lz: float, rr: float, z0: float, r0: float, emitter: dict,
    boundaries: dict, dt: float, n_steps: int, save_every: int,
) -> tuple[Project, int, int]:
    """軸を含む円柱 [0, lz] (軸方向) × [0, rr] (径方向) の rz / rz_x0 版と (径, 軸) の成分番号。

    boundaries は辺の名前 ("z0": z=0、"zl": z=lz、"rr": r=rr) → 境界条件 (edges 以外) の対応。
    軸の辺は自然境界のまま。エミッタは (z0, r0) の点。
    """
    if coord == "rz":
        poly, ri = [[0, 0], [lz, 0], [lz, rr], [0, rr]], 1
        edge = {"zl": 1, "rr": 2, "z0": 3}
    else:
        poly, ri = [[0, 0], [rr, 0], [rr, lz], [0, lz]], 0
        edge = {"z0": 0, "rr": 1, "zl": 2}
    zi = 1 - ri
    p1 = [0.0, 0.0]
    p1[ri], p1[zi] = r0, z0
    project = Project.model_validate(
        {
            "coord": coord,
            "geometry": {
                "domain": {"polygon": poly},
                "boundaries": [{"edges": [edge[k]], **bc} for k, bc in boundaries.items()],
            },
            "mesh": {"size": 0.001},
            "particles": {
                "species": {"preset": "electron"},
                "emitter": {**emitter, "kind": "point", "p1": p1},
                "dt": dt,
                "n_steps": n_steps,
                "save_every": save_every,
            },
        }
    )
    return project, ri, zi


@pytest.mark.parametrize("coord", ["rz", "rz_x0"])
def test_rz_near_axis_thermal_electrons_free_flight(coord):
    """軸から 1 µm の 2 eV 熱電子 (maxwell、vθ ≠ 0、無電場): 吸収されず、運動エネルギーが一定で、
    各フレームの位置が 3D の直線運動 r(t) = √((r0 + vr t)² + (vθ t)²)、z(t) = z0 + vz t と一致する。

    旧方式 (遠心力 vθ²/r = L²/r³ を今の位置で評価) では、軸のそばを通る粒子が 100〜2000 倍の
    エネルギーを得て 8 個中 6 個が壁で吸収されていた。
    """
    z0, r0 = 0.01, 1e-6
    dt, n_steps = 1e-10, 50
    emitter = {
        "n": 8, "energy_ev": 0.0, "energy_dist": "maxwell",
        "temperature_ev": 2.0, "seed": 4,
    }
    grounded = {"voltage": 0.0}
    project, ri, zi = _near_axis_project(
        coord, 0.02, 0.01, z0, r0, emitter,
        {"z0": grounded, "zl": grounded, "rr": grounded}, dt, n_steps, 1,
    )
    mesh = generate_mesh(project)
    sol = solve(project, mesh)
    assert np.all(sol.e_field == 0.0)  # 無電場
    result = trace(project, mesh, sol)

    _, v0 = _init_particles(project.particles.emitter, ME, vtheta=True)
    assert np.all(v0[:, 2] != 0.0)
    # 最初のステップで軸の反対側へ向かう (r + vr·dt < 0) 粒子を含む設定
    assert np.any(r0 + v0[:, ri] * dt < 0.0)

    assert not np.any(result.absorbed)
    e0_ev = 0.5 * ME * np.sum(v0**2, axis=1) / QE
    assert np.allclose(result.final_energy_ev, e0_ev, rtol=1e-12, atol=0.0)

    t = dt * np.arange(n_steps + 1)[None, :]
    xr = r0 + v0[:, ri, None] * t  # 3D 直線運動の (径, 周) 成分
    yt = v0[:, 2, None] * t
    r_exact = np.hypot(xr, yt)
    traj = result.trajectories
    assert traj.shape == (8, n_steps + 1, 2)
    assert np.allclose(traj[:, :, ri], r_exact, rtol=1e-9, atol=0.0)
    assert np.allclose(traj[:, :, zi], z0 + v0[:, zi, None] * t, rtol=1e-12, atol=0.0)

    # 最終速度の向き: 3D の速度を最終位置の局所座標で見た (vr, vz) の角度
    vel = np.zeros((8, 2))
    vel[:, ri] = (xr[:, -1] * v0[:, ri] + yt[:, -1] * v0[:, 2]) / r_exact[:, -1]
    vel[:, zi] = v0[:, zi]
    angle_exact = np.degrees(np.arctan2(vel[:, 1], vel[:, 0]))
    assert np.allclose(result.final_angle_deg, angle_exact, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("coord", ["rz", "rz_x0"])
def test_rz_near_axis_thermal_electrons_hit_electrodes(coord):
    """一様 Ez (陰極 z=0 が 0 V、陽極 z=L が V1) の中で、軸から 1 µm・陰極から 0.1 mm の 2 eV 熱電子が
    陰極か陽極に当たる。横方向は自由運動なので、衝突時の運動エネルギーは KE0 + V1·(z_hit − z0)/L、
    半径は r(tof) = √((r0 + vr·tof)² + (vθ·tof)²)、速度の向きは衝突時刻の 3D の速度を衝突位置の
    局所座標で見たものと一致する (吸収時の速度の回転の検証)。
    """
    lz, z0, r0 = 0.03, 1e-4, 1e-6
    dt, n_steps = 1e-11, 1500
    emitter = {
        "n": 16, "energy_ev": 0.0, "energy_dist": "maxwell",
        "temperature_ev": 2.0, "seed": 7,
    }
    project, ri, zi = _near_axis_project(
        coord, lz, 0.02, z0, r0, emitter,
        {"z0": {"voltage": 0.0}, "zl": {"voltage": V1}}, dt, n_steps, n_steps,
    )
    mesh = generate_mesh(project)
    sol = solve(project, mesh)
    result = trace(project, mesh, sol)

    _, v0 = _init_particles(project.particles.emitter, ME, vtheta=True)
    assert np.all(v0[:, 2] != 0.0)
    assert np.all(result.absorbed)
    hit = result.trajectories[:, -1, :]
    at_anode = np.isclose(hit[:, zi], lz, rtol=0.0, atol=1e-12)
    at_cathode = np.isclose(hit[:, zi], 0.0, rtol=0.0, atol=1e-12)
    assert np.all(at_anode | at_cathode)
    assert np.any(at_anode) and np.any(at_cathode)  # 陰極へ戻る粒子も含む設定

    e0_ev = 0.5 * ME * np.sum(v0**2, axis=1) / QE
    e_exact = e0_ev + V1 * (hit[:, zi] - z0) / lz
    # 誤差は衝突時刻の補間の O(dt²) (陰極で 4×10⁻⁵ 程度)。旧方式は 16 個中 11〜13 個が 10⁻³ を
    # 超え、最大 30〜40 倍ずれていた
    assert np.allclose(result.final_energy_ev, e_exact, rtol=1e-3, atol=0.0)

    tof = result.tof
    xr = r0 + v0[:, ri] * tof
    yt = v0[:, 2] * tof
    r_exact = np.hypot(xr, yt)
    assert np.allclose(hit[:, ri], r_exact, rtol=1e-5, atol=0.0)

    vel = np.zeros((16, 2))
    vel[:, ri] = (xr * v0[:, ri] + yt * v0[:, 2]) / r_exact
    vel[:, zi] = v0[:, zi] + (QE * V1 / (ME * lz)) * tof  # 陽極へ向かう一様な加速
    angle_exact = np.degrees(np.arctan2(vel[:, 1], vel[:, 0]))
    assert np.allclose(result.final_angle_deg, angle_exact, rtol=0.0, atol=1e-6)


# ---- 5. バリデーション -------------------------------------------------------------


def test_rz_validation_errors():
    # y < 0 の頂点を含む domain
    with pytest.raises(ValidationError, match="y"):
        Project.model_validate(
            {
                "coord": "rz",
                "geometry": {
                    "domain": {"polygon": [[0, -0.01], [LZ, -0.01], [LZ, B], [0, B]]},
                    "boundaries": [],
                },
                "mesh": {"size": 0.002},
            }
        )

    # 対称軸 (y=0) の辺への Dirichlet 指定
    with pytest.raises(ValidationError, match="対称軸"):
        Project.model_validate(
            {
                "coord": "rz",
                "geometry": {
                    "domain": {"polygon": [[0, 0], [LZ, 0], [LZ, B], [0, B]]},
                    "boundaries": [{"edges": [0], "voltage": 0.0}],
                },
                "mesh": {"size": 0.002},
            }
        )

    # pic + rz は対応済み (prompts/47): エラーにならず構築でき、rz モードになる
    project = _disk_project(
        pic={"dt": 1e-9, "n_steps": 10, "n_macro": 10, "frame_every": 10}
    )
    sim = PicSimulation(project)
    assert sim.rz and sim.ridx == 1
