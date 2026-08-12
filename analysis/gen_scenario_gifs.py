#!/usr/bin/env python3
"""
Scenario animation generator v3.

Key fixes in v3:
  - Car size to-scale: car_w = 4.5 m mapped to viewport, so 10 m gap looks like 10 m
  - Ghost cut-out: stationary obstacle drawn from t=0 (it was always there on the road);
    leader is BETWEEN ego and stationary, slides to adj lane at cut-out event
  - Viewport for cut-out excludes departed leader (prevents viewport ballooning)

Usage:
    python analysis/gen_scenario_gifs.py --mode single
    python analysis/gen_scenario_gifs.py --mode compare
    python analysis/gen_scenario_gifs.py --mode compare --scenario A --batch 3
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

from core.experiments import run_ablation_group
from core.controllers import get_all_controllers
from core.env import (
    HighwayCutInObstacleSim, UrbanStopObstacleSim, GhostCutOutObstacleSim,
    AdversarialStopGoObstacleSim,
)

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "gifs"
)
os.makedirs(OUT_DIR, exist_ok=True)

# Physical constants
REAL_CAR_LEN = 4.5   # metres
REAL_CAR_WID = 1.8   # metres


# ═══════════════════════════════════════════════════════════════════
# 1. Scenarios (from main.py)
# ═══════════════════════════════════════════════════════════════════
SCENARIOS = {
    "A": dict(
        name="highway_cutin",
        title="Scenario A: Highway Cut-in & Flee",
        subtitle="Obs 100 km/h cuts in → brakes → flees   |  Ego 120 km/h",
        cls=HighwayCutInObstacleSim,
        kwargs=dict(x0=30.0, v0=27.7, t_cutin=1.5),
        ego_v0=33.3, t_max=12.0,
        vis_type="cutin", t_event=1.5,
    ),
    "B": dict(
        name="urban_stop",
        title="Scenario B: Urban Stop & Go",
        subtitle="Obs brakes to stop, waits, restarts   |  Ego 50 km/h",
        cls=UrbanStopObstacleSim,
        kwargs=dict(x0=15.0, v0=14.0, t_cutin=1.0),
        ego_v0=14.0, t_max=30.0,
        vis_type="cutin", t_event=1.0,
    ),
    "C": dict(
        name="ghost_cutout_far",
        title="Scenario C: Ghost Cut-out (180 m)",
        subtitle="Leader swerves out → stationary at 180 m   |  Ego 80 km/h",
        cls=GhostCutOutObstacleSim,
        kwargs=dict(leader_v0=22.2, stat_x=180.0, t_cutout=1.5),
        ego_v0=22.2, t_max=15.0,
        vis_type="cutout", t_event=1.5,
    ),
    "D": dict(
        name="ghost_cutout_panic",
        title="Scenario D: Ghost Cut-out (Panic, 75 m)",
        subtitle="Leader swerves out → stationary at 75 m   |  Ego 80 km/h",
        cls=GhostCutOutObstacleSim,
        kwargs=dict(leader_v0=22.2, leader_x0=40.0, stat_x=75, t_cutout=1.5),
        ego_v0=22.2, t_max=20.0,
        vis_type="cutout", t_event=1.5,
    ),
    "E": dict(
        name="adversarial_stopgo",
        title="Scenario E: Adversarial Stop & Go",
        subtitle="Brake → accel → brake → flee   |  Ego 72 km/h",
        cls=AdversarialStopGoObstacleSim,
        kwargs=dict(ego_v0=20.0, v0=15.0, gap_at_cutin=15.0, t_cutin=1.0),
        ego_v0=20.0, t_max=12.0,
        vis_type="cutin", t_event=1.0,
    ),
}


# ═══════════════════════════════════════════════════════════════════
# 2. Controller batches
# ═══════════════════════════════════════════════════════════════════
BATCH_COLORS = {
    "batch1_delay": dict(
        PID_Standard="#ef5350", PID_Forward="#ffa726",
        MPC_NoAug="#ffee58", MPC_Aug="#66bb6a",
    ),
    "batch2_safety": dict(
        Track_Only="#ef5350", Hard_Bound="#ffa726", Soft_Bound="#66bb6a",
    ),
    "batch3_cost": dict(
        OSQP_Soft="#42a5f5", iLQR_Quad="#ffa726", iLQR_Exp="#66bb6a",
    ),
}

def get_batches():
    c = get_all_controllers()
    return dict(
        batch1_delay={
            "PID_Standard": c["PID_Standard"], "PID_Forward": c["PID_Forward"],
            "MPC_NoAug": c["MPC_NoAug_Track"], "MPC_Aug": c["MPC_Aug_Track"],
        },
        batch2_safety={
            "Track_Only": c["MPC_Aug_Track"], "Hard_Bound": c["MPC_Aug_Hard"],
            "Soft_Bound": c["MPC_Aug_Soft"],
        },
        batch3_cost={
            "OSQP_Soft": c["MPC_Aug_Soft"], "iLQR_Quad": c["iLQR_Aug_Quad"],
            "iLQR_Exp": c["iLQR_Aug_Exp"],
        },
    )


# ═══════════════════════════════════════════════════════════════════
# 3. Visual helpers
# ═══════════════════════════════════════════════════════════════════
EGO_LANE_Y = -0.45
ADJ_LANE_Y =  0.45
LANE_CHANGE_DUR = 0.5
ROAD_L, ROAD_R = 0.3, 9.7


def obs_lane_y(t, scfg):
    vtype, t_ev = scfg["vis_type"], scfg["t_event"]
    if vtype == "cutin":
        t0 = t_ev - LANE_CHANGE_DUR
        if t < t0:   return ADJ_LANE_Y
        if t < t_ev: return ADJ_LANE_Y + (t - t0) / LANE_CHANGE_DUR * (EGO_LANE_Y - ADJ_LANE_Y)
        return EGO_LANE_Y
    if vtype == "cutout":
        # Leader starts changing lanes well before cutout event,
        # so it's clearly in the adjacent lane by the time it passes STA.
        t_start = t_ev - 0.8
        t_end   = t_ev + 0.2
        if t < t_start: return EGO_LANE_Y
        if t < t_end:   return EGO_LANE_Y + (t - t_start) / (t_end - t_start) * (ADJ_LANE_Y - EGO_LANE_Y)
        return ADJ_LANE_Y
    return EGO_LANE_Y


def phase_label(v, a, active):
    if not active:                       return "HIDDEN",   "#555"
    if abs(v) < 0.1 and abs(a) < 0.3:   return "STOPPED",  "#ffa726"
    if a < -1:                           return "BRAKING",  "#ef5350"
    if a > 1:                            return "ACCEL",    "#66bb6a"
    return "COASTING", "#78909c"


def obs_color(a, active):
    if not active: return "#555"
    if a < -1:     return "#d32f2f"
    if a > 1:      return "#43a047"
    return "#f57c00"


def draw_car(ax, cx, cy, cw, ch, color, label=None, alpha=1.0):
    """Draw car rectangle at (cx, cy) with to-scale width cw."""
    if cx < ROAD_L - 1.5 or cx > ROAD_R + 1.5:
        return
    cw_draw = max(cw, 0.04)  # minimum visible width
    ax.add_patch(patches.FancyBboxPatch(
        (cx - cw_draw / 2, cy - ch / 2), cw_draw, ch,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="white", linewidth=0.5,
        alpha=alpha, zorder=5))
    if label:
        if cw_draw >= 0.20:
            # label inside car
            ax.text(cx, cy, label, ha="center", va="center",
                    fontsize=5, fontweight="bold", color="white", zorder=6, clip_on=True)
        else:
            # label above car (car too narrow for text inside)
            ax.text(cx, cy + ch / 2 + 0.05, label, ha="center", va="bottom",
                    fontsize=5, fontweight="bold", color=color, alpha=max(alpha, 0.7),
                    zorder=6, clip_on=True)


def extrapolate_leader(history, scfg):
    """
    Build full-length leader position array.
    Uses leader_v0 from scenario kwargs (constant velocity) to avoid the
    off-by-one bug: scenario.step() is called BEFORE get_state(), so the
    frame with t_arr < t_cutout may already have obs_v = 0 (switched).
    """
    t_arr = np.array(history["t"])
    ox = np.array(history["obs_x"])
    t_ev = scfg["t_event"]
    dt = t_arr[1] - t_arr[0] if len(t_arr) > 1 else 0.05
    lv = scfg["kwargs"].get("leader_v0", 22.2)  # constant — never use obs_v

    # Anchor: take a frame safely before the obs_x jump (step back by 2*dt)
    anchor_mask = t_arr < (t_ev - 2 * dt)
    if not np.any(anchor_mask):
        return np.full_like(t_arr, np.nan)
    ai = np.where(anchor_mask)[0][-1]
    lx0, t0 = ox[ai], t_arr[ai]

    # Constant-velocity extrapolation from anchor for the entire timeline
    ldr = lx0 + lv * (t_arr - t0)
    return ldr


# ═══════════════════════════════════════════════════════════════════
# 4. GIF generator
# ═══════════════════════════════════════════════════════════════════
def generate_gif(scfg, ctrl_dict, colors, filename):

    # ── simulate ────────────────────────────────────────────────
    print(f"    simulating {len(ctrl_dict)} ctrl(s)…", end=" ", flush=True)
    results = run_ablation_group(
        ctrl_dict, scfg["cls"], scfg["kwargs"], scfg["ego_v0"], scfg["t_max"])
    ctrl_names = list(results.keys())

    ref_name = max(ctrl_names, key=lambda n: results[n]["metrics"]["min_dist"])
    rh = results[ref_name]["history"]
    t_arr   = np.array(rh["t"])
    obs_x   = np.array(rh["obs_x"])
    obs_v   = np.array(rh["obs_v"])
    obs_a   = np.array(rh["obs_a"])
    obs_act = np.array(rh["obs_active"], dtype=bool)

    all_ego_x = {n: np.array(results[n]["history"]["ego_x"]) for n in ctrl_names}
    all_ego_v = {n: np.array(results[n]["history"]["ego_v"]) for n in ctrl_names}

    is_cutout = scfg["vis_type"] == "cutout"
    leader_xs = extrapolate_leader(rh, scfg) if is_cutout else None
    stat_x    = scfg["kwargs"].get("stat_x") if is_cutout else None

    # ── fixed viewport ──────────────────────────────────────────
    # For cut-out: viewport covers ego range + stationary obstacle only
    #   (leader can drive off-screen — keeps viewport tight)
    # For cut-in:  viewport covers ego range + obstacle range
    pools = [all_ego_x[n] for n in ctrl_names]
    if is_cutout:
        pools.append(np.array([stat_x]))
        # also include obs_x which covers leader pre-cutout positions
        pools.append(obs_x)
    else:
        pools.append(obs_x)

    all_pos = np.concatenate(pools)
    vmin  = float(np.nanmin(all_pos)) - 15
    vmax  = float(np.nanmax(all_pos)) + 15
    vspan = vmax - vmin
    rlen  = ROAD_R - ROAD_L

    def w2v(wx):
        return ROAD_L + (wx - vmin) / vspan * rlen

    # ── to-scale car dimensions ─────────────────────────────────
    car_w = REAL_CAR_LEN / vspan * rlen   # proportional to road
    car_h = 0.30                           # fixed y (schematic lateral axis)

    # ── subsampling ─────────────────────────────────────────────
    total = len(t_arr)
    skip  = max(2, total // 120)
    indices = list(range(0, total, skip))
    nf = len(indices)

    # ── figure ──────────────────────────────────────────────────
    fig, (ax_rd, ax_st) = plt.subplots(
        2, 1, figsize=(13, 5.5),
        gridspec_kw=dict(height_ratios=[2.8, 2.2]),
        facecolor="#1a1a2e")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.87, bottom=0.09, hspace=0.28)
    fig.suptitle(scfg["title"], fontsize=13, fontweight="bold", color="white", y=0.95)
    fig.text(0.5, 0.90, scfg["subtitle"], ha="center", fontsize=9, color="#aaa")

    # ── S-T axes ────────────────────────────────────────────────
    ax_st.set_facecolor("#0f0f23")
    ax_st.set_xlabel("Time [s]", color="white", fontsize=9)
    ax_st.set_ylabel("Position [m]", color="white", fontsize=9)
    ax_st.set_xlim(0, scfg["t_max"])
    ax_st.set_ylim(vmin, vmax)
    ax_st.tick_params(colors="white", labelsize=7)
    ax_st.grid(True, alpha=0.12, color="white")
    for sp in ax_st.spines.values(): sp.set_color("#444")

    # faint background traces
    for n in ctrl_names:
        ax_st.plot(t_arr, all_ego_x[n], color=colors[n], lw=0.7, alpha=0.18)
    if np.any(obs_act):
        ax_st.plot(t_arr[obs_act], obs_x[obs_act], color="#ef5350", lw=0.7, alpha=0.18)

    st_ego = {n: ax_st.plot([], [], color=colors[n], lw=1.8, label=n)[0] for n in ctrl_names}
    st_obs = ax_st.plot([], [], color="#ef5350", lw=1.8, ls="--", label="Obstacle")[0]

    # event marker
    ev_lbl = "Cut-out" if is_cutout else "Cut-in"
    ax_st.axvline(scfg["t_event"], color="#ffd54f", lw=0.7, ls=":", alpha=0.5)
    ax_st.text(scfg["t_event"] + 0.2, vmax - 5, ev_lbl,
               fontsize=7, color="#ffd54f", va="top")

    if stat_x is not None:
        ax_st.axhline(stat_x, color="#ef5350", lw=0.5, ls=":", alpha=0.35)
        ax_st.text(0.3, stat_x + 2, "Stationary", fontsize=6, color="#ef5350", alpha=0.5)

    if leader_xs is not None:
        pm = t_arr >= scfg["t_event"]
        if np.any(pm):
            ax_st.plot(t_arr[pm], leader_xs[pm],
                       color="#78909c", lw=0.7, ls="--", alpha=0.3, label="Leader")

    ncol = min(len(ctrl_names) + 2, 4)
    ax_st.legend(fontsize=6, loc="upper left", facecolor="#1a1a2e",
                 edgecolor="#444", labelcolor="white", ncol=ncol)

    tvl = [None]

    # ego y offsets (stagger in compare mode)
    nc = len(ctrl_names)
    if nc == 1:
        yoff = {ctrl_names[0]: EGO_LANE_Y}
    else:
        sp = min(0.06 * (nc - 1), 0.20)
        ys = np.linspace(EGO_LANE_Y - sp / 2, EGO_LANE_Y + sp / 2, nc)
        yoff = {n: ys[j] for j, n in enumerate(ctrl_names)}

    # ── animate ─────────────────────────────────────────────────
    def animate(fi):
        i = indices[fi]
        t_now = t_arr[i]

        ax_rd.clear()
        ax_rd.set_xlim(0, 10)
        ax_rd.set_ylim(-1.4, 1.4)
        ax_rd.set_aspect("equal")
        ax_rd.axis("off")
        ax_rd.set_facecolor("#1a1a2e")

        # road surface
        ax_rd.add_patch(patches.Rectangle(
            (ROAD_L - 0.1, -0.88), rlen + 0.2, 1.76,
            facecolor="#2d2d2d", edgecolor="none", zorder=1))
        for ey in (-0.85, 0.85):
            ax_rd.plot([ROAD_L - 0.1, ROAD_R + 0.1], [ey, ey],
                       color="white", lw=1.3, zorder=2)
        # centre dashes
        nd = max(10, int(vspan / 15))
        for di in range(nd + 1):
            wx = vmin + di * vspan / nd
            vx1, vx2 = w2v(wx), w2v(wx + vspan / nd * 0.35)
            if ROAD_L - 0.2 < vx1 < ROAD_R + 0.2:
                ax_rd.plot([vx1, vx2], [0, 0], color="#777", lw=1, alpha=0.45, zorder=2)

        # ════════════════════════════════════════════════════════
        #  Draw obstacles
        # ════════════════════════════════════════════════════════
        if is_cutout:
            # ── Ghost cut-out: 3 entities ──
            # Spatial order (front→back): STA ··· LDR ··· EGO
            # LDR's early lane change (starts 0.3s before t_event) ensures
            # it's already laterally offset when passing STA.

            # (a) Stationary obstacle — real position, always on road
            svx = w2v(stat_x)
            sta_a = 0.30 if t_now < scfg["t_event"] else 1.0
            draw_car(ax_rd, svx, EGO_LANE_Y, car_w, car_h, "#d32f2f", "STA", sta_a)
            if ROAD_L < svx < ROAD_R:
                ax_rd.text(svx, EGO_LANE_Y - car_h / 2 - 0.06,
                           "0 km/h", ha="center", fontsize=5, color="#bbb",
                           alpha=sta_a, zorder=7)

            # (b) Leader — extrapolated position (constant v after cut-out)
            if leader_xs is not None:
                lvx = w2v(leader_xs[i])
                lvy = obs_lane_y(t_now, scfg)
                la  = 0.65 if t_now > scfg["t_event"] + 0.6 else 0.9
                draw_car(ax_rd, lvx, lvy, car_w, car_h, "#78909c", "LDR", la)
                if ROAD_L < lvx < ROAD_R:
                    # leader always at leader_v0 (before and after cut-out)
                    lv_kmh = scfg["kwargs"].get("leader_v0", 22.2) * 3.6
                    ax_rd.text(lvx, lvy - car_h / 2 - 0.06,
                               f"{lv_kmh:.0f} km/h",
                               ha="center", fontsize=5, color="#bbb",
                               alpha=la, zorder=7)
        else:
            # ── Cut-in: single obstacle with lateral slide ──
            oly = obs_lane_y(t_now, scfg)
            ovx = w2v(obs_x[i])
            oc  = obs_color(obs_a[i], obs_act[i])
            draw_car(ax_rd, ovx, oly, car_w, car_h, oc, "OBS")
            if ROAD_L < ovx < ROAD_R:
                ax_rd.text(ovx, oly - car_h / 2 - 0.06,
                           f"{obs_v[i] * 3.6:.0f} km/h",
                           ha="center", fontsize=5, color="#bbb", zorder=7)

        # ════════════════════════════════════════════════════════
        #  Draw ego car(s)
        # ════════════════════════════════════════════════════════
        for n in ctrl_names:
            evx = w2v(all_ego_x[n][i])
            draw_car(ax_rd, evx, yoff[n], car_w, car_h, colors[n], n[:4])

        # ════════════════════════════════════════════════════════
        #  Gap annotation (single mode only)
        # ════════════════════════════════════════════════════════
        if nc == 1 and obs_act[i]:
            gap = obs_x[i] - all_ego_x[ctrl_names[0]][i]
            if 0 < gap:
                evx = w2v(all_ego_x[ctrl_names[0]][i])
                ovx = w2v(obs_x[i])
                # arrow between car edges
                e_right = evx + car_w / 2
                o_left  = ovx - car_w / 2
                if o_left > e_right + 0.02:
                    ax_rd.annotate("", xy=(o_left, 0.0), xytext=(e_right, 0.0),
                                   arrowprops=dict(arrowstyle="<->", color="#ffd54f", lw=1.2),
                                   zorder=7, clip_on=True)
                    ax_rd.text((evx + ovx) / 2, 0.12, f"{gap:.1f} m",
                               ha="center", fontsize=8, fontweight="bold",
                               color="#ffd54f", zorder=7)

        # ════════════════════════════════════════════════════════
        #  Badges
        # ════════════════════════════════════════════════════════
        ph, pc = phase_label(obs_v[i], obs_a[i], obs_act[i])
        ax_rd.text(ROAD_L, 1.2, ph, ha="left", va="top",
                   fontsize=8, fontweight="bold", color=pc, zorder=7,
                   bbox=dict(boxstyle="round,pad=0.25", fc="#222", alpha=0.8))
        ax_rd.text(ROAD_R, 1.2, f"t = {t_now:.1f} s", ha="right", va="top",
                   fontsize=9, fontweight="bold", color="white", zorder=7,
                   bbox=dict(boxstyle="round,pad=0.25", fc="#333", alpha=0.8))

        # legend strip (compare mode)
        if nc > 1:
            sp2 = min(2.2, rlen / nc)
            for ci, n in enumerate(ctrl_names):
                ax_rd.text(ROAD_L + ci * sp2, -1.18, f"■ {n}",
                           fontsize=6, color=colors[n], fontweight="bold",
                           va="top", zorder=7)

        # ════════════════════════════════════════════════════════
        #  S-T update
        # ════════════════════════════════════════════════════════
        for n in ctrl_names:
            st_ego[n].set_data(t_arr[:i + 1], all_ego_x[n][:i + 1])
        m = obs_act[:i + 1]
        if np.any(m):
            st_obs.set_data(t_arr[:i + 1][m], obs_x[:i + 1][m])
        if tvl[0] is not None:
            tvl[0].remove()
        tvl[0] = ax_st.axvline(t_now, color="white", lw=0.5, alpha=0.35)
        return []

    # ── save ────────────────────────────────────────────────────
    print(f"rendering {nf} frames…", end=" ", flush=True)
    anim = FuncAnimation(fig, animate, frames=nf, interval=100, blit=False)
    anim.save(os.path.join(OUT_DIR, filename), writer=PillowWriter(fps=10), dpi=120)
    plt.close(fig)
    print(f"✓  {filename}")


# ═══════════════════════════════════════════════════════════════════
# 5. CLI
# ═══════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="Generate scenario animation GIFs")
    p.add_argument("--mode", choices=["single", "compare"], default="single")
    p.add_argument("--scenario", choices=["all", "A", "B", "C", "D", "E"], default="all")
    p.add_argument("--batch", choices=["all", "1", "2", "3"], default="all")
    args = p.parse_args()

    scens = SCENARIOS if args.scenario == "all" else {args.scenario: SCENARIOS[args.scenario]}

    if args.mode == "single":
        print(f"\n[single] {len(scens)} GIF(s)…\n")
        c = get_all_controllers()
        one = {"iLQR_Exp": c["iLQR_Aug_Exp"]}
        clr = {"iLQR_Exp": "#4fc3f7"}
        for k, s in scens.items():
            print(f"  [{k}] {s['title']}")
            generate_gif(s, one, clr, f"{s['name']}.gif")

    elif args.mode == "compare":
        bmap = {"1": "batch1_delay", "2": "batch2_safety", "3": "batch3_cost"}
        batches = get_batches()
        brun = batches if args.batch == "all" else {bmap[args.batch]: batches[bmap[args.batch]]}
        total = len(scens) * len(brun)
        print(f"\n[compare] {total} GIF(s)…\n")
        for k, s in scens.items():
            for bn, bc in brun.items():
                print(f"  [{k} × {bn}] {s['title']}")
                generate_gif(s, bc, BATCH_COLORS[bn], f"{s['name']}_{bn}.gif")

    n = len([f for f in os.listdir(OUT_DIR) if f.endswith(".gif")])
    print(f"\nDone — {n} GIF(s) in {OUT_DIR}/")


if __name__ == "__main__":
    main()
