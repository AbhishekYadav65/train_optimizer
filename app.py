# app.py
# Hybrid Optimizer + Polished UI — demo-ready

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random, copy, time

# --- Page configuration ---
st.set_page_config(layout="wide", page_title="AI Train Traffic Optimizer (Hybrid + UI)")

# --- Grid config ---
GRID_ROWS, GRID_COLS = 3, 6
blocks = []
block_coords = {}
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        bid = f"B{r}-{c}"
        blocks.append(bid)
        block_coords[bid] = (c * 1.6, (GRID_ROWS - 1 - r) * 1.0)

# --- Load dataset ---
trains_df = pd.read_csv("trains.csv")

# --- Train class ---
class Train:
    def __init__(self, train_id, ttype, priority, start_time, origin, destination):
        self.id = int(train_id)
        self.type = ttype
        self.priority = int(priority)
        self.start_time = int(start_time)
        self.origin = origin
        self.destination = destination
        self.path = []
        self.path_index = 0
        self.pos_frac = 0.0
        self.delay = 0
        self.finished = False
        self.action = "not started"

    def copy_basic(self):
        t = Train(self.id, self.type, self.priority, self.start_time, self.origin, self.destination)
        t.path = list(self.path)
        t.path_index = self.path_index
        t.pos_frac = self.pos_frac
        t.delay = self.delay
        t.finished = self.finished
        t.action = self.action
        return t

# --- Path planning + speeds ---
def plan_path(train, row=None):
    if row is None:
        row = random.choice(range(GRID_ROWS))
    return [f"B{row}-{c}" for c in range(GRID_COLS)]

SPEED_MAP = {"Express": 0.55, "Local": 0.35, "Freight": 0.2}

# --- Init / Reset environment ---
def init_trains():
    trains = []
    for r in trains_df.itertuples():
        t = Train(r.train_id, r.type, r.priority, r.start_time, r.origin, r.destination)
        t.path = plan_path(t, row=random.choice([0,1,2]))
        trains.append(t)
    return trains

def reset_environment(preset=None):
    st.session_state.trains = init_trains()
    st.session_state.tick = 0
    st.session_state.breakdown_ticks = 0
    st.session_state.block_occupancy = {b: None for b in blocks}
    for t in st.session_state.trains:
        st.session_state.block_occupancy[t.path[t.path_index]] = t.id

    if preset == "rush":
        # Move Express trains to start early
        for t in st.session_state.trains:
            if t.type.lower() == "express":
                t.start_time = random.randint(0,2)
    elif preset == "freight":
        # Freight priority: make freight start early
        for t in st.session_state.trains:
            if t.type.lower() == "freight":
                t.start_time = random.randint(0,2)
    elif preset == "disruption":
        st.session_state.breakdown_ticks = 3

if "trains" not in st.session_state:
    reset_environment()

# --- Scoring / Metrics ---
def remaining_blocks(t):
    return max(0, len(t.path) - 1 - t.path_index)

def score_train(t):
    # This is your previously working hybrid scoring + rollout
    ALPHA, BETA, GAMMA = 120, 6, 4
    return ALPHA * t.priority - BETA * t.delay - GAMMA * remaining_blocks(t)

# --- Hybrid + 1-step Rollout conflict resolver (good version) ---
def greedy_conflict_resolution(requests):
    actions = {}
    for block, conts in requests.items():
        if not conts:
            continue
        chosen = max(conts, key=lambda t: (t.priority, -t.delay, -t.start_time))
        actions[chosen.id] = "proceed"
        for c in conts:
            if c.id != chosen.id:
                actions[c.id] = "hold"
                c.delay += 1
    return actions

def rollout_resolve(requests):
    actions = {}
    for block, conts in requests.items():
        if not conts:
            continue
        conts_sorted = sorted(conts, key=lambda t: (-score_train(t), -t.priority, -t.start_time))
        top = conts_sorted[0]
        if len(conts_sorted) == 1:
            actions[top.id] = "proceed"
            continue
        a, b = conts_sorted[0], conts_sorted[1]
        def sim(first, second):
            fa, fb = copy.deepcopy(first), copy.deepcopy(second)
            fa.pos_frac += SPEED_MAP.get(fa.type, 0.3)
            if fa.pos_frac >= 1.0:
                fa.pos_frac = 0.0
                fa.path_index = min(len(fa.path)-1, fa.path_index + 1)
            fb.delay += 1
            return fa.delay + fb.delay
        cost_ab = sim(a, b)
        cost_ba = sim(b, a)
        if cost_ab <= cost_ba:
            actions[a.id] = "proceed"
            actions[b.id] = "hold"
            b.delay += 1
        else:
            actions[b.id] = "proceed"
            actions[a.id] = "hold"
            a.delay += 1
        for c in conts_sorted[2:]:
            actions[c.id] = "hold"
            c.delay += 1
    return actions

# --- Main tick logic ---
def run_tick():
    tick = st.session_state.tick
    requests = {}
    for t in st.session_state.trains:
        if t.finished:
            t.action = "arrived"
            continue
        if tick < t.start_time:
            t.action = "not started"
            continue
        if t.path_index >= len(t.path) - 1 and t.pos_frac >= 1.0:
            t.finished, t.action = True, "arrived"
            continue
        if t.path_index + 1 < len(t.path):
            nb = t.path[t.path_index + 1]
            requests.setdefault(nb, []).append(t)

    if st.session_state.breakdown_ticks > 0:
        for conts in requests.values():
            for t in conts:
                t.action = "hold (breakdown)"
                t.delay += 1
        st.session_state.breakdown_ticks -= 1
        st.session_state.tick += 1
        return

    try:
        actions = rollout_resolve(requests)
    except Exception:
        actions = greedy_conflict_resolution(requests)

    # ensure at least one proceeds
    if not any(act == "proceed" for act in actions.values()):
        # choose the most urgent train
        active = [t for t in st.session_state.trains if not t.finished and st.session_state.tick >= t.start_time]
        if active:
            critical = max(active, key=lambda t: (t.priority, t.delay, -t.start_time))
            actions[critical.id] = "proceed"

    # apply actions
    for t in st.session_state.trains:
        act = actions.get(t.id, "hold")
        t.action = act
        if act == "proceed":
            t.pos_frac += SPEED_MAP.get(t.type, 0.3)
            if t.pos_frac >= 1.0:
                prev = t.path[t.path_index]
                if st.session_state.block_occupancy.get(prev) == t.id:
                    st.session_state.block_occupancy[prev] = None
                t.path_index += 1
                if t.path_index < len(t.path):
                    st.session_state.block_occupancy[t.path[t.path_index]] = t.id
                t.pos_frac = 0.0
        else:
            t.delay += 1
        if t.path_index >= len(t.path) - 1 and t.pos_frac == 0.0:
            t.finished, t.action = True, "arrived"

    st.session_state.tick += 1

# --- Visualization ---
def draw_grid(trains):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#ffffff")
    ax.axis("off")
    for bid, (x, y) in block_coords.items():
        ax.scatter(x, y, s=120, color="#333")
        ax.text(x, y - 0.25, bid, ha="center", fontsize=7, color="#666")
    for r in range(GRID_ROWS):
        for c in range(1, GRID_COLS):
            x1, y1 = block_coords[f"B{r}-{c-1}"]
            x2, y2 = block_coords[f"B{r}-{c}"]
            ax.plot([x1, x2], [y1, y2], color="#999", linestyle="--")
    cmap = {"Express": "#e53935", "Local": "#43a047", "Freight": "#1e88e5"}
    for t in trains:
        if t.finished:
            continue
        idx = t.path_index
        x_cur, y_cur = block_coords[t.path[idx]]
        if idx + 1 < len(t.path):
            x_n, y_n = block_coords[t.path[idx+1]]
            frac = min(1.0, max(0.0, t.pos_frac))
            dx = x_cur * (1 - frac) + x_n * frac
            dy = y_cur * (1 - frac) + y_n * frac + 0.12
        else:
            dx, dy = x_cur, y_cur + 0.12
        ax.scatter(dx, dy, s=220, color=cmap.get(t.type, "#666"))
        ax.text(dx, dy, str(t.id), ha="center", va="center", fontsize=8, color="white", weight="bold")
    # legend
    for i, (k, v) in enumerate(cmap.items()):
        ax.scatter([], [], s=120, color=v, label=k)
    ax.legend(loc="upper right")
    ax.set_xlim(-1, GRID_COLS * 1.6)
    ax.set_ylim(-1.2, GRID_ROWS * 1.2)
    return fig

# --- UI layout / polish ---
st.title("🚄 AI Train Traffic Optimizer — Hybrid + UI")

col1, col2, col3 = st.columns([2.5, 5, 2.5])

with col1:
    st.subheader("🎮 Controls")
    if st.button("▶ Next Tick"):
        run_tick()
    if st.button("▶ Auto-Play" if st.session_state.get("autoplay", False) else "▶ Auto-Play"):
        st.session_state.autoplay = not st.session_state.get("autoplay", False)
    if st.button("🔄 Reset"):
        reset_environment()
    st.subheader("⚠️ Scenarios")
    if st.button("Rush Hour"):
        reset_environment("rush")
    if st.button("Freight Priority"):
        reset_environment("freight")
    if st.button("Disruption"):
        reset_environment("disruption")

with col2:
    st.subheader("📍 Track Visualization")
    fig = draw_grid(st.session_state.trains)
    st.pyplot(fig, use_container_width=True)

with col3:
    st.subheader("📊 KPIs")
    avg_delay = sum(t.delay for t in st.session_state.trains) / max(1, len(st.session_state.trains))
    throughput = sum(1 for t in st.session_state.trains if t.finished)
    punctual = sum(1 for t in st.session_state.trains if t.finished and t.delay <= 5)
    punctual_rate = (punctual / max(1, len(st.session_state.trains)))*100

    st.metric("⏱ Time", st.session_state.tick)
    st.metric("📉 Avg Delay", f"{avg_delay:.1f} min", delta=f"{avg_delay:.1f} vs baseline")
    st.metric("🚆 Trains Completed", throughput)
    st.metric("🎯 Punctuality (<5 min)", f"{punctual_rate:.0f} %", delta=f"{punctual_rate:.0f} %")

st.markdown("---")
st.subheader("🚆 Train States")
rows = []
for t in st.session_state.trains:
    rows.append({"Train ID": t.id, "Type": t.type, "Block": t.path[t.path_index], "Delay": t.delay, "Action": t.action})
st.table(pd.DataFrame(rows))

st.markdown("---")
st.subheader("📊 Baseline vs Hybrid Comparison")
if st.button("▶ Run Full Simulation Comparison"):
    def simulate(strategy="baseline"):
        trains_copy = [t.copy_basic() for t in st.session_state.trains]
        tick = 0
        max_ticks = 400
        while any(not t.finished for t in trains_copy) and tick < max_ticks:
            requests = {}
            for t in trains_copy:
                if t.finished or tick < t.start_time:
                    continue
                if t.path_index >= len(t.path) - 1:
                    t.finished = True
                    continue
                nb = t.path[t.path_index + 1]
                requests.setdefault(nb, []).append(t)
            actions = {}
            if strategy == "baseline":
                for block, conts in requests.items():
                    chosen = min(conts, key=lambda x: x.start_time)
                    actions[chosen.id] = "proceed"
                    for c in conts:
                        if c.id != chosen.id:
                            actions[c.id] = "hold"
                            c.delay += 1
            else:
                for block, conts in requests.items():
                    chosen = max(conts, key=lambda t: score_train(t))
                    actions[chosen.id] = "proceed"
                    for c in conts:
                        if c.id != chosen.id:
                            actions[c.id] = "hold"
                            c.delay += 1
            for t in trains_copy:
                if actions.get(t.id) == "proceed":
                    t.pos_frac += SPEED_MAP.get(t.type, 0.3)
                    if t.pos_frac >= 1.0:
                        t.pos_frac = 0.0
                        t.path_index += 1
                        if t.path_index >= len(t.path) - 1:
                            t.finished = True
                else:
                    t.delay += 1
            tick += 1
        avg = sum(t.delay for t in trains_copy) / max(1, len(trains_copy))
        tp = sum(1 for t in trains_copy if t.finished)
        punctual = sum(1 for t in trains_copy if t.finished and t.delay <= 5)
        rate = (punctual / max(1, len(trains_copy))) * 100
        return avg, tp, rate

    base_avg, base_tp, base_rate = simulate("baseline")
    opt_avg, opt_tp, opt_rate = simulate("hybrid")

    comp = pd.DataFrame({
        "Strategy": ["Baseline (FCFS)", "Hybrid"],
        "Avg Delay (min)": [base_avg, opt_avg],
        "Throughput": [base_tp, opt_tp],
        "Punctuality %": [base_rate, opt_rate]
    })
    st.table(comp)
    fig, ax = plt.subplots(1,2,figsize=(9,3))
    ax[0].bar(["Baseline","Hybrid"], [base_avg, opt_avg], color=["#777","#1f77b4"])
    ax[0].set_ylabel("Avg Delay (min)")
    ax[1].bar(["Baseline","Hybrid"], [base_rate, opt_rate], color=["#777","#1f77b4"])
    ax[1].set_ylabel("Punctuality %")
    st.pyplot(fig)
    diff = base_avg - opt_avg
    if diff > 0:
        st.success(f"✅ Hybrid reduced avg delay by {diff:.2f} min and improved punctuality by {opt_rate - base_rate:.1f}%!")
    else:
        st.warning("⚠ No improvement in this run.")

# --- Auto-play ---
if st.session_state.get("autoplay", False):
    run_tick()
    time.sleep(1)
    st.rerun()
