import streamlit as st
import pandas as pd
import json

# Load train data
trains_df = pd.read_csv("trains.csv")
with open("network.json", "r") as f:
    network = json.load(f)

class Train:
    def __init__(self, train_id, ttype, priority, start_time, origin, destination):
        self.id = train_id
        self.type = ttype
        self.priority = priority
        self.start_time = start_time
        self.origin = origin
        self.destination = destination
        self.position = origin
        self.delay = 0
        self.finished = False
        self.action = "not started"

# Initialize session state
if "trains" not in st.session_state:
    st.session_state.trains = [
        Train(row.train_id, row.type, row.priority, row.start_time, row.origin, row.destination)
        for row in trains_df.itertuples()
    ]
    st.session_state.tick = 0

if "block_status" not in st.session_state:
    st.session_state.block_status = {block: None for block in network["blocks"]}

def decide_conflicts(requests):
    actions = {}
    for block, contenders in requests.items():
        contenders.sort(key=lambda t: (t.priority, t.delay), reverse=True)
        winner = contenders[0]
        actions[winner.id] = "proceed"
        for loser in contenders[1:]:
            actions[loser.id] = "hold"
            loser.delay += 1
    return actions

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

        if t.position == t.destination:
            t.finished = True
            t.action = "arrived"
            continue

        # Decide which block the train wants
        desired_block = "A-B" if t.position == "A" else "B-C"
        requests.setdefault(desired_block, []).append(t)

    # Conflict resolution
    if requests:
        actions = decide_conflicts(requests)
        for t in st.session_state.trains:
            if t.id in actions:
                t.action = actions[t.id]
                if actions[t.id] == "proceed":
                    if t.position == "A":
                        t.position = "B"
                    elif t.position == "B":
                        t.position = "C"
                    else:
                        t.finished = True
                        t.action = "arrived"
    st.session_state.tick += 1

# ==================== UI ==================== #
st.title("🚄 AI-Powered Train Traffic Control")

col1, col2 = st.columns(2)
with col1:
    if st.button("▶ Next Tick"):
        run_tick()
    if st.button("🔄 Reset"):
        st.session_state.trains = [
            Train(row.train_id, row.type, row.priority, row.start_time, row.origin, row.destination)
            for row in trains_df.itertuples()
        ]
        st.session_state.tick = 0
with col2:
    st.metric("⏱ Time", st.session_state.tick)
    avg_delay = sum(t.delay for t in st.session_state.trains) / len(st.session_state.trains)
    st.metric("📉 Avg Delay", f"{avg_delay:.1f} min")
    throughput = sum(1 for t in st.session_state.trains if t.finished)
    st.metric("🚆 Trains Completed", throughput)

# Show train states
train_data = [{
    "Train ID": t.id,
    "Type": t.type,
    "Pos": t.position,
    "Delay": t.delay,
    "Action": t.action
} for t in st.session_state.trains]
st.table(pd.DataFrame(train_data))
