# 🚄 AI Train Traffic Optimizer — Hybrid Priority + Rollout

A **simulation prototype** for optimizing train traffic in a grid network.  
The app compares **Baseline FCFS (First Come First Serve)** with a custom **Hybrid Optimizer (priority + rollout)**, and demonstrates how AI (Groq stub) can be integrated for future deployment.

---

## ✨ Features

- **Baseline vs Hybrid Comparison**  
  FCFS scheduling compared against a hybrid optimizer that:
  - Uses train **priority, delay, and distance remaining**.
  - Resolves conflicts with a **1-step rollout lookahead**.
  - Falls back to greedy if needed.

- **Simulation Controls**
  - ▶ Next Tick (advance simulation)
  - ▶ Toggle Auto-Play (auto simulation with delay slider)
  - 🔄 Reset
  - 🚧 Breakdown (simulate disruption)
  - ⏱ Random Delay


- **Track Visualization**  
  - Grid network with trains (Express, Local, Freight).  
  - Color-coded movement with live updates.

- **KPIs Dashboard**
  - Avg Delay (minutes)  
  - Trains Completed  
  - Punctuality Rate (<5min delay)  
  - Current Simulation Time  

- **Baseline vs Hybrid Results**  
  - Side-by-side comparison of strategies.  
  - Bar charts for **average delay** and **throughput**.  
  - Success/warning message on performance.

---


train_optimizer/

│── app.py # Main Streamlit application

│── trains.csv # Train dataset (id, type, priority, timings, origin, destination)

│── requirements.txt # Python dependencies


---

## 🚀 Installation & Run

### 1. Clone repo
```bash
git clone https://github.com/AbhishekYadav65/train_optimizer.git
cd train_optimizer


python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows PowerShell


pip install -r requirements.txt


streamlit run app.py




**🧪 Demo Flow (Hackathon / Judges)**

Launch app: streamlit run app.py

Show track visualization with trains loaded.

Run simulation step by step (▶ Next Tick).

Enable Auto-Play to demonstrate smooth flow.

Trigger scenarios:

🚧 Breakdown (simulate disruption)

⏱ Random Delay

Show Groq AI Mode toggle:

Logs: “Decisions made by Groq AI Engine (demo)”.

Explains future AI integration.

Run Baseline vs Hybrid Comparison:

Judges see bar charts + results message.
## 📂 Project Structure




🛠 Tech Stack

Python 3.9+

Streamlit (UI framework)

Pandas (data handling)

Matplotlib (visualization)

Custom Hybrid Algorithm (priority + rollout)

Groq AI Stub Integration (future AI extension)
