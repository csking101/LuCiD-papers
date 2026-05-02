# Coding Adventures

Hands-on interactive demos that bring paper concepts to life. Each adventure ties together ideas from multiple papers in the LuCiD collection into a single runnable project.

These are different from the per-paper visualisations (static PNGs, interactive HTMLs, Manim animations) -- adventures are **standalone programs** you run, interact with, and experiment with to build intuition.

---

## Adventures

| # | Adventure | Papers | Tech | Status |
|---|-----------|--------|------|--------|
| 1 | [Path-Finding Preference Game](01-pathfinding-preference-game/) | [1706](../papers/1706.03741/), [1707](../papers/1707.06347/), [2009](../papers/2009.01325/) | PyTorch + Rich | Done |

---

## Running

Each adventure is self-contained with its own `requirements.txt`. From the repo root:

```bash
source .venv/bin/activate
cd coding-adventures/<adventure-dir>
pip install -r requirements.txt
python app.py
```

All adventures include comprehensive test suites:

```bash
python -m pytest tests/ -v
```
