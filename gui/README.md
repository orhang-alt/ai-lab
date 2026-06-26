# AI Lab GUI

A local Streamlit dashboard over the lab. It's a thin, pure-Python layer that
*reads the lab from disk* and *reuses `core/`* — it never duplicates lab logic.

## Launch
```bash
cd ai-lab
./start.sh            # background, waits until healthy, prints the URL
./stop.sh             # kills the process and frees the port
./start.sh 8600       # custom port (stop with: ./stop.sh 8600)
```
Or run it in the foreground manually:
```bash
source .venv/bin/activate
pip install -e ".[gui]"          # streamlit + matplotlib + pandas (already installed)
streamlit run gui/app.py
```
Opens at http://localhost:8501. `start.sh` writes a PID to `.gui.pid` and logs to
`.gui.log` (both git-ignored).

## Screens
- **Dashboard** — scans every experiment (runs `run.py`; 🟢 done / ⚪ todo / 🔴 error),
  shows tier progress, embeds `ROADMAP.md`.
- **Playground** — interactive single neuron wired to `core.neuron.Neuron`: sliders
  for w₀/w₁/bias, activation picker, gate presets, live response surface + decision
  boundary + truth table.
- **Experiments** — per-experiment Overview (README), Code, Run (captures stdout),
  and an editable Notes tab saved back to `notes.md`.
- **Infobase** — renders the `infobase/*.md` notes.
- **Tests** — runs `pytest -q` and shows passed/skipped/failed.

## Layout
```
gui/
├── app.py            # entrypoint: st.navigation over the views
├── lab.py            # disk discovery + run helpers (the only "model" layer)
└── views/            # one file per screen
    ├── dashboard.py
    ├── playground.py
    ├── experiments.py
    ├── infobase.py
    └── tests.py
```

## Extending
Adding an experiment under `experiments/tierN_*/eNN_*/` makes it appear
automatically (Dashboard, Experiments). New `infobase/*.md` notes appear in the
Infobase screen with no code change.
