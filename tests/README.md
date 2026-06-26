# Tests

The safety net that proves `core/` is correct as it grows.

```bash
pytest -q                         # run all
pytest tests/test_e01_neuron.py   # one file
```

- `test_e01_neuron.py` — passes now: the Tier 0 neuron implements AND/OR/NOT.
- `test_engine_gradcheck.py` — **skips** until you implement `core/engine.py`
  (e04), then numerically verifies your autograd gradients.

As you promote code into `core/`, add a test here. Gradient checking
(`test_engine_gradcheck.py`) is the canonical way to validate backprop.
