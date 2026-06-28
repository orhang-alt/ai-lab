# Tests

The safety net that proves `core/` is correct as it grows.

```bash
pytest -q                         # run all
pytest tests/test_e01_neuron.py   # one file
```

- `test_e01_neuron.py` — passes now: the Tier 0 neuron implements AND/OR/NOT.
- `test_engine_gradcheck.py` — numerically verifies your autograd gradients.
- `test_core_math.py` — checks losses, optimizers, initializers, and MLP parameter
  plumbing.

As you promote code into `core/`, add a test here. Gradient checking
(`test_engine_gradcheck.py`) is the canonical way to validate backprop.
