# Tokamak Particle Orbit Explorer

Interactive Python demonstration of charged-particle orbits in tokamak and magnetic-mirror magnetic fields, using the Boris pusher and a guiding-center comparison.

## Features

- Boris particle pusher for arbitrary electromagnetic fields.
- Tokamak (large-aspect-ratio, circular) and magnetic-mirror field models.
- Guiding-center drift approximation for comparison.
- Numerical verification: energy conservation, magnetic-moment conservation, second-order convergence.
- Interactive matplotlib GUI.
- One-click validation figure suite.
- Jupyter notebook report.

## Install

```bash
pip install -r requirements.txt
```

## Run

Interactive GUI:
```bash
python -m src.gui
```

Generate validation figures:
```bash
python scripts/validation.py
```

Run tests:
```bash
pytest
```

Open the report:
```bash
jupyter notebook notebooks/demo.ipynb
```

## Project structure

- `src/fields.py` — magnetic and electric field definitions.
- `src/particles.py` — particle state.
- `src/integrators.py` — Boris and guiding-center steppers.
- `src/diagnostics.py` — energy, magnetic moment, classification, convergence.
- `src/gui.py` — interactive orbit explorer.
- `scripts/validation.py` — batch validation figure generator.
- `notebooks/demo.ipynb` — presentation report.
- `tests/` — unit tests.
