# Installation

DeepEntry release utilities require Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

The large data archive is not stored in the code repository. Download and unpack the companion archive (https://doi.org/10.5281/zenodo.19994995) next to this repository, or set `DEEPENTRY_DATA_ROOT` and `DEEPENTRY_MODEL_ROOT` to the relevant locations.
