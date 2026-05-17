# Installation

DeepEntry release utilities require Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

To load released model checkpoints, install the optional model dependency:

```bash
pip install -e ".[model]"
```

The large data archive is not stored in the code repository. Download and unpack the companion archive next to this repository, or set `DEEPENTRY_DATA_ROOT` and `DEEPENTRY_MODEL_ROOT` to the relevant locations.

Add the final public archive DOI to this file once the repository deposition is complete.
