# Installation

DeepEntry requires Python 3.9 or newer and PyTorch (>= 2.0).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Optional development dependencies (tests):

```bash
pip install -e ".[dev]"
pytest
```

The large data archive is not stored in the code repository. Download the
companion archive from the GitHub Release assets
(https://github.com/adistman/DeepEntry/releases) and unpack it. The LOVO56
configuration and training script use paths relative to the archive root, so
run them from inside the unpacked archive directory.
