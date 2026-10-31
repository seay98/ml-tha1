# Machine Learning — THA1 (Group 17)

Linear and quadratic regression of electrical power output against temperature, using Group 17's 20 observations. Data is included directly in the scripts.

- `code/a.i.py`: linear regression using least squares and gradient descent.
- `code/a.ii.py`: quadratic regression using gradient descent.

## Run

The environment and dependencies are managed with `uv`. Requires `uv` and Python 3.13 or newer. A.I also requires Tk support and a graphical display.

From the project directory, install dependencies and run:

```bash
uv sync
cd code
uv run a.i.py
uv run a.ii.py
```

Figures are saved in the current working directory (`code/` when using the commands above).
