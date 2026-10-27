# Machine Learning — THA1 (Group 17)

Simple linear regression of net hourly electrical output against temperature, using 20 observations.

`code/a.sl-regression.py` includes Z-score standardization, least-squares regression, and gradient descent with cost convergence tracking. It saves plots of the fitted lines and cost history in the current working directory.

## Run

Requires Python, NumPy, Matplotlib, and Tk support for the configured `TkAgg` backend.

```bash
pip install numpy matplotlib
python code/a.sl-regression.py
```
