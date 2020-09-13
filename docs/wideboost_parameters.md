Wideboost Parameters
====================

`wideboost` introduces two new parameters to Gradient Boosting: `'btype'` and `'extra_dims'`.  The [Overview](overview.md) describes Wide Boosting in more detail. The short summary is that Wide Boosting finds an $F$ to minimize the following expression:
$$
    L(Y,F(X)\beta)
$$
where $L$ is a loss function, $X$ is some inputs, $Y$ is a known output and $\beta$ is a fixed matrix that gets initialized at the start of fitting $F$.

## `'btype'`

The `'btype'` parameter describes how $\beta$ is initialized. It has 4 options:

* `'R'` fills $\beta$ with independent draws from a uniform random variable, $U(0,1)$.
* `'Rn'` fills $\beta$ with independent draws from a uniform random variable, $U(0,1)$ and normalizes each column of $\beta$ so that the columns of $\beta$ sum to one.
* `'I'` fills the first several rows and columns of $\beta$ with an identity matrix. The remaining entries are independent draws from a uniform random variable, $U(0,1)$.
* `'I'` fills the first several rows and columns of $\beta$ with an identity matrix. The remaining entries are independent draws from a uniform random variable, $U(0,1)$. The columns are then normalized to that each column of $\beta$ sums to one.

Based on our empirical experiments, we haven't found a consistent pattern as to which `'btype'` works best, so we recommend at least trying a couple to see which works.  Performance of `wideboost` is not extremely sensitive to this parameter, but it can make a difference when trying to get the best possible performance.

## `'extra_dims'`

The `'extra_dims'` parameter is an integer $\geq 0$. It controls how "wide" $F$ is.  If $F(X) \in \mathbb{R}^{1\times q}$, $\beta \in \mathbb{R}^{q \times d}$ and $Y \in \mathbb{R}^d$, the `'extra_dims'` parameter determines how much larger $q$ is than $d$.

Specifically $d + \mbox{extra_dims} = q$; thus `'extra_dims'` is exactly how many extra dimensions $F(X)$ has compared to $Y$.  Larger `'extra_dims'` gives a wider model.

We have found that wider models usually lead to better performance.  However, given that the current backend packages, `LightGBM` and `XGBoost`, fit additional trees for every added dimension, increasing `'extra_dims'` increases model fitting times.

## Notes

If `'btype' = 'I'` and `'extra_dims' = 0`, then Wide Boosting is equivalent to Gradient Boosting.