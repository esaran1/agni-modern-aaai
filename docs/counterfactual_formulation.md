# Counterfactual Formulation

Wildfire severity prediction under partial observation is a potential-outcomes problem.

For each patch-date pair `(p, t)`, define the potential severity outcomes:

- `Y(1)_(p,t)`: severity if a fire occurs.
- `Y(0)_(p,t) = 0`: severity if a fire does not occur.

Let `D_(p,t)` be the fire-occurrence indicator. The observed outcome is:

`Y_obs = D * Y(1) + (1 - D) * Y(0)`

The central difficulty is that `Y(1)` is only observed when `D = 1`. For non-fire rows, the severity that would have occurred under ignition is counterfactual.

## Selection Bias

Naive severity regression on the observed burned subset estimates:

`E[Y(1) | D = 1, X]`

This differs from the target quantity:

`E[Y(1) | X]`

because patches that ignite are systematically different from those that do not.

## Propensity Weighting

Define the propensity score:

`e(X) = P(D = 1 | X)`

Inverse-propensity weighting recovers the full-population treated expectation:

`E_IPW[Y(1)] = E[D * Y_obs / e(X)]`

Operationally, this means weighting severity regression losses by `1 / e(X)` for fire-positive rows. Rare fires with low propensity receive more weight because they are underrepresented in the observed severity sample.

## Expected Risk

Expected wildfire risk is:

`R(X) = P(D = 1 | X) * E[Y(1) | X]`

This directly motivates:

- Contribution 1: estimate `E[Y(1) | X]` with inverse-propensity correction.
- Contribution 2: optimize the composed risk score `R(X)` directly.
- Contribution 3: attach finite-sample uncertainty guarantees to `R(X)`.

## Future Work

A doubly robust estimator would combine outcome regression and IPW:

`E_DR[Y(1)] = E[m(X) + D * (Y_obs - m(X)) / e(X)]`

where `m(X)` is the severity model. This is intentionally left as future work.
