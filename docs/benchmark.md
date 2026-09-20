# Benchmark protocol

The first benchmark asks a narrow question:

**When a known visual transformation is introduced, does the measured drift signal track actual task degradation?**

## Protocol

1. Choose a fixed reference video window.
2. Run the baseline CV task on the unmodified frames.
3. Record the baseline task metric.
4. Apply one controlled appearance transformation at several severities.
5. Run the same CV task on each transformed window.
6. Extract the same features from reference and transformed windows.
7. Compute the drift statistic.
8. Record drift score, task metric, and task-metric delta.
9. Repeat across seeds where stochastic transformations are used.
10. Store raw results as CSV for later statistical analysis.

## Conditions

The initial generator contains brightness, Gaussian noise, blur, and contrast transformations.

These are benchmark controls, not claims that they represent every real deployment shift.

## Interpretation

The first result should be a relationship between:

- transformation severity,
- measured distributional drift,
- and downstream task degradation.

A high drift score with little task degradation is a useful negative case. A modest drift score with substantial task degradation is also important.

The benchmark should therefore not optimize thresholds before this relationship is measured.
