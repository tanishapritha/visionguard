# Research plan

## Central hypothesis

Semantic video reasoning may make visual-drift monitoring more operationally useful by identifying what changed, grounding explanations in temporal evidence, and estimating whether a change is relevant to downstream task performance.

A second hypothesis is that drift and uncertainty signals can selectively invoke expensive VLM reasoning while preserving acceptable task quality.

## Experimental rules

- Treat drift detection and harmful task degradation as separate variables.
- Measure downstream task performance under every controlled drift condition.
- Compare CV-only, VLM-always, and adaptive routing.
- Record VLM invocation rate and latency.
- Store exact frames used to support each VLM claim.
- Include large-drift/stable-task cases.
- Include degraded-task cases without a large aggregate drift score.
- Do not treat a VLM explanation as evidence unless cited frames support the claim.

## Primary metrics

Task: precision, recall, F1, mAP where applicable.

Drift: detection delay, false alarm rate, drift score versus task degradation.

System cost: VLM invocation rate, latency, total inference time, estimated model cost.

Evidence: supported, contradicted, and insufficient-evidence rates.

## First benchmark

Start with controlled appearance drift. Establish the relationship between known transformations, model outputs, drift statistics, and task performance before introducing semantic or behavioral drift.
