# VisionGuard

**Evidence-grounded, drift-aware video intelligence for deployed vision models.**

VisionGuard studies a practical reliability problem in deployed video systems:

> A vision model can continue running while the visual environment changes underneath it.

The system detects distributional change, measures whether that change correlates with task degradation, selectively invokes video-language reasoning for investigation, and attaches temporal evidence to the resulting claims.

## Research focus

VisionGuard is built around four questions:

1. Which drift signals best predict downstream model degradation?
2. Can a video-language model characterize detected drift more usefully than statistical signals alone?
3. Can claims about drift be grounded in temporally localized video evidence?
4. Can drift and uncertainty selectively route expensive reasoning without unacceptable loss in task quality?

The project treats these as empirical questions. No component is assumed to improve performance before evaluation.

## Experimental loop

```text
reference video
      |
      v
baseline CV model
      |
      +--------------------+
      |                    |
      v                    v
feature/output logs   controlled drift
      |                    |
      +---------+----------+
                v
          drift detection
                |
                v
       drift-to-performance
            analysis
                |
                v
          routing policy
           /          \
         CV       VLM investigation
                     |
                     v
             temporal evidence
                     |
                     v
             claim verification
```

## Status

Early research prototype. The first milestone is a reproducible benchmark connecting controlled visual drift to actual downstream model performance.

## Repository structure

- `src/visionguard/drift` — drift statistics and aggregation
- `src/visionguard/routing` — adaptive inference decisions
- `src/visionguard/evidence` — evidence and claim schemas
- `src/visionguard/vlm` — model-provider interface
- `benchmarks` — benchmark generation and evaluation
- `configs` — experiment configuration
- `tests` — unit and integration tests
- `docs` — research notes and methodology

## License

MIT
