# Model card: OULAD early warning (LightGBM)

## Intended use
Flag enrolled learners at day 28 for supportive outreach by instructors.
Not for grading, admissions, or any punitive decision.

## Training data
OULAD 2013B and 2013J presentations; students still enrolled at day 28 (11,809 rows).

## Features
Behaviour before day 28 (VLE activity, assessments, registration timing, prior attempts,
credits, module). Disability is excluded from inputs and used only for auditing.

## Metrics (from MLflow, local run)
| Metric | Validation (v1) | 2014 stream (mean over 13 batches, champion at the time) |
| --- | --- | --- |
| ROC AUC | 0.785 | 0.752 |
| Recall at threshold | 0.800 | 0.771 |
| Recall, disability = Y (n) | 0.843 (246) | see `data/state/stream_state.json` |
| Recall, disability = N (n) | 0.795 (2117) | see `data/state/stream_state.json` |
| Recall gap | 0.048 | 0.048 |

## Fairness gate
A new version ships only if its recall gap between disability groups is at most 0.10.
In the local replay, two challengers with higher AUC were blocked by this gate.

## Known limitations
Adult distance learners, 2013-2014; coarse self-declared disability flag;
simulated immediate labels; clicks are a proxy for engagement; per-batch group sizes can be small,
so single-batch gaps are noisy.
