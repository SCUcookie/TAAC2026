# Workflow Overview

## Goal

Train a model from user-provided training code, then evaluate it with user-provided evaluation code.

## Competition Constraints

- Training has no fixed attempt limit, but each run is time-consuming.
- The platform allows at most two models to train at the same time.
- Evaluation is limited to three successful submissions per day.
- Each successful evaluation returns both a score result and terminal logs.

## Operating Rules

- Keep training code and evaluation code clearly separated.
- Record every meaningful run in chronological order.
- Promote only checkpoints that have been locally verified first.
- Keep the workflow documentation in English.
