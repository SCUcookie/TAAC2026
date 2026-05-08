# Strategy

## Current Direction

- Keep the baseline architecture stable unless a small change clearly improves validation.
- Treat sequence truncation as a first-class experiment axis.
- Use short smoke runs before committing to long training runs.
- Record every result so the next decision is based on evidence, not memory.

## Promotion Criteria

- The checkpoint must complete local inference successfully.
- The inference output must match the required JSON format.
- Validation improvement should be stable, not a one-off spike.
- Do not spend a scarce evaluation slot on an unverified checkpoint.
