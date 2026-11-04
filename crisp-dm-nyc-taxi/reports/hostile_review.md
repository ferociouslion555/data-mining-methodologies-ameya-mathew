# Hostile review

- The data sample is not evidence of representativeness for the full population.
- One future-month holdout is not enough to establish long-term generalization.
- The analysis does not prove that any feature causes shorter or longer trips.
- Destination-based features are valid only when the destination is known at prediction time.
- The API contract is documented, but a production integration test, load test, and rollback test are still required.
- Claims are limited to the declared sample, feature set, and evaluation protocol.

## Evaluation-specific objections

- The June holdout is one future period, not evidence of year-round performance.
- The long-trip segment is systematically underpredicted, so the aggregate MAE hides an important operational weakness.
- The model uses destination coordinates, which are acceptable only when the destination is known at request time; a street-hail or pickup-only system would need a separate feature contract.
- No confidence or prediction intervals were evaluated.
- No ablation proves that haversine distance, cyclic time, or any other feature caused the observed improvement.
- No traffic, weather, event, road-network, borough, or neighborhood covariates were available.
