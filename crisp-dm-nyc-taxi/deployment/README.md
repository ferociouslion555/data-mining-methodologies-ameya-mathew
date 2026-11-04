# Deployment prototype

Run locally after installing the project requirements:

```bash
python train_model.py
uvicorn app:app --reload
```

Endpoints:

- `GET /health` reports whether the model artifact is loaded.
- `POST /predict` accepts validated trip features and returns an estimated duration.

The model artifact is trained by `train_model.py` from the raw sample and should be regenerated whenever the training data or feature contract changes. This is an educational prototype; it has not passed production integration, load, security, monitoring, or rollback testing.
