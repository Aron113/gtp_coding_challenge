## Run locally

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

## Endpoints

- `POST /square`
- `POST /move` for the SHOWDOWN Phase 1 bot
- `GET /health`

## Production start command

gunicorn -k uvicorn.workers.UvicornWorker main:app
