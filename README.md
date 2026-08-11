python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

Production start command:

gunicorn -k uvicorn.workers.UvicornWorker main:app

## showdown.py

A separate app for the SHOWDOWN bot challenge (POST /move, GET /health). Runs
independently from main.py, on its own port:

uvicorn showdown:app --reload --port 5000

Expose it over HTTPS for the coordinator to reach, e.g.:

cloudflared tunnel --url http://localhost:5000
# or
ngrok http 5000

Production start command:

gunicorn -k uvicorn.workers.UvicornWorker showdown:app
