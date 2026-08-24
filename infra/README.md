# Infrastructure

OralLens runs locally as two production-like containers: a CPU-only FastAPI/ML
service and an Nginx-served React application. Compose starts the frontend only
after the backend has loaded the frozen detector and reports ready.

## Run locally

1. Copy `.env.example` to `.env`.
2. Set `ORALLENS_MODEL_CHECKPOINT_PATH` to the absolute path of the frozen
   `checkpoint_best.pt` file. The file is bind-mounted read-only and is never
   copied into either image.
3. Run `docker compose up --build --wait` from the repository root.
4. Open `http://localhost:8080`.
5. Stop the stack with `docker compose down`.

The defaults expose the frontend and API only on the host loopback interface.
Change `ORALLENS_FRONTEND_ORIGIN` and `ORALLENS_API_BASE_URL` together when the
browser-facing addresses change.

## Runtime boundaries

- The backend runs as UID/GID `10001` with one Uvicorn worker and one concurrent
  inference slot.
- The container uses CPU-only Torch and TorchVision packages. The project-local
  CUDA training environment is unchanged.
- Uploaded inference copies and generated prediction JSON use `/tmp` tmpfs and
  are removed after each request.
- Production mode disables persistent/public scan history.
- Container source/config paths are rooted at `/opt/orallens`; only the absolute
  host checkpoint path varies by machine.
- `/health/live` reports process liveness. `/health/ready` succeeds only after
  the frozen model loads successfully.

This setup is intended for reproducible local verification. It does not add a
cloud platform, GPU runtime, database, Redis, or multi-worker inference.
