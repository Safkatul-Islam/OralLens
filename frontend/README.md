# Frontend

The frontend is a Vite + React + TypeScript interface for uploading an oral
image to the backend and reviewing the structured screening-support response.

## Current MVP

- Select and preview a JPEG or PNG image supported by the ML-backed MVP.
- Submit the image to `POST /scans`.
- Display prediction label, confidence, severity, model name, and mock/model
  mode.
- Draw returned detection boxes over the uploaded image.
- Display evidence summary, report text, limitations, next steps, and disclaimer.
- Show loading and safe error states.

## Runtime Configuration

The frontend uses `VITE_API_BASE_URL` and defaults to:

```text
http://127.0.0.1:8000
```

Example:

```powershell
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
npm run dev
```

## Local Development

Install dependencies from inside `frontend/`:

```powershell
npm install
```

Run the app:

```powershell
npm run dev
```

From the repository root, the checked-in helper script starts the frontend with
the default backend URL:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File frontend\scripts\run-dev.ps1
```

Build for production:

```powershell
npm run build
```

## Quality Checklist

### Is the implementation secure?

The browser only sends the selected file to the configured backend API. Model
paths, checkpoint paths, and output paths are never accepted from the frontend.
Backend validation remains the security boundary for upload type and size.

### Is the code clean and efficiently written?

The current MVP keeps the UI in one small React entrypoint because there is only
one workflow. The response contract is typed, and the detection overlay is
derived from backend coordinates instead of duplicating inference logic.

### Is the documentation clear?

This README documents setup, runtime configuration, current behavior, and
limitations. The API contract lives in `../docs/API.md`.

### Are there enough tests?

The frontend currently has a production build gate. API behavior and inference
contract are covered by backend tests. Component tests can be added when the UI
surface grows beyond the single MVP workflow.

## Status

MVP upload/result workflow is implemented and production build passes.
