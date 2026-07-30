# OralLens AI Frontend

React, TypeScript, and Vite interface for the OralLens AI experimental screening-support workflow.

The UI lets a user select an oral JPEG or PNG, preview it, submit it to FastAPI, overlay returned candidate boxes, and review prediction metadata, evidence, limitations, and next steps. It never presents the output as a diagnosis or treatment recommendation.

## Current behavior

- accepts JPEG and PNG through the file picker
- previews the selected image locally
- sends multipart field `file` to `POST /scans`
- reports concise backend validation failures
- displays model name, confidence, severity, detection count, and image size
- draws returned `xyxy` boxes over the image's natural coordinate system
- displays evidence, report text, limitations, next steps, and disclaimer
- labels mock and real-model output distinctly

The default API URL is `http://127.0.0.1:8000`. Override it with `VITE_API_BASE_URL` when needed.

## Requirements

- Node.js compatible with the versions pinned in `package-lock.json`
- project-local dependencies under `frontend/node_modules`
- running OralLens AI backend

## Install

From the repository root:

```powershell
Push-Location frontend
npm.cmd ci
Pop-Location
```

`npm ci` uses the committed lockfile and avoids modifying the resolved dependency graph during normal setup.

## Run locally

Start the frontend from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File frontend\scripts\run-dev.ps1
```

The script sets the API URL and starts Vite on `http://127.0.0.1:5173`. The backend CORS allowlist includes that origin and `http://localhost:5173`.

Start the ML-backed API separately:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File backend\scripts\run-ml-server.ps1
```

Do not assume either server is already running when beginning a new session.

## Production build

```powershell
Push-Location frontend
npm.cmd run build
Pop-Location
```

This runs TypeScript project compilation and the Vite production build. Generated `frontend/dist` output is ignored by git.

Latest production build: passed.

## Verification status

The real browser-to-backend-to-v2 path has been verified manually:

- request returned HTTP `201`
- the result showed a real MVP detector response
- 17 returned detections produced 17 SVG overlay boxes
- evidence, report, limitations, next steps, and disclaimer rendered
- no application-origin console errors were observed

This single scan proves integration behavior, not model quality or clinical reliability.

There is currently no dedicated frontend unit/component test suite. The TypeScript production build and manual browser E2E cover the current small interface, but automated interaction and accessibility tests remain a documented portfolio gap.

## Accessibility and safety notes

- upload, preview, result, and error regions use semantic labels or roles
- decorative icons are hidden from assistive technology
- errors are surfaced in an alert region
- result copy repeats the non-diagnostic boundary and appropriate professional follow-up

Future UI work should test keyboard flow, screen-reader output, color contrast, reduced motion if animation is introduced, box-label accessibility, and small-screen behavior.

## Related documents

- [Root project guide](../README.md)
- [Pipeline](../docs/PIPELINE.md)
- [Architecture](../docs/ARCHITECTURE.md)
- [API contract](../docs/API.md)
- [Backend guide](../backend/README.md)
