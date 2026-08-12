# OralLens AI Frontend

React, TypeScript, and Vite interface for the OralLens AI experimental screening-support workflow.

The UI lets a user select an oral JPEG or PNG, preview it, submit it to FastAPI, overlay returned candidate boxes, and review prediction metadata, evidence, limitations, and next steps. It never presents the output as a diagnosis or treatment recommendation.

## Current behavior

- accepts JPEG and PNG through the file picker
- rejects mismatched MIME type/filename-extension combinations before transmission while retaining backend validation as the authoritative security boundary
- previews the selected image locally
- sends multipart field `file` to `POST /scans`
- reports concise backend validation failures
- displays model name, raw three-decimal detector/mock score, severity, detection count, and image size
- labels the displayed score as an experimental ranking value, not a clinical probability
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

## Automated browser and accessibility tests

The Playwright suite runs against Chromium and starts/stops an in-process Vite server. API calls are intercepted at the browser boundary, so the suite does not require a running backend, checkpoint, dataset, or patient image. Test uploads are generated in memory.

Install the project-local browser once after `npm ci`:

```powershell
Push-Location frontend
$env:PLAYWRIGHT_BROWSERS_PATH = "0"
npx.cmd playwright install chromium
Pop-Location
```

Run the suite:

```powershell
Push-Location frontend
$env:PLAYWRIGHT_BROWSERS_PATH = "0"
npm.cmd test
Pop-Location
```

Coverage includes:

- initial semantics and automated WCAG 2.2 A/AA checks with axe-core
- keyboard order, visible upload focus, and minimum action-target height
- no-file and unsupported-file validation before network transmission
- image selection, preview, loading/disabled state, mocked HTTP `201`, v3 identity, raw score semantics, report content, and SVG overlay geometry
- structured backend errors and a safe fallback for non-JSON failures
- live status announcements for scanning and completion

Failure screenshots, videos, traces, and the HTML report are written below `frontend/test-results` and `frontend/playwright-report`; both paths are ignored by git.

Latest automated browser suite: `5 passed` in Chromium.

## Verification status

The real browser-to-backend-to-v3 path has been verified manually:

- request returned HTTP `201`
- the result showed a real MVP detector response
- the result identified model `orthodontic-plaque-mvp-v3`
- 15 returned detections produced 15 SVG overlay boxes
- the raw detector score and non-probability qualifier rendered without percentage conversion
- evidence, report, limitations, next steps, and disclaimer rendered
- no application-origin console errors were observed

This single scan proves integration behavior, not model quality or clinical reliability.

The automated suite verifies the isolated browser/API contract with mocked responses. The manual real-ML scan verifies integration with the local backend and checkpoint. Neither result measures model quality or clinical reliability.

## Accessibility and safety notes

- upload, preview, result, and error regions use semantic labels or roles
- decorative icons are hidden from assistive technology
- errors are surfaced in an alert region
- scanning and completion are announced through a polite status region
- upload and action controls have visible keyboard focus, and the action meets a 44-pixel minimum height
- the loading animation is disabled when reduced motion is requested
- result copy repeats the non-diagnostic boundary and appropriate professional follow-up

Automated axe checks detect only a subset of accessibility defects. Manual verification is still required with a screen reader such as NVDA, browser zoom/reflow, Windows high-contrast mode, touch input, and representative small screens. The visual box overlay is intentionally hidden from assistive technology because the same detection count and report are presented as text; future clinically meaningful spatial descriptions would require a separate accessible design.

## Related documents

- [Root project guide](../README.md)
- [Pipeline](../docs/PIPELINE.md)
- [Architecture](../docs/ARCHITECTURE.md)
- [API contract](../docs/API.md)
- [Backend guide](../backend/README.md)
