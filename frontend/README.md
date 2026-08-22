# OralLens AI Frontend

React, TypeScript, and Vite interface for the OralLens AI experimental screening-support workflow.

The UI accepts an oral JPEG or PNG, shows a local preview, submits it to FastAPI, overlays returned candidate boxes, and presents model metadata, evidence, limitations, and next steps. It does not present model output as diagnosis or treatment advice.

## Current behavior

- validates JPEG/PNG MIME-and-extension pairs before transmission
- retains backend validation as the authoritative boundary
- previews the selected image locally
- sends multipart field `file` to `POST /scans`
- displays concise request and backend failures
- renders model name, raw detector score, severity, count, and image size
- labels the score as an experimental ranking value, not a clinical probability
- draws pixel-space `xyxy` boxes over the natural image coordinate system
- displays evidence, report, limitations, next steps, and disclaimer
- renders technical abstention without a detector score or plaque-free claim

Default API URL: `http://127.0.0.1:8000`. Override it with `VITE_API_BASE_URL`.

## Install

```powershell
Push-Location frontend
npm.cmd ci
Pop-Location
```

`npm ci` uses the committed lockfile and installs only project-local dependencies.

## Run

From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File frontend\scripts\run-dev.ps1
```

The script starts Vite on `http://127.0.0.1:5173`. Start the model-backed API separately:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File backend\scripts\run-ml-server.ps1
```

Do not assume either server is already running.

## Production build

```powershell
Push-Location frontend
npm.cmd run build
Pop-Location
```

Generated `frontend/dist` output is ignored by git.

## Browser and accessibility tests

The Playwright suite starts an in-process Vite server and intercepts API requests, so it does not require the backend, checkpoint, dataset, or patient images. Test uploads are generated locally.

Install Chromium locally after `npm ci`:

```powershell
Push-Location frontend
$env:PLAYWRIGHT_BROWSERS_PATH = "0"
npx.cmd playwright install chromium
Pop-Location
```

Run:

```powershell
Push-Location frontend
$env:PLAYWRIGHT_BROWSERS_PATH = "0"
npm.cmd test
Pop-Location
```

Coverage includes:

- initial semantics and automated WCAG A/AA rules with axe-core
- keyboard order, visible focus, and minimum action-target height
- no-file and unsupported-file validation
- preview, loading state, mocked HTTP `201`, model metadata, score semantics, report content, and SVG geometry
- structured backend errors and safe non-JSON fallback
- technical abstention rendering
- live status announcements and reduced-motion behavior

Latest automated browser suite: `5 passed` in Chromium. Latest production build: passed.

## Real-model verification boundary

The final backend-to-model smoke used the frozen epoch-9 checkpoint and returned HTTP `201`, final model identity, and 14 class-1 detections above `0.80`. The API response contract consumed by the frontend did not change during model promotion.

The earlier manual browser-to-real-model scan established the full visual overlay path before final promotion. A browser run proves integration behavior only; it does not measure model quality or consumer-image reliability.

## Accessibility and safety notes

- semantic upload, status, error, result, and report regions
- decorative icons hidden from assistive technology
- validation failures surfaced in an alert region
- scanning and completion announced through a polite live region
- visible keyboard focus and 44-pixel minimum action height
- reduced-motion handling
- textual count/report alongside the decorative box overlay
- repeated non-diagnostic and non-probability wording

Automated axe checks do not replace manual NVDA, zoom/reflow, Windows high-contrast, touch, and representative small-screen testing.

Related: [Root guide](../README.md), [Pipeline](../docs/PIPELINE.md), [Architecture](../docs/ARCHITECTURE.md), [API](../docs/API.md), and [Backend guide](../backend/README.md).
