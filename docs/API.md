# OralLens AI API

## Scope

The FastAPI backend accepts oral-image uploads and returns experimental screening-support records. It does not return diagnoses, calibrated disease probabilities, treatment advice, or clinical rule-out decisions.

Default local base URL:

`http://127.0.0.1:8000`

## Runtime modes

| Mode | Default | Behavior |
|---|---:|---|
| `mock` | yes | deterministic placeholder for API development and tests |
| `ml` | no | frozen Faster R-CNN detector through the project-local ML package |

`backend\scripts\run-ml-server.ps1` selects ML mode and the final prediction config:

`ml/configs/orthodontic_plaque_detection_mvp_v4_originals_online_aug_predict.toml`

The active ML contract is epoch 9, threshold `0.80`, maximum 100 detections, and model identity `orthodontic-plaque-mvp-v4-originals-online-aug-epoch9`.

## `POST /scans`

Creates one scan record.

Request:

- content type: `multipart/form-data`
- field name: `file`
- API metadata allowlist: JPEG, PNG, or WebP
- ML runtime formats: JPEG or PNG
- default maximum upload size: 5 MiB

The backend verifies MIME type, filename extension, size, non-empty content, and magic bytes. The filename is display metadata only and is reduced to a basename.

Example response shape:

```json
{
  "id": "scan-uuid",
  "created_at": "2026-08-21T00:00:00Z",
  "original_filename": "oral-image.jpg",
  "content_type": "image/jpeg",
  "size_bytes": 123456,
  "sha256": "content-digest",
  "prediction": {
    "label": "possible_plaque",
    "display_name": "Plaque-positive peri-tooth region candidate",
    "confidence": 0.93,
    "severity": "medium",
    "is_mock": false,
    "model_name": "orthodontic-plaque-mvp-v4-originals-online-aug-epoch9",
    "prediction_count": 1,
    "detections": [
      {
        "box_xyxy": [120.0, 90.0, 210.0, 180.0],
        "label": 1,
        "score": 0.93
      }
    ]
  },
  "input_assessment": {
    "status": "supported",
    "reason_codes": [],
    "summary": "The image passed the configured technical-quality checks.",
    "image_width": 1024,
    "image_height": 768,
    "mean_luminance": 0.52,
    "luminance_stddev": 0.18
  },
  "evidence": {
    "kind": "summary",
    "summary": "Detected 1 plaque-positive peri-tooth region candidate with the detector."
  },
  "report": {
    "title": "AI Screening Support Report",
    "summary": "Experimental model output.",
    "limitations": ["This is not a diagnosis."],
    "recommended_next_steps": ["Consult a licensed dental professional for concerns."],
    "disclaimer": "OralLens AI is not a medical device."
  }
}
```

`prediction.confidence` is the maximum returned detector score. It is not a probability of plaque, disease, severity, risk, or patient outcome.

### Technical abstention

Before model construction, ML mode applies:

- EXIF orientation normalization
- minimum short side: 256 pixels
- maximum decoded pixels: 30,000,000
- mean luminance bounds: 0.05 to 0.98
- minimum luminance standard deviation: 0.02

If the image fails, the API still returns HTTP `201` with:

- `input_assessment.status = "unsupported"`
- one or more reason codes
- `prediction = null`
- retake guidance

This is an abstention, not a plaque-free finding.

### Supported zero-box response

If the image passes technical checks but no region reaches `0.80`, `prediction` remains present with:

- label `no_detection`
- confidence `0.0`
- zero detections
- explicit text that this does not establish plaque absence

## Read endpoints

### `GET /scans`

Returns all locally stored scan records in the typed list response.

### `GET /scans/{scan_id}`

Returns one record or HTTP `404` when the identifier is unknown.

## Error behavior

| Status | Meaning |
|---:|---|
| `400` | empty upload, signature mismatch, corrupt supported image, or invalid input |
| `413` | upload exceeds configured byte limit |
| `415` | unsupported MIME/extension pair or format unsupported by ML mode |
| `422` | FastAPI request validation failure |
| `404` | scan identifier not found |
| `500` | unexpected internal failure with generic client detail |

Errors use a structured envelope with a concise code/detail and request ID where available. Raw stack traces, filesystem paths, and exception internals are not returned.

## CORS

Default allowed origins:

- `http://127.0.0.1:5173`
- `http://localhost:5173`

Allowed methods and headers are explicit; wildcard CORS is not used.

## Persistence and privacy

The local JSON store records filename metadata, content type, size, digest, inference output, assessment, and report. Uploaded image bytes are not persisted by `JSONScanStore`; the ML temporary copy is removed after inference. Local records are ignored by git.

This storage design is for a single-user demonstration. It is not authenticated, encrypted clinical storage or a regulated retention system.

## Evidence boundary

Class `1` means a plaque-positive peri-tooth region under the AIRC annotation contract. The API wording does not claim discrete visible-plaque segmentation or diagnosis. See [Training and evaluation](TRAINING.md) for measured model performance and limitations.
