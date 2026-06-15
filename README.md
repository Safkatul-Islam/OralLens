# OralLens AI

OralLens AI is an end-to-end AI/ML portfolio project for dental image screening. The app will let a user upload an oral image, run computer vision inference, show confidence scores and visual evidence, and generate a structured AI-assisted screening report.

This project is designed as a production-style resume project, not a notebook-only demo. It will include model development, evaluation, backend serving, frontend visualization, documentation, and deployment-ready structure.

## Project Goal

Build a practical AI system that demonstrates:

- Computer vision model training or fine-tuning
- Image preprocessing and dataset preparation
- Model evaluation with real metrics
- Explainable prediction outputs
- Backend API serving
- Frontend user experience
- Responsible AI boundaries
- Reproducible local development
- Resume-ready documentation

## Responsible AI Scope

OralLens AI is not a diagnostic medical device. It is a learning and screening-support project that explains possible visual findings from sample images. Any real medical or dental concern should be reviewed by a qualified clinician.

## Planned Architecture

```text
User
  |
  v
Frontend upload UI
  |
  v
Backend API
  |
  +--> Image validation and storage
  |
  +--> ML inference pipeline
  |
  +--> Explainability output
  |
  +--> Structured AI report generator
  |
  v
Result dashboard and scan history
```

## Planned Modules

- `frontend/` - upload interface, prediction dashboard, result history
- `backend/` - API endpoints, request validation, inference orchestration
- `ml/` - data preparation, training, evaluation, inference utilities
- `docs/` - project brief, learning notes, model card, architecture notes
- `infra/` - Docker and deployment configuration

## Resume Story

Built an end-to-end AI dental image screening platform with computer vision inference, explainable visual outputs, structured report generation, backend API serving, frontend visualization, evaluation metrics, and production-ready documentation.

## Build Status

Project design phase is in progress.

