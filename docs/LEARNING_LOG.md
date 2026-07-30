# Engineering Learning Log

This log records the major decisions and lessons behind the current OralLens AI MVP. It is intentionally concise; detailed contracts and metrics live in their dedicated documents.

## 1. Start with an honest product boundary

The initial goal was narrowed from broad oral-health detection to experimental screening support. The language, schemas, UI, and reports consistently avoid diagnosis, treatment, clinical-validation, and medical-device claims.

**Lesson:** responsible AI starts with the intended-use boundary, not a disclaimer added after implementation.

## 2. Build a thin end-to-end vertical slice

The first usable architecture separated FastAPI routes, schemas, services, storage, and an inference protocol. A deterministic mock adapter enabled API and frontend development before a trained checkpoint was ready.

**Lesson:** a stable inference contract lets product engineering proceed without coupling routes to one model implementation.

## 3. Treat uploads and local paths as trust boundaries

Upload handling added bounded reads, MIME/extension/magic-byte checks, safe display filenames, hashing, explicit CORS, structured errors, and constrained ML temporary paths. Dataset code separately validates manifest paths, containment, missing files, and symlinks.

**Lesson:** ML applications inherit normal application-security risks plus model/data-specific filesystem risks.

## 4. Admit data deliberately

Only the verified Part 2 orthodontic-plaque material entered the workflow. Part 1 was excluded after its nested archive failed official integrity validation. Patient-aware split assignments were created before final evaluation.

**Lesson:** more data is not automatically better when integrity, license, label meaning, or leakage controls are uncertain.

## 5. Make the baseline small but complete

The v1 Faster R-CNN run exercised preparation, loading, training, validation, threshold selection, checkpointing, prediction, and tests. Its F1 was weak, but it provided a reproducible baseline and exposed false-positive behavior.

**Lesson:** a bounded baseline is valuable when it tests the whole scientific and software pipeline and its limitations are reported honestly.

## 6. Improve evidence without using the test split for decisions

The v2 run increased the bounded training work to three epochs and 512 batches per epoch. Loss decreased across all epochs. A complete validation sweep—not the held-out test—selected threshold `0.65`.

**Lesson:** the test split is a measurement boundary. Threshold and model decisions belong to training/validation evidence.

## 7. Diagnose data-contract failures before modifying data

The first held-out run failed on a rotated sample whose center/size values were individually normalized but whose derived corners crossed the boundary. A structured full-manifest audit found only two annotations in one test image, with a maximum crossing near `5e-7` and no post-clipping degeneration.

The fix introduced a documented `1e-6` tolerance, explicit TorchVision clipping, positive-area revalidation, and rejection of gross violations. It also translated shared target errors into evaluation-domain errors so the CLI no longer exposed a raw traceback for expected validation failures.

**Lesson:** do not clamp, skip, or delete a failing sample until the scope and cause are measured. The fix belongs at the shared target boundary and must preserve strict validation.

## 8. Freeze the operating point and measure held-out performance once

At threshold `0.65` and IoU `0.5`, complete validation F1 was `0.7583`. The frozen held-out test then achieved precision `0.7582`, recall `0.7037`, F1 `0.7299`, and mean matched IoU `0.7541`.

**Lesson:** a lower held-out result is information, not permission to tune on the test set.

## 9. Promote models through configuration boundaries

The backend was moved from v1 to v2 through its typed config and startup scripts. Route and service contracts did not change. A real integration smoke, backend suite, ML suite, frontend build, and manual browser scan verified the promotion.

**Lesson:** clean configuration boundaries make model promotion testable and limit changes across the application.

## 10. Documentation is part of the system

Planning-era documents had drifted from the working product. The documentation was reorganized so the root README explains outcomes, the architecture document owns system boundaries, component READMEs own local commands, and evidence documents own data/model claims.

**Lesson:** stale documentation can misrepresent a good implementation. Each fact needs an explicit documentary owner and links instead of duplicated promises.

## Current verified state

- end-to-end GPU-backed MVP: working
- ML suite: `132 passed, 1 skipped`
- backend suite: `23 passed, 1 skipped`
- real backend-to-ML smoke: `1 passed`
- frontend build and manual UI flow: passed
- v2 validation and frozen held-out artifacts: retained locally

## Next learning phase

The next goal is measurable trust: analyze failure modes, score calibration, image-quality robustness, dataset/source stratification, privacy boundaries, model provenance, and human oversight. These steps can strengthen an experimental prototype, but clinical trust would still require representative external evidence and appropriate independent review.

## Related documents

- [Architecture](ARCHITECTURE.md)
- [Project brief](PROJECT_BRIEF.md)
- [Dataset card](DATASET_CARD.md)
- [Training and evaluation](TRAINING.md)
- [Root project guide](../README.md)
