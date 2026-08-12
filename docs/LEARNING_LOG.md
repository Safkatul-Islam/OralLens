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

## 11. Measure the deployed policy, not only the model

The original evaluator measured all predictions above threshold, while application inference also applies a 25-result cap. A separate trustworthiness evaluator now references both historical evaluation config and deployed prediction config, retains per-image and per-patient evidence, and records checkpoint, manifest, config, runtime, and GPU provenance.

No validation or internal-test image reached the cap, so deployed and uncapped aggregate metrics are identical. Score-to-match ECE was `0.0949` on validation and `0.0971` on the then-designated frozen internal test split. Scores below `0.9` were overconfident as annotation-match indicators, patient-level F1 varied materially, and dark brightness-down samples repeatedly appeared among the largest false-negative cases. Blur also appeared in the internal test failure set.

The held-out trustworthiness pass used the exact frozen model, threshold, IoU, and cap for descriptive evidence only. It did not select or modify an operating point.

**Lesson:** trustworthy evaluation requires deployment-policy parity, provenance, stratified evidence, and visible failure cases. A detector score can be measured against annotation matching without being promoted to a disease probability or clinical-confidence claim.

## 12. Increase training coverage without confusing optimization with evidence

The v2 configuration used three epochs but capped each epoch at 512 of 3,834 training images and 64 of 468 validation images. V3 preserved the architecture, optimizer, seed, and epoch count while removing those caps. It also added atomic per-epoch last/best checkpoints and genuine resume state for the model, optimizer, and RNG.

V3 improved the internal test benchmark from F1 `0.7299` to `0.7795`, recall from `0.7037` to `0.7923`, and mean matched IoU from `0.7541` to `0.8102`. Validation and internal test F1 were similar (`0.7776` and `0.7795`).

The improvement had a tradeoff: test score-to-match ECE worsened from `0.0971` to `0.1878`. V3 is a better annotation-matching detector, but its raw scores are less reliable as match probabilities.

**Lesson:** more complete optimization can improve localization and recall without improving calibration. Accuracy, calibration, patient variation, and clinical meaning must be evaluated separately.

## 13. Treat stronger medical claims as an evidence-design problem

The current dataset has 5,160 images and 65,238 annotations but only 74 patient groups. Every manifest row has at least one positive annotation, and the data come from one specialized source with transformed variants. Those properties prevent clinical specificity, negative predictive value, multi-site generalization, and clinical-utility claims regardless of how many additional epochs are run.

The next phase was therefore defined claim-first: narrow the intended use, appoint dental/statistical/governance owners, establish a fit-for-purpose reference standard, collect representative positive and negative multi-site data, isolate calibration and external test cohorts, and evaluate the human-AI team before considering assistance claims.

**Lesson:** a stronger model metric does not authorize a stronger medical claim. Claims must be pre-specified and supported by representative, independent, clinically meaningful evidence.

## 14. Promote model identity and score meaning together

V3 was promoted through the backend's typed config and PowerShell/CMD startup boundaries without changing routes or response schemas. The inference config now carries an explicit validated model name through prediction artifacts, the adapter, stored scan records, tests, and the UI. Real checkpoint integration, full suites, the frontend build, and a browser scan verified the promoted path.

The first v3 browser check also exposed a trust issue: a raw detector score near `0.995` was rounded and displayed as `100% confidence`. The UI now shows a three-decimal `Detector score`, explicitly states that it is not a clinical probability, and the generated report uses the same ranking-score interpretation.

**Lesson:** model promotion is incomplete if users cannot identify the deployed model or if presentation turns an uncalibrated ranking value into an implied probability.

## 15. Automate the browser contract without overstating accessibility

The frontend gained a Playwright Chromium suite that intercepts the API boundary and uses in-memory image bytes rather than patient data. It verifies keyboard order and focus, unsupported-file rejection before transmission, loading and completion announcements, v3 identity, raw score semantics, report limitations, overlay geometry, and safe structured/non-JSON error behavior. Axe-core checks WCAG A/AA rules in the initial and completed-result states.

On Windows, Playwright's managed web-server teardown relies on a synchronous process-tree kill, which hung in the restricted development environment after all tests passed. The suite instead starts Vite through its project-local JavaScript API during global setup and returns an awaited in-process close callback. This keeps the suite self-contained without platform-specific shell cleanup or an additional dependency.

**Lesson:** browser automation should verify the user-facing contract and clean up deterministically. Automated rule checks improve coverage but cannot replace manual assistive-technology and human-factors evaluation.

## Current verified state

- end-to-end GPU-backed MVP: working
- ML suite: `150 passed, 1 skipped`
- backend suite: `24 passed, 1 skipped`
- real backend-to-ML smoke: `1 passed`
- frontend Playwright/axe suite: `5 passed`
- frontend build and manual UI flow: passed
- frontend dependency audit: `0` known vulnerabilities across `82` dependencies
- v2 historical and v3 full-coverage validation/test/trust artifacts: retained locally
- active application runtime: v3 at validation-selected threshold `0.85`
- UI/report score semantics: raw detector ranking value, not a percentage or clinical probability

## Next learning phase

The next goal is a claim-aligned clinical evidence foundation: define the target use and population, establish the dental reference standard and statistical plan, secure governance before collection, acquire representative positive/negative multi-site data, and reserve a new external evaluation boundary before v4 development. Controlled robustness, calibration, unsupported-input behavior, privacy, and human oversight remain required parts of that plan.

## Related documents

- [Architecture](ARCHITECTURE.md)
- [Project brief](PROJECT_BRIEF.md)
- [Dataset card](DATASET_CARD.md)
- [Training and evaluation](TRAINING.md)
- [Intended use and claims](INTENDED_USE_AND_CLAIMS.md)
- [Clinical evidence plan](CLINICAL_EVIDENCE_PLAN.md)
- [Data acquisition and annotation](DATA_ACQUISITION_AND_ANNOTATION.md)
- [Root project guide](../README.md)
