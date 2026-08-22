import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

const PNG_BYTES = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
  "base64",
);

const SCAN_RESPONSE = {
  id: "scan-browser-test",
  created_at: "2026-08-05T12:00:00Z",
  original_filename: "oral-image.png",
  content_type: "image/png",
  size_bytes: PNG_BYTES.length,
  sha256: "0".repeat(64),
  input_assessment: {
    status: "supported",
    reason_codes: [],
    summary: "The image passed the configured technical-quality checks.",
    image_width: 1,
    image_height: 1,
    mean_luminance: 0.5,
    luminance_stddev: 0.2,
  },
  prediction: {
    label: "plaque_candidate",
    display_name: "Plaque candidate",
    confidence: 0.8734,
    severity: "moderate",
    is_mock: false,
    model_name: "orthodontic-plaque-mvp-v3",
    prediction_count: 1,
    detections: [
      {
        box_xyxy: [0.1, 0.1, 0.9, 0.9],
        label: 1,
        score: 0.8734,
      },
    ],
  },
  evidence: {
    kind: "detector_output",
    summary: "One model-generated plaque candidate was returned.",
  },
  report: {
    title: "Experimental screening-support report",
    summary: "The detector identified one candidate region for review.",
    limitations: [
      "The detector is not clinically validated.",
      "The score is not a disease probability.",
    ],
    recommended_next_steps: ["Ask a qualified dental professional to review concerns."],
    disclaimer: "Experimental screening support only; not a diagnosis or treatment recommendation.",
  },
};

async function selectPng(page: Page) {
  await page.getByLabel("Choose oral image", { exact: true }).setInputFiles({
    name: "oral-image.png",
    mimeType: "image/png",
    buffer: PNG_BYTES,
  });
}

async function expectNoWcagViolations(page: Page) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22a", "wcag22aa"])
    .analyze();

  expect(results.violations).toEqual([]);
}

test("initial workspace is accessible by semantics and keyboard", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { level: 1, name: "Screening support console" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Image upload" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Scan result" })).toContainText("Ready for a scan");
  await expectNoWcagViolations(page);

  const fileInput = page.getByLabel("Choose oral image", { exact: true });
  await page.keyboard.press("Tab");
  await expect(fileInput).toBeFocused();
  await expect(page.locator(".dropzone")).toHaveCSS("outline-style", "solid");

  await page.keyboard.press("Tab");
  const scanButton = page.getByRole("button", { name: "Run scan" });
  await expect(scanButton).toBeFocused();
  const buttonBox = await scanButton.boundingBox();
  expect(buttonBox).not.toBeNull();
  expect(buttonBox!.height).toBeGreaterThanOrEqual(44);

  await page.keyboard.press("Enter");
  await expect(page.getByRole("alert")).toHaveText("Select a JPEG or PNG image first.");
  await expect(fileInput).toBeFocused();
});

test("JPEG and PNG are accepted while unsupported or mismatched files are rejected", async ({ page }) => {
  let scanRequestCount = 0;
  page.on("request", (request) => {
    if (request.url().endsWith("/scans")) scanRequestCount += 1;
  });
  await page.goto("/");

  const fileInput = page.getByLabel("Choose oral image", { exact: true });
  await fileInput.setInputFiles({
    name: "oral-image.gif",
    mimeType: "image/gif",
    buffer: Buffer.from("GIF89a"),
  });

  await expect(page.getByRole("alert")).toHaveText("Select a JPEG or PNG image.");
  await expect(fileInput).toHaveValue("");

  await fileInput.setInputFiles({
    name: "oral-image.png",
    mimeType: "image/jpeg",
    buffer: Buffer.from([0xff, 0xd8, 0xff, 0xd9]),
  });
  await expect(page.getByRole("alert")).toHaveText("Select a JPEG or PNG image.");
  await expect(fileInput).toHaveValue("");

  await fileInput.setInputFiles({
    name: "oral-image.jpeg",
    mimeType: "image/jpeg",
    buffer: Buffer.from([0xff, 0xd8, 0xff, 0xd9]),
  });
  await expect(page.getByText("oral-image.jpeg", { exact: true })).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(scanRequestCount).toBe(0);
});

test("successful scan exposes loading, v3 evidence, report, and overlay", async ({ page }) => {
  let releaseResponse: (() => void) | undefined;
  const responseGate = new Promise<void>((resolve) => {
    releaseResponse = resolve;
  });

  await page.route("http://127.0.0.1:8000/scans", async (route) => {
    expect(route.request().method()).toBe("POST");
    expect(route.request().headers()["content-type"]).toContain("multipart/form-data");
    await responseGate;
    await route.fulfill({
      status: 201,
      contentType: "application/json",
      body: JSON.stringify(SCAN_RESPONSE),
    });
  });

  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await selectPng(page);
  await expect(page.getByAltText("Selected oral scan")).toBeVisible();

  const scanButton = page.getByRole("button", { name: "Run scan" });
  await scanButton.click();
  try {
    const scanningButton = page.getByRole("button", { name: "Scanning" });
    await expect(scanningButton).toBeDisabled();
    await expect(scanningButton.locator(".spin")).toHaveCSS("animation-name", "none");
    await expect(page.getByRole("status")).toHaveText("Scanning image.");
  } finally {
    releaseResponse?.();
  }

  await expect(page.getByRole("heading", { level: 2, name: "Plaque candidate" })).toBeVisible();
  await expect(page.getByRole("status")).toHaveText("Scan complete.");
  await expect(page.getByText("orthodontic-plaque-mvp-v3", { exact: true })).toBeVisible();
  await expect(page.locator(".score")).toHaveText("0.873");
  await expect(page.getByText("not a clinical probability", { exact: false })).toBeVisible();
  await expect(page.getByText("The detector is not clinically validated.")).toBeVisible();
  await expect(page.getByText(SCAN_RESPONSE.report.disclaimer)).toBeVisible();
  await expect(page.locator(".overlay rect")).toHaveCount(1);
  await expect(page.locator(".overlay rect")).toHaveAttribute("x", "0.1");
  await expect(page.locator(".overlay rect")).toHaveAttribute("width", "0.8");
  await expectNoWcagViolations(page);
});

test("unsupported image is presented as abstention without a model score", async ({ page }) => {
  const unsupportedResponse = {
    ...SCAN_RESPONSE,
    input_assessment: {
      status: "unsupported",
      reason_codes: ["image_too_dark"],
      summary: "The image is too dark for reliable visual screening.",
      image_width: 640,
      image_height: 480,
      mean_luminance: 0.01,
      luminance_stddev: 0.08,
    },
    prediction: null,
    evidence: {
      kind: "summary",
      summary: "The image is too dark for reliable visual screening.",
    },
    report: {
      title: "Image Assessment Report",
      summary: "The condition detector did not run because the image was too dark.",
      limitations: [
        "The automated checks measure technical image properties only.",
        "This abstention is not a diagnosis or plaque-free finding.",
      ],
      recommended_next_steps: ["Retake the image with even lighting."],
      disclaimer: "Experimental screening support only; not a diagnosis.",
    },
  };

  await page.route("http://127.0.0.1:8000/scans", async (route) => {
    await route.fulfill({
      status: 201,
      contentType: "application/json",
      body: JSON.stringify(unsupportedResponse),
    });
  });

  await page.goto("/");
  await selectPng(page);
  await page.getByRole("button", { name: "Run scan" }).click();

  await expect(page.getByRole("heading", { level: 2, name: "Image not assessed" })).toBeVisible();
  await expect(page.getByRole("status")).toHaveText("Image unsupported. The condition scan did not run.");
  await expect(page.getByText("Detector not run", { exact: true })).toBeVisible();
  await expect(page.getByText("Image is too dark.", { exact: true })).toBeVisible();
  await expect(page.getByText("Not produced", { exact: true })).toHaveCount(2);
  await expect(page.locator(".score")).toHaveCount(0);
  await expect(page.locator(".overlay rect")).toHaveCount(0);
  await expectNoWcagViolations(page);
});

test("structured backend errors are announced without internal details", async ({ page }) => {
  await page.route("http://127.0.0.1:8000/scans", async (route) => {
    await route.fulfill({
      status: 415,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Only JPEG and PNG images are supported." }),
    });
  });
  await page.goto("/");
  await selectPng(page);
  await page.getByRole("button", { name: "Run scan" }).click();

  await expect(page.getByRole("alert")).toHaveText("Only JPEG and PNG images are supported.");
  await expect(page.getByRole("button", { name: "Run scan" })).toBeEnabled();
});

test("non-JSON backend failures use a safe generic message", async ({ page }) => {
  await page.route("http://127.0.0.1:8000/scans", async (route) => {
    await route.fulfill({
      status: 500,
      contentType: "text/plain",
      body: "internal path and traceback must not reach the interface",
    });
  });
  await page.goto("/");
  await selectPng(page);
  await page.getByRole("button", { name: "Run scan" }).click();

  await expect(page.getByRole("alert")).toHaveText("Scan request failed.");
  await expect(page.getByRole("alert")).not.toContainText("traceback");
  await expect(page.getByRole("button", { name: "Run scan" })).toBeEnabled();
});
