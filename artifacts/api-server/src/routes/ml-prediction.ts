import { Router, type IRouter } from "express";

const router: IRouter = Router();
const ML_API_URL = process.env["ML_API_URL"] ?? "http://127.0.0.1:8001";

router.get("/ml-prediction/models", async (_req, res) => {
  try {
    const response = await fetch(`${ML_API_URL}/models`, {
      signal: AbortSignal.timeout(5000),
    });
    const data: unknown = await response.json();

    if (!response.ok) {
      res.status(502).json({ error: "The ML service could not return model information." });
      return;
    }

    res.json(data);
  } catch {
    res.status(503).json({ error: "The ML service is unavailable. Please try again later." });
  }
});

router.post("/ml-prediction/predict", async (req, res) => {
  try {
    const response = await fetch(`${ML_API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
      signal: AbortSignal.timeout(15000),
    });
    const data: unknown = await response.json();

    if (!response.ok) {
      const detail =
        typeof data === "object" &&
        data !== null &&
        "detail" in data &&
        typeof data.detail === "string"
          ? data.detail
          : "The ML prediction could not be completed.";

      res.status(response.status === 422 ? 400 : 502).json({ error: detail });
      return;
    }

    res.json(data);
  } catch {
    res.status(503).json({ error: "The ML service is unavailable. Please try again later." });
  }
});
export default router;
