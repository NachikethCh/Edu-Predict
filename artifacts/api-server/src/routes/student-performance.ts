import { Router, type IRouter } from "express";
import {
  GetStudentPerformanceModelResponse,
  PredictStudentPerformanceBody,
  PredictStudentPerformanceCsvBody,
  PredictStudentPerformanceCsvResponse,
  PredictStudentPerformanceResponse,
} from "@workspace/api-zod";
import {
  getModelAvailability,
  getModelOverview,
  predictCsv,
  predictStudent,
} from "../lib/student-performance";

const router: IRouter = Router();

function getValidationMessage(error: {
  issues: { message: string }[];
}): string {
  return error.issues[0]?.message ?? "The request is not valid.";
}

router.get("/student-performance/model", (_req, res) => {
  const response = GetStudentPerformanceModelResponse.parse(getModelOverview());
  res.json(response);
});

router.post("/student-performance/predict", (req, res) => {
  const input = PredictStudentPerformanceBody.safeParse(req.body);
  if (!input.success) {
    res.status(400).json({ error: getValidationMessage(input.error) });
    return;
  }
  const availability = getModelAvailability(
    input.data.classLevel,
    input.data.stream,
    input.data.subject,
  );
  if (!availability.supported) {
    res.status(400).json({ error: availability.reason });
    return;
  }

  try {
    const result = PredictStudentPerformanceResponse.parse(
      predictStudent(
        input.data.classLevel,
        input.data.stream,
        input.data.subject,
        input.data.student,
      ),
    );
    res.json(result);
  } catch (error) {
    req.log.error({ err: error }, "Student performance prediction failed");
    res.status(500).json({
      error: "The prediction could not be completed. Please try again.",
    });
  }
});

router.post("/student-performance/predict-csv", (req, res) => {
  const input = PredictStudentPerformanceCsvBody.safeParse(req.body);
  if (!input.success) {
    res.status(400).json({ error: getValidationMessage(input.error) });
    return;
  }

  try {
    const result = PredictStudentPerformanceCsvResponse.parse(
      predictCsv(input.data.csvText),
    );
    res.json(result);
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "The CSV could not be read.";
    res.status(400).json({ error: message });
  }
});

export default router;
