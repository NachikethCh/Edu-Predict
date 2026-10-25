import { Router, type IRouter } from "express";
import healthRouter from "./health";
import studentPerformanceRouter from "./student-performance";
import mlPredictionRouter from "./ml-prediction";

const router: IRouter = Router();

router.use(healthRouter);
router.use(studentPerformanceRouter);
router.use(mlPredictionRouter);

export default router;
