import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { StudentFeatures } from "@workspace/api-zod";

export type Stream = "science" | "commerce" | "humanities";
export type ClassStream = Stream | null;
export type GradeBand = "needs_support" | "developing" | "on_track";
export const SUPPORTED_MODEL_ID = "uci-math-secondary";
const SUPPORTED_MATH_CLASSES = [9, 10, 11, 12];
const SUPPORTED_STREAMS: Stream[] = ["science", "commerce", "humanities"];

export const FEATURE_KEYS = [
  "g1",
  "age",
  "studytime",
  "failures",
  "absences",
  "schoolsup",
  "famsup",
  "paid",
  "higher",
  "internet",
  "goout",
  "traveltime",
  "health",
] as const satisfies readonly (keyof StudentFeatures)[];

const FEATURE_LABELS: Record<(typeof FEATURE_KEYS)[number], string> = {
  g1: "Previous assessment mark (source scale 0–20)",
  age: "Age",
  studytime: "Weekly study time",
  failures: "Past class failures",
  absences: "Absences",
  schoolsup: "School support",
  famsup: "Family support",
  paid: "Paid subject classes",
  higher: "Plans for higher education",
  internet: "Internet at home",
  goout: "Going out with friends",
  traveltime: "Travel time",
  health: "Self-reported health",
};

const TARGET_PASS_GRADE = 10;
const NEIGHBOR_COUNT = 9;
const TEST_FRACTION = 0.2;
const MODEL_SEED = 42;

type TrainingRecord = {
  features: StudentFeatures;
  target: number;
};

type Scaler = {
  means: number[];
  standardDeviations: number[];
};

type Model = {
  train: { vector: number[]; target: number }[];
  scaler: Scaler;
  metrics: ModelEvaluation;
};

export type FeatureImportance = {
  feature: string;
  label: string;
  importance: number;
};

export type ModelEvaluation = {
  subject: string;
  rows: number;
  trainRows: number;
  testRows: number;
  mae: number;
  rmse: number;
  r2: number;
  baselineMae: number;
  baselineRmse: number;
  passAccuracy: number;
  passPrecision: number;
  passRecall: number;
  passF1: number;
  featureImportance: FeatureImportance[];
};

export type ModelOverview = {
  datasetName: string;
  sourceUrl: string;
  doi: string;
  license: string;
  target: string;
  split: string;
  algorithm: string;
  seed: number;
  features: string[];
  modelSupport: ModelSupport[];
  evaluations: ModelEvaluation[];
};

export type ModelSupport = {
  modelId: string;
  subject: string;
  classLevels: number[];
  streams: Stream[];
  scope: string;
  classSpecific: boolean;
  kvValidated: boolean;
};

export type PredictionSignal = {
  feature: string;
  value: string;
  direction: "positive" | "attention";
  explanation: string;
};

export type PredictionResult = {
  classLevel: number;
  stream: ClassStream;
  subject: string;
  predictedGrade: number;
  gradeBand: GradeBand;
  neighborCount: number;
  similarStudentPassShare: number;
  similarStudentGradeRange: number[];
  topSignals: PredictionSignal[];
};

export type CsvRowPrediction = {
  rowNumber: number;
  studentId: string | null;
  classLevel: number;
  stream: ClassStream;
  subject: string;
  predictedGrade: number;
  gradeBand: GradeBand;
  similarStudentPassShare: number;
};

export type CsvRowError = {
  rowNumber: number;
  studentId: string | null;
  classLevel: number | null;
  subject: string | null;
  message: string;
};

export type CsvPredictionResult = {
  validCount: number;
  errorCount: number;
  predictions: CsvRowPrediction[];
  errors: CsvRowError[];
};

let trainedMathModel: Model | undefined;

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle<T>(values: T[], seed: number): T[] {
  const result = [...values];
  const random = seededRandom(seed);
  for (let index = result.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [result[index], result[swapIndex]] = [result[swapIndex], result[index]];
  }
  return result;
}

function parseDelimited(text: string, delimiter: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '"') {
        if (text[index + 1] === '"') {
          cell += '"';
          index += 1;
        } else {
          quoted = false;
        }
      } else {
        cell += character;
      }
      continue;
    }

    if (character === '"' && cell.length === 0) {
      quoted = true;
    } else if (character === delimiter) {
      row.push(cell.trim());
      cell = "";
    } else if (character === "\n" || character === "\r") {
      row.push(cell.trim());
      cell = "";
      if (row.some((value) => value.length > 0)) rows.push(row);
      row = [];
      if (character === "\r" && text[index + 1] === "\n") index += 1;
    } else {
      cell += character;
    }
  }

  if (quoted) throw new Error("The CSV contains an unfinished quoted field.");
  if (cell.length > 0 || row.length > 0) {
    row.push(cell.trim());
    if (row.some((value) => value.length > 0)) rows.push(row);
  }
  return rows;
}

function parseDataset(): TrainingRecord[] {
  const cwd = process.cwd();
  const directory = cwd.endsWith("/artifacts/api-server")
    ? cwd
    : resolve(cwd, "artifacts/api-server");
  const filename = "student-mat.csv";
  const text = readFileSync(resolve(directory, "data", filename), "utf8");
  const rows = parseDelimited(text.replace(/^\uFEFF/, ""), ";");
  const headers = rows[0]?.map((header) => header.toLowerCase());

  if (!headers?.includes("g1") || !headers.includes("g3")) {
    throw new Error(`The ${filename} dataset is missing G1 or G3.`);
  }

  const records = rows.slice(1).map((values, rowIndex) => {
    const raw = Object.fromEntries(
      headers.map((header, index) => [header, values[index]]),
    );
    const number = (field: string) => {
      const value = Number(raw[field]);
      if (!Number.isFinite(value)) {
        throw new Error(`Invalid ${field} value on dataset row ${rowIndex + 2}.`);
      }
      return value;
    };
    const yesNo = (field: string) => {
      if (raw[field] !== "yes" && raw[field] !== "no") {
        throw new Error(`Invalid ${field} value on dataset row ${rowIndex + 2}.`);
      }
      return raw[field] === "yes";
    };
    const features: StudentFeatures = {
      g1: number("g1"),
      age: number("age"),
      studytime: number("studytime"),
      failures: number("failures"),
      absences: number("absences"),
      schoolsup: yesNo("schoolsup"),
      famsup: yesNo("famsup"),
      paid: yesNo("paid"),
      higher: yesNo("higher"),
      internet: yesNo("internet"),
      goout: number("goout"),
      traveltime: number("traveltime"),
      health: number("health"),
    };
    return { features, target: number("g3") };
  });

  if (records.length < 20) {
    throw new Error(`The ${filename} dataset has too few usable records.`);
  }
  return records;
}

function toVector(features: StudentFeatures): number[] {
  return FEATURE_KEYS.map((key) =>
    typeof features[key] === "boolean" ? Number(features[key]) : features[key],
  );
}

function fitScaler(records: TrainingRecord[]): Scaler {
  const vectors = records.map(({ features }) => toVector(features));
  const means = FEATURE_KEYS.map(
    (_key, index) =>
      vectors.reduce((sum, vector) => sum + vector[index], 0) / vectors.length,
  );
  const standardDeviations = FEATURE_KEYS.map((_key, index) => {
    const variance =
      vectors.reduce(
        (sum, vector) => sum + (vector[index] - means[index]) ** 2,
        0,
      ) / vectors.length;
    return Math.sqrt(variance) || 1;
  });
  return { means, standardDeviations };
}

function scaleVector(vector: number[], scaler: Scaler): number[] {
  return vector.map(
    (value, index) =>
      (value - scaler.means[index]) / scaler.standardDeviations[index],
  );
}

function predictVector(
  vector: number[],
  trainingRows: Model["train"],
): { grade: number; neighbors: Model["train"] } {
  const neighbors = trainingRows
    .map((row) => ({
      ...row,
      distance: Math.sqrt(
        row.vector.reduce(
          (sum, value, index) => sum + (value - vector[index]) ** 2,
          0,
        ) / vector.length,
      ),
    }))
    .sort((left, right) => left.distance - right.distance)
    .slice(0, Math.min(NEIGHBOR_COUNT, trainingRows.length));

  const weightTotal = neighbors.reduce(
    (sum, row) => sum + 1 / (row.distance + 0.05),
    0,
  );
  const weightedGrade =
    neighbors.reduce(
      (sum, row) => sum + row.target / (row.distance + 0.05),
      0,
    ) / weightTotal;

  return {
    grade: Math.max(0, Math.min(20, weightedGrade)),
    neighbors,
  };
}

function meanAbsoluteError(actual: number[], predicted: number[]): number {
  return (
    actual.reduce((sum, value, index) => sum + Math.abs(value - predicted[index]), 0) /
    actual.length
  );
}

function rootMeanSquaredError(actual: number[], predicted: number[]): number {
  return Math.sqrt(
    actual.reduce((sum, value, index) => sum + (value - predicted[index]) ** 2, 0) /
      actual.length,
  );
}

function gradeBand(grade: number): GradeBand {
  if (grade < TARGET_PASS_GRADE) return "needs_support";
  if (grade < 14) return "developing";
  return "on_track";
}

function computePassMetrics(actual: number[], predicted: number[]) {
  let truePositive = 0;
  let trueNegative = 0;
  let falsePositive = 0;
  let falseNegative = 0;

  actual.forEach((grade, index) => {
    const actualPass = grade >= TARGET_PASS_GRADE;
    const predictedPass = predicted[index] >= TARGET_PASS_GRADE;
    if (actualPass && predictedPass) truePositive += 1;
    else if (!actualPass && !predictedPass) trueNegative += 1;
    else if (predictedPass) falsePositive += 1;
    else falseNegative += 1;
  });

  const accuracy = (truePositive + trueNegative) / actual.length;
  const precision =
    truePositive + falsePositive === 0
      ? 0
      : truePositive / (truePositive + falsePositive);
  const recall =
    truePositive + falseNegative === 0
      ? 0
      : truePositive / (truePositive + falseNegative);
  const f1 =
    precision + recall === 0
      ? 0
      : (2 * precision * recall) / (precision + recall);
  return { accuracy, precision, recall, f1 };
}

function calculateFeatureImportance(
  modelRows: Model["train"],
  testVectors: number[][],
  actual: number[],
  scaler: Scaler,
  courseSeed: number,
): FeatureImportance[] {
  const baselinePredictions = testVectors.map(
    (vector) => predictVector(vector, modelRows).grade,
  );
  const baselineMae = meanAbsoluteError(actual, baselinePredictions);

  return FEATURE_KEYS.map((key, featureIndex) => {
    const random = seededRandom(courseSeed + featureIndex + 11);
    const shuffledValues = testVectors.map((vector) => vector[featureIndex]);
    for (let index = shuffledValues.length - 1; index > 0; index -= 1) {
      const swapIndex = Math.floor(random() * (index + 1));
      [shuffledValues[index], shuffledValues[swapIndex]] = [
        shuffledValues[swapIndex],
        shuffledValues[index],
      ];
    }
    const permutedVectors = testVectors.map((vector, rowIndex) => {
      const changed = [...vector];
      changed[featureIndex] = shuffledValues[rowIndex];
      return changed;
    });
    const permutedMae = meanAbsoluteError(
      actual,
      permutedVectors.map((vector) => predictVector(vector, modelRows).grade),
    );
    return {
      feature: key,
      label: FEATURE_LABELS[key],
      importance: Math.max(0, permutedMae - baselineMae),
    };
  }).sort((left, right) => right.importance - left.importance);
}

function trainModel(): Model {
  const records = parseDataset();
  const split = shuffle(records, MODEL_SEED);
  const testCount = Math.max(1, Math.ceil(split.length * TEST_FRACTION));
  const testRecords = split.slice(0, testCount);
  const trainingRecords = split.slice(testCount);
  const scaler = fitScaler(trainingRecords);
  const train = trainingRecords.map(({ features, target }) => ({
    vector: scaleVector(toVector(features), scaler),
    target,
  }));
  const testVectors = testRecords.map(({ features }) =>
    scaleVector(toVector(features), scaler),
  );
  const actual = testRecords.map(({ target }) => target);
  const predictions = testVectors.map(
    (vector) => predictVector(vector, train).grade,
  );
  const meanTarget =
    trainingRecords.reduce((sum, row) => sum + row.target, 0) /
    trainingRecords.length;
  const baselinePredictions = actual.map(() => meanTarget);
  const targetMean =
    actual.reduce((sum, value) => sum + value, 0) / actual.length;
  const totalVariance = actual.reduce(
    (sum, value) => sum + (value - targetMean) ** 2,
    0,
  );
  const residualVariance = actual.reduce(
    (sum, value, index) => sum + (value - predictions[index]) ** 2,
    0,
  );
  const passMetrics = computePassMetrics(actual, predictions);
  const metrics: ModelEvaluation = {
    subject: "Mathematics",
    rows: records.length,
    trainRows: trainingRecords.length,
    testRows: testRecords.length,
    mae: meanAbsoluteError(actual, predictions),
    rmse: rootMeanSquaredError(actual, predictions),
    r2: totalVariance === 0 ? 0 : 1 - residualVariance / totalVariance,
    baselineMae: meanAbsoluteError(actual, baselinePredictions),
    baselineRmse: rootMeanSquaredError(actual, baselinePredictions),
    passAccuracy: passMetrics.accuracy,
    passPrecision: passMetrics.precision,
    passRecall: passMetrics.recall,
    passF1: passMetrics.f1,
    featureImportance: calculateFeatureImportance(
      train,
      testVectors,
      actual,
      scaler,
      MODEL_SEED,
    ),
  };

  return { train, scaler, metrics };
}

function getModel(): Model {
  if (trainedMathModel) return trainedMathModel;
  trainedMathModel = trainModel();
  return trainedMathModel;
}

export function getModelAvailability(
  classLevel: number,
  stream: ClassStream,
  subject: string,
): { supported: boolean; modelId: string | null; reason: string } {
  if (!Number.isInteger(classLevel) || classLevel < 1 || classLevel > 12) {
    return {
      supported: false,
      modelId: null,
      reason: "Choose a class from I to XII.",
    };
  }
  if (classLevel >= 11 && !stream) {
    return {
      supported: false,
      modelId: null,
      reason: "Choose a stream for Classes XI and XII.",
    };
  }
  if (classLevel <= 10 && stream) {
    return {
      supported: false,
      modelId: null,
      reason: "Streams are only used for Classes XI and XII.",
    };
  }
  if (stream && !SUPPORTED_STREAMS.includes(stream)) {
    return {
      supported: false,
      modelId: null,
      reason: "This stream does not have an evaluated model.",
    };
  }
  if (
    SUPPORTED_MATH_CLASSES.includes(classLevel) &&
    subject.trim().toLowerCase() === "mathematics"
  ) {
    return {
      supported: true,
      modelId: SUPPORTED_MODEL_ID,
      reason:
        "Research demo only: the source has no KV class or stream labels and is not validated for KV schools.",
    };
  }
  return {
    supported: false,
    modelId: null,
    reason: `No trained and evaluated model is available for Class ${classLevel}, ${subject}.`,
  };
}

export function getModelOverview(): ModelOverview {
  return {
    datasetName: "UCI Student Performance",
    sourceUrl: "https://archive.ics.uci.edu/dataset/320/student+performance",
    doi: "10.24432/C5TG7T",
    license: "CC BY 4.0",
    target: "G3 — final-period Mathematics grade (0–20), estimated from earlier academic and context fields",
    split: "Deterministic 80/20 train/test split; metrics use held-out test rows only",
    algorithm:
      "Standardized, inverse-distance weighted k-nearest-neighbors regression (k=9); compared with a training-mean baseline",
    seed: MODEL_SEED,
    features: FEATURE_KEYS.map((key) => FEATURE_LABELS[key]),
    modelSupport: [
      {
        modelId: SUPPORTED_MODEL_ID,
        subject: "Mathematics",
        classLevels: [...SUPPORTED_MATH_CLASSES],
        streams: [...SUPPORTED_STREAMS],
        scope:
          "Research demo using a secondary-school Mathematics sample; class and stream were not recorded in the source, and the model is not validated for KV schools.",
        classSpecific: false,
        kvValidated: false,
      },
    ],
    evaluations: [getModel().metrics],
  };
}

function formatFeatureValue(
  key: (typeof FEATURE_KEYS)[number],
  value: StudentFeatures[(typeof FEATURE_KEYS)[number]],
): string {
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (key === "g1") return `${value}/20`;
  if (key === "studytime") {
    return ["", "under 2 hours", "2–5 hours", "5–10 hours", "over 10 hours"][
      value
    ];
  }
  return String(value);
}

function getPredictionSignals(student: StudentFeatures): PredictionSignal[] {
  const signals: PredictionSignal[] = [
    {
      feature: "previous_marks",
      value: formatFeatureValue("g1", student.g1),
      direction: student.g1 >= TARGET_PASS_GRADE ? "positive" : "attention",
      explanation:
        "The first-period grade is the strongest model input; it is an early signal, not a final result.",
    },
  ];

  if (student.failures > 0) {
    signals.push({
      feature: "failures",
      value: `${student.failures}${student.failures === 4 ? " (3 or more)" : ""}`,
      direction: "attention",
      explanation:
        "Past class failures are associated with final-grade differences in this dataset; consider arranging subject support.",
    });
  }
  if (student.absences >= 5) {
    signals.push({
      feature: "absences",
      value: String(student.absences),
      direction: "attention",
      explanation:
        "Missed lessons can make it harder to keep up; check whether notes or a catch-up plan would help.",
    });
  }
  if (student.studytime >= 3) {
    signals.push({
      feature: "studytime",
      value: formatFeatureValue("studytime", student.studytime),
      direction: "positive",
      explanation:
        "A regular study schedule is a useful routine to maintain; the model does not treat it as a guarantee.",
    });
  } else if (signals.length < 3) {
    signals.push({
      feature: "studytime",
      value: formatFeatureValue("studytime", student.studytime),
      direction: "attention",
      explanation:
        "Try a short, consistent practice block and review progress with a teacher or trusted adult.",
    });
  }
  return signals.slice(0, 3);
}

export function predictStudent(
  classLevel: number,
  stream: ClassStream,
  subject: string,
  student: StudentFeatures,
): PredictionResult {
  const availability = getModelAvailability(classLevel, stream, subject);
  if (!availability.supported) throw new Error(availability.reason);
  const model = getModel();
  const vector = scaleVector(toVector(student), model.scaler);
  const result = predictVector(vector, model.train);
  const neighborGrades = result.neighbors
    .map((neighbor) => neighbor.target)
    .sort((left, right) => left - right);
  const lowIndex = Math.floor((neighborGrades.length - 1) * 0.2);
  const highIndex = Math.ceil((neighborGrades.length - 1) * 0.8);

  return {
    predictedGrade: Math.round(result.grade * 10) / 10,
    classLevel,
    stream,
    subject: "Mathematics",
    gradeBand: gradeBand(result.grade),
    neighborCount: result.neighbors.length,
    similarStudentPassShare:
      result.neighbors.filter(
        (neighbor) => neighbor.target >= TARGET_PASS_GRADE,
      ).length / result.neighbors.length,
    similarStudentGradeRange: [
      neighborGrades[lowIndex],
      neighborGrades[highIndex],
    ],
    topSignals: getPredictionSignals(student),
  };
}

const NUMERIC_RANGES: Record<
  (typeof FEATURE_KEYS)[number],
  { min: number; max: number; integer: boolean } | null
> = {
  g1: { min: 0, max: 20, integer: false },
  age: { min: 15, max: 22, integer: true },
  studytime: { min: 1, max: 4, integer: true },
  failures: { min: 0, max: 4, integer: true },
  absences: { min: 0, max: 93, integer: true },
  schoolsup: null,
  famsup: null,
  paid: null,
  higher: null,
  internet: null,
  goout: { min: 1, max: 5, integer: true },
  traveltime: { min: 1, max: 4, integer: true },
  health: { min: 1, max: 5, integer: true },
};

const CSV_FEATURE_COLUMNS: Record<(typeof FEATURE_KEYS)[number], string> = {
  g1: "previous_marks",
  age: "age",
  studytime: "study_time",
  failures: "failures",
  absences: "absences",
  schoolsup: "school_support",
  famsup: "family_support",
  paid: "paid_classes",
  higher: "higher_education_intent",
  internet: "internet_access",
  goout: "go_out",
  traveltime: "travel_time",
  health: "health",
};

const REQUIRED_CSV_HEADERS = [
  "class",
  "stream",
  "subject",
  ...Object.values(CSV_FEATURE_COLUMNS),
];

const ALLOWED_CSV_HEADERS = new Set([
  "student_id",
  ...REQUIRED_CSV_HEADERS,
]);

function parseStudentFields(values: Record<string, string>): StudentFeatures {
  const student: Record<(typeof FEATURE_KEYS)[number], number | boolean> = {
    g1: 0,
    age: 0,
    studytime: 0,
    failures: 0,
    absences: 0,
    schoolsup: false,
    famsup: false,
    paid: false,
    higher: false,
    internet: false,
    goout: 1,
    traveltime: 1,
    health: 1,
  };
  const booleanFields = new Set(["schoolsup", "famsup", "paid", "higher", "internet"]);

  for (const key of FEATURE_KEYS) {
    const raw = values[CSV_FEATURE_COLUMNS[key]]?.trim();
    if (!raw) throw new Error(`Missing value for ${FEATURE_LABELS[key]}.`);

    if (booleanFields.has(key)) {
      const normalized = raw.toLowerCase();
      if (["yes", "true", "1"].includes(normalized)) {
        student[key] = true;
      } else if (["no", "false", "0"].includes(normalized)) {
        student[key] = false;
      } else {
        throw new Error(`${FEATURE_LABELS[key]} must be yes/no or true/false.`);
      }
      continue;
    }

    const parsed = Number(raw);
    const range = NUMERIC_RANGES[key];
    if (!Number.isFinite(parsed) || !range) {
      throw new Error(`${FEATURE_LABELS[key]} must be a number.`);
    }
    if (range.integer && !Number.isInteger(parsed)) {
      throw new Error(`${FEATURE_LABELS[key]} must be a whole number.`);
    }
    if (parsed < range.min || parsed > range.max) {
      throw new Error(
        `${FEATURE_LABELS[key]} must be between ${range.min} and ${range.max}.`,
      );
    }
    student[key] = parsed;
  }
  return student as StudentFeatures;
}

function detectDelimiter(header: string): string {
  const commaCount = (header.match(/,/g) ?? []).length;
  const semicolonCount = (header.match(/;/g) ?? []).length;
  return semicolonCount > commaCount ? ";" : ",";
}

function parseClassLevel(value: string): number {
  const normalized = value.trim().replace(/^class\s+/i, "").toUpperCase();
  const romanNumerals: Record<string, number> = {
    I: 1,
    II: 2,
    III: 3,
    IV: 4,
    V: 5,
    VI: 6,
    VII: 7,
    VIII: 8,
    IX: 9,
    X: 10,
    XI: 11,
    XII: 12,
  };
  const numeric = Number(normalized);
  const classLevel = Number.isInteger(numeric)
    ? numeric
    : romanNumerals[normalized];
  if (!classLevel || classLevel < 1 || classLevel > 12) {
    throw new Error("Class must be a number from 1 to 12 or a Roman numeral I–XII.");
  }
  return classLevel;
}

function parseStream(value: string, classLevel: number): ClassStream {
  const normalized = value.trim().toLowerCase();
  if (classLevel <= 10 && !normalized) return null;
  if (!normalized && classLevel >= 11) {
    throw new Error("Stream is required for Classes XI and XII.");
  }
  if (SUPPORTED_STREAMS.includes(normalized as Stream)) {
    return normalized as Stream;
  }
  if (classLevel <= 10 && normalized) {
    throw new Error("Stream should be blank for Classes I to X.");
  }
  throw new Error("Stream must be Science, Commerce, or Humanities.");
}

export function predictCsv(csvText: string): CsvPredictionResult {
  if (csvText.length > 2_000_000) {
    throw new Error("The CSV is larger than the 2 MB limit.");
  }
  const firstLine = csvText.split(/\r?\n/, 1)[0] ?? "";
  const delimiter = detectDelimiter(firstLine);
  const rows = parseDelimited(csvText.replace(/^\uFEFF/, ""), delimiter);
  if (rows.length < 2) {
    throw new Error("Add a header row and at least one student row.");
  }
  if (rows.length - 1 > 500) {
    throw new Error("Upload no more than 500 student rows at a time.");
  }

  const headers = rows[0].map((header) => header.trim().toLowerCase());
  const missingColumns = REQUIRED_CSV_HEADERS.filter(
    (header) => !headers.includes(header),
  );
  if (missingColumns.length > 0) {
    throw new Error(
      `Missing required columns: ${missingColumns.join(", ")}.`,
    );
  }
  const unsupportedColumns = headers.filter(
    (header) => !ALLOWED_CSV_HEADERS.has(header),
  );
  if (unsupportedColumns.length > 0) {
    throw new Error(
      `Unsupported columns: ${unsupportedColumns.join(", ")}. Keep only documented template fields.`,
    );
  }

  const predictions: CsvRowPrediction[] = [];
  const errors: CsvRowError[] = [];
  rows.slice(1).forEach((values, index) => {
    const rowNumber = index + 2;
    const row = Object.fromEntries(
      headers.map((header, column) => [header, values[column] ?? ""]),
    );
    try {
      if (values.length !== headers.length) {
        throw new Error(
          `Expected ${headers.length} columns but found ${values.length}.`,
        );
      }
      const studentId = row.student_id?.trim() || null;
      const classLevel = parseClassLevel(row.class);
      const stream = parseStream(row.stream, classLevel);
      const requestedSubject = row.subject.trim();
      if (!requestedSubject) throw new Error("Subject is required.");
      const availability = getModelAvailability(
        classLevel,
        stream,
        requestedSubject,
      );
      if (!availability.supported) throw new Error(availability.reason);
      const student = parseStudentFields(row);
      const prediction = predictStudent(
        classLevel,
        stream,
        requestedSubject,
        student,
      );
      predictions.push({
        rowNumber,
        studentId,
        classLevel,
        stream,
        subject: prediction.subject,
        predictedGrade: prediction.predictedGrade,
        gradeBand: prediction.gradeBand,
        similarStudentPassShare: prediction.similarStudentPassShare,
      });
    } catch (error) {
      errors.push({
        rowNumber,
        studentId: row.student_id?.trim() || null,
        classLevel: (() => {
          try {
            return parseClassLevel(row.class ?? "");
          } catch {
            return null;
          }
        })(),
        subject: row.subject?.trim() || null,
        message:
          error instanceof Error ? error.message : "This row is not valid.",
      });
    }
  });

  return {
    validCount: predictions.length,
    errorCount: errors.length,
    predictions,
    errors,
  };
}

/**
 * Cross-validates the research Mathematics model without changing the production
 * model's existing deterministic 80/20 evaluation or prediction behavior.
 *
 * Each fold fits its scaler using training rows only. The "without_g1" run
 * removes the previous assessment mark from the feature vector to measure how
 * much the model relies on that input. Results are research diagnostics, not
 * evidence of validation for any particular school system.
 */
export function evaluateMathModelCrossValidation(foldCount = 5) {
  if (!Number.isInteger(foldCount) || foldCount < 2 || foldCount > 10) {
    throw new Error("foldCount must be an integer between 2 and 10.");
  }

  const records = parseDataset();
  if (records.length < foldCount) {
    throw new Error("The dataset must contain at least one row per fold.");
  }

  const shuffled = shuffle(records, MODEL_SEED);
  const featureSets = [
    { id: "all_features" as const, keys: [...FEATURE_KEYS] },
    {
      id: "without_g1" as const,
      keys: FEATURE_KEYS.filter((key) => key !== "g1"),
    },
  ];

  type FoldScores = {
    fold: number;
    trainRows: number;
    testRows: number;
    mae: number;
    rmse: number;
    r2: number;
    baselineMae: number;
    baselineRmse: number;
  };

  const experiments = featureSets.map(({ id, keys }) => {
    const actualAll: number[] = [];
    const predictedAll: number[] = [];
    const baselineAll: number[] = [];
    const folds: FoldScores[] = [];

    for (let fold = 0; fold < foldCount; fold += 1) {
      const testRecords = shuffled.filter((_record, index) => index % foldCount === fold);
      const trainingRecords = shuffled.filter((_record, index) => index % foldCount !== fold);
      if (testRecords.length === 0 || trainingRecords.length === 0) continue;

      const toSelectedVector = (features: StudentFeatures) =>
        keys.map((key) =>
          typeof features[key] === "boolean" ? Number(features[key]) : features[key],
        );

      const trainingVectors = trainingRecords.map(({ features }) =>
        toSelectedVector(features),
      );
      const means = keys.map(
        (_key, column) =>
          trainingVectors.reduce((sum, vector) => sum + vector[column], 0) /
          trainingVectors.length,
      );
      const standardDeviations = keys.map((_key, column) => {
        const variance = trainingVectors.reduce(
          (sum, vector) => sum + (vector[column] - means[column]) ** 2,
          0,
        ) / trainingVectors.length;
        return Math.sqrt(variance) || 1;
      });
      const scaleSelectedVector = (vector: number[]) =>
        vector.map((value, column) =>
          (value - means[column]) / standardDeviations[column],
        );

      const trainingRows: Model["train"] = trainingRecords.map(
        ({ features, target }) => ({
          vector: scaleSelectedVector(toSelectedVector(features)),
          target,
        }),
      );
      const actual = testRecords.map(({ target }) => target);
      const predicted = testRecords.map(({ features }) =>
        predictVector(
          scaleSelectedVector(toSelectedVector(features)),
          trainingRows,
        ).grade,
      );
      const trainingMean =
        trainingRecords.reduce((sum, record) => sum + record.target, 0) /
        trainingRecords.length;
      const baseline = actual.map(() => trainingMean);
      const targetMean = actual.reduce((sum, value) => sum + value, 0) / actual.length;
      const totalVariance = actual.reduce(
        (sum, value) => sum + (value - targetMean) ** 2,
        0,
      );
      const residualVariance = actual.reduce(
        (sum, value, index) => sum + (value - predicted[index]) ** 2,
        0,
      );

      folds.push({
        fold: fold + 1,
        trainRows: trainingRecords.length,
        testRows: testRecords.length,
        mae: meanAbsoluteError(actual, predicted),
        rmse: rootMeanSquaredError(actual, predicted),
        r2: totalVariance === 0 ? 0 : 1 - residualVariance / totalVariance,
        baselineMae: meanAbsoluteError(actual, baseline),
        baselineRmse: rootMeanSquaredError(actual, baseline),
      });
      actualAll.push(...actual);
      predictedAll.push(...predicted);
      baselineAll.push(...baseline);
    }

    const actualMean = actualAll.reduce((sum, value) => sum + value, 0) / actualAll.length;
    const totalVariance = actualAll.reduce(
      (sum, value) => sum + (value - actualMean) ** 2,
      0,
    );
    const residualVariance = actualAll.reduce(
      (sum, value, index) => sum + (value - predictedAll[index]) ** 2,
      0,
    );

    return {
      experiment: id,
      features: keys.map((key) => FEATURE_LABELS[key]),
      rows: actualAll.length,
      foldCount: folds.length,
      mae: meanAbsoluteError(actualAll, predictedAll),
      rmse: rootMeanSquaredError(actualAll, predictedAll),
      r2: totalVariance === 0 ? 0 : 1 - residualVariance / totalVariance,
      baselineMae: meanAbsoluteError(actualAll, baselineAll),
      baselineRmse: rootMeanSquaredError(actualAll, baselineAll),
      folds,
    };
  });

  return {
    datasetName: "UCI Student Performance — Mathematics",
    seed: MODEL_SEED,
    requestedFolds: foldCount,
    rows: records.length,
    method:
      "Deterministic shuffled k-fold cross-validation; each fold fits feature scaling on training rows only; KNN uses k=9.",
    warning:
      "Research-only evaluation. The source dataset does not identify KV class or stream and these results do not establish KV-school validity.",
    experiments,
  };
}

