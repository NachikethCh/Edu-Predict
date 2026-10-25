import assert from "node:assert/strict";
import test from "node:test";
import {
  getModelAvailability,
  getModelOverview,
  predictCsv,
  predictStudent,
  SUPPORTED_MODEL_ID,
} from "../src/lib/student-performance.ts";

const sampleStudent = {
  g1: 12,
  age: 17,
  studytime: 3,
  failures: 0,
  absences: 2,
  schoolsup: false,
  famsup: true,
  paid: false,
  higher: true,
  internet: true,
  goout: 2,
  traveltime: 1,
  health: 4,
};

const headers = [
  "student_id",
  "class",
  "stream",
  "subject",
  "previous_marks",
  "age",
  "study_time",
  "failures",
  "absences",
  "school_support",
  "family_support",
  "paid_classes",
  "higher_education_intent",
  "internet_access",
  "go_out",
  "travel_time",
  "health",
].join(",");

const validRow = [
  "KV-17",
  "IX",
  "",
  "Mathematics",
  12,
  17,
  3,
  0,
  2,
  "no",
  "yes",
  "no",
  "yes",
  "yes",
  2,
  1,
  4,
].join(",");

test("trains and evaluates only the Mathematics research model", () => {
  const overview = getModelOverview();
  assert.equal(overview.seed, 42);
  assert.equal(overview.evaluations.length, 1);
  assert.equal(overview.evaluations[0].subject, "Mathematics");
  assert.equal(overview.evaluations[0].rows, 395);
  assert.equal(
    overview.evaluations[0].trainRows + overview.evaluations[0].testRows,
    overview.evaluations[0].rows,
  );
  assert.ok(Number.isFinite(overview.evaluations[0].mae));
  assert.ok(Number.isFinite(overview.evaluations[0].rmse));
  assert.ok(Number.isFinite(overview.evaluations[0].r2));
  assert.ok(Number.isFinite(overview.evaluations[0].baselineMae));
  assert.ok(
    overview.evaluations[0].featureImportance.some(
      (item) => item.feature === "g1",
    ),
  );
  assert.equal(overview.modelSupport.length, 1);
  assert.equal(overview.modelSupport[0].modelId, SUPPORTED_MODEL_ID);
  assert.equal(overview.modelSupport[0].kvValidated, false);
});

test("supports only secondary Mathematics and rejects unrelated subjects", () => {
  assert.equal(
    getModelAvailability(9, null, "Mathematics").supported,
    true,
  );
  assert.equal(
    getModelAvailability(11, "science", "Mathematics").supported,
    true,
  );
  assert.equal(
    getModelAvailability(8, null, "Mathematics").supported,
    false,
  );
  assert.equal(
    getModelAvailability(10, null, "Science").supported,
    false,
  );
  assert.equal(
    getModelAvailability(9, null, "Portuguese").supported,
    false,
  );
  assert.equal(
    getModelAvailability(12, "humanities", "Physics").supported,
    false,
  );
  assert.equal(
    getModelAvailability(11, null, "Mathematics").supported,
    false,
  );

  const result = predictStudent(9, null, "Mathematics", sampleStudent);
  assert.equal(result.classLevel, 9);
  assert.equal(result.subject, "Mathematics");
  assert.ok(result.predictedGrade >= 0 && result.predictedGrade <= 20);
  assert.equal(result.neighborCount, 9);
  assert.ok(
    result.similarStudentPassShare >= 0 &&
      result.similarStudentPassShare <= 1,
  );
  assert.throws(
    () => predictStudent(9, null, "Science", sampleStudent),
    /No trained and evaluated model/,
  );
});

test("validates each CSV row and reports unsupported combinations without prediction", () => {
  const unsupportedRow = validRow.replace("Mathematics", "Science");
  const invalidMarks = validRow.replace(",12,17,", ",120,17,");
  const result = predictCsv(
    `${headers}\n${validRow}\n${unsupportedRow}\n${invalidMarks}`,
  );

  assert.equal(result.validCount, 1);
  assert.equal(result.errorCount, 2);
  assert.equal(result.predictions[0].rowNumber, 2);
  assert.equal(result.predictions[0].studentId, "KV-17");
  assert.equal(result.predictions[0].classLevel, 9);
  assert.equal(result.predictions[0].subject, "Mathematics");
  assert.match(result.errors[0].message, /No trained and evaluated model/);
  assert.match(result.errors[1].message, /between 0 and 20/);
});

test("rejects CSV files with missing required headers", () => {
  assert.throws(
    () => predictCsv("class,subject,previous_marks\n9,Mathematics,12"),
    /Missing required columns/,
  );
});
