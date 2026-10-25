import type { CsvPredictionResult } from "@workspace/api-client-react";

function escapeCsvCell(value: string | number): string {
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function serializeCsvResults(result: CsvPredictionResult): string {
  const lines = [
    [
      "rowNumber",
      "predictedGrade",
      "gradeBand",
      "similarStudentPassShare",
      "status",
      "error",
    ].join(","),
  ];

  result.predictions.forEach((row) => {
    lines.push(
      [
        row.rowNumber,
        row.predictedGrade,
        row.gradeBand,
        row.similarStudentPassShare,
        "valid",
        "",
      ]
        .map(escapeCsvCell)
        .join(","),
    );
  });
  result.errors.forEach((row) => {
    lines.push(
      [row.rowNumber, "", "", "", "error", row.message]
        .map(escapeCsvCell)
        .join(","),
    );
  });

  return lines.join("\r\n");
}
