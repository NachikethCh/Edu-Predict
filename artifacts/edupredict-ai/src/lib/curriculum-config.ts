export type SchoolStream = "science" | "commerce" | "humanities";

export type CurriculumEntry = {
  classLevel: number;
  stream: SchoolStream | null;
  subjects: string[];
};

export type CurriculumConfig = CurriculumEntry[];

export type StudentSelection = {
  classLevel: number;
  stream: SchoolStream | null;
  subject: string;
};

export type ModelSupportEntry = {
  modelId: string;
  subject: string;
  classLevels: number[];
  streams: SchoolStream[];
  scope: string;
  classSpecific: boolean;
  kvValidated: boolean;
};

export const SCHOOL_STREAMS: SchoolStream[] = [
  "science",
  "commerce",
  "humanities",
];

export const STREAM_LABELS: Record<SchoolStream, string> = {
  science: "Science",
  commerce: "Commerce",
  humanities: "Humanities",
};

const PRIMARY_SUBJECTS = [
  "English",
  "Hindi",
  "Mathematics",
  "Environmental Studies (EVS)",
];

const MIDDLE_SUBJECTS = [
  "English",
  "Hindi",
  "Mathematics",
  "Science",
  "Social Science",
  "Sanskrit",
  "Coding / Artificial Intelligence",
  "Other school-offered language or subject",
];

const SECONDARY_SUBJECTS = [
  "English",
  "Hindi",
  "Mathematics",
  "Science",
  "Social Science",
  "Sanskrit",
  "Artificial Intelligence",
  "Information Technology",
  "Computer Applications",
  "Other permitted CBSE elective or skill subject",
];

const SENIOR_SUBJECTS: Record<SchoolStream, string[]> = {
  science: [
    "English",
    "Hindi",
    "Other offered language",
    "Physics",
    "Chemistry",
    "Mathematics",
    "Biology",
    "Computer Science",
    "Informatics Practices",
    "Physical Education",
    "Other permitted elective",
  ],
  commerce: [
    "English",
    "Hindi",
    "Other offered language",
    "Accountancy",
    "Business Studies",
    "Economics",
    "Mathematics",
    "Applied Mathematics",
    "Informatics Practices",
    "Computer Science",
    "Entrepreneurship",
    "Physical Education",
    "Other permitted elective",
  ],
  humanities: [
    "English",
    "Hindi",
    "Other offered language",
    "History",
    "Geography",
    "Political Science",
    "Economics",
    "Sociology",
    "Psychology",
    "Mathematics",
    "Physical Education",
    "Other permitted elective",
  ],
};

export function buildDefaultCurriculum(): CurriculumConfig {
  const entries: CurriculumConfig = [];
  for (let classLevel = 1; classLevel <= 10; classLevel += 1) {
    let subjects: string[];
    if (classLevel <= 2) subjects = PRIMARY_SUBJECTS;
    else if (classLevel <= 5) subjects = PRIMARY_SUBJECTS;
    else if (classLevel <= 8) subjects = MIDDLE_SUBJECTS;
    else subjects = SECONDARY_SUBJECTS;
    entries.push({ classLevel, stream: null, subjects: [...subjects] });
  }
  for (const classLevel of [11, 12]) {
    for (const stream of SCHOOL_STREAMS) {
      entries.push({
        classLevel,
        stream,
        subjects: [...SENIOR_SUBJECTS[stream]],
      });
    }
  }
  return entries;
}

export function getCurriculumEntry(
  config: CurriculumConfig,
  classLevel: number,
  stream: SchoolStream | null,
): CurriculumEntry | undefined {
  return config.find(
    (entry) => entry.classLevel === classLevel && entry.stream === stream,
  );
}

export function getSubjectsFor(
  config: CurriculumConfig,
  classLevel: number,
  stream: SchoolStream | null,
): string[] {
  return getCurriculumEntry(config, classLevel, stream)?.subjects ?? [];
}

export function isValidCurriculumConfig(value: unknown): value is CurriculumConfig {
  if (!Array.isArray(value) || value.length !== 16) return false;
  const keys = new Set<string>();
  for (const entry of value) {
    if (
      typeof entry !== "object" ||
      entry === null ||
      !("classLevel" in entry) ||
      !("stream" in entry) ||
      !("subjects" in entry)
    ) {
      return false;
    }
    const item = entry as CurriculumEntry;
    if (!Number.isInteger(item.classLevel) || item.classLevel < 1 || item.classLevel > 12) {
      return false;
    }
    if (
      item.classLevel <= 10
        ? item.stream !== null
        : !SCHOOL_STREAMS.includes(item.stream as SchoolStream)
    ) {
      return false;
    }
    if (
      !Array.isArray(item.subjects) ||
      item.subjects.length > 40 ||
      item.subjects.some(
        (subject) =>
          typeof subject !== "string" ||
          subject.trim().length === 0 ||
          subject.trim().length > 80,
      )
    ) {
      return false;
    }
    const normalizedSubjects = item.subjects.map((subject) =>
      subject.trim().toLowerCase(),
    );
    if (new Set(normalizedSubjects).size !== normalizedSubjects.length) return false;
    const key = `${item.classLevel}:${item.stream ?? "none"}`;
    if (keys.has(key)) return false;
    keys.add(key);
  }
  return keys.size === 16;
}

export function findModelSupport(
  supportEntries: ModelSupportEntry[] | undefined,
  selection: StudentSelection,
): ModelSupportEntry | undefined {
  if (!supportEntries) return undefined;
  return supportEntries.find(
    (model) =>
      model.subject.trim().toLowerCase() === selection.subject.trim().toLowerCase() &&
      model.classLevels.includes(selection.classLevel) &&
      (selection.classLevel <= 10 ||
        (selection.stream !== null && model.streams.includes(selection.stream))),
  );
}

export function classLabel(classLevel: number): string {
  const romanNumerals = [
    "",
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
  ];
  return `Class ${romanNumerals[classLevel] ?? classLevel}`;
}
