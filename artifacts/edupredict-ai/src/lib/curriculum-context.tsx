import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  buildDefaultCurriculum,
  getSubjectsFor,
  isValidCurriculumConfig,
  type CurriculumConfig,
  type SchoolStream,
  type StudentSelection,
} from "./curriculum-config";

const CURRICULUM_STORAGE_KEY = "edupredict-kv-curriculum-v1";
const SELECTION_STORAGE_KEY = "edupredict-kv-selection-v1";

function readStoredCurriculum(): CurriculumConfig {
  try {
    const stored = localStorage.getItem(CURRICULUM_STORAGE_KEY);
    if (!stored) return buildDefaultCurriculum();
    const parsed: unknown = JSON.parse(stored);
    return isValidCurriculumConfig(parsed) ? parsed : buildDefaultCurriculum();
  } catch {
    return buildDefaultCurriculum();
  }
}

function readStoredSelection(): StudentSelection {
  const fallback: StudentSelection = {
    classLevel: 9,
    stream: null,
    subject: "Mathematics",
  };
  try {
    const stored = localStorage.getItem(SELECTION_STORAGE_KEY);
    if (!stored) return fallback;
    const parsed = JSON.parse(stored) as Partial<StudentSelection>;
    if (
      !Number.isInteger(parsed.classLevel) ||
      (parsed.classLevel ?? 0) < 1 ||
      (parsed.classLevel ?? 0) > 12
    ) {
      return fallback;
    }
    const classLevel = parsed.classLevel!;
    const stream =
      classLevel >= 11 &&
      (parsed.stream === "science" ||
        parsed.stream === "commerce" ||
        parsed.stream === "humanities")
        ? parsed.stream
        : null;
    return {
      classLevel,
      stream,
      subject: typeof parsed.subject === "string" ? parsed.subject : "",
    };
  } catch {
    return fallback;
  }
}

type CurriculumContextValue = {
  curriculum: CurriculumConfig;
  selection: StudentSelection;
  subjects: string[];
  setClassLevel: (classLevel: number) => void;
  setStream: (stream: SchoolStream) => void;
  setSubject: (subject: string) => void;
  saveSubjects: (
    classLevel: number,
    stream: SchoolStream | null,
    subjects: string[],
  ) => void;
  resetCurriculum: () => void;
};

const CurriculumContext = createContext<CurriculumContextValue | null>(null);

export function CurriculumProvider({ children }: { children: ReactNode }) {
  const [curriculum, setCurriculum] = useState(readStoredCurriculum);
  const [selection, setSelection] = useState(readStoredSelection);
  const subjects = useMemo(
    () => getSubjectsFor(curriculum, selection.classLevel, selection.stream),
    [curriculum, selection.classLevel, selection.stream],
  );

  useEffect(() => {
    localStorage.setItem(CURRICULUM_STORAGE_KEY, JSON.stringify(curriculum));
  }, [curriculum]);

  useEffect(() => {
    localStorage.setItem(SELECTION_STORAGE_KEY, JSON.stringify(selection));
  }, [selection]);

  useEffect(() => {
    if (subjects.includes(selection.subject)) return;
    setSelection((current) => ({
      ...current,
      subject: subjects.includes("Mathematics")
        ? "Mathematics"
        : (subjects[0] ?? ""),
    }));
  }, [selection.subject, subjects]);

  const setClassLevel = useCallback(
    (classLevel: number) => {
      setSelection((current) => {
        const stream =
          classLevel >= 11
            ? (current.stream ?? "science")
            : null;
        const nextSubjects = getSubjectsFor(curriculum, classLevel, stream);
        return {
          classLevel,
          stream,
          subject: nextSubjects.includes(current.subject)
            ? current.subject
            : nextSubjects.includes("Mathematics")
              ? "Mathematics"
              : (nextSubjects[0] ?? ""),
        };
      });
    },
    [curriculum],
  );

  const setStream = useCallback(
    (stream: SchoolStream) => {
      setSelection((current) => {
        const nextSubjects = getSubjectsFor(
          curriculum,
          current.classLevel,
          stream,
        );
        return {
          ...current,
          stream,
          subject: nextSubjects.includes(current.subject)
            ? current.subject
            : nextSubjects.includes("Mathematics")
              ? "Mathematics"
              : (nextSubjects[0] ?? ""),
        };
      });
    },
    [curriculum],
  );

  const setSubject = useCallback(
    (subject: string) =>
      setSelection((current) => ({ ...current, subject })),
    [],
  );

  const saveSubjects = useCallback(
    (
      classLevel: number,
      stream: SchoolStream | null,
      subjects: string[],
    ) => {
      setCurriculum((current) =>
        current.map((entry) =>
          entry.classLevel === classLevel && entry.stream === stream
            ? { ...entry, subjects }
            : entry,
        ),
      );
    },
    [],
  );

  const resetCurriculum = useCallback(() => {
    setCurriculum(buildDefaultCurriculum());
    setSelection({ classLevel: 9, stream: null, subject: "Mathematics" });
  }, []);

  const value = useMemo(
    () => ({
      curriculum,
      selection,
      subjects,
      setClassLevel,
      setStream,
      setSubject,
      saveSubjects,
      resetCurriculum,
    }),
    [
      curriculum,
      selection,
      subjects,
      setClassLevel,
      setStream,
      setSubject,
      saveSubjects,
      resetCurriculum,
    ],
  );

  return (
    <CurriculumContext.Provider value={value}>
      {children}
    </CurriculumContext.Provider>
  );
}

export function useCurriculum() {
  const context = useContext(CurriculumContext);
  if (!context) {
    throw new Error("useCurriculum must be used inside CurriculumProvider.");
  }
  return context;
}
