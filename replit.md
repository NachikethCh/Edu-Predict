# EduPredict AI

An evidence-aware student performance predictor that estimates final grades from first-period academic signals and supports practical, non-judgmental next steps.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the shared API server
- `pnpm --filter @workspace/edupredict-ai run dev` — run the EduPredict web app
- `pnpm --filter @workspace/api-server run typecheck` — check backend TypeScript
- `pnpm --filter @workspace/edupredict-ai run typecheck` — check frontend TypeScript
- `node --experimental-strip-types --test artifacts/api-server/test/student-performance.test.mjs` — run model and CSV tests
- `pnpm run typecheck` — full workspace check

## Stack

- pnpm workspace, Node.js 24, TypeScript
- Frontend: React, Vite, Wouter, TanStack Query
- API: Express 5, Zod validation, OpenAPI-generated client and schemas
- Model: deterministic, standardized inverse-distance weighted k-nearest-neighbors regression
- Dataset: UCI Student Performance, Mathematics and Portuguese language subsets
- No user account, persistent student records, paid service, or database is required

## Where things live

- `artifacts/edupredict-ai/` — responsive dashboard, prediction form, evaluation view, and CSV workflow
- `artifacts/api-server/src/lib/student-performance.ts` — dataset loading, training split, metrics, predictions, and CSV validation
- `artifacts/api-server/data/` — attributed UCI CSV source files bundled with the API
- `artifacts/api-server/src/routes/student-performance.ts` — model, prediction, and CSV endpoints
- `lib/api-spec/openapi.yaml` — source of truth for app API contracts
- `docs/project-report.md` — CBSE-inspired project write-up and completion templates

## Architecture decisions

- Predict G3 after the first grading period using G1 and selected early context; G2 and G3 are never model inputs.
- Keep student predictions in memory only for the active request; do not persist uploaded or entered student records.
- Report deterministic held-out metrics against a training-mean baseline and expose permutation feature importance.
- Use the public UCI dataset with explicit source and license attribution; do not present it as a CBSE or Indian student dataset.

## Product

Students and educators can review the project, estimate a final-grade range, see similar-record context and input signals, inspect held-out evaluation metrics, and validate or export CSV predictions. Results are educational support prompts, not official grades or individual judgments.

## Gotchas

- The UCI files use semicolon delimiters and quoted cells; do not assume a basic comma-separated parser.
- The API build copies `data/` into `dist/data/`; keep this step when changing the backend build.
- The API is mounted at `/api`; web requests must keep this prefix.
- The project has no deployment readiness until the frontend and API workflows have both been checked together.
