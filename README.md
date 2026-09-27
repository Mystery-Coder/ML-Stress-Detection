# MoodSense

Voice screening for emotion and low-mood indicators. The app is **Next.js + Supabase** (email/password auth, private audio storage, user-held encryption for results). A local **Python worker** runs Keras + librosa inference; `/api/analyze` only checks the session, fetches the WAV, and forwards bytes to that worker.

## Stack

- Next.js (App Router, TypeScript, Tailwind)
- Supabase Auth, Postgres + RLS, private Storage
- Local inference: `infer/app.py` (original MFCC + `.keras` models)
- Results: PBKDF2 → AES-256-GCM in the browser (`result_seals`)

Raw audio is access-control only (private bucket + RLS). Predictions are never stored in plaintext.

## Setup

```bash
cp .env.example .env.local
# Fill NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY, SUPABASE_SECRET_KEY
npm install
npx supabase db push   # or apply supabase/migrations in the dashboard
# terminal 1 — inference worker (Python 3.12 venv)
uv pip install -r requirements.txt
python -m infer.app
# terminal 2
npm run dev
```

Auth is **email + password only** (no OAuth). Enable Email in the Supabase dashboard.

## Scripts

| Command | Purpose |
|---|---|
| `npm run rls-test` | Owner vs other-user RLS denial |
| `python scripts/parity_test.py` then `npm run parity` | MFCC JS vs librosa |
| `python scripts/run_models.py` | Offline Keras folder eval (optional) |

Retention: `supabase/functions/cron-cleanup` deletes audio older than 30 days; sealed results stay.

Restore notes: `docs/restore-runbook.md`.
