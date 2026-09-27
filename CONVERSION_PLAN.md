# MoodSense v2 — Next.js + Supabase Conversion Plan

Migration plan for **ML-Stress-Detection (MoodSense)** from a Flask + Python monolith to a
**Next.js + Supabase** stack (Auth, Postgres, Storage). Course project combining **Healthcare
Tech** (server-side ML pipeline) and **Information Storage & Management** (access control,
user-key encryption, data lifecycle).

Status: **planning** · Target: end of semester demo

---

## 1. Decisions (locked)

| Topic | Decision |
|---|---|
| Frontend | Next.js (App Router, TypeScript, Tailwind v4) |
| Auth | **Supabase Auth, email sign-up/sign-in only** (no OAuth, no magic link) |
| Database | Supabase Postgres (profiles, tests, predictions, result_seals) |
| Storage | Supabase Storage — **private** bucket, RLS-gated, **raw audio NOT encrypted** (access-pattern protected) |
| Inference | **Server-side**, in a Next.js route handler: `@tensorflow/tfjs-node` + `librosa-wasm` (MFCC parity gated) |
| Results confidentiality | **User-held key**: results payload encrypted client-side (WebCrypto: PBKDF2 → AES-256-GCM), only ciphertext stored; unlock page requires the passcode |
| Python | Removed from the deployed stack. Remains only in Google Colab (training + model conversion), which is not a backend |
| Encryption stance | Audio protected by **access control** (private bucket + RLS + signed URLs). Results protected by **real encryption** (user key) — the *conclusions* are the sensitive data |

**Why audio is not client-side encrypted:** the server must read it to analyze it. Encrypting
it with the user's key would force client-side inference and contradict the server-side choice.
This is the documented tradeoff: *access control for the audio, zero-knowledge for the results.*

---

## 2. Target architecture

```
┌─ Browser (Next.js / React) ──────────────────────────────────────┐
│  · Supabase Auth (email sign-in / sign-up)                        │
│  · take_test flow: record 42 Qs → Web Audio API merge + split     │
│  · upload raw WAV → private Storage bucket (RLS)                  │
│  · /result/[testId]: "enter your key" → decrypt → Recharts        │
└───────────────┬───────────────────────────────────────────────────┘
                │ authenticated upload + POST /api/analyze
                ▼
┌─ Next.js server (Vercel) ─────────────────────────────────────────┐
│  Route Handler /api/analyze (nodejs runtime, maxDuration=60):      │
│    fetch WAV → split PCM chunks → librosa-wasm MFCC                │
│    → tfjs-node predict (emotion 259 / depression 10293)            │
│    → returns predictions JSON (server stores NOTHING about results)│
│  Service-role Supabase client (server-only key)                    │
└───────────────┬───────────────────────────────────────────────────┘
                ▼
┌─ Supabase ────────────────────────────────────────────────────────┐
│  Auth (email/password) · Postgres + RLS · Storage (private)       │
│  tables: profiles, tests, predictions(optional), result_seals      │
└─────────────────────────────────────────────────────────────────────┘

Migration/functional data flow (per test):
  1. User signs up/in with email        → Supabase Auth session
  2. Sets a passcode at test start      → stays in browser memory only
  3. Records answers                     → take_test.js port (MediaRecorder)
  4. Decode+merge via Web Audio API      → exported as merged WAV (PCM)
  5. Upload WAV to audio/{uid}/{test_id} → private bucket, owner RLS only
  6. POST /api/analyze {testId}          → server processes & returns predictions
  7. Browser seals predictions with passcode-derived key (WebCrypto)
     → upserts result_seals row (RLS: owner writes)
  8. Redirect to /result/[testId]        → re-enter passcode → decrypt → charts
     (wrong key / tampered seal → "Invalid key" denial, AEAD auth failure)
```

---

## 3. Security model (the ISM deliverable)

| Data | Where | Protection | Who can read |
|---|---|---|---|
| Raw audio (`merged.wav`) | Storage, `audio/{uid}/{test_id}/` | Private bucket, RLS `storage.objects` policies, TLS, signed URLs (15 min) | Owner + server (service role, during analysis) |
| Predictions JSON (transient) | In-memory only | Never persisted server-side | Browser only, then discarded |
| Encoded results | Postgres `result_seals` | **AES-256-GCM** sealed with PBKDF2(passcode) key | **Only the key-holder** — not the server operator, not a DB breach |
| Key material | Never stored | Non-extractable WebCrypto key, derived per-session | Never persisted anywhere |

Threat model summary:

- **Storage/DB breach** → attacker gets raw audio (access-control only, exposed) and sealed
  result blobs (unreadable). This is the accepted tradeoff.
- **Server operator** → can run analysis, can read raw audio; **cannot** read any stored results.
- **Lost passcode** → results unrecoverable by design (demo this; optional break-glass escrow in
  Supabase Vault is a documented tradeoff, not default).

---

## 4. New repository layout

```
moodsense-v2/
├── src/
│   ├── app/
│   │   ├── login/page.tsx  register/page.tsx
│   │   ├── dashboard/page.tsx               (history of tests)
│   │   ├── test/page.tsx                    (recording flow + passcode setup)
│   │   ├── result/[testId]/page.tsx         (unlock page + charts)
│   │   ├── api/analyze/route.ts             (server-side inference)
│   │   └── layout.tsx
│   ├── components/
│   │   ├── Recorder.tsx                     (port of take_test.js)
│   │   ├── QuestionFlow.tsx
│   │   └── charts/  EmotionBar.tsx, EmotionLine.tsx, DepressionLine.tsx, EmotionPie.tsx
│   ├── lib/
│   │   ├── supabase/ client.ts, server.ts
│   │   ├── audio/ merge.ts, split.ts, wav.ts
│   │   ├── crypto/ seal.ts, unlock.ts, pbkdf2.ts
│   │   └── inference/ mfcc.ts, models.ts, labels.json
│   └── middleware.ts                        (session guard)
├── supabase/
│   ├── migrations/ 0001_init.sql, 0002_rls.sql, 0003_storage.sql
│   ├── functions/   cron-cleanup/           (retention: delete audio > 30 days)
│   └── config.toml
├── models/tfjs_model/  emotion/  depression/   (tfjs snapshot of the .keras files)
├── scripts/
│   ├── convert_keras_to_tfjs.py
│   ├── extract_labels.py                    (lb-emotion.sav / lb-depression.sav → labels.json)
│   └── parity_test.{py,mjs}                 (librosa vs librosa-wasm MFCC correlation)
└── next.config.mjs
```

---

## 5. Phases & milestones

### M0 — Scaffolding & Supabase setup (wk 1)

- [ ] `npx create-next-app@latest` (TS, App Router, Tailwind, src dir)
- [ ] `supabase init` + create a Supabase project; `supabase link`
- [ ] Enable **Email** auth provider only. Supabase dashboard → Authentication → Sign In / Up → email enabled; **disable** all other providers (Google, GitHub, Apple…).
  - Decide email confirmation: keep default on (realistic) or disable "Confirm email" for demo frictionlessness.
- [ ] `npm i @supabase/supabase-js @supabase/ssr recharts wav-encoder @tensorflow/tfjs-node`
- [ ] `.env.local`:
  ```
  NEXT_PUBLIC_SUPABASE_URL=…
  NEXT_PUBLIC_SUPABASE_ANON_KEY=…
  SUPABASE_SERVICE_ROLE_KEY=…   # server-side ONLY — never in the browser bundle
  ```
- **Acceptance:** app boots; Supabase Storage + a placeholder table visible in dashboard.

### M1 — Email sign-up / sign-in (wk 2)

Browser client (`src/lib/supabase/client.ts`):

```ts
import { createBrowserClient } from '@supabase/ssr';

export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );
}
```

Server client for route handlers + server components (`src/lib/supabase/server.ts`):

```ts
import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';

export function createClient() {
  const cookieStore = cookies();
  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() { return cookieStore.getAll(); },
        setAll(all) {
          try { all.forEach(({ name, value, options }) => cookieStore.set(name, value, options)); }
          catch {}
        },
      },
    }
  );
}
```

Session guard (`src/middleware.ts`) — refresh session, protect `/test`, `/dashboard`, `/result/*`:

```ts
import { createServerClient } from '@supabase/ssr';
import { NextResponse, type NextRequest } from 'next/server';

export async function middleware(request: NextRequest) {
  let response = NextResponse.next({ request });
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() { return request.cookies.getAll(); },
        setAll(all) {
          all.forEach(({ name, value }) => request.cookies.set(name, value));
          response = NextResponse.next({ request });
          all.forEach(({ name, value, options }) => response.cookies.set(name, value, options));
        },
      },
    }
  );
  const { data } = await supabase.auth.getUser();
  const isAuthPage = request.nextUrl.pathname.startsWith('/login') || request.nextUrl.pathname.startsWith('/register');
  if (!data.user && !isAuthPage && request.nextUrl.pathname !== '/') {
    return NextResponse.redirect(new URL('/login', request.url));
  }
  if (data.user && isAuthPage) return NextResponse.redirect(new URL('/dashboard', request.url));
  return response;
}

export const config = { matcher: ['/login', '/register', '/dashboard', '/test', '/result/:path*'] };
```

Login / register pages: call `supabase.auth.signUp({ email, password })` and
`supabase.auth.signInWithPassword(...)`; handle errors; show success → redirect to `/dashboard`.

- **Acceptance:** sign up + sign in with any email, protected routes redirect anonymous users to `/login`. No OAuth buttons anywhere.

### M2 — Schema & RLS (wk 2–3)

`supabase/migrations/0001_init.sql`:

```sql
-- profiles: 1:1 with auth.users
create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  created_at timestamptz default now()
);

-- auto-create profile on signup
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  insert into public.profiles (id) values (new.id);
  return new;
end $$;

create trigger on_auth_user_created after insert on auth.users
  for each row execute procedure public.handle_new_user();

-- one test session = one completed questionnaire run
create table public.tests (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  status text not null default 'recording'
    check (status in ('recording','processing','completed','error')),
  audio_path text,
  started_at timestamptz default now(),
  completed_at timestamptz
);
create index on public.tests (user_id, started_at desc);

-- optional per-chunk predictions (kept for analytics; the sensitive
-- conclusions live ONLY in result_seals)
create table public.predictions (
  id uuid primary key default gen_random_uuid(),
  test_id uuid not null references public.tests(id) on delete cascade,
  chunk_type text not null check (chunk_type in ('emotion_3s','depression_2m')),
  chunk_index int not null,
  predicted_label text not null,
  probs jsonb,
  created_at timestamptz default now()
);
create index on public.predictions (test_id, chunk_type);
```

`supabase/migrations/0002_rls.sql`:

```sql
alter table public.tests      enable row level security;
alter table public.predictions enable row level security;
alter table public.result_seals enable row level security;
alter table public.profiles   enable row level security;

-- profiles: user reads/updates own
create policy "profiles read own" on public.profiles
  for select using (auth.uid() = id);
create policy "profiles update own" on public.profiles
  for update using (auth.uid() = id);

-- tests: owner-only
create policy "tests read own" on public.tests
  for select using (auth.uid() = user_id);
create policy "tests insert own" on public.tests
  for insert with check (auth.uid() = user_id);
create policy "tests update own" on public.tests
  for update using (auth.uid() = user_id);

-- predictions: derived from owned tests
create policy "predictions read own" on public.predictions
  for select using (exists (
    select 1 from public.tests t where t.id = predictions.test_id and t.user_id = auth.uid()));
create policy "predictions insert own" on public.predictions
  for insert with check (exists (
    select 1 from public.tests t where t.id = predictions.test_id and t.user_id = auth.uid()));
```

`0003_storage.sql` (bucket + object RLS):

```sql
insert into storage.buckets (id, name, public, file_size_limit)
values ('audio', 'audio', false, 104857600);

-- owner-scoped paths: audio/{user_id}/{test_id}/...
create policy "audio read own" on storage.objects
  for select using (
    bucket_id = 'audio' and auth.uid()::text = (storage.foldername(name))[1]);
create policy "audio insert own" on storage.objects
  for insert with check (
    bucket_id = 'audio' and auth.uid()::text = (storage.foldername(name))[1]);
create policy "audio update own" on storage.objects
  for update using (
    bucket_id = 'audio' and auth.uid()::text = (storage.foldername(name))[1]);
create policy "audio delete own" on storage.objects
  for delete using (
    bucket_id = 'audio' and auth.uid()::text = (storage.foldername(name))[1]);
```

- **Acceptance (ISM demo):** RLS denial tests — anon and *other-user* tokens get zero rows /
  zero objects; the owner sees their own only. Record this as a test script (`scripts/rls_test.mjs`).

### M3 — Audio pipeline (wk 3–4)

Port `take_test.js` → `Recorder.tsx` (MediaRecorder, WebM, 128 kbps — same as current).

Then replace pydub (`audio_adder` + `split_audio_to_folders`) with **browser-side processing**
using the Web Audio API — the decode is the expensive step and it's free in the browser:

- `src/lib/audio/merge.ts`: decode each recorded blob with
  `OfflineAudioContext` / `AudioBuffer`, concatenate into one `AudioBuffer`.
- `src/lib/audio/wav.ts`: encode merged buffer → 16-bit PCM WAV
  (use `wav-encoder`, or ~40 lines of manual PCM write — WAV is trivial: 44-byte header + frames).
- Upload: `supabase.storage.from('audio').upload(`${userId}/${testId}/merged.wav`, wavBlob)`.

No ffmpeg anywhere. The server later splits the WAV by pure byte math.

- **Acceptance:** browser records → merged → uploaded; single `merged.wav` visible in the
  private bucket only for the owner.

### M4 — Model conversion & parity gate (wk 4–5) ⚠️ risk gate

Convert the two `.keras` models (once, in Colab or locally):

```bash
pip install tensorflowjs
tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model \
  Model/emotion.keras      models/tfjs_model/emotion
tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model \
  Model/depression.keras   models/tfjs_model/depression
```

Label encoders → `src/lib/inference/labels.json`
(`scripts/extract_labels.py`, read the pickles; verify `med`, not `mid`):

```json
{
  "emotion":     ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"],
  "depression":  ["high", "low", "med"]
}
```

**MFCC parity test (do this before wiring inference):** replicate the training-time
`extract_mfcc()` exactly — `sr=44100`, `offset=0.5`, `duration=3` (emotion) / `120`
(depression), `n_mfcc=13/20`, `np.mean(..., axis=0)` over coefficients — in the JS lib
(`librosa-wasm` preferred; `meyda` fallback). Compare vectors on ~50 real clips:

- ✅ correlation ≈ 0.99 on the mean features → proceed.
- ❌ → **retrain in Colab using the JS/WASM feature extractor** (swap the extractor in the
  existing notebooks, reuse everything else). Budget 1–3 days of Colab compute.

- **Acceptance:** `scripts/parity_test.mjs` prints the correlation report; decision recorded.

### M5 — Inference route `/api/analyze` (wk 5–6)

Route handler (`src/app/api/analyze/route.ts`):

```ts
export const runtime = 'nodejs';       // tfjs-node needs Node, not edge
export const maxDuration = 60;         // hobby max; Pro fluid up to 300 s
```

Pipeline (mirrors `class_predictor()` in `Webapp.py`):

```ts
// model singleton — load once, reuse across requests
let models: { emotion: tf.LayersModel; depression: tf.LayersModel } | null = null;
async function getModels() {
  if (!models) models = {
    emotion:    await tf.loadLayersModel('file://models/tfjs_model/emotion/model.json'),
    depression: await tf.loadLayersModel('file://models/tfjs_model/depression/model.json'),
  };
  return models;
}
```

1. Verify the user session (`server.ts` client) — the requesting user must own the test.
2. Fetch `audio/{uid}/{testId}/merged.wav` via the **service-role** client.
3. Split WAV into 3 s and 2 min chunks by PCM frame offsets
   (`frames = durationSec * sampleRate`; mono 16-bit → 2 bytes/frame) — `src/lib/audio/split.ts`.
4. MFCC per chunk (`librosa-wasm`) with the parity-approved params;
   pad/trim to **259** (emotion) / **10293** (depression); reshape `(1, n, 1)`.
5. `model.predict(tensor)` → `argmax` → label via `labels.json`.
   Batch the ~200 three-second chunks in a few `predict` calls for speed.
6. Return `{ predictions: { emotion: [...], depression: [...] } }` — the server stores
   **nothing** about results.

`next.config.mjs` — ensure tfjs-weight `.bin` files are bundled:

```js
const nextConfig = {
  outputFileTracingIncludes: { '/api/analyze': ['./models/**'] },
  serverExternalPackages: ['@tensorflow/tfjs-node'],
};
```

- **Acceptance:** POST returns predictions matching the Flask baseline
  (compare against `evaluation_results.json`: 120 emotion predictions + 3 depression `low`
  predictions for the 6-min fixture).

### M6 — Sealed results & unlock page (wk 6)

WebCrypto, all in the browser — the passcode and key never leave the client.

`src/lib/crypto/seal.ts`:

```ts
async function deriveKey(passcode: string, salt: Uint8Array) {
  const material = await crypto.subtle.importKey(
    'raw', new TextEncoder().encode(passcode), 'PBKDF2', false, ['deriveKey']);
  return crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt, iterations: 150_000, hash: 'SHA-256' },
    material, { name: 'AES-GCM', length: 256 },
    false,                                    // non-extractable — key can never be exported
    ['encrypt', 'decrypt']);
}

export async function sealResults(passcode: string, payload: unknown) {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const iv   = crypto.getRandomValues(new Uint8Array(12));
  const key  = await deriveKey(passcode, salt);
  const ct   = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv }, key,
    new TextEncoder().encode(JSON.stringify(payload)));
  return { ciphertext: ct, iv, salt };
}
```

Store the seal (RLS: owner-insert via `tests` join):

```ts
const { error } = await supabase.from('result_seals').upsert({
  test_id, ciphertext, iv, salt,
});
```

Unlock page (`/result/[testId]`): render "Enter your key to unlock your results" →
fetch the seal row (owner RLS) → `deriveKey(passcode, salt)` → `crypto.subtle.decrypt(...)`.
**Any** wrong passcode or tampered bytes → `OperationError` → show "Invalid key — access denied".
Decrypted payload drives the charts.

- **Acceptance (ISM demo):** DB row is sealed ciphertext; wrong key denied; correct key renders.
  `result_seals` unreadable even with service-role access.

### M7 — Result dashboard (wk 6–7)

Port `analyze_emotions()` (matplotlib) → Recharts:

| Flask (matplotlib) | Recharts component |
|---|---|
| emotion_counts bar chart | `EmotionBar.tsx` (BarChart) |
| emotion over time (3 s ticks, ordered `["calm","fearful","disgust","sad","angry","happy","surprised","neutral"]`) | `EmotionLine.tsx` (LineChart) |
| depression over time (2 min ticks, `["low","med","high"]`) | `DepressionLine.tsx` |
| emotion pie | `EmotionPie.tsx` |

Reuse the exact index orderings/step logic from `analyze_emotions()` so outputs match the
Flask version. Add a `/dashboard` page listing the user's tests.

### M8 — Hardening & deployment (wk 7–9)

- [ ] Retention: `supabase/functions/cron-cleanup` (triggered `cron: */15`) deletes
  `audio/{uid}/…` objects with `tests.started_at < now() - interval '30 days'` and flags the
  test as purged. Deploy via `supabase functions deploy cron-cleanup`.
- [ ] Audit log table (`who, what, test_id, ip, at`) written on result *unlock* attempts and
  exports (great for the course write-up on auditing).
- [ ] Signed URLs with 15-min expiry for any controlled playback/download.
- [ ] Backups: enable PGBackups / PITR; write a restore runbook in `docs/`.
- [ ] Deploy: `supabase db push` → Vercel project connected to the Supabase env vars →
  `next build` clean.
- [ ] `.gitignore`: `models/tfjs_model/**` if over repo limits (tfjs shards ~4 MB — verify);
  else commit with `git-lfs` or add a `scripts/download_models.mjs`.

---

## 6. Mapping of current code → new home

| Current file / function | New location / replacement |
|---|---|
| `Webapp.py` routes + `templates/*` | `src/app/*` pages (App Router) |
| `take_test.js` | `src/components/Recorder.tsx` + `QuestionFlow.tsx` |
| `audio_adder()` (pydub merge) | `src/lib/audio/merge.ts` (Web Audio API) |
| `split_audio_to_folders()` | `src/lib/audio/split.ts` (WAV PCM slicing, server-side) |
| `extract_mfcc()` (librosa) | `src/lib/inference/mfcc.ts` (librosa-wasm, parity-gated) |
| `class_predictor()` | `src/app/api/analyze/route.ts` (tfjs-node) |
| `Model/*.keras`, `Model/lb-*.sav` | `models/tfjs_model/*` + `src/lib/inference/labels.json` |
| `predictions.json` | transient in-memory → sealed in `result_seals` |
| `analyze_emotions()` matplotlib | `src/components/charts/*` (Recharts) |
| `run_models.py` | optional `scripts/evaluate.mjs` |
| `SECRET_KEY` env | Supabase Auth session + Vault for any server secrets |

---

## 7. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| MFCC parity fails (accuracy drop) | Medium–High | Parity gate at M4 before wiring inference; retrain path removes risk |
| Vercel function duration limits | Low | maxDuration=60 (Hobby) / 300 (Pro); batch predictions; heavy decode already client-side |
| tfjs `.bin` weights not bundled on Vercel | Medium | `outputFileTracingIncludes` in `next.config.mjs`; verify with deployed smoke test |
| librosa-wasm performance on 2-min chunks | Low | 2-min chunks are few (≈5); parallelize |
| Lost passcode → lost results | Certain (by design) | Document + demo; optional break-glass escrow in Vault (documented tradeoff) |
| Audio exposed on storage breach | Accepted tradeoff | Raw audio is access-control-only by decision; add retention purge |
| RLS misconfig | Medium | `scripts/rls_test.mjs` denial tests in CI or pre-demo checklist |

---

## 8. Acceptance checklist (demo script)

1. [ ] Sign up with email → confirm → dashboard
2. [ ] Anon / second user blocked from the first user's audio + seals (RLS)
3. [ ] Take test: set passcode → record → browser merges/splits → uploads `merged.wav`
4. [ ] `/api/analyze` runs server-side; predictions match Flask baseline fixture
5. [ ] `result_seals` row = sealed ciphertext (unreadable via dashboard even w/ service role)
6. [ ] Result page: wrong passcode → **Invalid key**; correct passcode → charts render
7. [ ] Retention job removes audio after policy window; sealed results remain
8. [ ] `supabase db push` + Vercel deploy green; email auth only, no OAuth

---

## 9. ISM report outline (content generated by this build)

1. Data model: ERD + rationale (UUID PKs, cascade deletes, JSONB for probabilities)
2. Access control matrix: resource × role × operation → RLS policy
3. Decision record: access-control for audio vs. user-key encryption for results (and why E2E audio is incompatible with server-side inference)
4. Key management: PBKDF2 iterations, non-extractable WebCrypto keys, salt/IV handling, lost-key semantics, optional break-glass
5. Data lifecycle: create → store → process → access → retain 30 days → purge
6. Audit & privacy: unlock-access audit log, dummy/synthetic demo data only, no real PII
7. Backup & recovery: PITR + restore runbook