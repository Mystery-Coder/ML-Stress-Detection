-- MoodSense schema: profiles, tests, predictions, sealed results, audit log.

create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  created_at timestamptz default now()
);

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id) values (new.id);
  return new;
end;
$$;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();

create table public.tests (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  status text not null default 'recording'
    check (status in ('recording', 'processing', 'completed', 'error', 'purged')),
  audio_path text,
  started_at timestamptz default now(),
  completed_at timestamptz
);

create index tests_user_started_idx on public.tests (user_id, started_at desc);

create table public.predictions (
  id uuid primary key default gen_random_uuid(),
  test_id uuid not null references public.tests(id) on delete cascade,
  chunk_type text not null check (chunk_type in ('emotion_3s', 'depression_2m')),
  chunk_index int not null,
  predicted_label text not null,
  probs jsonb,
  created_at timestamptz default now()
);

create index predictions_test_chunk_idx on public.predictions (test_id, chunk_type);

-- AES-GCM seal. ciphertext, iv, and salt are base64 text so supabase-js can upsert them.
create table public.result_seals (
  test_id uuid primary key references public.tests(id) on delete cascade,
  ciphertext text not null,
  iv text not null,
  salt text not null,
  created_at timestamptz default now()
);

create table public.audit_log (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete set null,
  action text not null,
  test_id uuid,
  ip text,
  created_at timestamptz default now()
);
