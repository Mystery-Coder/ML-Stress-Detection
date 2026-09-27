-- Owner-only access. Service role bypasses RLS (used by cron-cleanup).

alter table public.profiles enable row level security;
alter table public.tests enable row level security;
alter table public.predictions enable row level security;
alter table public.result_seals enable row level security;
alter table public.audit_log enable row level security;

create policy "profiles read own" on public.profiles
  for select using (auth.uid() = id);

create policy "profiles update own" on public.profiles
  for update using (auth.uid() = id);

create policy "tests read own" on public.tests
  for select using (auth.uid() = user_id);

create policy "tests insert own" on public.tests
  for insert with check (auth.uid() = user_id);

create policy "tests update own" on public.tests
  for update using (auth.uid() = user_id);

create policy "predictions read own" on public.predictions
  for select using (exists (
    select 1 from public.tests t
    where t.id = predictions.test_id and t.user_id = auth.uid()
  ));

create policy "predictions insert own" on public.predictions
  for insert with check (exists (
    select 1 from public.tests t
    where t.id = predictions.test_id and t.user_id = auth.uid()
  ));

create policy "result_seals read own" on public.result_seals
  for select using (exists (
    select 1 from public.tests t
    where t.id = result_seals.test_id and t.user_id = auth.uid()
  ));

create policy "result_seals insert own" on public.result_seals
  for insert with check (exists (
    select 1 from public.tests t
    where t.id = result_seals.test_id and t.user_id = auth.uid()
  ));

-- Upsert from the browser needs UPDATE as well as INSERT.
create policy "result_seals update own" on public.result_seals
  for update using (exists (
    select 1 from public.tests t
    where t.id = result_seals.test_id and t.user_id = auth.uid()
  ))
  with check (exists (
    select 1 from public.tests t
    where t.id = result_seals.test_id and t.user_id = auth.uid()
  ));

create policy "audit_log read own" on public.audit_log
  for select using (auth.uid() = user_id);

create policy "audit_log insert own" on public.audit_log
  for insert with check (auth.uid() = user_id);
