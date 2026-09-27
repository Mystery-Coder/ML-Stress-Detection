-- Private audio bucket, 100 MB. Object keys are {user_id}/{test_id}/...

insert into storage.buckets (id, name, public, file_size_limit)
values ('audio', 'audio', false, 104857600);

create policy "audio read own" on storage.objects
  for select using (
    bucket_id = 'audio'
    and auth.uid()::text = (storage.foldername(name))[1]
  );

create policy "audio insert own" on storage.objects
  for insert with check (
    bucket_id = 'audio'
    and auth.uid()::text = (storage.foldername(name))[1]
  );

create policy "audio update own" on storage.objects
  for update using (
    bucket_id = 'audio'
    and auth.uid()::text = (storage.foldername(name))[1]
  );

create policy "audio delete own" on storage.objects
  for delete using (
    bucket_id = 'audio'
    and auth.uid()::text = (storage.foldername(name))[1]
  );
