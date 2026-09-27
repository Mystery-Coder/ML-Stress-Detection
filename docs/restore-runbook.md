# MoodSense restore runbook (PITR)

Postgres point-in-time recovery brings back `profiles`, `tests`, `predictions`, `result_seals`, and `audit_log`. It does not restore Storage objects. Raw audio in the private `audio` bucket is access-controlled and deleted after 30 days by `cron-cleanup`. Sealed results stay in `result_seals` and are only readable with the user's passcode.

## Enable PITR

1. Supabase Dashboard → Project → Database → Backups.
2. Enable Point in Time Recovery (paid add-on). Confirm WAL archiving is active and note the retention window.
3. Keep daily backups on as well.

## Restore

1. Pick the timestamp just before the bad change. Stay inside the PITR window.
2. Restore to a **new** project (or a branch) so production stays up until the copy is checked.
3. On the restored project, confirm:
   - `result_seals` row count matches the expected time.
   - RLS is still enabled on `profiles`, `tests`, `predictions`, `result_seals`, `audit_log`.
   - `storage.buckets` still has private bucket `audio` (100 MB).
4. Do not try to decrypt seals. Ciphertext without the user passcode is expected.
5. Audio files are not in the database backup. If an object was deleted from Storage, PITR will not bring it back. Re-link the app only after you accept that gap.
6. Point the app at the restored project: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY`, `SUPABASE_SECRET_KEY`.
7. Redeploy `cron-cleanup` (`verify_jwt = false`) and confirm the schedule is still `*/15`.

## After cutover

Spot-check one owner: their tests and seal load; another user and the anon key see nothing. Leave `result_seals` in place even when `tests.status` is `purged`.
