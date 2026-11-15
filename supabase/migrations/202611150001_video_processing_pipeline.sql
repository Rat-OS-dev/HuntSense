create table if not exists public.video_processing_jobs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  source_path text not null,
  source_title text not null,
  status text not null default 'queued',
  duration_seconds numeric,
  error_message text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists video_processing_jobs_queue_idx on public.video_processing_jobs(status, created_at);
alter table public.video_processing_jobs enable row level security;
drop policy if exists "Users can view their own video processing jobs" on public.video_processing_jobs;
create policy "Users can view their own video processing jobs" on public.video_processing_jobs for select to authenticated using (auth.uid() = user_id);
drop policy if exists "Users can create their own video processing jobs" on public.video_processing_jobs;
create policy "Users can create their own video processing jobs" on public.video_processing_jobs for insert to authenticated with check (auth.uid() = user_id);
drop policy if exists "Users can delete their own video processing jobs" on public.video_processing_jobs;
create policy "Users can delete their own video processing jobs" on public.video_processing_jobs for delete to authenticated using (auth.uid() = user_id);

create table if not exists public.video_fight_clips (
  id uuid primary key default gen_random_uuid(),
  match_id uuid not null references public.match_analyses(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  storage_path text not null,
  start_sec numeric not null,
  end_sec numeric not null,
  created_at timestamptz not null default now()
);
create index if not exists video_fight_clips_match_idx on public.video_fight_clips(match_id, start_sec);
alter table public.video_fight_clips enable row level security;
drop policy if exists "Users can view their own fight clips" on public.video_fight_clips;
create policy "Users can view their own fight clips" on public.video_fight_clips for select to authenticated using (auth.uid() = user_id);
alter table public.video_fight_clips add column if not exists analysis_notes text;
