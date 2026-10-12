-- ELEVATE in Amsterdam: survey layer (pilot), PostgreSQL schema for Supabase.
-- Run once in the Supabase SQL editor. Re-running is safe: objects are created if missing and
-- functions are replaced.
--
-- Design
-- * Anonymous: a random session id made in the browser, an optional stakeholder role, nothing else.
--   No names, no e-mail, no free text, no IP address in these tables.
-- * Visitors (the anon / publishable key) may only INSERT. They cannot read, change or delete rows.
--   Results are published only as aggregates, through SECURITY DEFINER functions.
-- * Every answer is checked by constraints; a trigger caps the rows one session can add.
-- * Pilot only: the page says that responses are not research findings without an ethics-approved study.

create extension if not exists pgcrypto;

-- ---------------------------------------------------------------- sessions
create table if not exists public.survey_sessions (
  id          uuid primary key,
  created_at  timestamptz not null default now(),
  consent     boolean not null check (consent),
  role        text check (role in ('resident', 'planner', 'public_health', 'municipality',
                                   'designer', 'civic', 'researcher', 'other')),
  amsterdam   text check (amsterdam in ('lives', 'works', 'visits', 'no')),
  client      text check (char_length(client) <= 40)
);

-- ------------------------------------------- WP6: reactions to AI pictures
-- Each pair shows the 2025 photo and its AI-edited version side by side, sides randomised and
-- unlabelled; then the edited one is revealed and rated for realism.
create table if not exists public.wp6_responses (
  id          uuid primary key default gen_random_uuid(),
  session_id  uuid not null references public.survey_sessions (id) on delete cascade,
  created_at  timestamptz not null default now(),
  image_id    text not null check (image_id ~ '^recording_[0-9_-]+_(front|back|left|right)$'),
  gap_type    text not null check (gap_type in ('crossing', 'trees', 'seating', 'greenery', 'cycle')),
  edited_side text not null check (edited_side in ('left', 'right')),
  choice      text not null check (choice in ('left', 'right', 'none')),
  realism     smallint check (realism between 1 and 5),
  rt_ms       integer check (rt_ms between 0 and 3600000)
);

-- ------------------------------- WP3: pairwise perception of street views
create table if not exists public.wp3_pairwise (
  id          uuid primary key default gen_random_uuid(),
  session_id  uuid not null references public.survey_sessions (id) on delete cascade,
  created_at  timestamptz not null default now(),
  question    text not null check (question in ('pleasant', 'safe')),
  left_id     text not null check (left_id ~ '^recording_[0-9_-]+_(front|back|left|right)$'),
  right_id    text not null check (right_id ~ '^recording_[0-9_-]+_(front|back|left|right)$'),
  choice      text not null check (choice in ('left', 'right', 'equal')),
  rt_ms       integer check (rt_ms between 0 and 3600000),
  check (left_id <> right_id)
);

-- ------------------------------------------- WP1: indicator priorities
-- The draft list of 15 indicators (WP1); a ranking is the ordered top five, by draft rank number.
create table if not exists public.wp1_indicators (
  rank_id     smallint primary key check (rank_id between 1 and 15),
  indicator   text not null
);
insert into public.wp1_indicators (rank_id, indicator) values
  (1, 'pedestrian crossing at intersections'), (2, 'green view (vegetation share)'),
  (3, 'shade on walkway'), (4, 'seating'), (5, 'sidewalk present'),
  (6, 'street lighting (pole present)'), (7, 'intersection density (from images)'),
  (8, 'cycle infrastructure'), (9, 'sidewalk obstruction'), (10, 'active frontage'),
  (11, 'perceived beauty / safety / liveliness'), (12, 'sidewalk parking'),
  (13, 'accessibility barrier'), (14, 'litter'), (15, 'surface damage, traffic calming, sidewalk gap')
on conflict (rank_id) do update set indicator = excluded.indicator;

create or replace function public.valid_top_five(r smallint[]) returns boolean
language sql immutable as $$
  select cardinality(r) = 5
     and (select count(distinct x) from unnest(r) x) = 5
     and (select bool_and(x between 1 and 15) from unnest(r) x)
$$;

create table if not exists public.wp1_rankings (
  id          uuid primary key default gen_random_uuid(),
  session_id  uuid not null references public.survey_sessions (id) on delete cascade,
  created_at  timestamptz not null default now(),
  top_five    smallint[] not null check (public.valid_top_five(top_five))
);

-- --------------------------------------------------------- spam guard
create or replace function public.cap_rows_per_session() returns trigger
language plpgsql security definer set search_path = public as $$
declare n integer;
begin
  execute format('select count(*) from %I.%I where session_id = $1', tg_table_schema, tg_table_name)
    into n using new.session_id;
  if n >= 150 then
    raise exception 'this session has already answered enough, thank you';
  end if;
  return new;
end $$;

drop trigger if exists cap_wp6 on public.wp6_responses;
create trigger cap_wp6 before insert on public.wp6_responses for each row execute function public.cap_rows_per_session();
drop trigger if exists cap_wp3 on public.wp3_pairwise;
create trigger cap_wp3 before insert on public.wp3_pairwise for each row execute function public.cap_rows_per_session();
drop trigger if exists cap_wp1 on public.wp1_rankings;
create trigger cap_wp1 before insert on public.wp1_rankings for each row execute function public.cap_rows_per_session();

-- ---------------------------------------- row level security: insert only
alter table public.survey_sessions enable row level security;
alter table public.wp6_responses   enable row level security;
alter table public.wp3_pairwise    enable row level security;
alter table public.wp1_rankings    enable row level security;
alter table public.wp1_indicators  enable row level security;

revoke all on public.survey_sessions, public.wp6_responses, public.wp3_pairwise, public.wp1_rankings,
              public.wp1_indicators from anon, authenticated;
grant insert on public.survey_sessions, public.wp6_responses, public.wp3_pairwise, public.wp1_rankings
  to anon, authenticated;

drop policy if exists insert_sessions on public.survey_sessions;
create policy insert_sessions on public.survey_sessions for insert to anon, authenticated with check (consent);
drop policy if exists insert_wp6 on public.wp6_responses;
create policy insert_wp6 on public.wp6_responses for insert to anon, authenticated with check (true);
drop policy if exists insert_wp3 on public.wp3_pairwise;
create policy insert_wp3 on public.wp3_pairwise for insert to anon, authenticated with check (true);
drop policy if exists insert_wp1 on public.wp1_rankings;
create policy insert_wp1 on public.wp1_rankings for insert to anon, authenticated with check (true);

-- ------------------------------------------- published aggregates only
create or replace function public.survey_overview() returns jsonb
language sql stable security definer set search_path = public as $$
  select jsonb_build_object(
    'sessions',     (select count(*) from survey_sessions),
    'wp6_answers',  (select count(*) from wp6_responses),
    'wp3_answers',  (select count(*) from wp3_pairwise),
    'wp1_rankings', (select count(*) from wp1_rankings),
    'roles', coalesce((select jsonb_object_agg(coalesce(role, 'not given'), n)
                         from (select role, count(*) n from survey_sessions group by role) r), '{}'::jsonb),
    'since',        (select min(created_at) from survey_sessions)
  )
$$;

-- WP6: per picture pair, how often the edited street was preferred, and how realistic it looked
create or replace function public.wp6_summary()
returns table (image_id text, gap_type text, n bigint, prefer_edited bigint, prefer_photo bigint,
               no_preference bigint, realism_n bigint, realism_mean numeric)
language sql stable security definer set search_path = public as $$
  select image_id, gap_type, count(*),
         count(*) filter (where choice = edited_side),
         count(*) filter (where choice <> edited_side and choice <> 'none'),
         count(*) filter (where choice = 'none'),
         count(realism), round(avg(realism), 2)
  from wp6_responses group by image_id, gap_type order by image_id
$$;

-- WP3: per view and question, comparisons, wins, ties and win rate (ties count half)
create or replace function public.wp3_scores()
returns table (image_id text, question text, comparisons bigint, wins bigint, ties bigint, win_rate numeric)
language sql stable security definer set search_path = public as $$
  with sides as (
    select question, left_id as image_id, (choice = 'left') as won, (choice = 'equal') as tied from wp3_pairwise
    union all
    select question, right_id, (choice = 'right'), (choice = 'equal') from wp3_pairwise
  )
  select image_id, question, count(*), count(*) filter (where won), count(*) filter (where tied),
         round((count(*) filter (where won) + 0.5 * count(*) filter (where tied))::numeric / count(*), 3)
  from sides group by image_id, question order by question, image_id
$$;

-- WP1: how often each indicator is in a top five, its Borda points (5 for first ... 1 for fifth),
-- and its mean position among the rankings that list it; optionally for one stakeholder role
create or replace function public.wp1_summary(for_role text default null)
returns table (rank_id smallint, indicator text, listed bigint, borda bigint, mean_position numeric)
language sql stable security definer set search_path = public as $$
  with picks as (
    select p.x as rank_id, p.pos
    from wp1_rankings w
    join survey_sessions s on s.id = w.session_id
    cross join lateral unnest(w.top_five) with ordinality as p(x, pos)
    where for_role is null or s.role = for_role
  )
  select i.rank_id, i.indicator, count(p.pos), coalesce(sum(6 - p.pos), 0)::bigint, round(avg(p.pos), 2)
  from wp1_indicators i left join picks p on p.rank_id = i.rank_id
  group by i.rank_id, i.indicator order by 4 desc, 1
$$;

revoke all on function public.survey_overview(), public.wp6_summary(), public.wp3_scores(),
                       public.wp1_summary(text), public.cap_rows_per_session(), public.valid_top_five(smallint[])
  from public;
grant execute on function public.survey_overview(), public.wp6_summary(), public.wp3_scores(),
                          public.wp1_summary(text) to anon, authenticated;
grant execute on function public.valid_top_five(smallint[]) to anon, authenticated;
