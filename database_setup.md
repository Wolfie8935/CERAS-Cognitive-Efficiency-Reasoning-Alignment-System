# Database Setup Guide — CERAS

This file contains everything needed to replicate the Supabase backend for this project.

---

## 1. Create Supabase Project

1. Go to [https://supabase.com](https://supabase.com) and sign in
2. Click **New Project**
   - **Name:** `major-project-agent` (or any name you prefer)
   - **Region:** Closest to you
   - **Plan:** Free
3. Wait ~2 minutes for provisioning

---

## 2. Configure Authentication

1. Go to **Authentication → Providers → Email**
2. Set:
   - **Enable Email Signup:** ✅ ON
   - **Confirm email:** ❌ OFF (for development)
3. Click **Save**

---

## 3. Environment Variables

### Frontend — `frontend/.env`
```env
VITE_SUPABASE_URL=https://<your-project-ref>.supabase.co
VITE_SUPABASE_ANON_KEY=<your-anon-public-key>
```

### Backend — `.env` (root)
```env
SUPABASE_URL=https://<your-project-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>
SUPABASE_JWT_SECRET=<your-jwt-secret>
```

> **Where to find these values:**
> - Go to **Project Settings → API**
> - `Project URL` → `VITE_SUPABASE_URL` and `SUPABASE_URL`
> - `anon public` key → `VITE_SUPABASE_ANON_KEY`
> - `service_role` key (click 👁 to reveal) → `SUPABASE_SERVICE_ROLE_KEY`
> - **Project Settings → API → JWT Settings** → `JWT Secret` → `SUPABASE_JWT_SECRET`

> ⚠️ **Never commit `.env` files or the `service_role` key to GitHub.**

---

## 4. Database Schema

Go to **SQL Editor** in your Supabase dashboard, paste the entire block below, and click **▶ Run**.

```sql
-- ================================================
-- CERAS DATABASE SCHEMA
-- ================================================

-- 1. PROFILES
create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text not null,
  display_name text,
  avatar_url text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- 2. API KEY VAULT
create table public.api_keys (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  provider text not null,
  api_key text not null,
  key_label text,
  is_active boolean default true,
  last_verified_at timestamptz,
  is_valid boolean,
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  unique(user_id, provider, key_label)
);

-- 3. CHAT SESSIONS
create table public.chat_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  session_title text,
  main_provider text,
  verifier_provider text,
  main_model text,
  verifier_model text,
  created_at timestamptz default now()
);

-- 4. CHAT MESSAGES
create table public.chat_messages (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.chat_sessions(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  prompt text not null,
  final_steps jsonb,
  strategy_used text,
  llm_calls_used integer,
  created_at timestamptz default now()
);

-- 5. SESSION METRICS
create table public.session_metrics (
  id uuid primary key default gen_random_uuid(),
  message_id uuid not null references public.chat_messages(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  cepm_score float,
  cnn_score float,
  fused_score float,
  confidence float,
  readiness text,
  formulation_time float,
  runtime float,
  total_tokens integer,
  est_prompt_tokens integer,
  est_response_tokens integer,
  prompt_length integer,
  character_count integer,
  sentence_count integer,
  unique_word_ratio float,
  concept_density float,
  prompt_quality float,
  keystrokes integer,
  prompt_type integer,
  typing_speed_wpm float,
  typing_speed_cpm float,
  backspace_count integer,
  pause_count integer,
  avg_pause_duration float,
  total_pauses_ms float,
  typing_duration_ms float,
  burst_count integer,
  api_provider_main text,
  api_provider_verifier text,
  model_main text,
  model_verifier text,
  created_at timestamptz default now()
);

-- 6. TYPING ANALYTICS
create table public.typing_analytics (
  id uuid primary key default gen_random_uuid(),
  message_id uuid references public.chat_messages(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  wpm float,
  cpm float,
  backspace_count integer,
  pause_count integer,
  avg_pause_ms float,
  total_pauses_ms float,
  duration_ms float,
  burst_count integer,
  recorded_at timestamptz default now()
);

-- 7. USER ACTIVITY LOG
create table public.user_activity_log (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  action text not null,
  metadata jsonb,
  ip_address text,
  user_agent text,
  created_at timestamptz default now()
);

-- ================================================
-- ROW LEVEL SECURITY (users see only their own data)
-- ================================================
alter table public.profiles enable row level security;
alter table public.api_keys enable row level security;
alter table public.chat_sessions enable row level security;
alter table public.chat_messages enable row level security;
alter table public.session_metrics enable row level security;
alter table public.typing_analytics enable row level security;
alter table public.user_activity_log enable row level security;

create policy "users_own_profile"   on public.profiles           for all using (auth.uid() = id);
create policy "users_own_keys"      on public.api_keys           for all using (auth.uid() = user_id);
create policy "users_own_sessions"  on public.chat_sessions      for all using (auth.uid() = user_id);
create policy "users_own_messages"  on public.chat_messages      for all using (auth.uid() = user_id);
create policy "users_own_metrics"   on public.session_metrics    for all using (auth.uid() = user_id);
create policy "users_own_typing"    on public.typing_analytics   for all using (auth.uid() = user_id);
create policy "users_own_logs"      on public.user_activity_log  for all using (auth.uid() = user_id);

-- ================================================
-- AUTO-CREATE PROFILE TRIGGER
-- ================================================
create or replace function public.handle_new_user()
returns trigger as $$
begin
  insert into public.profiles (id, email)
  values (new.id, new.email);
  return new;
end;
$$ language plpgsql security definer;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();
```

After running, verify in **Table Editor** that these 7 tables exist:
- `profiles`
- `api_keys`
- `chat_sessions`
- `chat_messages`
- `session_metrics`
- `typing_analytics`
- `user_activity_log`

---

## 4b. New Features Schema (follow-ups, reports, plans, token cost)

If the base schema above is already applied, run this block in the SQL Editor to add support for follow-up messages, session reports, learning plans, and token cost per user.

```sql
-- ================================================
-- CERAS: New features (run on existing schema)
-- ================================================

-- 1. Add cost_usd to session_metrics
ALTER TABLE public.session_metrics
  ADD COLUMN IF NOT EXISTS cost_usd numeric(12, 8) DEFAULT NULL;

-- 2. Follow-up messages (Socratic follow-up thread per chat message)
CREATE TABLE IF NOT EXISTS public.followup_messages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id uuid NOT NULL REFERENCES public.chat_messages(id) ON DELETE CASCADE,
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  role text NOT NULL CHECK (role IN ('user', 'assistant')),
  content text NOT NULL,
  prompt_tokens integer DEFAULT 0,
  completion_tokens integer DEFAULT 0,
  cost_usd numeric(12, 8) DEFAULT NULL,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE public.followup_messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY "users_own_followup"
  ON public.followup_messages FOR ALL
  USING (auth.uid() = user_id);

-- 3. Learning plans (2-week plan per chat message)
CREATE TABLE IF NOT EXISTS public.learning_plans (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id uuid NOT NULL REFERENCES public.chat_messages(id) ON DELETE CASCADE,
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  plan_text text NOT NULL,
  prompt_tokens integer DEFAULT 0,
  completion_tokens integer DEFAULT 0,
  cost_usd numeric(12, 8) DEFAULT NULL,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE public.learning_plans ENABLE ROW LEVEL SECURITY;

CREATE POLICY "users_own_learning_plans"
  ON public.learning_plans FOR ALL
  USING (auth.uid() = user_id);

-- 4. Session reports (downloaded report content)
CREATE TABLE IF NOT EXISTS public.session_reports (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id uuid NOT NULL REFERENCES public.chat_sessions(id) ON DELETE CASCADE,
  message_id uuid NOT NULL REFERENCES public.chat_messages(id) ON DELETE CASCADE,
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  report_content text NOT NULL,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE public.session_reports ENABLE ROW LEVEL SECURITY;

CREATE POLICY "users_own_session_reports"
  ON public.session_reports FOR ALL
  USING (auth.uid() = user_id);

-- 5. View: token cost per user (session + followup + plan)
CREATE OR REPLACE VIEW public.user_token_cost_summary AS
SELECT user_id, 'session' AS feature, id AS source_id, cost_usd, created_at
FROM public.session_metrics
WHERE cost_usd IS NOT NULL AND cost_usd > 0
UNION ALL
SELECT user_id, 'followup' AS feature, id AS source_id, cost_usd, created_at
FROM public.followup_messages
WHERE cost_usd IS NOT NULL AND cost_usd > 0
UNION ALL
SELECT user_id, 'plan' AS feature, id AS source_id, cost_usd, created_at
FROM public.learning_plans
WHERE cost_usd IS NOT NULL AND cost_usd > 0;
```

After running, you should have: `session_metrics.cost_usd`, tables `followup_messages`, `learning_plans`, `session_reports`, and view `user_token_cost_summary`. To get total cost per user: `SELECT user_id, SUM(cost_usd) AS total_cost_usd FROM public.user_token_cost_summary GROUP BY user_id;`

---

## 5. Install Dependencies

```bash
# Frontend
cd frontend
npm install @supabase/supabase-js react-router-dom

# Backend
pip install supabase python-jose[cryptography]
```

---

## 6. Table Overview

| Table | Purpose |
|---|---|
| `profiles` | Extended user info (display name, avatar) |
| `api_keys` | Saved Groq / Gemini / OpenAI keys per user |
| `chat_sessions` | Groups of related prompts with model config |
| `chat_messages` | Individual prompts + LLM responses (JSONB) |
| `session_metrics` | All CEPM, CNN, fusion scores + timing + tokens + cost_usd |
| `typing_analytics` | WPM, CPM, backspaces, pauses, bursts per session |
| `user_activity_log` | Login, logout, run_session audit trail |
| `followup_messages` | Socratic follow-up thread per chat message (role, content, tokens, cost) |
| `learning_plans` | 2-week learning plan per chat message (plan_text, tokens, cost) |
| `session_reports` | Downloaded session report content per session/message |

**View:** `user_token_cost_summary` — unions cost from session_metrics, followup_messages, learning_plans for per-user token cost queries.

All tables use **Row Level Security** — users can only read/write their own rows.
