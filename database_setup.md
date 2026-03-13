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
| `session_metrics` | All CEPM, CNN, fusion scores + timing + tokens |
| `typing_analytics` | WPM, CPM, backspaces, pauses, bursts per session |
| `user_activity_log` | Login, logout, run_session audit trail |

All tables use **Row Level Security** — users can only read/write their own rows.
