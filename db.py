"""
Supabase Python client for server-side database operations.
Uses service_role key to bypass RLS for admin operations.
"""

import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")


def get_supabase_client() -> Client:
    """Get a Supabase client with service_role privileges."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env"
        )
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def save_session_to_db(
    user_id: str,
    prompt: str,
    result: dict,
    config: dict,
    typing_analytics: dict = None,
):
    """
    Save a complete session (chat + metrics + typing) to Supabase.
    Called from the /api/save-session endpoint.
    """
    client = get_supabase_client()

    # 1. Create chat session
    session_data = client.table("chat_sessions").insert({
        "user_id": user_id,
        "session_title": prompt[:80],
        "main_provider": config.get("main_provider"),
        "verifier_provider": config.get("verifier_provider"),
        "main_model": config.get("main_model"),
        "verifier_model": config.get("verifier_model"),
    }).execute()

    session_id = session_data.data[0]["id"]

    # 2. Create chat message
    msg_data = client.table("chat_messages").insert({
        "session_id": session_id,
        "user_id": user_id,
        "prompt": prompt,
        "final_steps": result.get("final_steps", []),
        "strategy_used": result.get("strategy_used", ""),
        "llm_calls_used": result.get("llm_calls_used", 0),
    }).execute()

    message_id = msg_data.data[0]["id"]

    # 3. Save session metrics
    features = result.get("features", {})
    ta = typing_analytics or {}
    client.table("session_metrics").insert({
        "message_id": message_id,
        "user_id": user_id,
        "cepm_score": result.get("cepm_score"),
        "cnn_score": result.get("cnn_score"),
        "fused_score": result.get("fused_score"),
        "confidence": result.get("confidence"),
        "readiness": result.get("readiness"),
        "formulation_time": result.get("formulation_time"),
        "runtime": result.get("runtime"),
        "total_tokens": result.get("total_tokens"),
        "prompt_length": features.get("prompt_length"),
        "character_count": features.get("character_count"),
        "sentence_count": features.get("sentence_count"),
        "unique_word_ratio": features.get("unique_word_ratio"),
        "concept_density": features.get("concept_density"),
        "prompt_quality": features.get("prompt_quality"),
        "keystrokes": features.get("keystrokes"),
        "prompt_type": features.get("prompt_type"),
        "typing_speed_wpm": ta.get("wpm"),
        "typing_speed_cpm": ta.get("cpm"),
        "backspace_count": ta.get("backspaceCount"),
        "pause_count": ta.get("pauseCount"),
        "avg_pause_duration": ta.get("avgPauseDuration"),
        "total_pauses_ms": ta.get("totalPauses"),
        "typing_duration_ms": ta.get("duration"),
        "burst_count": ta.get("burstCount"),
        "api_provider_main": config.get("main_provider"),
        "api_provider_verifier": config.get("verifier_provider"),
        "model_main": config.get("main_model"),
        "model_verifier": config.get("verifier_model"),
    }).execute()

    # 4. Save typing analytics
    if typing_analytics:
        client.table("typing_analytics").insert({
            "message_id": message_id,
            "user_id": user_id,
            "wpm": ta.get("wpm"),
            "cpm": ta.get("cpm"),
            "backspace_count": ta.get("backspaceCount"),
            "pause_count": ta.get("pauseCount"),
            "avg_pause_ms": ta.get("avgPauseDuration"),
            "total_pauses_ms": ta.get("totalPauses"),
            "duration_ms": ta.get("duration"),
            "burst_count": ta.get("burstCount"),
        }).execute()

    return {"session_id": session_id, "message_id": message_id}
