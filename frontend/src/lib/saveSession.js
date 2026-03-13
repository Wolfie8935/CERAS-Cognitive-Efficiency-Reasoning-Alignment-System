import { supabase } from '../lib/supabase';

/**
 * Save a complete session (chat + metrics + typing analytics) to Supabase.
 * Called after receiving results from the backend.
 */
export async function saveSession({ userId, prompt, result, config, typingAnalytics }) {
  if (!userId) return null;

  try {
    // 1. Create chat session
    const { data: session, error: sessionErr } = await supabase
      .from('chat_sessions')
      .insert({
        user_id: userId,
        session_title: prompt.substring(0, 80),
        main_provider: config.main_provider,
        verifier_provider: config.verifier_provider,
        main_model: config.main_model,
        verifier_model: config.verifier_model,
      })
      .select()
      .single();

    if (sessionErr) throw sessionErr;

    // 2. Create chat message
    const { data: message, error: msgErr } = await supabase
      .from('chat_messages')
      .insert({
        session_id: session.id,
        user_id: userId,
        prompt,
        final_steps: result.final_steps || [],
        strategy_used: result.strategy_used || '',
        llm_calls_used: result.llm_calls_used || 0,
      })
      .select()
      .single();

    if (msgErr) throw msgErr;

    // 3. Save session metrics
    const features = result.features || {};
    const { error: metricsErr } = await supabase
      .from('session_metrics')
      .insert({
        message_id: message.id,
        user_id: userId,
        cepm_score: result.cepm_score,
        cnn_score: result.cnn_score,
        fused_score: result.fused_score,
        confidence: result.confidence,
        readiness: result.readiness,
        formulation_time: result.formulation_time,
        runtime: result.runtime,
        total_tokens: result.total_tokens,
        est_prompt_tokens: Math.round((result.total_tokens || 0) * 0.3),
        est_response_tokens: Math.round((result.total_tokens || 0) * 0.7),
        prompt_length: features.prompt_length,
        character_count: features.character_count,
        sentence_count: features.sentence_count,
        unique_word_ratio: features.unique_word_ratio,
        concept_density: features.concept_density,
        prompt_quality: features.prompt_quality,
        keystrokes: features.keystrokes,
        prompt_type: features.prompt_type,
        typing_speed_wpm: typingAnalytics?.wpm || 0,
        typing_speed_cpm: typingAnalytics?.cpm || 0,
        backspace_count: typingAnalytics?.backspaceCount || 0,
        pause_count: typingAnalytics?.pauseCount || 0,
        avg_pause_duration: typingAnalytics?.avgPauseDuration || 0,
        total_pauses_ms: typingAnalytics?.totalPauses || 0,
        typing_duration_ms: typingAnalytics?.duration || 0,
        burst_count: typingAnalytics?.burstCount || 0,
        api_provider_main: config.main_provider,
        api_provider_verifier: config.verifier_provider,
        model_main: config.main_model,
        model_verifier: config.verifier_model,
      });

    if (metricsErr) console.error('Metrics save error:', metricsErr);

    // 4. Save typing analytics
    if (typingAnalytics) {
      const { error: typingErr } = await supabase
        .from('typing_analytics')
        .insert({
          message_id: message.id,
          user_id: userId,
          wpm: typingAnalytics.wpm || 0,
          cpm: typingAnalytics.cpm || 0,
          backspace_count: typingAnalytics.backspaceCount || 0,
          pause_count: typingAnalytics.pauseCount || 0,
          avg_pause_ms: typingAnalytics.avgPauseDuration || 0,
          total_pauses_ms: typingAnalytics.totalPauses || 0,
          duration_ms: typingAnalytics.duration || 0,
          burst_count: typingAnalytics.burstCount || 0,
        });

      if (typingErr) console.error('Typing analytics save error:', typingErr);
    }

    // 5. Log activity
    await supabase.from('user_activity_log').insert({
      user_id: userId,
      action: 'run_session',
      metadata: {
        session_id: session.id,
        provider: config.main_provider,
        model: config.main_model,
        fused_score: result.fused_score,
      },
    });

    return { session, message };
  } catch (err) {
    console.error('Error saving session:', err);
    return null;
  }
}
