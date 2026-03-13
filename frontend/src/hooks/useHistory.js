import { useCallback, useEffect, useState } from 'react';
import { supabase } from '../lib/supabase';

export default function useHistory(userId) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const fetchSessions = useCallback(async () => {
    if (!userId) return;
    setLoading(true);
    try {
      let query = supabase
        .from('chat_sessions')
        .select(`
          *,
          chat_messages (
            id, prompt, final_steps, strategy_used, llm_calls_used, created_at,
            session_metrics (*)
          )
        `)
        .eq('user_id', userId)
        .order('created_at', { ascending: false })
        .limit(50);

      if (searchQuery.trim()) {
        // Search within messages for prompt text
        query = supabase
          .from('chat_sessions')
          .select(`
            *,
            chat_messages!inner (
              id, prompt, final_steps, strategy_used, llm_calls_used, created_at,
              session_metrics (*)
            )
          `)
          .eq('user_id', userId)
          .ilike('chat_messages.prompt', `%${searchQuery.trim()}%`)
          .order('created_at', { ascending: false })
          .limit(50);
      }

      const { data, error } = await query;
      if (error) throw error;
      setSessions(data || []);
    } catch (err) {
      console.error('Error fetching history:', err);
    } finally {
      setLoading(false);
    }
  }, [userId, searchQuery]);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  const deleteSession = async (sessionId) => {
    try {
      const { error } = await supabase
        .from('chat_sessions')
        .delete()
        .eq('id', sessionId)
        .eq('user_id', userId);
      if (error) throw error;
      setSessions(prev => prev.filter(s => s.id !== sessionId));
    } catch (err) {
      console.error('Error deleting session:', err);
    }
  };

  return {
    sessions,
    loading,
    searchQuery,
    setSearchQuery,
    refresh: fetchSessions,
    deleteSession,
  };
}
