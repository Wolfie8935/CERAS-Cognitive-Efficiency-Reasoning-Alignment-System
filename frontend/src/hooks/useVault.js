import { useCallback, useEffect, useState } from 'react';
import { supabase } from '../lib/supabase';

export default function useVault(userId) {
  const [keys, setKeys] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchKeys = useCallback(async () => {
    if (!userId) return;
    setLoading(true);
    try {
      const { data, error } = await supabase
        .from('api_keys')
        .select('*')
        .eq('user_id', userId)
        .eq('is_active', true)
        .order('created_at', { ascending: false });
      if (error) throw error;
      setKeys(data || []);
    } catch (err) {
      console.error('Error fetching keys:', err);
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchKeys();
  }, [fetchKeys]);

  const saveKey = async (provider, apiKey, label = 'default') => {
    if (!userId) return;
    try {
      // Upsert: update if same provider+label exists, else insert
      const { data, error } = await supabase
        .from('api_keys')
        .upsert(
          {
            user_id: userId,
            provider,
            api_key: apiKey,
            key_label: label,
            is_active: true,
            updated_at: new Date().toISOString(),
          },
          { onConflict: 'user_id,provider,key_label' }
        )
        .select();
      if (error) throw error;
      await fetchKeys();
      return data;
    } catch (err) {
      console.error('Error saving key:', err);
      throw err;
    }
  };

  const deleteKey = async (keyId) => {
    try {
      const { error } = await supabase
        .from('api_keys')
        .delete()
        .eq('id', keyId)
        .eq('user_id', userId);
      if (error) throw error;
      setKeys(prev => prev.filter(k => k.id !== keyId));
    } catch (err) {
      console.error('Error deleting key:', err);
    }
  };

  const updateVerification = async (keyId, isValid) => {
    try {
      const { error } = await supabase
        .from('api_keys')
        .update({
          is_valid: isValid,
          last_verified_at: new Date().toISOString(),
        })
        .eq('id', keyId)
        .eq('user_id', userId);
      if (error) throw error;
      await fetchKeys();
    } catch (err) {
      console.error('Error updating verification:', err);
    }
  };

  // Helper: get active key for a provider
  const getKeyForProvider = (provider) => {
    const found = keys.find(k => k.provider === provider && k.is_active);
    return found?.api_key || '';
  };

  return {
    keys,
    loading,
    saveKey,
    deleteKey,
    updateVerification,
    getKeyForProvider,
    refresh: fetchKeys,
  };
}
