import { useState } from 'react';
import { checkConnection } from '../api';
import './VaultPage.css';

export default function VaultPage({ vault, config, setConfig, onClose }) {
    const { keys, saveKey, deleteKey, updateVerification, loading } = vault;
    const [newKey, setNewKey] = useState({ provider: 'Groq', api_key: '', label: 'default' });
    const [saving, setSaving] = useState(false);
    const [testing, setTesting] = useState({});
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    const handleSave = async () => {
        if (!newKey.api_key.trim()) return;
        setError('');
        setSuccess('');
        setSaving(true);
        try {
            await saveKey(newKey.provider, newKey.api_key.trim(), newKey.label || 'default');
            // Also update live config
            const configKey = `${newKey.provider.toLowerCase()}_api_key`;
            setConfig(prev => ({ ...prev, [configKey]: newKey.api_key.trim() }));
            setSuccess(`${newKey.provider} key saved successfully!`);
            setNewKey({ provider: 'Groq', api_key: '', label: 'default' });
        } catch (err) {
            setError(err.message);
        } finally {
            setSaving(false);
        }
    };

    const handleTest = async (key) => {
        setTesting(prev => ({ ...prev, [key.id]: true }));
        try {
            const result = await checkConnection(key.provider, key.api_key);
            await updateVerification(key.id, result.connected);
        } catch {
            await updateVerification(key.id, false);
        } finally {
            setTesting(prev => ({ ...prev, [key.id]: false }));
        }
    };

    const handleUseKey = (key) => {
        const configKey = `${key.provider.toLowerCase()}_api_key`;
        setConfig(prev => ({ ...prev, [configKey]: key.api_key }));
    };

    const handleDelete = async (keyId) => {
        await deleteKey(keyId);
    };

    const maskKey = (k) => {
        if (!k || k.length < 10) return '***';
        return k.substring(0, 6) + '••••••' + k.substring(k.length - 4);
    };

    return (
        <div className="vault-page">
            <div className="vault-header">
                <h3>🔒 API Key Vault</h3>
                <button className="vault-close" onClick={onClose}>✕</button>
            </div>

            {/* Add New Key */}
            <div className="vault-add-section">
                <h4>Add New Key</h4>
                <div className="vault-add-form">
                    <select
                        value={newKey.provider}
                        onChange={(e) => setNewKey(prev => ({ ...prev, provider: e.target.value }))}
                    >
                        <option value="Groq">Groq</option>
                        <option value="Gemini">Gemini</option>
                        <option value="OpenAI">OpenAI</option>
                    </select>
                    <input
                        type="password"
                        placeholder="Paste your API key..."
                        value={newKey.api_key}
                        onChange={(e) => setNewKey(prev => ({ ...prev, api_key: e.target.value }))}
                    />
                    <input
                        type="text"
                        placeholder="Label (optional)"
                        value={newKey.label}
                        onChange={(e) => setNewKey(prev => ({ ...prev, label: e.target.value }))}
                        className="vault-label-input"
                    />
                    <button
                        className="vault-save-btn"
                        onClick={handleSave}
                        disabled={saving || !newKey.api_key.trim()}
                    >
                        {saving ? '...' : '💾 Save'}
                    </button>
                </div>
                {error && <div className="vault-error">{error}</div>}
                {success && <div className="vault-success">{success}</div>}
            </div>

            {/* Saved Keys */}
            <div className="vault-keys-section">
                <h4>Saved Keys ({keys.length})</h4>
                {loading && <p className="vault-loading">Loading...</p>}

                {!loading && keys.length === 0 && (
                    <p className="vault-empty">No keys saved yet. Add one above!</p>
                )}

                {keys.map(key => (
                    <div key={key.id} className="vault-key-card">
                        <div className="vault-key-top">
                            <span className="vault-key-provider">{key.provider}</span>
                            <span className="vault-key-label">{key.key_label || 'default'}</span>
                            {key.is_valid === true && <span className="vault-badge green">✓ Valid</span>}
                            {key.is_valid === false && <span className="vault-badge red">✗ Invalid</span>}
                            {key.is_valid == null && <span className="vault-badge grey">? Untested</span>}
                        </div>
                        <div className="vault-key-value">{maskKey(key.api_key)}</div>
                        {key.last_verified_at && (
                            <div className="vault-key-verified">
                                Last tested: {new Date(key.last_verified_at).toLocaleString()}
                            </div>
                        )}
                        <div className="vault-key-actions">
                            <button className="vault-action-btn use" onClick={() => handleUseKey(key)}>
                                ⚡ Use
                            </button>
                            <button
                                className="vault-action-btn test"
                                onClick={() => handleTest(key)}
                                disabled={testing[key.id]}
                            >
                                {testing[key.id] ? '...' : '🧪 Test'}
                            </button>
                            <button className="vault-action-btn delete" onClick={() => handleDelete(key.id)}>
                                🗑️
                            </button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
