import { useEffect, useRef, useState } from 'react';
import logo from '../../assets/ceras_logo.png';
import { checkConnection, checkHealth } from '../api';
import { useAuth } from '../context/AuthContext';
import { GEMINI_MODELS, GROQ_MODELS, OPENAI_MODELS } from '../data/examples';
import './Sidebar.css';

export default function Sidebar({ config, setConfig, isOpen, onClose, user, onOpenHistory, onOpenVault }) {
    const { signOut } = useAuth();
    const [statuses, setStatuses] = useState({
        groq: 'Waiting',
        gemini: 'Waiting',
        openai: 'Waiting',
    });
    const [modelsLoaded, setModelsLoaded] = useState(false);
    const [modelsLoading, setModelsLoading] = useState(true);
    const debounceRef = useRef({});

    // Poll model loading status
    useEffect(() => {
        let alive = true;
        const poll = async () => {
            try {
                const data = await checkHealth();
                if (!alive) return;
                setModelsLoaded(data.models_loaded);
                setModelsLoading(data.models_loading);
                if (!data.models_loaded && !data.model_error) {
                    setTimeout(poll, 2000);
                }
            } catch {
                if (alive) setTimeout(poll, 3000);
            }
        };
        poll();
        return () => { alive = false; };
    }, []);

    // Auto-check connection when keys change (debounced to avoid flooding backend)
    const handleKeyChange = (provider, value) => {
        const key = provider.toLowerCase();
        setConfig(prev => ({ ...prev, [`${key}_api_key`]: value }));

        // Clear any pending check for this provider
        if (debounceRef.current[key]) clearTimeout(debounceRef.current[key]);

        if (value.length > 10) {
            setStatuses(prev => ({ ...prev, [key]: 'Checking...' }));
            debounceRef.current[key] = setTimeout(async () => {
                try {
                    const result = await checkConnection(provider, value);
                    setStatuses(prev => ({ ...prev, [key]: result.connected ? 'Connected' : 'Not Connected' }));
                } catch {
                    setStatuses(prev => ({ ...prev, [key]: 'Not Connected' }));
                }
            }, 800);
        } else {
            setStatuses(prev => ({ ...prev, [key]: 'Waiting' }));
        }
    };

    const getModelsForProvider = (provider) => {
        if (provider === 'Groq') return GROQ_MODELS;
        if (provider === 'Gemini') return GEMINI_MODELS;
        return OPENAI_MODELS;
    };

    const getStatusColor = (s) => {
        if (s === 'Connected') return 'green';
        if (s === 'Not Connected') return 'red';
        if (s === 'Checking...') return 'blue pulse';
        return 'grey';
    };

    const handleSignOut = async () => {
        try {
            await signOut();
        } catch (err) {
            console.error('Sign out error:', err);
        }
    };

    return (
        <aside className={`sidebar ${isOpen ? 'sidebar--open' : ''}`}>
            {/* Brand */}
            <div className="sidebar-brand">
                <button className="sidebar-close-btn" onClick={onClose} aria-label="Close sidebar">✕</button>
                <img src={logo} alt="CERAS Logo" />
                <h2>CERAS</h2>
                <p>Cognitive Efficiency & Reasoning Alignment System</p>
            </div>

            <div className="sidebar-content">
                {/* User Card */}
                {user && (
                    <div className="sidebar-user-card">
                        <div className="user-avatar">
                            {(user.user_metadata?.display_name || user.email || '?')[0].toUpperCase()}
                        </div>
                        <div className="user-info">
                            <span className="user-name">{user.user_metadata?.display_name || 'User'}</span>
                            <span className="user-email">{user.email}</span>
                        </div>
                        <button className="user-logout-btn" onClick={handleSignOut} title="Sign Out">
                            ⏻
                        </button>
                    </div>
                )}

                {/* Navigation */}
                <div className="sidebar-nav">
                    <button className="sidebar-nav-btn" onClick={onOpenHistory}>
                        <span>📜</span> Chat History
                    </button>
                    <button className="sidebar-nav-btn" onClick={onOpenVault}>
                        <span>🔒</span> API Key Vault
                    </button>
                </div>

                {/* Model Loading Status */}
                {modelsLoading && !modelsLoaded && (
                    <div className="model-loading-banner">
                        <div className="spinner-mini" />
                        <span>Loading ML models...</span>
                    </div>
                )}
                {modelsLoaded && (
                    <div className="models-ready-banner">
                        <span>✓</span>
                        <span>Models ready</span>
                    </div>
                )}

                {/* API Keys */}
                <div className="sidebar-section">
                    <div className="sidebar-section-title"><span>🔑</span> API Configuration</div>
                    {['Groq', 'Gemini', 'OpenAI'].map(provider => (
                        <div className="api-key-group" key={provider}>
                            <label>{provider} API Key</label>
                            <input
                                className="api-key-input"
                                type="password"
                                placeholder={`Enter ${provider} key...`}
                                value={config[`${provider.toLowerCase()}_api_key`] || ''}
                                onChange={e => handleKeyChange(provider, e.target.value)}
                            />
                        </div>
                    ))}
                </div>

                {/* Main Model Selection */}
                <div className="sidebar-section">
                    <div className="sidebar-section-title"><span>🤖</span> Model Selection</div>
                    <div className="model-select-group">
                        <div className="group-label">Main Reasoner</div>
                        <div className="model-row">
                            <div className="provider-radio-group">
                                {['Groq', 'Gemini', 'OpenAI'].map(p => (
                                    <label className="provider-radio" key={p}>
                                        <input
                                            type="radio"
                                            name="main_provider"
                                            checked={config.main_provider === p}
                                            onChange={() => {
                                                const models = getModelsForProvider(p);
                                                setConfig(prev => ({ ...prev, main_provider: p, main_model: models[0] }));
                                            }}
                                        />
                                        {p}
                                    </label>
                                ))}
                            </div>
                            <div className="model-dropdown">
                                <select
                                    value={config.main_model}
                                    onChange={e => setConfig(prev => ({ ...prev, main_model: e.target.value }))}
                                >
                                    {getModelsForProvider(config.main_provider).map(m => (
                                        <option key={m} value={m}>{m}</option>
                                    ))}
                                </select>
                            </div>
                        </div>
                    </div>

                    <div className="model-select-group">
                        <div className="group-label">Verifier Model</div>
                        <div className="model-row">
                            <div className="provider-radio-group">
                                {['Groq', 'Gemini', 'OpenAI'].map(p => (
                                    <label className="provider-radio" key={p}>
                                        <input
                                            type="radio"
                                            name="verifier_provider"
                                            checked={config.verifier_provider === p}
                                            onChange={() => {
                                                const models = getModelsForProvider(p);
                                                setConfig(prev => ({ ...prev, verifier_provider: p, verifier_model: models[1] || models[0] }));
                                            }}
                                        />
                                        {p}
                                    </label>
                                ))}
                            </div>
                            <div className="model-dropdown">
                                <select
                                    value={config.verifier_model}
                                    onChange={e => setConfig(prev => ({ ...prev, verifier_model: e.target.value }))}
                                >
                                    {getModelsForProvider(config.verifier_provider).map(m => (
                                        <option key={m} value={m}>{m}</option>
                                    ))}
                                </select>
                            </div>
                        </div>
                    </div>
                </div>

                {/* System Status */}
                <div className="sidebar-section">
                    <div className="sidebar-section-title"><span>⚙️</span> System Status</div>
                    <div className="status-box">
                        {['Groq', 'Gemini', 'OpenAI'].map(p => {
                            const key = p.toLowerCase();
                            const status = statuses[key];
                            return (
                                <div className="status-row" key={p}>
                                    <span className="status-label">{p} API</span>
                                    <span className="status-indicator">
                                        <span className={`status-dot ${getStatusColor(status)}`} />
                                        {status}
                                    </span>
                                </div>
                            );
                        })}
                        <div className="status-row">
                            <span className="status-label">Telemetry</span>
                            <span className="status-indicator">
                                <span className="status-dot blue" />
                                Tracking
                            </span>
                        </div>
                    </div>
                </div>

                {/* Architecture */}
                <div className="sidebar-section">
                    <div className="sidebar-section-title"><span>🏗️</span> Architecture</div>
                    <div className="arch-box">
                        {[
                            { name: 'ToT-LLM', desc: 'Tree-of-Thoughts Reasoning Engine' },
                            { name: 'CEPM', desc: 'Cognitive Engagement Modeling' },
                            { name: 'CNN-Vis', desc: 'Behavioral Signal Analysis' },
                            { name: 'Fusion Layer', desc: 'Multi-Modal Signal Integration' },
                        ].map(a => (
                            <div className="arch-item" key={a.name}>
                                <strong>✦ {a.name}</strong>
                                <span>{a.desc}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="sidebar-footer">v2.0.0 • Neural Learning Stack</div>
        </aside>
    );
}
