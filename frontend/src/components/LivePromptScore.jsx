import { useMemo } from 'react';
import './LivePromptScore.css';

/**
 * Mirrors server.py extract_ceras_features — runs client-side for real-time feedback.
 */
function computeFeatures(text) {
    const words = text.trim().split(/\s+/).filter(Boolean);
    const wordCount = Math.max(words.length, 0);
    const charCount = text.length;
    const sentenceCount = Math.max((text.match(/[.!?]/g) || []).length, wordCount > 0 ? 1 : 0);
    const uniqueRatio = wordCount > 0 ? new Set(words.map(w => w.toLowerCase())).size / wordCount : 0;
    const conceptDensity = wordCount > 0 ? words.filter(w => w.length > 6).length / wordCount : 0;
    const promptQuality = Math.min(wordCount / 150, 1);

    let promptType = 0;
    if (wordCount >= 120) promptType = 3;
    else if (wordCount >= 60) promptType = 2;
    else if (wordCount >= 20) promptType = 1;

    return { wordCount, charCount, sentenceCount, uniqueRatio, conceptDensity, promptQuality, promptType };
}

const PROMPT_TYPE_LABELS = ['Short', 'Medium', 'Detailed', 'Comprehensive'];
const PROMPT_TYPE_CLASSES = ['short', 'medium', 'detailed', 'comprehensive'];

export default function LivePromptScore({ prompt, analytics }) {
    const features = useMemo(() => computeFeatures(prompt), [prompt]);

    if (!prompt.trim()) {
        return (
            <div className="live-score-panel">
                <div className="live-score-empty">
                    <span>⌨️</span>
                    Start typing to see real-time prompt analysis
                </div>
            </div>
        );
    }

    const scorePercent = Math.round(features.promptQuality * 100);
    const circumference = 2 * Math.PI * 56; // radius 56
    const offset = circumference - (features.promptQuality * circumference);

    const getScoreColor = (v) => {
        if (v >= 0.75) return '#22c55e';
        if (v >= 0.50) return '#60a5fa';
        if (v >= 0.25) return '#f59e0b';
        return '#ef4444';
    };

    const getWpmClass = (wpm) => {
        if (wpm > 80) return 'good';
        if (wpm > 40) return '';
        return '';
    };

    const getDelRatioClass = (ratio) => {
        if (ratio > 0.4) return 'alert';
        if (ratio > 0.2) return 'warn';
        return '';
    };

    return (
        <div className="live-score-panel">
            {/* Header */}
            <div className="live-score-header">
                <div className="live-score-title">
                    <span className="live-dot" />
                    Live Prompt Analysis
                </div>
                <span className={`prompt-type-badge ${PROMPT_TYPE_CLASSES[features.promptType]}`}>
                    {PROMPT_TYPE_LABELS[features.promptType]}
                </span>
            </div>

            {/* Body: Gauge + Feature Bars */}
            <div className="live-score-body">
                {/* Radial Gauge */}
                <div className="gauge-container">
                    <svg className="gauge-svg" viewBox="0 0 128 128">
                        <circle className="gauge-bg" cx="64" cy="64" r="56" />
                        <circle
                            className="gauge-fill"
                            cx="64" cy="64" r="56"
                            style={{
                                stroke: getScoreColor(features.promptQuality),
                                strokeDasharray: circumference,
                                strokeDashoffset: offset,
                            }}
                        />
                    </svg>
                    <div className="gauge-center">
                        <div className="gauge-value" style={{ color: getScoreColor(features.promptQuality) }}>
                            {scorePercent}
                        </div>
                        <div className="gauge-label">Quality</div>
                    </div>
                </div>

                {/* Feature Bars */}
                <div className="feature-bars">
                    <div className="feature-row">
                        <span className="feature-name">Words</span>
                        <div className="feature-bar-track">
                            <div className="feature-bar-fill blue" style={{ width: `${Math.min(features.wordCount / 150 * 100, 100)}%` }} />
                        </div>
                        <span className="feature-value">{features.wordCount}</span>
                    </div>
                    <div className="feature-row">
                        <span className="feature-name">Sentences</span>
                        <div className="feature-bar-track">
                            <div className="feature-bar-fill purple" style={{ width: `${Math.min(features.sentenceCount / 10 * 100, 100)}%` }} />
                        </div>
                        <span className="feature-value">{features.sentenceCount}</span>
                    </div>
                    <div className="feature-row">
                        <span className="feature-name">Unique Ratio</span>
                        <div className="feature-bar-track">
                            <div className="feature-bar-fill cyan" style={{ width: `${features.uniqueRatio * 100}%` }} />
                        </div>
                        <span className="feature-value">{(features.uniqueRatio * 100).toFixed(0)}%</span>
                    </div>
                    <div className="feature-row">
                        <span className="feature-name">Concept Density</span>
                        <div className="feature-bar-track">
                            <div className="feature-bar-fill emerald" style={{ width: `${features.conceptDensity * 100}%` }} />
                        </div>
                        <span className="feature-value">{(features.conceptDensity * 100).toFixed(0)}%</span>
                    </div>
                </div>
            </div>

            {/* Typing Dynamics Strip */}
            <div className="typing-dynamics">
                <div className="dynamic-stat">
                    <div className={`dynamic-stat-value ${getWpmClass(analytics.wpm)}`}>
                        {analytics.wpm}
                    </div>
                    <div className="dynamic-stat-label">WPM</div>
                </div>
                <div className="dynamic-stat">
                    <div className="dynamic-stat-value">
                        {analytics.totalKeystrokes}
                    </div>
                    <div className="dynamic-stat-label">Keystrokes</div>
                </div>
                <div className="dynamic-stat">
                    <div className={`dynamic-stat-value ${analytics.isHesitating ? 'hesitation-flash' : ''}`}>
                        {analytics.hesitations}
                    </div>
                    <div className="dynamic-stat-label">Hesitations</div>
                </div>
                <div className="dynamic-stat">
                    <div className={`dynamic-stat-value ${getDelRatioClass(analytics.deletionRatio)}`}>
                        {(analytics.deletionRatio * 100).toFixed(0)}%
                    </div>
                    <div className="dynamic-stat-label">Edit Rate</div>
                </div>
                <div className="dynamic-stat">
                    <div className="dynamic-stat-value">
                        {analytics.sessionDuration}s
                    </div>
                    <div className="dynamic-stat-label">Time</div>
                </div>
            </div>
        </div>
    );
}
