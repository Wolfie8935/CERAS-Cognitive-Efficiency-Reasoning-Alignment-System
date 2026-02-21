import './LoadingOverlay.css';

export default function LoadingOverlay() {
    return (
        <div className="loading-overlay">
            <div className="loading-card">
                <div className="loading-brain">
                    <div className="loading-brain-ring" />
                    <div className="loading-brain-ring" />
                    <div className="loading-brain-ring" />
                    <div className="loading-brain-emoji">🧠</div>
                </div>
                <div className="loading-title">Reasoning Engine Active</div>
                <div className="loading-subtitle">
                    Processing your prompt through the CERAS neural pipeline
                </div>
                <div className="loading-steps">
                    <div className="loading-step" style={{ animationDelay: '0s' }}>
                        <span className="loading-step-dot" />
                        Tree-of-Thoughts decomposition
                    </div>
                    <div className="loading-step" style={{ animationDelay: '0.1s' }}>
                        <span className="loading-step-dot" />
                        Multi-verifier validation pipeline
                    </div>
                    <div className="loading-step" style={{ animationDelay: '0.2s' }}>
                        <span className="loading-step-dot" />
                        CEPM & CNN cognitive scoring
                    </div>
                    <div className="loading-step" style={{ animationDelay: '0.3s' }}>
                        <span className="loading-step-dot" />
                        Fusion layer integration
                    </div>
                </div>
            </div>
        </div>
    );
}
