import './PromptInput.css';

export default function PromptInput({ prompt, setPrompt, onRun, onNewProblem, loading, modelsLoaded, modelError, onKeyDown, onPaste }) {

    const getButtonText = () => {
        if (loading) return '⏳ Running Reasoning Engine...';
        if (modelError) return '⚠ Model Loading Error';
        if (!modelsLoaded) return '⏳ Models Loading...';
        return '▶ Run Learning Session';
    };

    const handlePaste = (e) => {
        const pastedText = e.clipboardData?.getData('text') || '';
        if (pastedText && onPaste) {
            // Let the paste happen first, then simulate analytics
            setTimeout(() => onPaste(pastedText), 0);
        }
    };

    return (
        <div className="prompt-input-section">
            <hr />
            <div className="prompt-actions-top">
                <button className="new-problem-btn" onClick={onNewProblem}>
                    ✦ New Problem
                </button>
            </div>
            <textarea
                className="prompt-textarea"
                placeholder="Enter your learning question or problem..."
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                onKeyDown={onKeyDown}
                onPaste={handlePaste}
                rows={6}
            />
            <button
                className={`run-btn${modelError ? ' run-btn-error' : ''}`}
                onClick={onRun}
                disabled={loading || !prompt.trim() || !modelsLoaded}
            >
                {getButtonText()}
            </button>
            {modelError && (
                <div className="model-error-msg">
                    <span>⚠</span> {modelError}
                    <span className="error-hint"> — Retrying automatically...</span>
                </div>
            )}
        </div>
    );
}
