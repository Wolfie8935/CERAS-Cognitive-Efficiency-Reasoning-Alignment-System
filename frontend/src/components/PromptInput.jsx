import './PromptInput.css';

export default function PromptInput({ prompt, setPrompt, onRun, onNewProblem, loading, modelsLoaded, onKeyDown }) {
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
                rows={6}
            />
            <button
                className="run-btn"
                onClick={onRun}
                disabled={loading || !prompt.trim() || !modelsLoaded}
            >
                {loading ? '⏳ Running Reasoning Engine...' : !modelsLoaded ? '⏳ Models Loading...' : '▶ Run Learning Session'}
            </button>
        </div>
    );
}
