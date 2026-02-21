import { BAD_EXAMPLES, GOOD_EXAMPLES } from '../data/examples';
import './ExampleCards.css';

export default function ExampleCards({ onSelect }) {
    return (
        <>
            {/* Good Examples */}
            <div className="examples-section fade-in-up" style={{ animationDelay: '0.1s' }}>
                <h2 className="examples-section-title">🌟 GOOD EXAMPLES TO PROMPT (High CE Score)</h2>
                <p className="examples-section-caption">
                    These prompts are detailed and structured, leading to higher cognitive efficiency scores.
                </p>
                <div className="good-examples-grid">
                    {GOOD_EXAMPLES.map(ex => (
                        <div className="example-card" key={ex.id} style={{ background: ex.gradient }}>
                            <div className="example-card-header">
                                <span>✨</span> {ex.title}
                            </div>
                            <div className="example-card-body">{ex.text}</div>
                            <button className="example-card-btn" onClick={() => onSelect(ex.text)}>
                                Select: {ex.title}
                            </button>
                        </div>
                    ))}
                </div>
            </div>

            {/* Bad Examples */}
            <div className="examples-section fade-in-up" style={{ animationDelay: '0.2s' }}>
                <h2 className="examples-section-title">⚠️ BAD EXAMPLES TO PROMPT (Low CE Score)</h2>
                <p className="examples-section-caption">
                    These prompts look normal but lack depth, structure, or analytical clarity.
                </p>
                <div className="bad-examples-grid">
                    {BAD_EXAMPLES.map(ex => (
                        <div className="bad-example-card" key={ex.id}>
                            <div>{ex.text}</div>
                            <button className="bad-example-btn" onClick={() => onSelect(ex.text)}>
                                Use: {ex.label}
                            </button>
                        </div>
                    ))}
                </div>
            </div>
        </>
    );
}
