import { useState } from 'react';
import './PromptGuide.css';

export default function PromptGuide() {
    const [open, setOpen] = useState(false);

    return (
        <div className="prompt-guide-wrapper">
            <button className="prompt-guide-toggle" onClick={() => setOpen(o => !o)}>
                <span>🎓 Guide: How to Write the Perfect Prompt</span>
                <span className={`chevron ${open ? 'open' : ''}`}>▼</span>
            </button>
            {open && (
                <div className="prompt-guide-content">
                    <h3>🔑 Key Principles of Cognitive Efficiency</h3>
                    <p>To get the best results from CERAS (and any LLM), focus on these core elements:</p>
                    <ol>
                        <li><strong>Context & Role:</strong> Define <em>who</em> the model is (e.g., "Act as a senior physicist") and <em>what</em> the situation is.</li>
                        <li><strong>Explicit Constraints:</strong> Set boundaries. Mention word counts, specific formats (JSON, Markdown), or stylistic requirements.</li>
                        <li><strong>Chain of Thought:</strong> Ask the model to "explain its reasoning" or "break down the problem step-by-step" before giving the final answer.</li>
                        <li><strong>Few-Shot Examples:</strong> Providing 1-2 examples of the desired output format is the single most effective way to guide behavior.</li>
                        <li><strong>Iterative Refinement:</strong> Use the <strong>Diagnostics</strong> below to see where your prompt lacks density or clarity, then refine it.</li>
                    </ol>
                    <p><em>Tip: Use the "Good Examples" below to see these principles in action!</em></p>
                </div>
            )}
        </div>
    );
}
