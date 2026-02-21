import './Header.css';

export default function Header() {
    return (
        <header className="header fade-in-up">
            <h1 className="header-title">CERAS</h1>
            <p className="header-subtitle">Cognitive Efficiency Reasoning Alignment System</p>
            <div className="header-description">
                <b>CERAS</b> is an advanced adaptive learning environment designed to optimize how you learn and solve complex problems.
                By fusing <b>Large Language Model (LLM)</b> reasoning capabilities with real-time <b>Cognitive Efficiency</b> metrics,
                current behavioral diagnostics, and neuro-fuzzy alignment, CERAS provides a personalized learning experience.
                It analyzes your input complexity, structure, and intent to guide you through deep concepts with tailored roadmaps,
                ensuring you don't just get answers, but truly master the material.
            </div>
        </header>
    );
}
