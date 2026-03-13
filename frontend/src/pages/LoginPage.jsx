import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import './LoginPage.css';

export default function LoginPage() {
    const { signIn, signUp } = useAuth();
    const [isSignUp, setIsSignUp] = useState(false);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [displayName, setDisplayName] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [success, setSuccess] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setSuccess('');
        setLoading(true);

        try {
            if (isSignUp) {
                await signUp(email, password, displayName);
                setSuccess('Account created successfully! You are now logged in.');
            } else {
                await signIn(email, password);
            }
        } catch (err) {
            setError(err.message || 'An error occurred');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="login-page">
            <div className="login-particles">
                {Array.from({ length: 20 }, (_, i) => (
                    <div key={i} className="particle" style={{
                        left: `${Math.random() * 100}%`,
                        top: `${Math.random() * 100}%`,
                        animationDelay: `${Math.random() * 5}s`,
                        animationDuration: `${3 + Math.random() * 4}s`,
                    }} />
                ))}
            </div>

            <div className="login-card">
                <div className="login-brand">
                    <div className="login-logo-glow">
                        <span className="login-logo-text">C</span>
                    </div>
                    <h1>CERAS</h1>
                    <p>Cognitive Efficiency & Reasoning Alignment System</p>
                </div>

                <div className="login-tabs">
                    <button
                        className={`login-tab ${!isSignUp ? 'active' : ''}`}
                        onClick={() => { setIsSignUp(false); setError(''); setSuccess(''); }}
                    >
                        Sign In
                    </button>
                    <button
                        className={`login-tab ${isSignUp ? 'active' : ''}`}
                        onClick={() => { setIsSignUp(true); setError(''); setSuccess(''); }}
                    >
                        Sign Up
                    </button>
                </div>

                <form className="login-form" onSubmit={handleSubmit}>
                    {isSignUp && (
                        <div className="login-field">
                            <label htmlFor="displayName">Display Name</label>
                            <input
                                id="displayName"
                                type="text"
                                placeholder="Your name"
                                value={displayName}
                                onChange={(e) => setDisplayName(e.target.value)}
                                required={isSignUp}
                            />
                        </div>
                    )}

                    <div className="login-field">
                        <label htmlFor="email">Email</label>
                        <input
                            id="email"
                            type="email"
                            placeholder="you@example.com"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                            autoComplete="email"
                        />
                    </div>

                    <div className="login-field">
                        <label htmlFor="password">Password</label>
                        <input
                            id="password"
                            type="password"
                            placeholder="••••••••"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            minLength={6}
                            autoComplete={isSignUp ? 'new-password' : 'current-password'}
                        />
                    </div>

                    {error && <div className="login-error">{error}</div>}
                    {success && <div className="login-success">{success}</div>}

                    <button
                        type="submit"
                        className="login-submit"
                        disabled={loading}
                    >
                        {loading ? (
                            <span className="login-spinner" />
                        ) : (
                            isSignUp ? 'Create Account' : 'Sign In'
                        )}
                    </button>
                </form>

                <div className="login-footer">
                    {isSignUp ? (
                        <p>Already have an account?{' '}
                            <button className="login-link" onClick={() => setIsSignUp(false)}>Sign In</button>
                        </p>
                    ) : (
                        <p>Don't have an account?{' '}
                            <button className="login-link" onClick={() => setIsSignUp(true)}>Sign Up</button>
                        </p>
                    )}
                </div>
            </div>
        </div>
    );
}
