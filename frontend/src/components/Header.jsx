import { useAuth } from '../context/AuthContext';
import './Header.css';

export default function Header({ user, onOpenHistory, onOpenVault }) {
    const { signOut } = useAuth();

    return (
        <header className="header">
            <div className="header-left">
                <h1 className="header-title">CERAS Dashboard</h1>
            </div>
            <div className="header-right">
                {user && (
                    <>
                        <button className="header-icon-btn" onClick={onOpenHistory} title="Chat History">
                            📜
                        </button>
                        <button className="header-icon-btn" onClick={onOpenVault} title="API Key Vault">
                            🔒
                        </button>
                        <div className="header-user">
                            <div className="header-avatar">
                                {(user.user_metadata?.display_name || user.email || '?')[0].toUpperCase()}
                            </div>
                            <button className="header-logout" onClick={signOut} title="Sign Out">
                                Sign Out
                            </button>
                        </div>
                    </>
                )}
            </div>
        </header>
    );
}
