/**
 * Sign in / sign up.
 *
 * The crowd runs behind it — the first thing anyone sees should already be the
 * thing itself, not a form on a blank page.
 */

import { useState, type FormEvent } from 'react';
import { useAuth } from './AuthProvider.js';
import { CrowdPane } from '../crowd/CrowdPane.js';

type Mode = 'in' | 'up';

export function AuthScreen() {
  const { signIn, signUp, configured } = useAuth();
  const [mode, setMode] = useState<Mode>('in');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setNotice(null);
    setBusy(true);
    try {
      if (mode === 'in') await signIn(email, password);
      else setNotice(await signUp(email, password));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="auth">
      <div className="auth-bg" aria-hidden>
        <CrowdPane side="user" />
      </div>

      <form className="auth-card" onSubmit={submit}>
        <h1>null</h1>
        <p className="auth-sub">
          It hears you. This is the first interface that shows you what it heard.
        </p>

        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          autoComplete="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          autoComplete={mode === 'in' ? 'current-password' : 'new-password'}
          required
          minLength={6}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        {error && <p className="auth-error">{error}</p>}
        {notice && <p className="auth-notice">{notice}</p>}

        <button type="submit" disabled={busy}>
          {busy ? '…' : mode === 'in' ? 'Sign in' : 'Create account'}
        </button>

        <button
          type="button"
          className="auth-link"
          onClick={() => {
            setMode(mode === 'in' ? 'up' : 'in');
            setError(null);
            setNotice(null);
          }}
        >
          {mode === 'in' ? 'No account? Sign up' : 'Already have an account? Sign in'}
        </button>

        {!configured && (
          <p className="auth-warn">
            Supabase isn’t configured — running a local session. Set
            <code> VITE_SUPABASE_URL </code> and <code> VITE_SUPABASE_ANON_KEY </code>
            in <code>.env</code> for real accounts.
          </p>
        )}
      </form>
    </div>
  );
}
