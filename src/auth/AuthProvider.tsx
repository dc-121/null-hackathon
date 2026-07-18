/**
 * Auth state.
 *
 * The only React state in the app that matters — everything per-frame lives in
 * the mutable store instead. This one is fine: it changes a handful of times
 * per session.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import { isConfigured, supabase } from './supabase.js';

export interface AuthUser {
  id: string;
  email: string;
  /** True when we're running without Supabase configured. */
  local?: boolean;
}

interface AuthValue {
  user: AuthUser | null;
  loading: boolean;
  configured: boolean;
  signIn(email: string, password: string): Promise<void>;
  signUp(email: string, password: string): Promise<string | null>;
  signOut(): Promise<void>;
}

const AuthContext = createContext<AuthValue | null>(null);

const LOCAL_KEY = 'null-hackathon.local-session';

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!supabase) {
      const saved = localStorage.getItem(LOCAL_KEY);
      if (saved) setUser({ id: 'local', email: saved, local: true });
      setLoading(false);
      return;
    }

    supabase.auth.getSession().then(({ data }) => {
      const s = data.session;
      if (s?.user) setUser({ id: s.user.id, email: s.user.email ?? '' });
      setLoading(false);
    });

    const { data: sub } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ? { id: session.user.id, email: session.user.email ?? '' } : null);
    });
    return () => sub.subscription.unsubscribe();
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    if (!supabase) {
      localStorage.setItem(LOCAL_KEY, email);
      setUser({ id: 'local', email, local: true });
      return;
    }
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) throw new Error(error.message);
  }, []);

  const signUp = useCallback(async (email: string, password: string) => {
    if (!supabase) {
      localStorage.setItem(LOCAL_KEY, email);
      setUser({ id: 'local', email, local: true });
      return null;
    }
    const { data, error } = await supabase.auth.signUp({ email, password });
    if (error) throw new Error(error.message);
    // Depending on project settings this may require email confirmation, in
    // which case there's no session yet and the user needs telling.
    if (!data.session) return 'Check your email to confirm your account.';
    return null;
  }, []);

  const signOut = useCallback(async () => {
    localStorage.removeItem(LOCAL_KEY);
    if (supabase) await supabase.auth.signOut();
    setUser(null);
  }, []);

  const value = useMemo<AuthValue>(
    () => ({ user, loading, configured: isConfigured, signIn, signUp, signOut }),
    [user, loading, signIn, signUp, signOut]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider');
  return ctx;
}
