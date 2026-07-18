/**
 * Supabase client.
 *
 * Deliberately optional: if the env vars aren't set the app still runs in a
 * local session so nobody's blocked on someone else provisioning a project.
 * `isConfigured` is surfaced in the UI so that state is never silent.
 */

import { createClient, type SupabaseClient } from '@supabase/supabase-js';

const url = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

export const isConfigured = Boolean(url && anonKey);

export const supabase: SupabaseClient | null = isConfigured
  ? createClient(url!, anonKey!)
  : null;
