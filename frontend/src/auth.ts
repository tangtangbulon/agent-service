import type { AuthSession } from './types';

const STORAGE_KEY = 'insurance-mall-auth-v3';

function isValidSession(value: unknown): value is AuthSession {
  if (!value || typeof value !== 'object') return false;
  const session = value as Partial<AuthSession>;
  return (
    typeof session.accessToken === 'string' &&
    typeof session.expiresAt === 'number' &&
    session.expiresAt > Date.now() &&
    typeof session.user?.id === 'number'
  );
}

export function readSession(): AuthSession | null {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    const session: unknown = value ? JSON.parse(value) : null;
    if (isValidSession(session)) return session;
  } catch {
    // 存储内容损坏时按未登录处理
  }
  window.localStorage.removeItem(STORAGE_KEY);
  return null;
}

export function saveSession(session: AuthSession): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
}

export function clearSession(): void {
  window.localStorage.removeItem(STORAGE_KEY);
}
