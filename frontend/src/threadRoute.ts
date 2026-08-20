export type ThreadRoute =
  | { kind: 'blank' }
  | { kind: 'thread'; threadId: string }
  | { kind: 'not-found' };

export type HistoryMode = 'push' | 'replace' | 'none';

export function parseThreadRoute(pathname = window.location.pathname): ThreadRoute {
  if (pathname === '/' || pathname === '') return { kind: 'blank' };

  const match = /^\/threads\/([^/]+)\/?$/.exec(pathname);
  if (!match?.[1]) return { kind: 'not-found' };

  try {
    const threadId = decodeURIComponent(match[1]);
    return threadId ? { kind: 'thread', threadId } : { kind: 'not-found' };
  } catch {
    return { kind: 'not-found' };
  }
}

export function threadPath(threadId: string) {
  return `/threads/${encodeURIComponent(threadId)}`;
}

export function writeBlankRoute(mode: Exclude<HistoryMode, 'none'> = 'push') {
  const method = mode === 'replace' ? 'replaceState' : 'pushState';
  window.history[method]({ kind: 'blank' }, '', '/');
}

export function writeThreadRoute(
  threadId: string,
  mode: Exclude<HistoryMode, 'none'> = 'push',
) {
  const method = mode === 'replace' ? 'replaceState' : 'pushState';
  window.history[method]({ kind: 'thread', threadId }, '', threadPath(threadId));
}
