import { afterEach, describe, expect, it } from 'vitest';
import {
  parseThreadRoute,
  threadPath,
  writeBlankRoute,
  writeThreadRoute,
} from './threadRoute';

afterEach(() => {
  window.history.replaceState(null, '', '/');
});

describe('thread routes', () => {
  it('parses blank, direct and invalid paths without accepting extra segments', () => {
    expect(parseThreadRoute('/')).toEqual({ kind: 'blank' });
    expect(parseThreadRoute('/threads/thread%20one')).toEqual({
      kind: 'thread',
      threadId: 'thread one',
    });
    expect(parseThreadRoute('/threads/thread-1/')).toEqual({
      kind: 'thread',
      threadId: 'thread-1',
    });
    expect(parseThreadRoute('/threads/thread-1/messages')).toEqual({ kind: 'not-found' });
    expect(parseThreadRoute('/threads/%E0%A4%A')).toEqual({ kind: 'not-found' });
  });

  it('builds encoded canonical paths and writes push or replace history entries', () => {
    expect(threadPath('thread one')).toBe('/threads/thread%20one');

    writeThreadRoute('thread one');
    expect(window.location.pathname).toBe('/threads/thread%20one');
    expect(window.history.state).toEqual({ kind: 'thread', threadId: 'thread one' });

    writeBlankRoute('replace');
    expect(window.location.pathname).toBe('/');
    expect(window.history.state).toEqual({ kind: 'blank' });
  });
});
