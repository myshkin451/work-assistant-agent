import { describe, expect, it, vi } from 'vitest';
import {
  createInitialRun,
  getAccountUsage,
  parseProductEvent,
  readSseEvents,
  streamRunEvents,
  updateThread,
} from './api';
import type { ProductEvent } from './types';

function event(seq: number, type: ProductEvent['type'], data: Record<string, unknown> = {}) {
  return {
    event_id: `event-${seq}`,
    run_id: 'run-1',
    thread_id: 'thread-1',
    seq,
    type,
    occurred_at: '2026-08-12T12:00:00Z',
    data,
  };
}

function sseResponse(events: Array<Record<string, unknown>>) {
  const body = events
    .map((item) => `id: ${String(item.seq)}\nevent: ${String(item.type)}\ndata: ${JSON.stringify(item)}\n\n`)
    .join('');
  return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } });
}

describe('product event boundary', () => {
  it('accepts only the documented product envelope', () => {
    expect(parseProductEvent(event(1, 'run.started', { status: 'running' }))).toMatchObject({
      seq: 1,
      type: 'run.started',
    });
    expect(
      parseProductEvent({
        ...event(2, 'message.delta'),
        type: 'provider.reasoning',
        data: { reasoning: 'private' },
      }),
    ).toBeNull();
    expect(parseProductEvent({ ...event(2, 'message.delta'), seq: 0 })).toBeNull();
  });

  it('ignores malformed and private SSE data instead of rendering it', async () => {
    const response = new Response(
      [
        'event: provider.reasoning\ndata: {"reasoning":"private"}\n\n',
        `event: message.delta\ndata: ${JSON.stringify(event(1, 'message.delta', { delta: '公开内容' }))}\n\n`,
      ].join(''),
    );
    const received: ProductEvent[] = [];
    await readSseEvents(response.body!, (item) => received.push(item));
    expect(received).toHaveLength(1);
    expect(received[0]?.data).toEqual({ delta: '公开内容' });
  });

  it('reconnects from the last accepted seq and drops replay duplicates', async () => {
    vi.useFakeTimers();
    const requests: string[] = [];
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockImplementationOnce(async (input) => {
        requests.push(String(input));
        return sseResponse([event(1, 'run.started', { status: 'running' })]);
      })
      .mockImplementationOnce(async (input) => {
        requests.push(String(input));
        return sseResponse([
          event(1, 'run.started', { status: 'running' }),
          event(2, 'run.completed', { status: 'completed' }),
        ]);
      });
    vi.stubGlobal('fetch', fetchMock);
    const received: number[] = [];
    const controller = new AbortController();
    const streaming = streamRunEvents(
      'run-1',
      0,
      { onEvent: (item) => received.push(item.seq) },
      controller.signal,
    );

    await vi.advanceTimersByTimeAsync(250);
    await streaming;

    expect(requests).toEqual([
      '/api/runs/run-1/events?after_seq=0',
      '/api/runs/run-1/events?after_seq=1',
    ]);
    expect(received).toEqual([1, 2]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    for (const [url, init] of fetchMock.mock.calls) {
      expect(String(url)).not.toContain('principal');
      expect(init).toEqual(
        expect.objectContaining({
          credentials: 'include',
          headers: { Accept: 'text/event-stream' },
        }),
      );
    }
    vi.useRealTimers();
  });

  it.each([401, 403])('does not retry an authorization failure with status %s', async (status) => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ detail: { code: 'access_denied' } }), {
        status,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const controller = new AbortController();

    await expect(
      streamRunEvents('run-1', 3, { onEvent: vi.fn() }, controller.signal),
    ).rejects.toMatchObject({ status });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/runs/run-1/events?after_seq=3',
      expect.objectContaining({ credentials: 'include' }),
    );
  });
});

describe('thread workspace API boundary', () => {
  it('reads only the current account scope with credentialed no-store GET', async () => {
    const response = {
      account: {
        display_name: '当前用户',
        organization: null,
        extensions: { session_expires_at: null, permission_summary: null },
      },
      scope: {
        range: '30d',
        from_at: null,
        to_at: '2026-08-20T12:00:00Z',
        thread_id: 'thread/with space',
      },
      runs: { total: 0, completed: 0, failed: 0, cancelled: 0, active: 0 },
      model_calls: { value: 0, availability: 'complete' },
      retries: { value: 0, availability: 'complete' },
      input_tokens: { value: 0, availability: 'complete' },
      output_tokens: { value: 0, availability: 'complete' },
      cached_tokens: { value: 0, availability: 'complete' },
      reasoning_tokens: { value: 0, availability: 'complete' },
      total_tokens: { value: 0, availability: 'complete' },
    };
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify(response), {
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(getAccountUsage('30d', 'thread/with space')).resolves.toEqual(response);
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/account/usage?range=30d&thread_id=thread%2Fwith+space',
      expect.objectContaining({
        credentials: 'include',
        cache: 'no-store',
        headers: { Accept: 'application/json' },
      }),
    );
  });

  it('sends only the frozen initial-run fields and receives summary plus run', async () => {
    const thread = {
      thread_id: '11111111-1111-4111-8111-111111111111',
      title: '首问',
      created_at: '2026-08-20T12:00:00Z',
      updated_at: '2026-08-20T12:00:00Z',
    };
    const run = {
      run_id: 'run-1',
      thread_id: thread.thread_id,
      status: 'running',
      last_seq: 0,
      created_at: thread.created_at,
      completed_at: null,
    };
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ thread, run }), {
        status: 201,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(createInitialRun(thread.thread_id, '第一问', 'stable-key')).resolves.toEqual({
      thread,
      run,
    });
    expect(fetchMock).toHaveBeenCalledWith(
      `/api/threads/${thread.thread_id}/initial-run`,
      expect.objectContaining({
        method: 'POST',
        credentials: 'include',
        body: JSON.stringify({ message: '第一问', idempotency_key: 'stable-key' }),
      }),
    );
  });

  it('patches only the normalized title field', async () => {
    const updated = {
      thread_id: 'thread/with space',
      title: '新标题',
      created_at: '2026-08-20T12:00:00Z',
      updated_at: '2026-08-20T12:01:00Z',
    };
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify(updated), {
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(updateThread(updated.thread_id, updated.title)).resolves.toEqual(updated);
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/threads/thread%2Fwith%20space',
      expect.objectContaining({
        method: 'PATCH',
        credentials: 'include',
        body: JSON.stringify({ title: updated.title }),
      }),
    );
  });
});
