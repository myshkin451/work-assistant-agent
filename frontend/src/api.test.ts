import { describe, expect, it, vi } from 'vitest';
import { parseProductEvent, readSseEvents, streamRunEvents } from './api';
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
  });
});
