import {
  productEventTypes,
  type ProductEvent,
  type ProductEventType,
  type RunView,
  type ThreadSnapshot,
  type ThreadSummary,
} from './types';

const configuredBase = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() ?? '';
export const API_BASE_URL = configuredBase.replace(/\/$/, '');

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

const apiUrl = (path: string) => `${API_BASE_URL}${path}`;

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), {
    ...init,
    credentials: 'include',
    cache: 'no-store',
    headers: {
      Accept: 'application/json',
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    throw new ApiError(response.status, publicErrorMessage(response.status));
  }

  return (await response.json()) as T;
}

function publicErrorMessage(status: number) {
  if (status === 401) return '当前请求未通过身份认证，请完成认证后刷新页面。';
  if (status === 403) return '当前身份无权访问这项会话或运行。';
  if (status === 404) return '请求的会话不存在。';
  if (status === 409) return '这个会话已有任务正在运行。';
  if (status === 422) return '提交内容不符合要求。';
  if (status >= 500) return '服务暂时不可用，请稍后重试。';
  return '请求没有成功，请重试。';
}

export const listThreads = async () => {
  const response = await requestJson<{ items: ThreadSummary[] }>('/api/threads');
  return response.items;
};

export const createThread = (title?: string) =>
  requestJson<ThreadSnapshot>('/api/threads', {
    method: 'POST',
    body: JSON.stringify(title ? { title } : {}),
  });

export const getThread = (threadId: string) =>
  requestJson<ThreadSnapshot>(`/api/threads/${encodeURIComponent(threadId)}`);

export const createRun = (threadId: string, message: string, idempotencyKey: string) =>
  requestJson<RunView>(`/api/threads/${encodeURIComponent(threadId)}/runs`, {
    method: 'POST',
    body: JSON.stringify({ message, idempotency_key: idempotencyKey }),
  });

export const cancelRun = (runId: string) =>
  requestJson<RunView>(`/api/runs/${encodeURIComponent(runId)}/cancel`, {
    method: 'POST',
  });

const knownEventTypes = new Set<string>(productEventTypes);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export function parseProductEvent(value: unknown): ProductEvent | null {
  if (!isRecord(value)) return null;
  if (
    typeof value.event_id !== 'string' ||
    typeof value.run_id !== 'string' ||
    typeof value.thread_id !== 'string' ||
    typeof value.seq !== 'number' ||
    !Number.isSafeInteger(value.seq) ||
    value.seq < 1 ||
    typeof value.type !== 'string' ||
    !knownEventTypes.has(value.type) ||
    typeof value.occurred_at !== 'string' ||
    !isRecord(value.data)
  ) {
    return null;
  }

  return {
    event_id: value.event_id,
    run_id: value.run_id,
    thread_id: value.thread_id,
    seq: value.seq,
    type: value.type as ProductEventType,
    occurred_at: value.occurred_at,
    data: value.data,
  };
}

export async function readSseEvents(
  body: ReadableStream<Uint8Array>,
  onEvent: (event: ProductEvent) => void,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const flushBlock = (block: string) => {
    const data = block
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart())
      .join('\n');
    if (!data) return;
    try {
      const event = parseProductEvent(JSON.parse(data) as unknown);
      if (event) onEvent(event);
    } catch {
      // Invalid or private provider payloads are deliberately ignored.
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done }).replaceAll('\r\n', '\n');
    let boundary = buffer.indexOf('\n\n');
    while (boundary >= 0) {
      flushBlock(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf('\n\n');
    }
    if (done) break;
  }
  if (buffer.trim()) flushBlock(buffer);
}

const waitForReconnect = (delay: number, signal: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(resolve, delay);
    signal.addEventListener(
      'abort',
      () => {
        window.clearTimeout(timeout);
        reject(new DOMException('Aborted', 'AbortError'));
      },
      { once: true },
    );
  });

type StreamCallbacks = {
  onEvent: (event: ProductEvent) => void;
  onConnectionChange?: (state: 'connecting' | 'live' | 'reconnecting') => void;
};

export async function streamRunEvents(
  runId: string,
  initialAfterSeq: number,
  callbacks: StreamCallbacks,
  signal: AbortSignal,
): Promise<void> {
  let afterSeq = initialAfterSeq;
  let attempts = 0;
  callbacks.onConnectionChange?.('connecting');

  while (!signal.aborted) {
    try {
      const response = await fetch(
        apiUrl(`/api/runs/${encodeURIComponent(runId)}/events?after_seq=${afterSeq}`),
        {
          // Keep the cross-origin loopback GET CORS-simple. The server response
          // already disables caching; a request Cache-Control header would add
          // an unnecessary preflight before every reconnect.
          headers: { Accept: 'text/event-stream' },
          credentials: 'include',
          cache: 'no-store',
          signal,
        },
      );
      if (!response.ok || !response.body) {
        if (response.status >= 400 && response.status < 500) {
          throw new ApiError(response.status, publicErrorMessage(response.status));
        }
        throw new Error('SSE connection unavailable');
      }

      attempts = 0;
      callbacks.onConnectionChange?.('live');
      let terminal = false;
      await readSseEvents(response.body, (event) => {
        if (event.run_id !== runId || event.seq <= afterSeq) return;
        afterSeq = event.seq;
        callbacks.onEvent(event);
        terminal =
          event.type === 'run.completed' ||
          event.type === 'run.failed' ||
          event.type === 'run.cancelled';
      });
      if (terminal || signal.aborted) return;
    } catch (error) {
      if (signal.aborted || (error instanceof DOMException && error.name === 'AbortError')) return;
      if (error instanceof ApiError) throw error;
    }

    attempts += 1;
    callbacks.onConnectionChange?.('reconnecting');
    await waitForReconnect(Math.min(250 * 2 ** (attempts - 1), 2_000), signal);
  }
}
