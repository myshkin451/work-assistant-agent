import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from './App';
import type { Message, ProductEvent, RunSnapshot, RunView, ThreadSnapshot, ThreadSummary } from './types';

const timestamp = '2026-08-20T12:00:00Z';

function jsonResponse(value: unknown, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

const threadA: ThreadSummary = {
  thread_id: 'thread-1',
  title: '会话一',
  created_at: timestamp,
  updated_at: timestamp,
};

const threadB: ThreadSummary = {
  ...threadA,
  thread_id: 'thread-2',
  title: '会话二',
};

function snapshotFor(thread: ThreadSummary, content: string): ThreadSnapshot {
  return {
    ...thread,
    messages: content
      ? [
          {
            message_id: `message-${thread.thread_id}`,
            role: 'user',
            content,
            created_at: timestamp,
            run_id: null,
          },
        ]
      : [],
    runs: [],
    active_run: null,
  };
}

function event(
  seq: number,
  type: ProductEvent['type'],
  data: Record<string, unknown> = {},
): ProductEvent {
  return {
    event_id: `event-${seq}`,
    run_id: 'run-stream',
    thread_id: threadA.thread_id,
    seq,
    type,
    occurred_at: timestamp,
    data,
  };
}

function eventBlock(item: ProductEvent) {
  return `id: ${item.seq}\nevent: ${item.type}\ndata: ${JSON.stringify(item)}\n\n`;
}

describe('conversation workspace navigation', () => {
  beforeEach(() => {
    window.history.replaceState(null, '', '/');
  });

  it('keeps a new conversation local until the first valid question', async () => {
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      if (String(input) === '/api/threads' && (init?.method ?? 'GET') === 'GET') {
        return jsonResponse({ items: [threadA] });
      }
      throw new Error(`Unexpected request: ${init?.method ?? 'GET'} ${String(input)}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText('今天想处理什么？')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '新建对话' }));
    await user.click(screen.getByRole('button', { name: '新建对话' }));

    expect(window.location.pathname).toBe('/');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(
      fetchMock.mock.calls.some(([, init]) => (init?.method ?? 'GET') === 'POST'),
    ).toBe(false);
  });

  it('locks navigation during initial admission and refreshes history if location changes', async () => {
    const initialKey = '33333333-3333-4333-8333-333333333333';
    const initialThreadId = '44444444-4444-4444-8444-444444444444';
    vi.spyOn(crypto, 'randomUUID')
      .mockReturnValueOnce(initialKey)
      .mockReturnValueOnce(initialThreadId);
    let listReads = 0;
    let resolveAdmission!: (response: Response) => void;
    const admission = new Promise<Response>((resolve) => {
      resolveAdmission = resolve;
    });
    const admittedThread: ThreadSummary = {
      ...threadA,
      thread_id: initialThreadId,
      title: '首问已创建',
    };
    const admittedRun: RunView = {
      run_id: 'admitted-run',
      thread_id: initialThreadId,
      status: 'running',
      last_seq: 0,
      created_at: timestamp,
      completed_at: null,
    };
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/threads' && method === 'GET') {
        listReads += 1;
        return jsonResponse({ items: listReads === 1 ? [threadA] : [admittedThread, threadA] });
      }
      if (url === `/api/threads/${initialThreadId}/initial-run` && method === 'POST') {
        return admission;
      }
      if (url === '/api/threads/thread-1' && method === 'GET') {
        return jsonResponse(snapshotFor(threadA, '原有会话正文'));
      }
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText('今天想处理什么？');
    await user.type(screen.getByLabelText('输入消息'), '创建后切换');
    await user.click(screen.getByRole('button', { name: '发送消息' }));
    expect(screen.getByRole('button', { name: '新建对话' })).toBeDisabled();
    expect(screen.getByRole('button', { name: threadA.title })).toBeDisabled();

    // Browser traversal is outside the disabled in-app navigation controls.
    window.history.pushState(null, '', '/threads/thread-1');
    window.dispatchEvent(new PopStateEvent('popstate'));
    expect(await screen.findByText('原有会话正文')).toBeInTheDocument();

    resolveAdmission(jsonResponse({ thread: admittedThread, run: admittedRun }, 201));
    await waitFor(() => expect(listReads).toBe(2));
    expect(window.location.pathname).toBe('/threads/thread-1');
    expect(screen.getByText('原有会话正文')).toBeInTheDocument();
    expect(screen.queryByText('创建后切换')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: threadA.title })).toBeEnabled();
    expect(
      fetchMock.mock.calls.some(([input]) =>
        String(input).startsWith('/api/runs/admitted-run/events'),
      ),
    ).toBe(false);
  });

  it('supports direct loading, switching and browser back/forward', async () => {
    window.history.replaceState(null, '', '/threads/thread-1');
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/threads' && method === 'GET') {
        return jsonResponse({ items: [threadA, threadB] });
      }
      if (url === '/api/threads/thread-1' && method === 'GET') {
        return jsonResponse(snapshotFor(threadA, '会话一正文'));
      }
      if (url === '/api/threads/thread-2' && method === 'GET') {
        return jsonResponse(snapshotFor(threadB, '会话二正文'));
      }
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText('会话一正文')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: threadB.title }));
    expect(await screen.findByText('会话二正文')).toBeInTheDocument();
    expect(window.location.pathname).toBe('/threads/thread-2');

    window.history.back();
    await waitFor(() => expect(window.location.pathname).toBe('/threads/thread-1'));
    expect(await screen.findByText('会话一正文')).toBeInTheDocument();

    window.history.forward();
    await waitFor(() => expect(window.location.pathname).toBe('/threads/thread-2'));
    expect(await screen.findByText('会话二正文')).toBeInTheDocument();
  });

  it('renames inline and renders the updated history title exactly once', async () => {
    window.history.replaceState(null, '', '/threads/thread-1');
    const updated = { ...threadA, title: '重命名后的会话' };
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/threads' && method === 'GET') return jsonResponse({ items: [threadA] });
      if (url === '/api/threads/thread-1' && method === 'GET') {
        return jsonResponse(snapshotFor(threadA, '用于重命名的正文'));
      }
      if (url === '/api/threads/thread-1' && method === 'PATCH') return jsonResponse(updated);
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText('用于重命名的正文')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '重命名当前对话' }));
    const titleInput = screen.getByLabelText('对话标题');
    await user.clear(titleInput);
    await user.type(titleInput, updated.title);
    await user.click(screen.getByRole('button', { name: '保存' }));

    const historyButton = await screen.findByRole('button', { name: updated.title });
    expect(within(historyButton).getAllByText(updated.title)).toHaveLength(1);
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/threads/thread-1',
      expect.objectContaining({
        method: 'PATCH',
        body: JSON.stringify({ title: updated.title }),
      }),
    );
  });

  it('opens an accessible mobile drawer and closes it with Escape', async () => {
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      if (String(input) === '/api/threads' && (init?.method ?? 'GET') === 'GET') {
        return jsonResponse({ items: [threadA] });
      }
      throw new Error(`Unexpected request: ${init?.method ?? 'GET'} ${String(input)}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText('今天想处理什么？');
    const trigger = screen.getByRole('button', { name: '打开对话导航' });
    await user.click(trigger);
    const drawer = screen.getByRole('dialog', { name: '对话导航' });
    expect(drawer).toHaveAttribute('aria-modal', 'true');
    fireEvent.keyDown(drawer, { key: 'Escape' });
    expect(screen.queryByRole('dialog', { name: '对话导航' })).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it('stops following stream output after the user scrolls away from the bottom', async () => {
    window.history.replaceState(null, '', '/threads/thread-1');
    const activeRun: RunView = {
      run_id: 'run-stream',
      thread_id: threadA.thread_id,
      status: 'running',
      last_seq: 1,
      created_at: timestamp,
      completed_at: null,
    };
    const question: Message = {
      message_id: 'stream-question',
      role: 'user',
      content: '请流式回答',
      created_at: timestamp,
      run_id: activeRun.run_id,
    };
    const activeSnapshot: ThreadSnapshot = {
      ...threadA,
      messages: [question],
      runs: [
        {
          ...activeRun,
          events: [event(1, 'run.started', { status: 'running' })],
        } as RunSnapshot,
      ],
      active_run: activeRun,
    };
    let streamController!: ReadableStreamDefaultController<Uint8Array>;
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        streamController = controller;
      },
    });
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/threads' && method === 'GET') return jsonResponse({ items: [threadA] });
      if (url === '/api/threads/thread-1' && method === 'GET') return jsonResponse(activeSnapshot);
      if (url === '/api/runs/run-stream/events?after_seq=1') {
        return new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } });
      }
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const scrollTo = vi.spyOn(Element.prototype, 'scrollTo');
    const user = userEvent.setup();
    const { container } = render(<App />);

    expect(await screen.findByText('请流式回答')).toBeInTheDocument();
    const scrollRegion = container.querySelector<HTMLElement>('.conversation-scroll');
    expect(scrollRegion).not.toBeNull();
    Object.defineProperties(scrollRegion!, {
      scrollHeight: { configurable: true, value: 1_000 },
      clientHeight: { configurable: true, value: 400 },
      scrollTop: { configurable: true, value: 120, writable: true },
    });
    fireEvent.scroll(scrollRegion!);
    expect(await screen.findByRole('button', { name: '回到最新' })).toBeInTheDocument();
    const callsBeforeDelta = scrollTo.mock.calls.length;

    streamController.enqueue(
      new TextEncoder().encode(eventBlock(event(2, 'message.delta', { delta: '**未闭合流式内容' }))),
    );
    expect(await screen.findByText('**未闭合流式内容')).toBeInTheDocument();
    expect(scrollTo).toHaveBeenCalledTimes(callsBeforeDelta);

    await user.click(screen.getByRole('button', { name: '回到最新' }));
    expect(scrollTo).toHaveBeenLastCalledWith({ top: 1_000, behavior: 'smooth' });
    streamController.enqueue(
      new TextEncoder().encode(eventBlock(event(3, 'run.completed', { status: 'completed' }))),
    );
    streamController.close();
  });
});
