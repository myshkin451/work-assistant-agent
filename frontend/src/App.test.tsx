import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from './App';
import type {
  Message,
  ProductEvent,
  RunFailureCode,
  RunSnapshot,
  RunView,
  ThreadSnapshot,
  ThreadSummary,
} from './types';

const timestamp = '2026-08-12T12:00:00Z';

function jsonResponse(value: unknown, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function event(
  seq: number,
  type: ProductEvent['type'],
  data: Record<string, unknown> = {},
  runId = 'run-1',
  threadId = 'thread-1',
) {
  return {
    event_id: `event-${seq}`,
    run_id: runId,
    thread_id: threadId,
    seq,
    type,
    occurred_at: timestamp,
    data,
  };
}

function privateEvent(seq: number) {
  return {
    event_id: `private-${seq}`,
    run_id: 'run-1',
    thread_id: 'thread-1',
    seq,
    type: 'provider.reasoning',
    occurred_at: timestamp,
    data: { reasoning: '不应显示的内部思考' },
  };
}

function sseResponse(events: Array<Record<string, unknown>>) {
  const body = events
    .map((item) => `id: ${String(item.seq)}\nevent: ${String(item.type)}\ndata: ${JSON.stringify(item)}\n\n`)
    .join('');
  return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } });
}

const summary: ThreadSummary = {
  thread_id: 'thread-1',
  title: '查询上海当前时间',
  created_at: timestamp,
  updated_at: timestamp,
};

const running: RunView = {
  run_id: 'run-1',
  thread_id: 'thread-1',
  status: 'running',
  last_seq: 0,
  created_at: timestamp,
  completed_at: null,
};

const userMessage = {
  message_id: 'message-user',
  role: 'user' as const,
  content: '请查询当前上海时间，并说明结果来自哪里。',
  created_at: timestamp,
  run_id: 'run-1',
};

const assistantMessage = {
  message_id: 'message-assistant',
  role: 'assistant' as const,
  content: '上海当前时间为 2026 年 8 月 12 日 20:00。',
  created_at: timestamp,
  run_id: 'run-1',
};

function runSnapshot(run: RunView, events: ProductEvent[]): RunSnapshot {
  return { ...run, events };
}

describe('employee chat vertical slice', () => {
  beforeEach(() => {
    window.history.replaceState(null, '', '/threads/thread-1');
  });

  it('blocks all interaction when the initial request is unauthenticated', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse({ detail: { code: 'authentication_required' } }, 401),
    );
    vi.stubGlobal('fetch', fetchMock);
    render(<App />);

    expect(
      await screen.findByText('登录已失效，请重新登录。'),
    ).toBeInTheDocument();
    expect(screen.getByLabelText('输入消息')).toBeDisabled();
    expect(screen.queryByText('今天想处理什么？')).not.toBeInTheDocument();
    for (const button of screen.getAllByRole('button', { name: '新建对话' })) {
      expect(button).toBeDisabled();
    }
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/threads',
      expect.objectContaining({ credentials: 'include' }),
    );
  });

  it('clears another principal\'s rendered state when SSE reauthorization is forbidden', async () => {
    const privateMessage: Message = {
      ...userMessage,
      content: 'A 主体的私密会话正文',
    };
    const activeRun = { ...running, last_seq: 1 };
    const privateSnapshot: ThreadSnapshot = {
      ...summary,
      title: 'A 主体私密历史',
      messages: [privateMessage],
      runs: [runSnapshot(activeRun, [event(1, 'run.started', { status: 'running' })])],
      active_run: activeRun,
    };
    let resolveSse!: (response: Response) => void;
    const deferredSse = new Promise<Response>((resolve) => {
      resolveSse = resolve;
    });
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/threads' && method === 'GET') return jsonResponse({ items: [summary] });
      if (url === '/api/threads/thread-1' && method === 'GET') return jsonResponse(privateSnapshot);
      if (url === '/api/runs/run-1/events?after_seq=1') {
        return deferredSse;
      }
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<App />);

    expect(await screen.findByText(privateMessage.content)).toBeInTheDocument();
    expect(screen.getByText(privateSnapshot.title)).toBeInTheDocument();
    resolveSse(jsonResponse({ detail: { code: 'run_forbidden' } }, 403));
    expect(
      await screen.findByText('你没有权限查看这个对话。'),
    ).toBeInTheDocument();
    expect(screen.queryByText(privateMessage.content)).not.toBeInTheDocument();
    expect(screen.queryByText(privateSnapshot.title)).not.toBeInTheDocument();
    expect(screen.getByLabelText('输入消息')).toBeDisabled();
    expect(screen.queryByRole('button', { name: '停止运行' })).not.toBeInTheDocument();
  });

  it('keeps owned history and ignores a late request after a route-level 403', async () => {
    const otherSummary: ThreadSummary = {
      ...summary,
      thread_id: 'thread-2',
      title: '触发重新鉴权',
    };
    const idleSnapshot: ThreadSnapshot = {
      ...summary,
      messages: [userMessage],
      runs: [],
      active_run: null,
    };
    let resolveCreateRun!: (response: Response) => void;
    const deferredCreateRun = new Promise<Response>((resolve) => {
      resolveCreateRun = resolve;
    });
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/threads' && method === 'GET') {
        return jsonResponse({ items: [summary, otherSummary] });
      }
      if (url === '/api/threads/thread-1' && method === 'GET') return jsonResponse(idleSnapshot);
      if (url === '/api/threads/thread-1/runs' && method === 'POST') {
        return deferredCreateRun;
      }
      if (url === '/api/threads/thread-2' && method === 'GET') {
        return jsonResponse({ detail: { code: 'thread_forbidden' } }, 403);
      }
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText(userMessage.content)).toBeInTheDocument();
    const composer = screen.getByLabelText('输入消息');
    await user.type(composer, 'A 主体尚未发送完成的草稿');
    await user.click(screen.getByRole('button', { name: '发送消息' }));
    await user.click(screen.getByRole('button', { name: otherSummary.title }));

    expect(
      await screen.findByText('无法打开这个对话。你可以选择最近对话，或新建一个对话。'),
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: summary.title })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: otherSummary.title })).toBeInTheDocument();
    resolveCreateRun(jsonResponse({ detail: { code: 'service_failure' } }, 500));
    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(
        '无法打开这个对话。你可以选择最近对话，或新建一个对话。',
      );
    });
    expect(composer).toHaveValue('');
    expect(composer).toBeEnabled();
    expect(screen.queryByText(userMessage.content)).not.toBeInTheDocument();
    expect(screen.queryByText('A 主体尚未发送完成的草稿')).not.toBeInTheDocument();
  });

  it('connects SSE before the first snapshot resolves and renders deduplicated product events', async () => {
    window.history.replaceState(null, '', '/');
    const initialKey = '11111111-1111-4111-8111-111111111111';
    const initialThreadId = '22222222-2222-4222-8222-222222222222';
    vi.spyOn(crypto, 'randomUUID')
      .mockReturnValueOnce(initialKey)
      .mockReturnValueOnce(initialThreadId);
    const initialSummary: ThreadSummary = { ...summary, thread_id: initialThreadId };
    const initialRunning: RunView = { ...running, thread_id: initialThreadId };
    const emptyList = { items: [] as ThreadSummary[] };
    const withUser: ThreadSnapshot = {
      ...initialSummary,
      messages: [userMessage],
      runs: [runSnapshot(initialRunning, [])],
      active_run: initialRunning,
    };
    const completedSnapshot: ThreadSnapshot = {
      ...initialSummary,
      messages: [userMessage, assistantMessage],
      runs: [
        runSnapshot(
          { ...initialRunning, status: 'completed', last_seq: 9, completed_at: timestamp },
          [
            event(1, 'run.started', { status: 'running' }, 'run-1', initialThreadId),
            event(2, 'tool.started', {
              tool_call_id: 'tool-1',
              name: 'get_current_time',
              label: '查询指定时区的当前时间',
              input_summary: 'Asia/Shanghai',
            }, 'run-1', initialThreadId),
            event(3, 'tool.finished', {
              tool_call_id: 'tool-1',
              name: 'get_current_time',
              label: '查询指定时区的当前时间',
              output_summary: '已取得 Asia/Shanghai 当前时间',
            }, 'run-1', initialThreadId),
            event(5, 'message.delta', { delta: '上海当前时间' }, 'run-1', initialThreadId),
            event(6, 'message.delta', { delta: '为 2026 年 8 月 12 日 20:00。' }, 'run-1', initialThreadId),
            event(7, 'source.added', {
              source_id: 'source-1',
              label: '系统时钟 · Asia/Shanghai',
              description: '由只读时间工具返回',
            }, 'run-1', initialThreadId),
            event(8, 'message.completed', { message: assistantMessage }, 'run-1', initialThreadId),
            event(9, 'run.completed', { status: 'completed' }, 'run-1', initialThreadId),
          ],
        ),
      ],
      active_run: null,
    };
    let snapshotReads = 0;
    let staleSnapshotResolved = false;
    let resolveStaleSnapshot!: (response: Response) => void;
    const staleSnapshot = new Promise<Response>((resolve) => {
      resolveStaleSnapshot = resolve;
    });
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/threads' && method === 'GET') {
        return jsonResponse(snapshotReads === 0 ? emptyList : { items: [initialSummary] });
      }
      if (url === `/api/threads/${initialThreadId}/initial-run` && method === 'POST') {
        return jsonResponse({ thread: initialSummary, run: initialRunning }, 201);
      }
      if (url === `/api/threads/${initialThreadId}` && method === 'GET') {
        snapshotReads += 1;
        return snapshotReads === 1 ? staleSnapshot : jsonResponse(completedSnapshot);
      }
      if (url === '/api/runs/run-1/events?after_seq=0') {
        return sseResponse([
          event(1, 'run.started', { status: 'running' }, 'run-1', initialThreadId),
          event(2, 'tool.started', {
            tool_call_id: 'tool-1',
            name: 'get_current_time',
            label: '查询指定时区的当前时间',
            input_summary: 'Asia/Shanghai',
          }, 'run-1', initialThreadId),
          event(3, 'tool.finished', {
            tool_call_id: 'tool-1',
            name: 'get_current_time',
            label: '查询指定时区的当前时间',
            output_summary: '已取得 Asia/Shanghai 当前时间',
          }, 'run-1', initialThreadId),
          { ...privateEvent(4), thread_id: initialThreadId },
          event(5, 'message.delta', { delta: '上海当前时间' }, 'run-1', initialThreadId),
          event(5, 'message.delta', { delta: '重复内容不应显示' }, 'run-1', initialThreadId),
          event(6, 'message.delta', { delta: '为 2026 年 8 月 12 日 20:00。' }, 'run-1', initialThreadId),
          event(7, 'source.added', {
            source_id: 'source-1',
            label: '系统时钟 · Asia/Shanghai',
            description: '由只读时间工具返回',
          }, 'run-1', initialThreadId),
          event(8, 'message.completed', { message: assistantMessage }, 'run-1', initialThreadId),
          event(9, 'run.completed', { status: 'completed' }, 'run-1', initialThreadId),
        ]);
      }
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText('今天想处理什么？')).toBeInTheDocument();
    await user.type(
      screen.getByLabelText('输入消息'),
      '请查询当前上海时间，并说明结果来自哪里。',
    );
    await user.click(screen.getByRole('button', { name: '发送消息' }));

    expect(await screen.findByText(assistantMessage.content)).toBeInTheDocument();
    expect(staleSnapshotResolved).toBe(false);
    expect(screen.getByText('查询指定时区的当前时间')).toBeInTheDocument();
    expect(screen.getByText('系统时钟 · Asia/Shanghai')).toBeInTheDocument();
    expect(screen.queryByText('重复内容不应显示')).not.toBeInTheDocument();
    expect(screen.queryByText('不应显示的内部思考')).not.toBeInTheDocument();
    expect(window.location.pathname).toBe(`/threads/${initialThreadId}`);
    expect(fetchMock).toHaveBeenCalledWith(
      `/api/threads/${initialThreadId}/initial-run`,
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ message: userMessage.content, idempotency_key: initialKey }),
      }),
    );
    expect(
      fetchMock.mock.calls.some(
        ([input, init]) => String(input) === '/api/threads' && init?.method === 'POST',
      ),
    ).toBe(false);
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/runs/run-1/events?after_seq=0',
      expect.objectContaining({ credentials: 'include' }),
    );
    staleSnapshotResolved = true;
    resolveStaleSnapshot(jsonResponse(withUser));
    await waitFor(() => expect(screen.queryByRole('button', { name: '停止运行' })).not.toBeInTheDocument());
    expect(screen.getByText(assistantMessage.content)).toBeInTheDocument();
  });

  it('keeps a newer same-thread run active when an older refresh returns late', async () => {
    const firstRun = { ...running, last_seq: 1 };
    const firstSnapshot: ThreadSnapshot = {
      ...summary,
      messages: [userMessage],
      runs: [runSnapshot(firstRun, [event(1, 'run.started', { status: 'running' })])],
      active_run: firstRun,
    };
    const secondRun: RunView = {
      ...running,
      run_id: 'run-2',
      created_at: '2026-08-12T12:01:00Z',
    };
    let threadReads = 0;
    let resolveOldRefresh!: (response: Response) => void;
    const oldRefresh = new Promise<Response>((resolve) => {
      resolveOldRefresh = resolve;
    });
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/threads' && method === 'GET') return jsonResponse({ items: [summary] });
      if (url === '/api/threads/thread-1' && method === 'GET') {
        threadReads += 1;
        if (threadReads === 1) return jsonResponse(firstSnapshot);
        return oldRefresh;
      }
      if (url === '/api/runs/run-1/events?after_seq=1') {
        return sseResponse([event(2, 'run.completed', { status: 'completed' })]);
      }
      if (url === '/api/threads/thread-1/runs' && method === 'POST') {
        return jsonResponse(secondRun);
      }
      if (url === '/api/runs/run-2/events?after_seq=0') {
        return new Promise<Response>(() => undefined);
      }
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    const composer = await screen.findByLabelText('输入消息');
    await waitFor(() => expect(composer).toBeEnabled());
    await user.type(composer, '继续查询下一项');
    await user.click(screen.getByRole('button', { name: '发送消息' }));
    expect(await screen.findByRole('button', { name: '停止运行' })).toBeInTheDocument();

    resolveOldRefresh(jsonResponse(firstSnapshot));
    await waitFor(() => expect(screen.getByRole('button', { name: '停止运行' })).toBeInTheDocument());
    expect(composer).toBeDisabled();
    expect(screen.getByText('继续查询下一项')).toBeInTheDocument();
  });

  it('restores an active run and continues SSE from the snapshot last seq', async () => {
    const activeEvents = [
      event(1, 'run.started', { status: 'running' }),
      event(2, 'tool.started', {
        tool_call_id: 'tool-1',
        name: 'get_current_time',
        label: '查询指定时区的当前时间',
        input_summary: 'Asia/Shanghai',
      }),
      event(3, 'tool.finished', {
        tool_call_id: 'tool-1',
        name: 'get_current_time',
        label: '查询指定时区的当前时间',
        output_summary: '已取得 Asia/Shanghai 当前时间',
      }),
      event(4, 'source.added', {
        source_id: 'source-1',
        label: '系统时钟 · Asia/Shanghai',
        description: '由只读时间工具返回',
      }),
    ];
    const activeRun = { ...running, last_seq: 4 };
    const activeSnapshot: ThreadSnapshot = {
      ...summary,
      messages: [userMessage],
      runs: [runSnapshot(activeRun, activeEvents)],
      active_run: activeRun,
    };
    const completedSnapshot: ThreadSnapshot = {
      ...summary,
      messages: [userMessage, assistantMessage],
      runs: [
        runSnapshot(
          { ...running, status: 'completed', last_seq: 7, completed_at: timestamp },
          [
            ...activeEvents,
            event(5, 'message.delta', { delta: assistantMessage.content }),
            event(6, 'message.completed', { message: assistantMessage }),
            event(7, 'run.completed', { status: 'completed' }),
          ],
        ),
      ],
      active_run: null,
    };
    let threadReads = 0;
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/threads' && method === 'GET') return jsonResponse({ items: [summary] });
      if (url === '/api/threads/thread-1' && method === 'GET') {
        threadReads += 1;
        return jsonResponse(threadReads === 1 ? activeSnapshot : completedSnapshot);
      }
      if (url === '/api/runs/run-1/events?after_seq=4') {
        return sseResponse([
          event(4, 'source.added', {
            source_id: 'source-1',
            label: '重复来源',
            description: '不应覆盖已恢复来源',
          }),
          event(5, 'message.delta', { delta: assistantMessage.content }),
          event(6, 'message.completed', { message: assistantMessage }),
          event(7, 'run.completed', { status: 'completed' }),
        ]);
      }
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<App />);

    expect(await screen.findByText(assistantMessage.content)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/runs/run-1/events?after_seq=4',
      expect.objectContaining({ credentials: 'include' }),
    );
    expect(screen.queryByText('重复来源')).not.toBeInTheDocument();
  });

  it('keeps a recovered terminal event authoritative when its background refresh is offline', async () => {
    const activeRun = { ...running, last_seq: 1 };
    const activeSnapshot: ThreadSnapshot = {
      ...summary,
      messages: [userMessage],
      runs: [runSnapshot(activeRun, [event(1, 'run.started', { status: 'running' })])],
      active_run: activeRun,
    };
    let listReads = 0;
    let threadReads = 0;
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/threads' && method === 'GET') {
        listReads += 1;
        if (listReads === 1) return jsonResponse({ items: [summary] });
        throw new TypeError('offline background list refresh');
      }
      if (url === '/api/threads/thread-1' && method === 'GET') {
        threadReads += 1;
        if (threadReads === 1) return jsonResponse(activeSnapshot);
        throw new TypeError('offline background snapshot refresh');
      }
      if (url === '/api/runs/run-1/events?after_seq=1') {
        return sseResponse([
          event(2, 'message.delta', { delta: assistantMessage.content }),
          event(3, 'message.completed', { message: assistantMessage }),
          event(4, 'run.completed', { status: 'completed' }),
        ]);
      }
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<App />);

    expect(await screen.findByText(assistantMessage.content)).toBeInTheDocument();
    expect(screen.queryByText('已完成')).not.toBeInTheDocument();
    await waitFor(() => expect(listReads).toBeGreaterThan(1));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('rebuilds every historical run with its own tool, source and terminal state', async () => {
    const cities = [
      ['run-shanghai', 'Asia/Shanghai', '上海时间', '上海现在是 20:00。'],
      ['run-london', 'Europe/London', '伦敦时间', '伦敦现在是 13:00。'],
      ['run-new-york', 'America/New_York', '纽约时间', '纽约现在是 08:00。'],
    ] as const;
    const messages: Message[] = [];
    const runs: RunSnapshot[] = [];

    cities.forEach(([runId, timezone, question, answer], index) => {
      const user: Message = {
        message_id: `user-${index}`,
        role: 'user',
        content: question,
        created_at: timestamp,
        run_id: runId,
      };
      const assistant: Message = {
        message_id: `assistant-${index}`,
        role: 'assistant',
        content: answer,
        created_at: timestamp,
        run_id: runId,
      };
      const run: RunView = {
        ...running,
        run_id: runId,
        status: 'completed',
        last_seq: 6,
        completed_at: timestamp,
      };
      messages.push(user, assistant);
      runs.push(
        runSnapshot(run, [
          event(1, 'run.started', { status: 'running' }, runId),
          event(
            2,
            'tool.started',
            {
              tool_call_id: `tool-${index}`,
              name: 'get_current_time',
              label: '查询指定时区的当前时间',
              input_summary: timezone,
            },
            runId,
          ),
          event(
            3,
            'tool.finished',
            {
              tool_call_id: `tool-${index}`,
              name: 'get_current_time',
              label: '查询指定时区的当前时间',
              output_summary: `已取得 ${timezone} 当前时间`,
            },
            runId,
          ),
          event(
            4,
            'source.added',
            {
              source_id: `source-${index}`,
              label: `系统时钟 · ${timezone}`,
              description: '由只读时间工具返回',
            },
            runId,
          ),
          event(5, 'message.completed', { message: assistant }, runId),
          event(6, 'run.completed', { status: 'completed' }, runId),
        ]),
      );
    });

    const historySnapshot: ThreadSnapshot = {
      ...summary,
      messages,
      runs,
      active_run: null,
    };
    const otherSummary: ThreadSummary = {
      ...summary,
      thread_id: 'thread-2',
      title: '其他对话',
    };
    const otherSnapshot: ThreadSnapshot = {
      ...otherSummary,
      messages: [],
      runs: [],
      active_run: null,
    };
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/threads' && method === 'GET') {
        return jsonResponse({ items: [summary, otherSummary] });
      }
      if (url === '/api/threads/thread-1' && method === 'GET') return jsonResponse(historySnapshot);
      if (url === '/api/threads/thread-2' && method === 'GET') return jsonResponse(otherSnapshot);
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText('纽约现在是 08:00。')).toBeInTheDocument();
    for (const [, timezone, , answer] of cities) {
      expect(screen.getByText(`已取得 ${timezone} 当前时间`)).toBeInTheDocument();
      expect(screen.getByText(`系统时钟 · ${timezone}`)).toBeInTheDocument();
      expect(screen.getByText(answer)).toBeInTheDocument();
    }
    expect(screen.queryByText('已完成')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: '其他对话' }));
    expect(await screen.findByText('今天想处理什么？')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '查询上海当前时间' }));
    expect(await screen.findByText('伦敦现在是 13:00。')).toBeInTheDocument();

    expect(
      fetchMock.mock.calls.filter(([, init]) => (init?.method ?? 'GET') === 'POST'),
    ).toHaveLength(0);
  });

  it('keeps stream_unavailable as connection state instead of a run failure', async () => {
    const activeRun = { ...running, last_seq: 1 };
    const activeSnapshot: ThreadSnapshot = {
      ...summary,
      messages: [userMessage],
      runs: [runSnapshot(activeRun, [event(1, 'run.started', { status: 'running' })])],
      active_run: activeRun,
    };
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/threads' && method === 'GET') return jsonResponse({ items: [summary] });
      if (url === '/api/threads/thread-1' && method === 'GET') return jsonResponse(activeSnapshot);
      if (url === '/api/runs/run-1/events?after_seq=1') return jsonResponse({}, 404);
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<App />);

    expect(await screen.findByText('连接中断')).toBeInTheDocument();
    expect(
      screen.getByText('连接中断，刷新页面后可继续查看。'),
    ).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '重试' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '停止运行' })).toBeInTheDocument();
  });

  it('renders every frozen run failure code with a deterministic message', async () => {
    const cases: Array<[RunFailureCode, string]> = [
      ['run_timeout', '等待时间过长，这次回答没有完成。'],
      ['agent_execution_failed', '这次没有生成完整回答，请重试。'],
      ['service_restarted', '服务刚刚恢复，请重试这条消息。'],
      ['model_step_limit', '这次没有生成完整回答，请重试。'],
      ['tool_call_limit', '这次没有生成完整回答，请重试。'],
      ['repeated_tool_call', '这次没有生成完整回答，请重试。'],
      ['no_progress', '这次没有生成完整回答，请重试。'],
      ['tool_not_allowed', '当前请求暂时无法处理。'],
      ['result_schema_invalid', '这次回答不完整，未予展示。'],
      ['source_validation_failed', '来源未通过校验，这次回答未予展示。'],
    ];
    const messages: Message[] = [];
    const runs: RunSnapshot[] = [];
    cases.forEach(([failureCode], index) => {
      const runId = `failed-${index}`;
      messages.push({
        message_id: `failed-message-${index}`,
        role: 'user',
        content: `failure ${index}`,
        created_at: timestamp,
        run_id: runId,
      });
      runs.push(
        runSnapshot(
          {
            ...running,
            run_id: runId,
            status: 'failed',
            last_seq: 2,
            completed_at: timestamp,
          },
          [
            event(1, 'run.started', { status: 'running' }, runId),
            event(2, 'run.failed', { status: 'failed', error_code: failureCode }, runId),
          ],
        ),
      );
    });
    const failureSnapshot: ThreadSnapshot = {
      ...summary,
      messages,
      runs,
      active_run: null,
    };
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/threads' && method === 'GET') return jsonResponse({ items: [summary] });
      if (url === '/api/threads/thread-1' && method === 'GET') return jsonResponse(failureSnapshot);
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<App />);

    const expectedCounts = new Map<string, number>();
    for (const [, message] of cases) {
      expectedCounts.set(message, (expectedCounts.get(message) ?? 0) + 1);
    }
    for (const [message, count] of expectedCounts) {
      expect(await screen.findAllByText(message)).toHaveLength(count);
    }
    expect(screen.getAllByRole('button', { name: '重试' })).toHaveLength(cases.length);
  });

  it('retries service_restarted as a new run and preserves the failed turn', async () => {
    const failedRun: RunView = {
      ...running,
      status: 'failed',
      last_seq: 3,
      completed_at: timestamp,
    };
    const failedEvents = [
      event(1, 'run.started', { status: 'running' }),
      event(2, 'tool.started', {
        tool_call_id: 'tool-old',
        name: 'get_current_time',
        label: '查询指定时区的当前时间',
        input_summary: 'Asia/Shanghai',
      }),
      event(3, 'run.failed', { status: 'failed', error_code: 'service_restarted' }),
    ];
    const failedSnapshot: ThreadSnapshot = {
      ...summary,
      messages: [userMessage],
      runs: [runSnapshot(failedRun, failedEvents)],
      active_run: null,
    };
    const retryRunView: RunView = {
      ...running,
      run_id: 'run-2',
      last_seq: 0,
    };
    const retryUser: Message = {
      ...userMessage,
      message_id: 'message-user-retry',
      run_id: 'run-2',
    };
    const retryAssistant: Message = {
      ...assistantMessage,
      message_id: 'message-assistant-retry',
      run_id: 'run-2',
      content: '重试后已取得上海当前时间。',
    };
    const retryEvents = [
      event(1, 'run.started', { status: 'running' }, 'run-2'),
      event(
        2,
        'tool.started',
        {
          tool_call_id: 'tool-new',
          name: 'get_current_time',
          label: '查询指定时区的当前时间',
          input_summary: 'Asia/Shanghai',
        },
        'run-2',
      ),
      event(
        3,
        'tool.finished',
        {
          tool_call_id: 'tool-new',
          name: 'get_current_time',
          label: '查询指定时区的当前时间',
          output_summary: '已取得 Asia/Shanghai 当前时间',
        },
        'run-2',
      ),
      event(4, 'message.completed', { message: retryAssistant }, 'run-2'),
      event(5, 'run.completed', { status: 'completed' }, 'run-2'),
    ];
    const retryActiveSnapshot: ThreadSnapshot = {
      ...summary,
      messages: [userMessage, retryUser],
      runs: [runSnapshot(failedRun, failedEvents), runSnapshot(retryRunView, [])],
      active_run: retryRunView,
    };
    const completedRetryRun: RunView = {
      ...retryRunView,
      status: 'completed',
      last_seq: 5,
      completed_at: timestamp,
    };
    const retryCompletedSnapshot: ThreadSnapshot = {
      ...summary,
      messages: [userMessage, retryUser, retryAssistant],
      runs: [runSnapshot(failedRun, failedEvents), runSnapshot(completedRetryRun, retryEvents)],
      active_run: null,
    };
    let threadReads = 0;
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/threads' && method === 'GET') return jsonResponse({ items: [summary] });
      if (url === '/api/threads/thread-1' && method === 'GET') {
        threadReads += 1;
        if (threadReads === 1) return jsonResponse(failedSnapshot);
        if (threadReads === 2) return jsonResponse(retryActiveSnapshot);
        return jsonResponse(retryCompletedSnapshot);
      }
      if (url === '/api/threads/thread-1/runs' && method === 'POST') {
        return jsonResponse(retryRunView);
      }
      if (url === '/api/runs/run-2/events?after_seq=0') return sseResponse(retryEvents);
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText('服务刚刚恢复，请重试这条消息。')).toBeInTheDocument();
    expect(screen.getByText('已停止')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '重试' }));

    expect(await screen.findByText(retryAssistant.content)).toBeInTheDocument();
    expect(screen.getByText('服务刚刚恢复，请重试这条消息。')).toBeInTheDocument();
    const runPosts = fetchMock.mock.calls.filter(
      ([input, init]) =>
        String(input) === '/api/threads/thread-1/runs' && (init?.method ?? 'GET') === 'POST',
    );
    expect(runPosts).toHaveLength(1);
    const requestBody = JSON.parse(String(runPosts[0]?.[1]?.body)) as {
      message: string;
      idempotency_key: string;
    };
    expect(requestBody.message).toBe(userMessage.content);
    expect(requestBody.idempotency_key).toMatch(/^[0-9a-f-]{36}$/i);
  });

  it('cancels the active run and leaves a clear terminal state', async () => {
    const activeSnapshot: ThreadSnapshot = {
      ...summary,
      messages: [userMessage],
      runs: [runSnapshot({ ...running, last_seq: 1 }, [event(1, 'run.started', { status: 'running' })])],
      active_run: { ...running, last_seq: 1 },
    };
    const cancelled: RunView = {
      ...running,
      status: 'cancelled',
      last_seq: 2,
      completed_at: timestamp,
    };
    const cancelledSnapshot: ThreadSnapshot = {
      ...summary,
      messages: [userMessage],
      runs: [
        runSnapshot(cancelled, [
          event(1, 'run.started', { status: 'running' }),
          event(2, 'run.cancelled', { status: 'cancelled' }),
        ]),
      ],
      active_run: null,
    };
    let threadReads = 0;
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/threads' && method === 'GET') return jsonResponse({ items: [summary] });
      if (url === '/api/threads/thread-1' && method === 'GET') {
        threadReads += 1;
        return jsonResponse(threadReads === 1 ? activeSnapshot : cancelledSnapshot);
      }
      if (url === '/api/runs/run-1/events?after_seq=1') {
        return sseResponse([]);
      }
      if (url === '/api/runs/run-1/cancel' && method === 'POST') return jsonResponse(cancelled);
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    const stop = await screen.findByRole('button', { name: '停止运行' });
    await user.click(stop);
    expect(await screen.findByText('已停止回答。')).toBeInTheDocument();
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/runs/run-1/cancel',
        expect.objectContaining({ method: 'POST' }),
      ),
    );
  });

  it('does not abort a newly selected active stream when an old cancel returns late', async () => {
    const threadB: ThreadSummary = {
      ...summary,
      thread_id: 'thread-2',
      title: '活动会话 B',
    };
    const runB: RunView = {
      ...running,
      run_id: 'run-2',
      thread_id: 'thread-2',
      last_seq: 1,
    };
    const messageB: Message = {
      ...userMessage,
      message_id: 'message-b',
      run_id: 'run-2',
      content: '会话 B 的问题',
    };
    const snapshotA: ThreadSnapshot = {
      ...summary,
      messages: [userMessage],
      runs: [runSnapshot({ ...running, last_seq: 1 }, [event(1, 'run.started', { status: 'running' })])],
      active_run: { ...running, last_seq: 1 },
    };
    const snapshotB: ThreadSnapshot = {
      ...threadB,
      messages: [messageB],
      runs: [
        runSnapshot(runB, [
          event(1, 'run.started', { status: 'running' }, 'run-2', 'thread-2'),
        ]),
      ],
      active_run: runB,
    };
    const cancelledA: RunView = {
      ...running,
      status: 'cancelled',
      last_seq: 2,
      completed_at: timestamp,
    };
    let resolveCancel!: (response: Response) => void;
    const cancelResponse = new Promise<Response>((resolve) => {
      resolveCancel = resolve;
    });
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/threads' && method === 'GET') {
        return jsonResponse({ items: [summary, threadB] });
      }
      if (url === '/api/threads/thread-1' && method === 'GET') return jsonResponse(snapshotA);
      if (url === '/api/threads/thread-2' && method === 'GET') return jsonResponse(snapshotB);
      if (url === '/api/runs/run-1/events?after_seq=1') {
        return new Promise<Response>(() => undefined);
      }
      if (url === '/api/runs/run-1/cancel' && method === 'POST') return cancelResponse;
      if (url === '/api/runs/run-2/events?after_seq=1') {
        return new Promise<Response>(() => undefined);
      }
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole('button', { name: '停止运行' }));
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/runs/run-1/cancel',
        expect.objectContaining({ method: 'POST' }),
      ),
    );
    await user.click(screen.getByRole('button', { name: '活动会话 B' }));
    expect(await screen.findByText(messageB.content)).toBeInTheDocument();
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/runs/run-2/events?after_seq=1',
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      ),
    );

    const bRequest = fetchMock.mock.calls.find(
      ([input]) => String(input) === '/api/runs/run-2/events?after_seq=1',
    );
    const bSignal = bRequest?.[1]?.signal;
    expect(bSignal).toBeInstanceOf(AbortSignal);
    const threadListReadsBeforeCancelSettles = fetchMock.mock.calls.filter(
      ([input, init]) => String(input) === '/api/threads' && (init?.method ?? 'GET') === 'GET',
    ).length;
    resolveCancel(jsonResponse(cancelledA));
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.filter(
          ([input, init]) =>
            String(input) === '/api/threads' && (init?.method ?? 'GET') === 'GET',
        ).length,
      ).toBeGreaterThan(threadListReadsBeforeCancelSettles),
    );
    expect(bSignal?.aborted).toBe(false);
  });
});
