import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
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
  it('blocks all interaction when the initial request is unauthenticated', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse({ detail: { code: 'authentication_required' } }, 401),
    );
    vi.stubGlobal('fetch', fetchMock);
    render(<App />);

    expect(
      await screen.findByText('当前请求未通过身份认证，请完成认证后刷新页面。'),
    ).toBeInTheDocument();
    expect(screen.getByLabelText('向 Work Assistant 提问')).toBeDisabled();
    expect(screen.queryByText('从一个真实工具开始')).not.toBeInTheDocument();
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
      await screen.findByText('当前身份无权访问这项会话或运行。'),
    ).toBeInTheDocument();
    expect(screen.queryByText(privateMessage.content)).not.toBeInTheDocument();
    expect(screen.queryByText(privateSnapshot.title)).not.toBeInTheDocument();
    expect(screen.getByLabelText('向 Work Assistant 提问')).toBeDisabled();
    expect(screen.queryByRole('button', { name: '停止运行' })).not.toBeInTheDocument();
  });

  it('does not let a late request restore state after access is blocked', async () => {
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
    const composer = screen.getByLabelText('向 Work Assistant 提问');
    await user.type(composer, 'A 主体尚未发送完成的草稿');
    await user.click(screen.getByRole('button', { name: '发送消息' }));
    await user.click(screen.getByRole('button', { name: otherSummary.title }));

    expect(
      await screen.findByText('当前身份无权访问这项会话或运行。'),
    ).toBeInTheDocument();
    resolveCreateRun(jsonResponse({ detail: { code: 'service_failure' } }, 500));
    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(
        '当前身份无权访问这项会话或运行。',
      );
    });
    expect(composer).toHaveValue('');
    expect(composer).toBeDisabled();
    expect(screen.queryByText(userMessage.content)).not.toBeInTheDocument();
    expect(screen.queryByText('A 主体尚未发送完成的草稿')).not.toBeInTheDocument();
  });

  it('creates a real run and renders deduplicated product events, tools and sources', async () => {
    const emptyList = { items: [] as ThreadSummary[] };
    const emptySnapshot: ThreadSnapshot = { ...summary, messages: [], runs: [], active_run: null };
    const withUser: ThreadSnapshot = {
      ...summary,
      messages: [userMessage],
      runs: [runSnapshot(running, [])],
      active_run: running,
    };
    const completedSnapshot: ThreadSnapshot = {
      ...summary,
      messages: [userMessage, assistantMessage],
      runs: [
        runSnapshot(
          { ...running, status: 'completed', last_seq: 9, completed_at: timestamp },
          [
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
            event(5, 'message.delta', { delta: '上海当前时间' }),
            event(6, 'message.delta', { delta: '为 2026 年 8 月 12 日 20:00。' }),
            event(7, 'source.added', {
              source_id: 'source-1',
              label: '系统时钟 · Asia/Shanghai',
              description: '由只读时间工具返回',
            }),
            event(8, 'message.completed', { message: assistantMessage }),
            event(9, 'run.completed', { status: 'completed' }),
          ],
        ),
      ],
      active_run: null,
    };
    let snapshotReads = 0;
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/threads' && method === 'GET') {
        return jsonResponse(snapshotReads === 0 ? emptyList : { items: [summary] });
      }
      if (url === '/api/threads' && method === 'POST') return jsonResponse(emptySnapshot);
      if (url === '/api/threads/thread-1/runs' && method === 'POST') return jsonResponse(running);
      if (url === '/api/threads/thread-1' && method === 'GET') {
        snapshotReads += 1;
        return jsonResponse(snapshotReads === 1 ? withUser : completedSnapshot);
      }
      if (url === '/api/runs/run-1/events?after_seq=0') {
        return sseResponse([
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
          privateEvent(4),
          event(5, 'message.delta', { delta: '上海当前时间' }),
          event(5, 'message.delta', { delta: '重复内容不应显示' }),
          event(6, 'message.delta', { delta: '为 2026 年 8 月 12 日 20:00。' }),
          event(7, 'source.added', {
            source_id: 'source-1',
            label: '系统时钟 · Asia/Shanghai',
            description: '由只读时间工具返回',
          }),
          event(8, 'message.completed', { message: assistantMessage }),
          event(9, 'run.completed', { status: 'completed' }),
        ]);
      }
      throw new Error(`Unexpected request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText('从一个真实工具开始')).toBeInTheDocument();
    await user.type(
      screen.getByLabelText('向 Work Assistant 提问'),
      '请查询当前上海时间，并说明结果来自哪里。',
    );
    await user.click(screen.getByRole('button', { name: '发送消息' }));

    expect(await screen.findByText(assistantMessage.content)).toBeInTheDocument();
    expect(screen.getByText('查询指定时区的当前时间')).toBeInTheDocument();
    expect(screen.getByText('系统时钟 · Asia/Shanghai')).toBeInTheDocument();
    expect(screen.queryByText('重复内容不应显示')).not.toBeInTheDocument();
    expect(screen.queryByText('不应显示的内部思考')).not.toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/runs/run-1/events?after_seq=0',
      expect.objectContaining({ credentials: 'include' }),
    );
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
    expect(await screen.findByText('已完成')).toBeInTheDocument();
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
    expect(screen.getAllByText('已完成')).toHaveLength(3);

    await user.click(screen.getByRole('button', { name: '其他对话' }));
    expect(await screen.findByText('从一个真实工具开始')).toBeInTheDocument();
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
      screen.getByText('实时连接暂不可用。运行状态未被改为失败；请刷新页面重新连接。'),
    ).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '重新运行' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '停止运行' })).toBeInTheDocument();
  });

  it('renders every frozen run failure code with a deterministic message', async () => {
    const cases: Array<[RunFailureCode, string]> = [
      ['run_timeout', '本次运行超时，未能完成。'],
      ['agent_execution_failed', 'Agent 执行失败，未能完成本次运行。'],
      ['service_restarted', '服务已重启，原运行已安全结束。'],
      ['model_step_limit', 'Agent 已达到本次推理步数上限，运行已安全停止。'],
      ['tool_call_limit', 'Agent 已达到本次工具调用上限，运行已安全停止。'],
      ['repeated_tool_call', '检测到重复工具调用，运行已安全停止。'],
      ['no_progress', 'Agent 连续未取得新进展，运行已安全停止。'],
      ['tool_not_allowed', 'Agent 请求了未获授权的工具，运行已安全停止。'],
      ['result_schema_invalid', 'Agent 返回结果不符合约定，未保存为回答。'],
      ['source_validation_failed', '回答来源校验失败，未保存为回答。'],
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

    for (const [, message] of cases) {
      expect(await screen.findByText(message)).toBeInTheDocument();
    }
    expect(screen.getAllByRole('button', { name: '重新运行' })).toHaveLength(cases.length);
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

    expect(await screen.findByText('服务已重启，原运行已安全结束。')).toBeInTheDocument();
    expect(screen.getByText('已停止')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '重新运行' }));

    expect(await screen.findByText(retryAssistant.content)).toBeInTheDocument();
    expect(screen.getByText('服务已重启，原运行已安全结束。')).toBeInTheDocument();
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
    expect(await screen.findByText('本次运行已停止。')).toBeInTheDocument();
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
