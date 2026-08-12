import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { App } from './App';
import type { ProductEvent, RunView, ThreadSnapshot, ThreadSummary } from './types';

const timestamp = '2026-08-12T12:00:00Z';

function jsonResponse(value: unknown, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function event(seq: number, type: ProductEvent['type'], data: Record<string, unknown> = {}) {
  return {
    event_id: `event-${seq}`,
    run_id: 'run-1',
    thread_id: 'thread-1',
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

describe('employee chat vertical slice', () => {
  it('creates a real run and renders deduplicated product events, tools and sources', async () => {
    const emptyList = { items: [] as ThreadSummary[] };
    const emptySnapshot: ThreadSnapshot = { ...summary, messages: [], active_run: null };
    const withUser: ThreadSnapshot = { ...summary, messages: [userMessage], active_run: running };
    const completedSnapshot: ThreadSnapshot = {
      ...summary,
      messages: [userMessage, assistantMessage],
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
      expect.objectContaining({ credentials: 'omit' }),
    );
  });

  it('restores an active run from the thread snapshot and replays from seq zero', async () => {
    const activeSnapshot: ThreadSnapshot = {
      ...summary,
      messages: [userMessage],
      active_run: { ...running, last_seq: 4 },
    };
    const completedSnapshot: ThreadSnapshot = {
      ...summary,
      messages: [userMessage, assistantMessage],
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
      if (url === '/api/runs/run-1/events?after_seq=0') {
        return sseResponse([
          event(1, 'run.started', { status: 'running' }),
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
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/runs/run-1/events?after_seq=0',
      expect.objectContaining({ credentials: 'omit' }),
    );
  });

  it('cancels the active run and leaves a clear terminal state', async () => {
    const activeSnapshot: ThreadSnapshot = {
      ...summary,
      messages: [userMessage],
      active_run: running,
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
      if (url === '/api/runs/run-1/events?after_seq=0') {
        return sseResponse([event(1, 'run.started', { status: 'running' })]);
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
});
