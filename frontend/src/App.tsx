import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from 'react';
import {
  ApiError,
  cancelRun,
  createRun,
  createThread,
  getThread,
  listThreads,
  streamRunEvents,
} from './api';
import {
  isTerminalStatus,
  type Message,
  type ProductEvent,
  type RunProjection,
  type RunStatus,
  type RunView,
  type SourceReference,
  type ThreadSnapshot,
  type ThreadSummary,
  type ToolProgress,
} from './types';

const EMPTY_TITLE = '新对话';

const iconPaths = {
  plus: 'M12 5v14M5 12h14',
  send: 'M12 19V5M6 11l6-6 6 6',
  stop: 'M8 8h8v8H8z',
  time: 'M12 7v5l3 2 M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18',
  source: 'M5 4h14v16H5z M8 8h8M8 12h8M8 16h5',
  chat: 'M4 5h16v12H8l-4 3z',
} as const;

function Icon({ name }: { name: keyof typeof iconPaths }) {
  return (
    <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d={iconPaths[name]} />
    </svg>
  );
}

const getString = (data: Record<string, unknown>, key: string) =>
  typeof data[key] === 'string' ? data[key] : undefined;

const createProjection = (run: RunView): RunProjection => ({
  run,
  assistantText: '',
  tools: [],
  sources: [],
  lastSeq: 0,
  connection: 'connecting',
  cancelling: false,
});

function upsertTool(tools: ToolProgress[], tool: ToolProgress) {
  const index = tools.findIndex((item) => item.tool_call_id === tool.tool_call_id);
  if (index < 0) return [...tools, tool];
  return tools.map((item, itemIndex) => (itemIndex === index ? { ...item, ...tool } : item));
}

function upsertSource(sources: SourceReference[], source: SourceReference) {
  const index = sources.findIndex((item) => item.source_id === source.source_id);
  if (index < 0) return [...sources, source];
  return sources.map((item, itemIndex) => (itemIndex === index ? source : item));
}

function statusFromTerminalEvent(type: ProductEvent['type']): RunStatus | undefined {
  if (type === 'run.completed') return 'completed';
  if (type === 'run.failed') return 'failed';
  if (type === 'run.cancelled') return 'cancelled';
  return undefined;
}

function applyEvent(projection: RunProjection, event: ProductEvent): RunProjection {
  if (
    event.run_id !== projection.run.run_id ||
    event.thread_id !== projection.run.thread_id ||
    event.seq <= projection.lastSeq
  ) {
    return projection;
  }

  const next: RunProjection = {
    ...projection,
    lastSeq: event.seq,
    run: { ...projection.run, last_seq: event.seq },
  };

  if (event.type === 'run.started') {
    return { ...next, run: { ...next.run, status: 'running' } };
  }
  if (event.type === 'message.delta') {
    const delta = getString(event.data, 'delta');
    return delta ? { ...next, assistantText: `${next.assistantText}${delta}` } : next;
  }
  if (event.type === 'tool.started') {
    const toolCallId = getString(event.data, 'tool_call_id');
    const name = getString(event.data, 'name');
    const label = getString(event.data, 'label');
    if (!toolCallId || !name || !label) return next;
    const inputSummary = getString(event.data, 'input_summary');
    return {
      ...next,
      tools: upsertTool(next.tools, {
        tool_call_id: toolCallId,
        name,
        label,
        ...(inputSummary ? { input_summary: inputSummary } : {}),
        status: 'running',
      }),
    };
  }
  if (event.type === 'tool.finished') {
    const toolCallId = getString(event.data, 'tool_call_id');
    const name = getString(event.data, 'name');
    const label = getString(event.data, 'label');
    const outputSummary = getString(event.data, 'output_summary');
    if (!toolCallId || !name || !label || !outputSummary) return next;
    const existing = next.tools.find((tool) => tool.tool_call_id === toolCallId);
    return {
      ...next,
      tools: upsertTool(next.tools, {
        ...existing,
        tool_call_id: toolCallId,
        name,
        label,
        output_summary: outputSummary,
        status: 'completed',
      }),
    };
  }
  if (event.type === 'source.added') {
    const sourceId = getString(event.data, 'source_id');
    const label = getString(event.data, 'label');
    const description = getString(event.data, 'description');
    if (!sourceId || !label || !description) return next;
    return { ...next, sources: upsertSource(next.sources, { source_id: sourceId, label, description }) };
  }
  if (event.type === 'message.completed') {
    const value = event.data.message;
    if (typeof value === 'object' && value !== null && 'content' in value) {
      const content = (value as { content?: unknown }).content;
      if (typeof content === 'string') return { ...next, assistantText: content };
    }
    return next;
  }

  const terminalStatus = statusFromTerminalEvent(event.type);
  if (terminalStatus) {
    return {
      ...next,
      connection: 'closed',
      cancelling: false,
      errorCode: getString(event.data, 'error_code'),
      run: { ...next.run, status: terminalStatus },
    };
  }
  return next;
}

function titleFromMessage(message: string) {
  const collapsed = message.replace(/\s+/g, ' ').trim();
  return collapsed.length > 28 ? `${collapsed.slice(0, 28)}…` : collapsed || EMPTY_TITLE;
}

function optimisticMessage(content: string, idempotencyKey: string): Message {
  return {
    message_id: `pending-${idempotencyKey}`,
    role: 'user',
    content,
    created_at: new Date().toISOString(),
    run_id: null,
  };
}

function mergeOptimisticMessage(snapshot: ThreadSnapshot, pending: Message) {
  const persisted = snapshot.messages.some(
    (message) => message.role === 'user' && message.content === pending.content,
  );
  return persisted ? snapshot : { ...snapshot, messages: [...snapshot.messages, pending] };
}

function displayError(error: unknown) {
  if (error instanceof ApiError) return error.message;
  return '无法连接服务，请确认后端已经启动。';
}

export function App() {
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [snapshot, setSnapshot] = useState<ThreadSnapshot | null>(null);
  const [projection, setProjection] = useState<RunProjection | null>(null);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const selectedThreadRef = useRef<string | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const submittingRef = useRef(false);
  const conversationEndRef = useRef<HTMLDivElement | null>(null);

  const refreshThreads = useCallback(async () => {
    const items = await listThreads();
    setThreads(items);
    return items;
  }, []);

  const refreshSnapshot = useCallback(async (threadId: string) => {
    const fresh = await getThread(threadId);
    if (selectedThreadRef.current === threadId) setSnapshot(fresh);
    return fresh;
  }, []);

  const connectRun = useCallback(
    (run: RunView, initialAfterSeq = 0) => {
      streamAbortRef.current?.abort();
      const controller = new AbortController();
      streamAbortRef.current = controller;
      setProjection(createProjection(run));

      void streamRunEvents(
        run.run_id,
        initialAfterSeq,
        {
          onConnectionChange: (connection) => {
            setProjection((current) =>
              current?.run.run_id === run.run_id ? { ...current, connection } : current,
            );
          },
          onEvent: (event) => {
            setProjection((current) =>
              current?.run.run_id === run.run_id ? applyEvent(current, event) : current,
            );
            if (event.type === 'message.completed') {
              const message = event.data.message;
              if (typeof message === 'object' && message !== null) {
                const completedMessage = message as Message;
                if (
                  typeof completedMessage.message_id === 'string' &&
                  typeof completedMessage.content === 'string' &&
                  completedMessage.role === 'assistant'
                ) {
                  setSnapshot((current) => {
                    if (!current || current.thread_id !== event.thread_id) return current;
                    const messages = current.messages.some(
                      (item) => item.message_id === completedMessage.message_id,
                    )
                      ? current.messages.map((item) =>
                          item.message_id === completedMessage.message_id ? completedMessage : item,
                        )
                      : [...current.messages, completedMessage];
                    return { ...current, messages };
                  });
                }
              }
            }
          },
        },
        controller.signal,
      )
        .then(async () => {
          if (controller.signal.aborted) return;
          await Promise.all([refreshSnapshot(run.thread_id), refreshThreads()]);
        })
        .catch((streamError: unknown) => {
          if (controller.signal.aborted) return;
          setError(displayError(streamError));
          setProjection((current) =>
            current?.run.run_id === run.run_id
              ? { ...current, connection: 'closed', errorCode: 'stream_unavailable' }
              : current,
          );
        });
    },
    [refreshSnapshot, refreshThreads],
  );

  const openThread = useCallback(
    async (threadId: string) => {
      streamAbortRef.current?.abort();
      selectedThreadRef.current = threadId;
      setLoading(true);
      setError(null);
      setProjection(null);
      try {
        const fresh = await getThread(threadId);
        if (selectedThreadRef.current !== threadId) return;
        setSnapshot(fresh);
        if (fresh.active_run && !isTerminalStatus(fresh.active_run.status)) {
          connectRun(fresh.active_run, 0);
        }
      } catch (openError) {
        if (selectedThreadRef.current === threadId) setError(displayError(openError));
      } finally {
        if (selectedThreadRef.current === threadId) setLoading(false);
      }
    },
    [connectRun],
  );

  useEffect(() => {
    let disposed = false;
    void refreshThreads()
      .then((items) => {
        if (disposed) return;
        if (items[0]) return openThread(items[0].thread_id);
        setLoading(false);
      })
      .catch((initialError: unknown) => {
        if (!disposed) {
          setError(displayError(initialError));
          setLoading(false);
        }
      });
    return () => {
      disposed = true;
      streamAbortRef.current?.abort();
    };
  }, [openThread, refreshThreads]);

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ block: 'end', behavior: 'smooth' });
  }, [projection?.assistantText, projection?.tools.length, snapshot?.messages.length]);

  const startNewThread = useCallback(async () => {
    if (submittingRef.current) return;
    streamAbortRef.current?.abort();
    setError(null);
    setLoading(true);
    try {
      const created = await createThread();
      selectedThreadRef.current = created.thread_id;
      setSnapshot(created);
      setProjection(null);
      await refreshThreads();
    } catch (newThreadError) {
      setError(displayError(newThreadError));
    } finally {
      setLoading(false);
    }
  }, [refreshThreads]);

  const submitMessage = useCallback(async () => {
    const message = input.trim();
    if (!message || submittingRef.current || (projection && !isTerminalStatus(projection.run.status))) {
      return;
    }
    submittingRef.current = true;
    setError(null);
    setInput('');
    const idempotencyKey = crypto.randomUUID();
    const pending = optimisticMessage(message, idempotencyKey);
    let target = snapshot;

    try {
      if (!target) {
        target = await createThread(titleFromMessage(message));
        selectedThreadRef.current = target.thread_id;
        setSnapshot(target);
      }
      const threadId = target.thread_id;
      setSnapshot((current) =>
        current?.thread_id === threadId
          ? { ...current, messages: [...current.messages, pending] }
          : current,
      );
      const run = await createRun(threadId, message, idempotencyKey);
      try {
        const fresh = await getThread(threadId);
        if (selectedThreadRef.current === threadId) {
          setSnapshot(mergeOptimisticMessage(fresh, pending));
        }
      } catch {
        // The optimistic user message remains visible while the event stream proceeds.
      }
      await refreshThreads();
      connectRun(run, 0);
    } catch (submitError) {
      setInput(message);
      setSnapshot((current) =>
        current
          ? { ...current, messages: current.messages.filter((item) => item.message_id !== pending.message_id) }
          : current,
      );
      setError(displayError(submitError));
    } finally {
      submittingRef.current = false;
    }
  }, [connectRun, input, projection, refreshThreads, snapshot]);

  const stopRun = useCallback(async () => {
    if (!projection || isTerminalStatus(projection.run.status) || projection.cancelling) return;
    const runId = projection.run.run_id;
    setProjection((current) => (current ? { ...current, cancelling: true } : current));
    setError(null);
    try {
      const run = await cancelRun(runId);
      setProjection((current) =>
        current?.run.run_id === runId
          ? {
              ...current,
              run,
              cancelling: run.status !== 'cancelled',
              connection: run.status === 'cancelled' ? 'closed' : current.connection,
            }
          : current,
      );
      if (run.status === 'cancelled') {
        streamAbortRef.current?.abort();
        await Promise.all([refreshSnapshot(run.thread_id), refreshThreads()]);
      }
    } catch (cancelError) {
      setProjection((current) => (current ? { ...current, cancelling: false } : current));
      setError(displayError(cancelError));
    }
  }, [projection, refreshSnapshot, refreshThreads]);

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    void submitMessage();
  };

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void submitMessage();
    }
  };

  const active = projection !== null && !isTerminalStatus(projection.run.status);
  const displayedMessages = useMemo(() => snapshot?.messages ?? [], [snapshot]);

  return (
    <div className="app-shell">
      <Sidebar
        threads={threads}
        activeThreadId={snapshot?.thread_id ?? null}
        onOpenThread={(threadId) => void openThread(threadId)}
        onNewThread={() => void startNewThread()}
      />

      <main className="chat-main">
        <header className="topbar">
          <div>
            <strong>{snapshot?.title || 'Work Assistant'}</strong>
            <span>对话、工具与来源</span>
          </div>
          <div className="mobile-actions">
            <label className="sr-only" htmlFor="mobile-thread-select">
              选择历史会话
            </label>
            <select
              id="mobile-thread-select"
              value={snapshot?.thread_id ?? ''}
              onChange={(event) => event.target.value && void openThread(event.target.value)}
            >
              <option value="">历史会话</option>
              {threads.map((thread) => (
                <option key={thread.thread_id} value={thread.thread_id}>
                  {thread.title}
                </option>
              ))}
            </select>
            <button className="mobile-new" type="button" onClick={() => void startNewThread()}>
              <Icon name="plus" />
              <span className="sr-only">新建对话</span>
            </button>
          </div>
        </header>

        <section className="conversation" aria-live="polite" aria-busy={active}>
          {loading && !snapshot ? <LoadingState /> : null}
          {!loading && displayedMessages.length === 0 && !projection ? <EmptyState /> : null}

          {displayedMessages.map((message) => {
            const isProjectedAssistant =
              message.role === 'assistant' && message.run_id === projection?.run.run_id;
            if (isProjectedAssistant && projection) {
              return <AssistantTurn key={message.message_id} projection={projection} />;
            }
            return <MessageTurn key={message.message_id} message={message} />;
          })}

          {projection &&
          !displayedMessages.some(
            (message) => message.role === 'assistant' && message.run_id === projection.run.run_id,
          ) ? (
            <AssistantTurn projection={projection} />
          ) : null}

          {error ? (
            <div className="page-error" role="alert">
              {error}
            </div>
          ) : null}
          <div ref={conversationEndRef} />
        </section>

        <Composer
          value={input}
          active={active}
          cancelling={projection?.cancelling ?? false}
          disabled={loading}
          onChange={setInput}
          onKeyDown={handleComposerKeyDown}
          onSubmit={handleSubmit}
          onStop={() => void stopRun()}
        />
      </main>
    </div>
  );
}

type SidebarProps = {
  threads: ThreadSummary[];
  activeThreadId: string | null;
  onOpenThread: (threadId: string) => void;
  onNewThread: () => void;
};

function Sidebar({ threads, activeThreadId, onOpenThread, onNewThread }: SidebarProps) {
  return (
    <aside className="sidebar" aria-label="对话导航">
      <div className="brand">
        <span className="brand-mark">W</span>
        <div>
          <strong>Work Assistant</strong>
          <span>Agent 对话核心</span>
        </div>
      </div>
      <button className="new-thread" type="button" onClick={onNewThread}>
        <Icon name="plus" />
        新建对话
      </button>
      <div className="history">
        <p>最近对话</p>
        {threads.length === 0 ? (
          <span className="history-empty">还没有对话</span>
        ) : (
          threads.map((thread) => (
            <button
              className={thread.thread_id === activeThreadId ? 'history-item active' : 'history-item'}
              type="button"
              key={thread.thread_id}
              onClick={() => onOpenThread(thread.thread_id)}
              aria-current={thread.thread_id === activeThreadId ? 'page' : undefined}
            >
              <Icon name="chat" />
              <span>{thread.title || EMPTY_TITLE}</span>
            </button>
          ))
        )}
      </div>
    </aside>
  );
}

function LoadingState() {
  return (
    <div className="empty-state" role="status">
      <span className="loading-dot" />
      正在载入会话…
    </div>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <span className="empty-icon">
        <Icon name="time" />
      </span>
      <h1>从一个真实工具开始</h1>
      <p>例如：请查询当前上海时间，并说明结果来自哪里。</p>
    </div>
  );
}

function MessageTurn({ message }: { message: Message }) {
  if (message.role === 'user') {
    return (
      <article className="user-turn">
        <div className="user-bubble">{message.content}</div>
      </article>
    );
  }
  return (
    <article className="assistant-turn">
      <span className="assistant-avatar">W</span>
      <div className="assistant-content">
        <strong className="assistant-name">Work Assistant</strong>
        <div className="answer-text">{message.content}</div>
      </div>
    </article>
  );
}

function AssistantTurn({ projection }: { projection: RunProjection }) {
  const { run, tools, sources, assistantText, connection, cancelling, errorCode } = projection;
  const isWorking = !isTerminalStatus(run.status);
  return (
    <article className="assistant-turn">
      <span className="assistant-avatar">W</span>
      <div className="assistant-content">
        <div className="assistant-heading">
          <strong className="assistant-name">Work Assistant</strong>
          <RunStatusLabel
            status={run.status}
            connection={connection}
            cancelling={cancelling}
          />
        </div>
        {tools.length > 0 ? <ToolRun tools={tools} working={isWorking} /> : null}
        {assistantText ? (
          <div className="answer-text">{assistantText}</div>
        ) : isWorking ? (
          <div className="thinking" role="status">
            <span />
            <span />
            <span />
            正在处理
          </div>
        ) : null}
        {run.status === 'failed' || errorCode === 'stream_unavailable' ? (
          <div className="run-notice error" role="alert">
            {friendlyRunError(errorCode)}
          </div>
        ) : null}
        {run.status === 'cancelled' ? <div className="run-notice">本次运行已停止。</div> : null}
        {sources.length > 0 ? (
          <div className="sources" aria-label="回答来源">
            {sources.map((source) => (
              <span className="source-chip" key={source.source_id} title={source.description}>
                <Icon name="source" />
                {source.label}
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </article>
  );
}

function friendlyRunError(errorCode?: string) {
  if (errorCode === 'permission_denied') return '当前身份没有完成此操作的权限。';
  if (errorCode === 'timeout') return '任务等待时间过长，请稍后重试。';
  if (errorCode === 'stream_unavailable') return '实时连接已断开，请刷新页面恢复会话。';
  return '本次运行没有完成，请重试。';
}

function RunStatusLabel({
  status,
  connection,
  cancelling,
}: {
  status: RunStatus;
  connection: RunProjection['connection'];
  cancelling: boolean;
}) {
  let label = '准备中';
  if (cancelling) label = '正在停止';
  else if (connection === 'reconnecting') label = '正在重新连接';
  else if (status === 'running') label = '运行中';
  else if (status === 'completed') label = '已完成';
  else if (status === 'failed') label = '失败';
  else if (status === 'cancelled') label = '已停止';
  return <span className={`run-status ${status}`}>{label}</span>;
}

function ToolRun({ tools, working }: { tools: ToolProgress[]; working: boolean }) {
  const completed = tools.filter((tool) => tool.status === 'completed').length;
  return (
    <details className="tool-run" open={working}>
      <summary>
        <span className={working ? 'status-orb pulsing' : 'status-orb'} />
        <strong>{working ? '正在调用工具' : `已完成 ${completed} 个工具步骤`}</strong>
        <span>查看详情</span>
      </summary>
      <div className="tool-list">
        {tools.map((tool) => (
          <div className="tool-step" key={tool.tool_call_id}>
            <span className={tool.status === 'completed' ? 'tool-check completed' : 'tool-check'}>
              {tool.status === 'completed' ? '✓' : '·'}
            </span>
            <div>
              <strong>{tool.label}</strong>
              {tool.output_summary || tool.input_summary ? (
                <span>{tool.output_summary || tool.input_summary}</span>
              ) : null}
            </div>
            <span>{tool.status === 'completed' ? '完成' : '运行中'}</span>
          </div>
        ))}
      </div>
    </details>
  );
}

type ComposerProps = {
  value: string;
  active: boolean;
  cancelling: boolean;
  disabled: boolean;
  onChange: (value: string) => void;
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  onSubmit: (event: FormEvent) => void;
  onStop: () => void;
};

function Composer({
  value,
  active,
  cancelling,
  disabled,
  onChange,
  onKeyDown,
  onSubmit,
  onStop,
}: ComposerProps) {
  return (
    <div className="composer-dock">
      <form className="composer" onSubmit={onSubmit}>
        <label className="sr-only" htmlFor="message-input">
          向 Work Assistant 提问
        </label>
        <textarea
          id="message-input"
          value={value}
          placeholder="向 Work Assistant 提问…"
          rows={2}
          disabled={disabled || active}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={onKeyDown}
        />
        <div className="composer-footer">
          <span>Enter 发送，Shift + Enter 换行</span>
          {active ? (
            <button
              className="stop-button"
              type="button"
              onClick={onStop}
              disabled={cancelling}
              aria-label={cancelling ? '正在停止' : '停止运行'}
            >
              <Icon name="stop" />
            </button>
          ) : (
            <button
              className="send-button"
              type="submit"
              disabled={disabled || value.trim().length === 0}
              aria-label="发送消息"
            >
              <Icon name="send" />
            </button>
          )}
        </div>
      </form>
      <p>工具结果和重要信息应通过来源继续核对。</p>
    </div>
  );
}
