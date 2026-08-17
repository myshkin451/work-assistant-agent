import {
  Fragment,
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
  isRunFailureCode,
  isTerminalStatus,
  type Message,
  type ProductEvent,
  type RunProjection,
  type RunSnapshot,
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
  connection: isTerminalStatus(run.status) ? 'closed' : 'connecting',
  cancelling: false,
});

type ProjectionMap = Record<string, RunProjection>;

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
    const errorCode = event.data.error_code;
    return {
      ...next,
      connection: 'closed',
      cancelling: false,
      ...(isRunFailureCode(errorCode) ? { failureCode: errorCode } : {}),
      run: { ...next.run, status: terminalStatus },
    };
  }
  return next;
}

function projectionFromSnapshot(run: RunSnapshot): RunProjection {
  const { events, ...view } = run;
  const projected = events.reduce(applyEvent, createProjection(view));
  return {
    ...projected,
    run: { ...projected.run, ...view },
    lastSeq: run.last_seq,
    connection: isTerminalStatus(run.status) ? 'closed' : 'connecting',
  };
}

function projectionsFromSnapshot(snapshot: ThreadSnapshot): ProjectionMap {
  const projections = Object.fromEntries(
    snapshot.runs.map((run) => [run.run_id, projectionFromSnapshot(run)]),
  ) as ProjectionMap;

  if (snapshot.active_run && !projections[snapshot.active_run.run_id]) {
    projections[snapshot.active_run.run_id] = createProjection(snapshot.active_run);
  }

  for (const message of snapshot.messages) {
    if (message.role !== 'assistant' || !message.run_id) continue;
    const projection = projections[message.run_id];
    if (projection && !projection.assistantText) {
      projections[message.run_id] = { ...projection, assistantText: message.content };
    }
  }
  return projections;
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
    (message) =>
      message.role === 'user' &&
      (pending.run_id ? message.run_id === pending.run_id : message.content === pending.content),
  );
  return persisted ? snapshot : { ...snapshot, messages: [...snapshot.messages, pending] };
}

function displayError(error: unknown) {
  if (error instanceof ApiError) return error.message;
  return '无法连接服务，请确认后端已经启动。';
}

function isAccessError(error: unknown): error is ApiError {
  return error instanceof ApiError && (error.status === 401 || error.status === 403);
}

export function App() {
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [snapshot, setSnapshot] = useState<ThreadSnapshot | null>(null);
  const [projectionsByRunId, setProjectionsByRunId] = useState<ProjectionMap>({});
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [accessBlocked, setAccessBlocked] = useState(false);
  const selectedThreadRef = useRef<string | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const submittingRef = useRef(false);
  const accessBlockedRef = useRef(false);
  const conversationEndRef = useRef<HTMLDivElement | null>(null);

  const blockAccess = useCallback((accessError: unknown) => {
    if (accessBlockedRef.current) return true;
    if (!isAccessError(accessError)) return false;
    accessBlockedRef.current = true;
    submittingRef.current = false;
    streamAbortRef.current?.abort();
    selectedThreadRef.current = null;
    setThreads([]);
    setSnapshot(null);
    setProjectionsByRunId({});
    setInput('');
    setSubmitting(false);
    setLoading(false);
    setAccessBlocked(true);
    setError(displayError(accessError));
    return true;
  }, []);

  const refreshThreads = useCallback(async () => {
    if (accessBlockedRef.current) return [];
    const items = await listThreads();
    if (accessBlockedRef.current) return [];
    setThreads(items);
    return items;
  }, []);

  const refreshSnapshot = useCallback(async (threadId: string) => {
    if (accessBlockedRef.current) return null;
    const fresh = await getThread(threadId);
    if (!accessBlockedRef.current && selectedThreadRef.current === threadId) {
      setSnapshot(fresh);
      setProjectionsByRunId(projectionsFromSnapshot(fresh));
    }
    return fresh;
  }, []);

  const connectRun = useCallback(
    (run: RunView, initialProjection: RunProjection = createProjection(run)) => {
      if (accessBlockedRef.current || isTerminalStatus(run.status)) return;
      streamAbortRef.current?.abort();
      const controller = new AbortController();
      streamAbortRef.current = controller;
      const startingProjection = {
        ...initialProjection,
        run: { ...run, ...initialProjection.run },
        connection: 'connecting' as const,
      };
      setProjectionsByRunId((current) => ({
        ...current,
        [run.run_id]: startingProjection,
      }));

      const isCurrentStream = () =>
        !controller.signal.aborted &&
        streamAbortRef.current === controller &&
        selectedThreadRef.current === run.thread_id;

      void (async () => {
        try {
          await streamRunEvents(
            run.run_id,
            startingProjection.lastSeq,
            {
              onConnectionChange: (connection) => {
                if (!isCurrentStream()) return;
                setProjectionsByRunId((current) => {
                  const projection = current[run.run_id];
                  return projection
                    ? { ...current, [run.run_id]: { ...projection, connection } }
                    : current;
                });
              },
              onEvent: (event) => {
                if (!isCurrentStream()) return;
                setProjectionsByRunId((current) => {
                  const projection = current[run.run_id];
                  return projection
                    ? { ...current, [run.run_id]: applyEvent(projection, event) }
                    : current;
                });
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
                              item.message_id === completedMessage.message_id
                                ? completedMessage
                                : item,
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
          );
        } catch (streamError) {
          if (!isCurrentStream()) return;
          if (blockAccess(streamError)) return;
          setProjectionsByRunId((current) => {
            const projection = current[run.run_id];
            return projection
              ? { ...current, [run.run_id]: { ...projection, connection: 'unavailable' } }
              : current;
          });
          return;
        }

        if (!isCurrentStream()) return;
        try {
          await Promise.all([refreshSnapshot(run.thread_id), refreshThreads()]);
          if (isCurrentStream()) setError(null);
        } catch (refreshError) {
          if (blockAccess(refreshError)) return;
          // A persisted terminal event is already authoritative. A background
          // snapshot/list refresh may fail during the same short outage and
          // must not leave a stale global service alert after recovery.
        }
      })();
    },
    [blockAccess, refreshSnapshot, refreshThreads],
  );

  const openThread = useCallback(
    async (threadId: string) => {
      if (accessBlockedRef.current) return;
      streamAbortRef.current?.abort();
      selectedThreadRef.current = threadId;
      setLoading(true);
      setError(null);
      setSnapshot(null);
      setProjectionsByRunId({});
      try {
        const fresh = await getThread(threadId);
        if (accessBlockedRef.current) return;
        if (selectedThreadRef.current !== threadId) return;
        const projections = projectionsFromSnapshot(fresh);
        setSnapshot(fresh);
        setProjectionsByRunId(projections);
        if (fresh.active_run && !isTerminalStatus(fresh.active_run.status)) {
          connectRun(
            fresh.active_run,
            projections[fresh.active_run.run_id] ?? createProjection(fresh.active_run),
          );
        }
      } catch (openError) {
        if (!blockAccess(openError) && selectedThreadRef.current === threadId) {
          setError(displayError(openError));
        }
      } finally {
        if (selectedThreadRef.current === threadId) setLoading(false);
      }
    },
    [blockAccess, connectRun],
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
          if (!blockAccess(initialError)) {
            setError(displayError(initialError));
            setLoading(false);
          }
        }
      });
    return () => {
      disposed = true;
      streamAbortRef.current?.abort();
    };
  }, [blockAccess, openThread, refreshThreads]);

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ block: 'end', behavior: 'smooth' });
  }, [
    projectionsByRunId,
    snapshot?.active_run?.run_id,
    snapshot?.messages.length,
    snapshot?.thread_id,
  ]);

  const startNewThread = useCallback(async () => {
    if (submittingRef.current || accessBlockedRef.current) return;
    streamAbortRef.current?.abort();
    setError(null);
    setLoading(true);
    try {
      const created = await createThread();
      if (accessBlockedRef.current) return;
      selectedThreadRef.current = created.thread_id;
      setSnapshot(created);
      setProjectionsByRunId(projectionsFromSnapshot(created));
      await refreshThreads();
    } catch (newThreadError) {
      if (!blockAccess(newThreadError)) setError(displayError(newThreadError));
    } finally {
      setLoading(false);
    }
  }, [blockAccess, refreshThreads]);

  const activeProjection = useMemo(() => {
    const activeRunId = snapshot?.active_run?.run_id;
    if (activeRunId) return projectionsByRunId[activeRunId] ?? null;
    return (
      Object.values(projectionsByRunId).find(
        (projection) => !isTerminalStatus(projection.run.status),
      ) ?? null
    );
  }, [projectionsByRunId, snapshot?.active_run?.run_id]);

  const startMessage = useCallback(async (
    rawMessage: string,
    restoreComposerOnFailure: boolean,
  ) => {
    const message = rawMessage.trim();
    if (
      !message ||
      accessBlockedRef.current ||
      submittingRef.current ||
      (activeProjection && !isTerminalStatus(activeProjection.run.status))
    ) {
      return;
    }
    submittingRef.current = true;
    setSubmitting(true);
    setError(null);
    const idempotencyKey = crypto.randomUUID();
    const pending = optimisticMessage(message, idempotencyKey);
    let target = snapshot;

    try {
      if (!target) {
        target = await createThread(titleFromMessage(message));
        if (accessBlockedRef.current) return;
        selectedThreadRef.current = target.thread_id;
        setSnapshot(target);
        setProjectionsByRunId(projectionsFromSnapshot(target));
      }
      const threadId = target.thread_id;
      setSnapshot((current) =>
        current?.thread_id === threadId
          ? { ...current, messages: [...current.messages, pending] }
          : current,
      );
      const run = await createRun(threadId, message, idempotencyKey);
      if (accessBlockedRef.current) return;
      const persistedPending = { ...pending, run_id: run.run_id };
      const initialProjection = createProjection(run);
      if (selectedThreadRef.current === threadId) {
        setSnapshot((current) => {
          if (!current || current.thread_id !== threadId) return current;
          return {
            ...current,
            messages: current.messages.map((item) =>
              item.message_id === pending.message_id ? persistedPending : item,
            ),
            runs: current.runs.some((item) => item.run_id === run.run_id)
              ? current.runs
              : [...current.runs, { ...run, events: [] }],
            active_run: run,
          };
        });
        setProjectionsByRunId((current) => ({
          ...current,
          [run.run_id]: initialProjection,
        }));
      }

      let projectionToConnect = initialProjection;
      let shouldConnect = !isTerminalStatus(run.status);
      try {
        const fresh = await getThread(threadId);
        if (selectedThreadRef.current === threadId) {
          const merged = mergeOptimisticMessage(fresh, persistedPending);
          const projections = projectionsFromSnapshot(merged);
          setSnapshot(merged);
          setProjectionsByRunId(projections);
          projectionToConnect = projections[run.run_id] ?? initialProjection;
          shouldConnect =
            fresh.active_run?.run_id === run.run_id &&
            !isTerminalStatus(projectionToConnect.run.status);
        }
      } catch (snapshotError) {
        if (blockAccess(snapshotError)) return;
        // The optimistic user message remains visible while the event stream proceeds.
      }
      if (selectedThreadRef.current === threadId && shouldConnect) {
        connectRun(run, projectionToConnect);
      }
      void refreshThreads().catch((refreshError: unknown) => {
        if (blockAccess(refreshError)) return;
        if (selectedThreadRef.current === threadId) setError(displayError(refreshError));
      });
    } catch (submitError) {
      if (blockAccess(submitError)) return;
      if (restoreComposerOnFailure) setInput(message);
      setSnapshot((current) =>
        current
          ? { ...current, messages: current.messages.filter((item) => item.message_id !== pending.message_id) }
          : current,
      );
      setError(displayError(submitError));
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
    }
  }, [activeProjection, blockAccess, connectRun, refreshThreads, snapshot]);

  const submitMessage = useCallback(async () => {
    const message = input.trim();
    if (!message) return;
    setInput('');
    await startMessage(message, true);
  }, [input, startMessage]);

  const retryRun = useCallback(
    async (runId: string) => {
      const original = snapshot?.messages.find(
        (message) => message.role === 'user' && message.run_id === runId,
      );
      if (!original) {
        setError('无法找到这次运行的原始问题。');
        return;
      }
      await startMessage(original.content, false);
    },
    [snapshot?.messages, startMessage],
  );

  const stopRun = useCallback(async () => {
    if (
      accessBlockedRef.current ||
      !activeProjection ||
      isTerminalStatus(activeProjection.run.status) ||
      activeProjection.cancelling
    ) {
      return;
    }
    const runId = activeProjection.run.run_id;
    const targetThreadId = activeProjection.run.thread_id;
    const targetController = streamAbortRef.current;
    setProjectionsByRunId((current) => {
      const projection = current[runId];
      return projection
        ? { ...current, [runId]: { ...projection, cancelling: true } }
        : current;
    });
    setError(null);
    try {
      const run = await cancelRun(runId);
      if (accessBlockedRef.current) return;
      setProjectionsByRunId((current) => {
        const projection = current[runId];
        return projection
          ? {
              ...current,
              [runId]: {
                ...projection,
                run,
                cancelling: run.status !== 'cancelled',
                connection: run.status === 'cancelled' ? 'closed' : projection.connection,
              },
            }
          : current;
      });
      if (run.status === 'cancelled') {
        if (
          selectedThreadRef.current === run.thread_id &&
          streamAbortRef.current === targetController
        ) {
          targetController?.abort();
          await Promise.all([refreshSnapshot(run.thread_id), refreshThreads()]);
        } else {
          await refreshThreads();
        }
      }
    } catch (cancelError) {
      if (blockAccess(cancelError)) return;
      setProjectionsByRunId((current) => {
        const projection = current[runId];
        return projection
          ? { ...current, [runId]: { ...projection, cancelling: false } }
          : current;
      });
      if (selectedThreadRef.current === targetThreadId) {
        setError(displayError(cancelError));
      }
    }
  }, [activeProjection, blockAccess, refreshSnapshot, refreshThreads]);

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

  const active = activeProjection !== null && !isTerminalStatus(activeProjection.run.status);
  const displayedMessages = useMemo(() => snapshot?.messages ?? [], [snapshot]);
  const displayedRunIds = useMemo(
    () =>
      new Set(
        displayedMessages
          .filter((message) => message.role === 'user' && message.run_id)
          .map((message) => message.run_id as string),
      ),
    [displayedMessages],
  );

  return (
    <div className="app-shell">
      <Sidebar
        threads={threads}
        activeThreadId={snapshot?.thread_id ?? null}
        disabled={accessBlocked}
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
              disabled={accessBlocked}
            >
              <option value="">历史会话</option>
              {threads.map((thread) => (
                <option key={thread.thread_id} value={thread.thread_id}>
                  {thread.title}
                </option>
              ))}
            </select>
            <button
              className="mobile-new"
              type="button"
              onClick={() => void startNewThread()}
              disabled={accessBlocked}
            >
              <Icon name="plus" />
              <span className="sr-only">新建对话</span>
            </button>
          </div>
        </header>

        <section className="conversation" aria-live="polite" aria-busy={active}>
          {loading && !snapshot ? <LoadingState /> : null}
          {!loading &&
          !error &&
          displayedMessages.length === 0 &&
          Object.keys(projectionsByRunId).length === 0 ? (
            <EmptyState />
          ) : null}

          {displayedMessages.map((message) => {
            if (message.role === 'assistant' && message.run_id && projectionsByRunId[message.run_id]) {
              return null;
            }
            if (message.role === 'user' && message.run_id) {
              const projection = projectionsByRunId[message.run_id];
              return (
                <Fragment key={message.message_id}>
                  <MessageTurn message={message} />
                  {projection ? (
                    <AssistantTurn
                      projection={projection}
                      retryDisabled={active || submitting || accessBlocked}
                      onRetry={() => void retryRun(projection.run.run_id)}
                    />
                  ) : null}
                </Fragment>
              );
            }
            return <MessageTurn key={message.message_id} message={message} />;
          })}

          {Object.values(projectionsByRunId)
            .filter((projection) => !displayedRunIds.has(projection.run.run_id))
            .map((projection) => (
              <AssistantTurn
                key={projection.run.run_id}
                projection={projection}
                retryDisabled={active || submitting || accessBlocked}
                onRetry={() => void retryRun(projection.run.run_id)}
              />
            ))}

          {error ? (
            <div className="page-error" role="alert">
              {error}
            </div>
          ) : null}
          <div className="conversation-end" ref={conversationEndRef} />
        </section>

        <Composer
          value={input}
          active={active}
          cancelling={activeProjection?.cancelling ?? false}
          disabled={loading || submitting || accessBlocked}
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
  disabled: boolean;
};

function Sidebar({ threads, activeThreadId, onOpenThread, onNewThread, disabled }: SidebarProps) {
  return (
    <aside className="sidebar" aria-label="对话导航">
      <div className="brand">
        <span className="brand-mark">W</span>
        <div>
          <strong>Work Assistant</strong>
          <span>Agent 对话核心</span>
        </div>
      </div>
      <button className="new-thread" type="button" onClick={onNewThread} disabled={disabled}>
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
              disabled={disabled}
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

function AssistantTurn({
  projection,
  retryDisabled,
  onRetry,
}: {
  projection: RunProjection;
  retryDisabled: boolean;
  onRetry: () => void;
}) {
  const { run, tools, sources, assistantText, connection, cancelling, failureCode } = projection;
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
        {tools.length > 0 ? (
          <ToolRun tools={tools} working={isWorking} runStatus={run.status} />
        ) : null}
        {assistantText ? (
          <div className="answer-text">{assistantText}</div>
        ) : isWorking && connection !== 'unavailable' ? (
          <div className="thinking" role="status">
            <span />
            <span />
            <span />
            正在处理
          </div>
        ) : null}
        {run.status === 'failed' ? (
          <div className="run-notice error run-notice-row" role="alert">
            <span>{friendlyRunError(failureCode)}</span>
            <button className="retry-button" type="button" onClick={onRetry} disabled={retryDisabled}>
              重新运行
            </button>
          </div>
        ) : null}
        {isWorking && connection === 'unavailable' ? (
          <div className="run-notice connection" role="status">
            实时连接暂不可用。运行状态未被改为失败；请刷新页面重新连接。
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

function friendlyRunError(failureCode: RunProjection['failureCode']) {
  if (failureCode === 'run_timeout') return '本次运行超时，未能完成。';
  if (failureCode === 'agent_execution_failed') return 'Agent 执行失败，未能完成本次运行。';
  if (failureCode === 'service_restarted') return '服务已重启，原运行已安全结束。';
  if (failureCode === 'model_step_limit') return 'Agent 已达到本次推理步数上限，运行已安全停止。';
  if (failureCode === 'tool_call_limit') return 'Agent 已达到本次工具调用上限，运行已安全停止。';
  if (failureCode === 'repeated_tool_call') return '检测到重复工具调用，运行已安全停止。';
  if (failureCode === 'no_progress') return 'Agent 连续未取得新进展，运行已安全停止。';
  if (failureCode === 'tool_not_allowed') return 'Agent 请求了未获授权的工具，运行已安全停止。';
  if (failureCode === 'result_schema_invalid') return 'Agent 返回结果不符合约定，未保存为回答。';
  if (failureCode === 'source_validation_failed') return '回答来源校验失败，未保存为回答。';
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
  if (status === 'completed') label = '已完成';
  else if (status === 'failed') label = '失败';
  else if (status === 'cancelled') label = '已停止';
  else if (cancelling) label = '正在停止';
  else if (connection === 'reconnecting') label = '正在重新连接';
  else if (connection === 'unavailable') label = '连接中断';
  else if (status === 'running') label = '运行中';
  return <span className={`run-status ${status}`}>{label}</span>;
}

function ToolRun({
  tools,
  working,
  runStatus,
}: {
  tools: ToolProgress[];
  working: boolean;
  runStatus: RunStatus;
}) {
  const completed = tools.filter((tool) => tool.status === 'completed').length;
  const stopped = isTerminalStatus(runStatus) ? tools.length - completed : 0;
  return (
    <details className="tool-run" open={working}>
      <summary>
        <span
          className={
            working ? 'status-orb pulsing' : stopped > 0 ? 'status-orb stopped' : 'status-orb'
          }
        />
        <strong>
          {working
            ? '正在调用工具'
            : stopped > 0
              ? `${completed} 个完成，${stopped} 个已停止`
              : `已完成 ${completed} 个工具步骤`}
        </strong>
        <span>查看详情</span>
      </summary>
      <div className="tool-list">
        {tools.map((tool) => (
          <div className="tool-step" key={tool.tool_call_id}>
            <span
              className={
                tool.status === 'completed'
                  ? 'tool-check completed'
                  : isTerminalStatus(runStatus)
                    ? 'tool-check stopped'
                    : 'tool-check'
              }
            >
              {tool.status === 'completed' ? '✓' : isTerminalStatus(runStatus) ? '×' : '·'}
            </span>
            <div>
              <strong>{tool.label}</strong>
              {tool.output_summary || tool.input_summary ? (
                <span>{tool.output_summary || tool.input_summary}</span>
              ) : null}
            </div>
            <span>
              {tool.status === 'completed'
                ? '完成'
                : isTerminalStatus(runStatus)
                  ? '已停止'
                  : '运行中'}
            </span>
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
