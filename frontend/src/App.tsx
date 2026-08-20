import {
  Fragment,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type RefObject,
} from 'react';
import {
  ApiError,
  cancelRun,
  createInitialRun,
  createRun,
  getThread,
  listThreads,
  streamRunEvents,
  updateThread,
} from './api';
import { MarkdownMessage } from './MarkdownMessage';
import {
  parseThreadRoute,
  threadPath,
  type HistoryMode,
  writeBlankRoute,
  writeThreadRoute,
} from './threadRoute';
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

const publicSourceCopy: Record<string, string> = {
  'System clock with IANA time data': 'IANA 时区系统时钟',
  'Current server clock converted with the requested IANA timezone.':
    '由系统时钟按指定 IANA 时区换算。',
};

const iconPaths = {
  plus: 'M12 5v14M5 12h14',
  send: 'M12 19V5M6 11l6-6 6 6',
  stop: 'M8 8h8v8H8z',
  time: 'M12 7v5l3 2 M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18',
  source: 'M5 4h14v16H5z M8 8h8M8 12h8M8 16h5',
  chat: 'M4 5h16v12H8l-4 3z',
  close: 'M6 6l12 12M18 6L6 18',
  down: 'M6 9l6 6 6-6',
  edit: 'M4 20h4L19 9l-4-4L4 16v4z M13.5 6.5l4 4',
  menu: 'M4 7h16M4 12h16M4 17h16',
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

function mergeProjectionMapsMonotonic(
  current: ProjectionMap,
  incoming: ProjectionMap,
): ProjectionMap {
  const next = { ...current };
  for (const [runId, candidate] of Object.entries(incoming)) {
    const existing = current[runId];
    if (!existing) {
      next[runId] = candidate;
      continue;
    }
    if (candidate.lastSeq < existing.lastSeq) continue;
    if (candidate.lastSeq === existing.lastSeq) {
      next[runId] = {
        ...existing,
        assistantText: existing.assistantText || candidate.assistantText,
        tools: existing.tools.length >= candidate.tools.length ? existing.tools : candidate.tools,
        sources:
          existing.sources.length >= candidate.sources.length
            ? existing.sources
            : candidate.sources,
        ...(isTerminalStatus(candidate.run.status)
          ? { run: candidate.run, connection: 'closed' as const, cancelling: false }
          : {}),
      };
      continue;
    }
    next[runId] = {
      ...candidate,
      connection: isTerminalStatus(candidate.run.status) ? 'closed' : existing.connection,
      cancelling: existing.cancelling,
    };
  }
  return next;
}

function messagesMatch(left: Message, right: Message) {
  if (left.message_id === right.message_id) return true;
  if (left.role !== right.role) return false;
  if (left.run_id && right.run_id) return left.run_id === right.run_id;
  return left.role === 'user' && left.content === right.content;
}

function mergeThreadSnapshotMonotonic(
  current: ThreadSnapshot | null,
  incoming: ThreadSnapshot,
  pending?: Message,
) {
  const base = pending ? mergeOptimisticMessage(incoming, pending) : incoming;
  if (!current || current.thread_id !== incoming.thread_id) return base;

  const messages = [...base.messages];
  for (const message of current.messages) {
    if (!messages.some((candidate) => messagesMatch(candidate, message))) messages.push(message);
  }

  const runs = [...base.runs];
  for (const run of current.runs) {
    const index = runs.findIndex((candidate) => candidate.run_id === run.run_id);
    if (index < 0) runs.push(run);
    else if ((runs[index]?.last_seq ?? 0) < run.last_seq) runs[index] = run;
  }

  let activeRun = base.active_run;
  if (activeRun) {
    const currentRun = current.runs.find((candidate) => candidate.run_id === activeRun?.run_id);
    if (
      currentRun &&
      isTerminalStatus(currentRun.status) &&
      currentRun.last_seq >= activeRun.last_seq
    ) {
      activeRun = null;
    }
  }
  if (current.active_run && current.active_run.run_id !== activeRun?.run_id) {
    const mergedCurrentRun = runs.find(
      (candidate) => candidate.run_id === current.active_run?.run_id,
    );
    const currentCreatedAt = Date.parse(current.active_run.created_at);
    const incomingCreatedAt = activeRun ? Date.parse(activeRun.created_at) : Number.NaN;
    const currentIsNoOlder =
      !activeRun ||
      !Number.isFinite(currentCreatedAt) ||
      !Number.isFinite(incomingCreatedAt) ||
      currentCreatedAt >= incomingCreatedAt;
    if (
      mergedCurrentRun &&
      !isTerminalStatus(mergedCurrentRun.status) &&
      currentIsNoOlder
    ) {
      activeRun = current.active_run;
    }
  }
  if (!activeRun && current.active_run) {
    const incomingRun = base.runs.find(
      (candidate) => candidate.run_id === current.active_run?.run_id,
    );
    if (!incomingRun || !isTerminalStatus(incomingRun.status)) activeRun = current.active_run;
  }

  return { ...base, messages, runs, active_run: activeRun };
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
  return '暂时无法连接，请稍后重试。';
}

function isAccessError(error: unknown): error is ApiError {
  return error instanceof ApiError && (error.status === 401 || error.status === 403);
}

type InitialAttempt = {
  threadId: string;
  idempotencyKey: string;
  message: string;
};

const FOLLOW_BOTTOM_THRESHOLD = 80;

export function App() {
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [snapshot, setSnapshot] = useState<ThreadSnapshot | null>(null);
  const [projectionsByRunId, setProjectionsByRunId] = useState<ProjectionMap>({});
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [admittingInitialRun, setAdmittingInitialRun] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [accessBlocked, setAccessBlocked] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [renameInput, setRenameInput] = useState('');
  const [renameSaving, setRenameSaving] = useState(false);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  const selectedThreadRef = useRef<string | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const submittingRef = useRef(false);
  const accessBlockedRef = useRef(false);
  const initialAttemptRef = useRef<InitialAttempt | null>(null);
  const conversationScrollRef = useRef<HTMLDivElement | null>(null);
  const followOutputRef = useRef(true);
  const mobileNavButtonRef = useRef<HTMLButtonElement | null>(null);

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
    setAdmittingInitialRun(false);
    setLoading(false);
    setAccessBlocked(true);
    setMobileNavOpen(false);
    setRenaming(false);
    setRenameSaving(false);
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
      setSnapshot((current) => mergeThreadSnapshotMonotonic(current, fresh));
      setProjectionsByRunId((current) =>
        mergeProjectionMapsMonotonic(current, projectionsFromSnapshot(fresh)),
      );
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
    async (threadId: string, historyMode: HistoryMode = 'push') => {
      if (accessBlockedRef.current) return;
      if (historyMode !== 'none' && window.location.pathname !== threadPath(threadId)) {
        writeThreadRoute(threadId, historyMode);
      }
      streamAbortRef.current?.abort();
      selectedThreadRef.current = threadId;
      followOutputRef.current = true;
      setShowJumpToLatest(false);
      setMobileNavOpen(false);
      setRenaming(false);
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
        if (selectedThreadRef.current !== threadId) return;
        if (openError instanceof ApiError && (openError.status === 403 || openError.status === 404)) {
          // One inaccessible URL is a route-level failure. Keep the caller's
          // own thread list available; active-run authorization failures still
          // use the global fail-closed path in connectRun.
          setError('无法打开这个对话。你可以选择最近对话，或新建一个对话。');
        } else if (!blockAccess(openError)) {
          setError(displayError(openError));
        }
      } finally {
        if (selectedThreadRef.current === threadId) setLoading(false);
      }
    },
    [blockAccess, connectRun],
  );

  const showBlankThread = useCallback((historyMode: HistoryMode = 'push') => {
    if (accessBlockedRef.current) return;
    streamAbortRef.current?.abort();
    selectedThreadRef.current = null;
    initialAttemptRef.current = null;
    followOutputRef.current = true;
    if (
      historyMode !== 'none' &&
      (window.location.pathname !== '/' || historyMode === 'replace')
    ) {
      writeBlankRoute(historyMode);
    }
    setSnapshot(null);
    setProjectionsByRunId({});
    setInput('');
    setLoading(false);
    setError(null);
    setMobileNavOpen(false);
    setRenaming(false);
    setShowJumpToLatest(false);
  }, []);

  useEffect(() => {
    let disposed = false;

    const applyLocation = () => {
      if (disposed || accessBlockedRef.current) return;
      const route = parseThreadRoute();
      if (route.kind === 'thread') {
        void openThread(route.threadId, 'none');
        return;
      }
      if (route.kind === 'blank') {
        showBlankThread('none');
        return;
      }

      streamAbortRef.current?.abort();
      selectedThreadRef.current = null;
      setSnapshot(null);
      setProjectionsByRunId({});
      setLoading(false);
      setError('这个对话链接无效。请从最近对话进入，或新建一个对话。');
    };

    const handlePopState = () => applyLocation();
    window.addEventListener('popstate', handlePopState);
    void refreshThreads()
      .then(() => applyLocation())
      .catch((initialError: unknown) => {
        if (!disposed && !blockAccess(initialError)) {
          setError(displayError(initialError));
          setLoading(false);
        }
      });

    return () => {
      disposed = true;
      window.removeEventListener('popstate', handlePopState);
      streamAbortRef.current?.abort();
    };
  }, [blockAccess, openThread, refreshThreads, showBlankThread]);

  useLayoutEffect(() => {
    if (!followOutputRef.current) return;
    const scrollContainer = conversationScrollRef.current;
    scrollContainer?.scrollTo({ top: scrollContainer.scrollHeight, behavior: 'auto' });
  }, [
    projectionsByRunId,
    snapshot?.active_run?.run_id,
    snapshot?.messages.length,
    snapshot?.thread_id,
  ]);

  const handleConversationScroll = useCallback(() => {
    const scrollContainer = conversationScrollRef.current;
    if (!scrollContainer) return;
    const distanceFromBottom =
      scrollContainer.scrollHeight - scrollContainer.scrollTop - scrollContainer.clientHeight;
    const shouldFollow = distanceFromBottom <= FOLLOW_BOTTOM_THRESHOLD;
    followOutputRef.current = shouldFollow;
    setShowJumpToLatest(!shouldFollow);
  }, []);

  const scrollToLatest = useCallback((behavior: ScrollBehavior = 'smooth') => {
    followOutputRef.current = true;
    setShowJumpToLatest(false);
    const scrollContainer = conversationScrollRef.current;
    scrollContainer?.scrollTo({ top: scrollContainer.scrollHeight, behavior });
  }, []);

  const startNewThread = useCallback(() => {
    if (submittingRef.current || accessBlockedRef.current) return;
    showBlankThread(window.location.pathname === '/' ? 'none' : 'push');
  }, [showBlankThread]);

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
    let targetThreadId = snapshot?.thread_id ?? null;
    const isInitialRun = targetThreadId === null;
    if (isInitialRun) setAdmittingInitialRun(true);

    try {
      let run: RunView;
      let persistedPending: Message;

      if (isInitialRun) {
        const reusable = initialAttemptRef.current;
        const attempt =
          reusable?.message === message
            ? reusable
            : {
                threadId: crypto.randomUUID(),
                idempotencyKey,
                message,
              };
        initialAttemptRef.current = attempt;
        targetThreadId = attempt.threadId;
        const now = new Date().toISOString();
        selectedThreadRef.current = targetThreadId;
        setSnapshot({
          thread_id: targetThreadId,
          title: titleFromMessage(message),
          created_at: now,
          updated_at: now,
          messages: [pending],
          runs: [],
          active_run: null,
        });
        setProjectionsByRunId({});

        const created = await createInitialRun(
          attempt.threadId,
          message,
          attempt.idempotencyKey,
        );
        if (accessBlockedRef.current) return;
        if (
          created.thread.thread_id !== attempt.threadId ||
          created.run.thread_id !== attempt.threadId
        ) {
          throw new Error('Initial run response does not match the requested thread');
        }
        initialAttemptRef.current = null;
        run = created.run;
        persistedPending = { ...pending, run_id: run.run_id };
        if (selectedThreadRef.current !== attempt.threadId) {
          void refreshThreads().catch((refreshError: unknown) => {
            blockAccess(refreshError);
          });
          return;
        }
        writeThreadRoute(created.thread.thread_id, 'replace');
        setSnapshot({
          ...created.thread,
          messages: [persistedPending],
          runs: [{ ...run, events: [] }],
          active_run: run,
        });
      } else {
        const threadId = targetThreadId;
        if (!threadId) throw new Error('Existing run requires a thread');
        setSnapshot((current) =>
          current?.thread_id === threadId
            ? { ...current, messages: [...current.messages, pending] }
            : current,
        );
        run = await createRun(threadId, message, idempotencyKey);
        if (accessBlockedRef.current) return;
        persistedPending = { ...pending, run_id: run.run_id };
      }

      const threadId = targetThreadId;
      if (!threadId) throw new Error('Run response is missing a thread');
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

      if (selectedThreadRef.current === threadId && !isTerminalStatus(run.status)) {
        connectRun(run, initialProjection);
      }
      void getThread(threadId)
        .then((fresh) => {
          if (accessBlockedRef.current || selectedThreadRef.current !== threadId) return;
          const projections = projectionsFromSnapshot(fresh);
          setSnapshot((current) =>
            mergeThreadSnapshotMonotonic(current, fresh, persistedPending),
          );
          setProjectionsByRunId((current) =>
            mergeProjectionMapsMonotonic(current, projections),
          );
        })
        .catch((snapshotError: unknown) => {
          blockAccess(snapshotError);
          // The optimistic message and live event stream remain authoritative
          // while a background snapshot is temporarily unavailable.
        });
      void refreshThreads().catch((refreshError: unknown) => {
        if (blockAccess(refreshError)) return;
        if (selectedThreadRef.current === threadId) setError(displayError(refreshError));
      });
    } catch (submitError) {
      if (blockAccess(submitError)) return;
      const stillSelected = selectedThreadRef.current === targetThreadId;
      if (isInitialRun && submitError instanceof ApiError && submitError.status < 500) {
        initialAttemptRef.current = null;
      }
      if (restoreComposerOnFailure && stillSelected) setInput(message);
      setSnapshot((current) =>
        current?.thread_id === targetThreadId
          ? isInitialRun
            ? null
            : {
                ...current,
                messages: current.messages.filter(
                  (item) => item.message_id !== pending.message_id,
                ),
              }
          : current,
      );
      if (isInitialRun && stillSelected) {
        selectedThreadRef.current = null;
        setProjectionsByRunId({});
      }
      if (stillSelected) setError(displayError(submitError));
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
      if (isInitialRun) setAdmittingInitialRun(false);
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
        setError('无法重试这条消息，请重新发送问题。');
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

  const beginRename = useCallback(() => {
    if (!snapshot || accessBlockedRef.current) return;
    setRenameInput(snapshot.title);
    setRenaming(true);
    setError(null);
  }, [snapshot]);

  const saveRename = useCallback(
    async (event: FormEvent) => {
      event.preventDefault();
      if (!snapshot || renameSaving || accessBlockedRef.current) return;
      const title = renameInput.trim();
      if (!title) {
        setError('对话标题不能为空。');
        return;
      }
      if (title.length > 200) {
        setError('对话标题不能超过 200 个字符。');
        return;
      }

      const threadId = snapshot.thread_id;
      setRenameSaving(true);
      setError(null);
      try {
        const updated = await updateThread(threadId, title);
        if (selectedThreadRef.current !== threadId || accessBlockedRef.current) return;
        setSnapshot((current) =>
          current?.thread_id === threadId ? { ...current, ...updated } : current,
        );
        setThreads((current) =>
          current.map((thread) => (thread.thread_id === threadId ? updated : thread)),
        );
        setRenaming(false);
      } catch (renameError) {
        if (!blockAccess(renameError) && selectedThreadRef.current === threadId) {
          setError(displayError(renameError));
        }
      } finally {
        setRenameSaving(false);
      }
    },
    [blockAccess, renameInput, renameSaving, snapshot],
  );

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
        disabled={accessBlocked || admittingInitialRun}
        onOpenThread={(threadId) => void openThread(threadId)}
        onNewThread={startNewThread}
      />

      <main className="chat-main">
        <header className="topbar">
          <button
            ref={mobileNavButtonRef}
            className="mobile-menu-button"
            type="button"
            aria-label="打开对话导航"
            aria-expanded={mobileNavOpen}
            aria-controls="mobile-conversation-drawer"
            onClick={() => setMobileNavOpen(true)}
            disabled={accessBlocked}
          >
            <Icon name="menu" />
          </button>
          <div className="topbar-title">
            {renaming && snapshot ? (
              <form className="rename-form" onSubmit={(event) => void saveRename(event)}>
                <label className="sr-only" htmlFor="thread-title-input">
                  对话标题
                </label>
                <input
                  id="thread-title-input"
                  value={renameInput}
                  maxLength={200}
                  autoFocus
                  disabled={renameSaving}
                  onChange={(event) => setRenameInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Escape') setRenaming(false);
                  }}
                />
                <button type="submit" disabled={renameSaving || !renameInput.trim()}>
                  保存
                </button>
                <button type="button" onClick={() => setRenaming(false)} disabled={renameSaving}>
                  取消
                </button>
              </form>
            ) : (
              <div className="title-row">
                <strong>{snapshot?.title || '新对话'}</strong>
                {snapshot ? (
                  <button type="button" aria-label="重命名当前对话" onClick={beginRename}>
                    <Icon name="edit" />
                  </button>
                ) : null}
              </div>
            )}
          </div>
        </header>

        <div className="conversation-region">
          <div
            className="conversation-scroll"
            ref={conversationScrollRef}
            onScroll={handleConversationScroll}
          >
            <section
              className="conversation"
              aria-label="对话内容"
              aria-live="polite"
              aria-busy={active}
            >
              {loading && !snapshot ? <LoadingState /> : null}
              {!loading &&
              !error &&
              displayedMessages.length === 0 &&
              Object.keys(projectionsByRunId).length === 0 ? (
                <EmptyState />
              ) : null}

              {displayedMessages.map((message) => {
                if (
                  message.role === 'assistant' &&
                  message.run_id &&
                  projectionsByRunId[message.run_id]
                ) {
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
            </section>
          </div>
          {showJumpToLatest ? (
            <button
              className="jump-to-latest"
              type="button"
              onClick={() => scrollToLatest()}
            >
              <Icon name="down" />
              回到最新
            </button>
          ) : null}
        </div>

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

      {mobileNavOpen ? (
        <MobileDrawer
          triggerRef={mobileNavButtonRef}
          threads={threads}
          activeThreadId={snapshot?.thread_id ?? null}
          disabled={accessBlocked || admittingInitialRun}
          onClose={() => setMobileNavOpen(false)}
          onOpenThread={(threadId) => void openThread(threadId)}
          onNewThread={startNewThread}
        />
      ) : null}
    </div>
  );
}

type ThreadNavigationProps = {
  threads: ThreadSummary[];
  activeThreadId: string | null;
  onOpenThread: (threadId: string) => void;
  onNewThread: () => void;
  disabled: boolean;
};

function ThreadNavigation({
  threads,
  activeThreadId,
  onOpenThread,
  onNewThread,
  disabled,
}: ThreadNavigationProps) {
  return (
    <>
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
              className={
                thread.thread_id === activeThreadId ? 'history-item active' : 'history-item'
              }
              type="button"
              key={thread.thread_id}
              onClick={() => onOpenThread(thread.thread_id)}
              aria-current={thread.thread_id === activeThreadId ? 'page' : undefined}
              disabled={disabled}
            >
              <span>{thread.title || EMPTY_TITLE}</span>
            </button>
          ))
        )}
      </div>
    </>
  );
}

function Sidebar(props: ThreadNavigationProps) {
  return (
    <aside className="sidebar" aria-label="对话导航">
      <div className="brand">
        <strong>工作助手</strong>
      </div>
      <ThreadNavigation {...props} />
    </aside>
  );
}

type MobileDrawerProps = ThreadNavigationProps & {
  triggerRef: RefObject<HTMLButtonElement | null>;
  onClose: () => void;
};

function MobileDrawer({ triggerRef, onClose, ...navigationProps }: MobileDrawerProps) {
  const drawerRef = useRef<HTMLElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    const trigger = triggerRef.current;
    document.body.style.overflow = 'hidden';
    closeButtonRef.current?.focus();
    return () => {
      document.body.style.overflow = previousOverflow;
      trigger?.focus();
    };
  }, [triggerRef]);

  const handleKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key !== 'Tab') return;

    const focusable = Array.from(
      drawerRef.current?.querySelectorAll<HTMLElement>(
        'button:not(:disabled), a[href], input:not(:disabled), [tabindex]:not([tabindex="-1"])',
      ) ?? [],
    );
    const first = focusable[0];
    const last = focusable.at(-1);
    if (!first || !last) return;
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  return (
    <div
      className="mobile-drawer-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <aside
        id="mobile-conversation-drawer"
        ref={drawerRef}
        className="mobile-drawer"
        role="dialog"
        aria-modal="true"
        aria-label="对话导航"
        onKeyDown={handleKeyDown}
      >
        <div className="mobile-drawer-heading">
          <div className="brand compact">
            <strong>工作助手</strong>
          </div>
          <button ref={closeButtonRef} type="button" aria-label="关闭对话导航" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
        <ThreadNavigation {...navigationProps} />
      </aside>
    </div>
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
      <h1>今天想处理什么？</h1>
      <p>可以直接提问，也可以继续最近的对话。</p>
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
      <div className="assistant-content">
        <MarkdownMessage content={message.content} />
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
      <div className="assistant-content">
        <div className="assistant-heading">
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
          <MarkdownMessage content={assistantText} />
        ) : isWorking && connection !== 'unavailable' ? (
          <div className="thinking" role="status">
            <span />
            <span />
            <span />
            正在回答…
          </div>
        ) : null}
        {run.status === 'failed' ? (
          <div className="run-notice error run-notice-row" role="alert">
            <span>{friendlyRunError(failureCode)}</span>
            <button className="retry-button" type="button" onClick={onRetry} disabled={retryDisabled}>
              重试
            </button>
          </div>
        ) : null}
        {isWorking && connection === 'unavailable' ? (
          <div className="run-notice connection" role="status">
            连接中断，刷新页面后可继续查看。
          </div>
        ) : null}
        {run.status === 'cancelled' ? <div className="run-notice">已停止回答。</div> : null}
        {sources.length > 0 ? (
          <div className="sources" aria-label="回答来源">
            <strong>参考来源</strong>
            <ol>
              {sources.map((source, index) => (
                <li key={source.source_id}>
                  <span className="source-index">{index + 1}.</span>
                  <div>
                    <span>{publicSourceCopy[source.label] ?? source.label}</span>
                    <small>{publicSourceCopy[source.description] ?? source.description}</small>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        ) : null}
      </div>
    </article>
  );
}

function friendlyRunError(failureCode: RunProjection['failureCode']) {
  if (failureCode === 'run_timeout') return '等待时间过长，这次回答没有完成。';
  if (failureCode === 'service_restarted') return '服务刚刚恢复，请重试这条消息。';
  if (failureCode === 'tool_not_allowed') return '当前请求暂时无法处理。';
  if (failureCode === 'result_schema_invalid') return '这次回答不完整，未予展示。';
  if (failureCode === 'source_validation_failed') return '来源未通过校验，这次回答未予展示。';
  return '这次没有生成完整回答，请重试。';
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
  if (isTerminalStatus(status)) return null;
  let label = '正在准备回答…';
  if (cancelling) label = '正在停止…';
  else if (connection === 'reconnecting') label = '正在恢复连接…';
  else if (connection === 'unavailable') label = '连接中断';
  else if (status === 'running') label = '正在回答…';
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
  const activeTool = [...tools].reverse().find((tool) => tool.status !== 'completed');
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
            ? `正在查询${activeTool ? `：${activeTool.label}` : '…'}`
            : stopped > 0
              ? '查询已停止'
              : '查询记录'}
        </strong>
        <span>详情</span>
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
                ? '已查询'
                : isTerminalStatus(runStatus)
                  ? '已停止'
                  : '查询中'}
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
          输入消息
        </label>
        <textarea
          id="message-input"
          value={value}
          placeholder="输入消息"
          rows={1}
          disabled={disabled || active}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={onKeyDown}
        />
        <div className="composer-actions">
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
    </div>
  );
}
