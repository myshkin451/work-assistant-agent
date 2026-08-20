export type ThreadSummary = {
  thread_id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type Message = {
  message_id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  run_id: string | null;
};

export type RunStatus = 'created' | 'running' | 'completed' | 'failed' | 'cancelled';

export type UsageAvailability =
  | 'complete'
  | 'partial'
  | 'unavailable'
  | 'unknown'
  | 'pending';

export type UsageMetric = {
  value: number | null;
  availability: UsageAvailability;
};

export type RunUsage = {
  schema_version: '1.0.0' | null;
  state: 'final' | 'unknown' | 'pending';
  model_call_count: number | null;
  retry_count: number | null;
  input_tokens: UsageMetric;
  output_tokens: UsageMetric;
  cached_tokens: UsageMetric;
  reasoning_tokens: UsageMetric;
  total_tokens: UsageMetric;
  time_to_first_visible_ms: number | null;
  generation_duration_ms: number | null;
  run_duration_ms: number | null;
  error_category: string | null;
};

export type RunView = {
  run_id: string;
  thread_id: string;
  status: RunStatus;
  last_seq: number;
  created_at: string;
  completed_at: string | null;
  usage: RunUsage;
};

export type AccountUsageRange = '7d' | '30d' | 'all';

export type AccountUsageResponse = {
  account: {
    display_name: string;
    organization: string | null;
    extensions: {
      session_expires_at: string | null;
      permission_summary: string | null;
    };
  };
  scope: {
    range: AccountUsageRange;
    from_at: string | null;
    to_at: string;
    thread_id: string | null;
  };
  runs: {
    total: number;
    completed: number;
    failed: number;
    cancelled: number;
    active: number;
  };
  model_calls: UsageMetric;
  retries: UsageMetric;
  input_tokens: UsageMetric;
  output_tokens: UsageMetric;
  cached_tokens: UsageMetric;
  reasoning_tokens: UsageMetric;
  total_tokens: UsageMetric;
};

export type RunSnapshot = RunView & {
  events: ProductEvent[];
};

export type ThreadSnapshot = ThreadSummary & {
  messages: Message[];
  runs: RunSnapshot[];
  active_run: RunView | null;
};

export type InitialRunResponse = {
  thread: ThreadSummary;
  run: RunView;
};

export const productEventTypes = [
  'run.started',
  'tool.started',
  'tool.finished',
  'message.delta',
  'source.added',
  'message.completed',
  'run.completed',
  'run.failed',
  'run.cancelled',
] as const;

export type ProductEventType = (typeof productEventTypes)[number];

export type ProductEvent = {
  event_id: string;
  run_id: string;
  thread_id: string;
  seq: number;
  type: ProductEventType;
  occurred_at: string;
  data: Record<string, unknown>;
};

export type ToolProgress = {
  tool_call_id: string;
  name: string;
  label: string;
  input_summary?: string;
  output_summary?: string;
  status: 'running' | 'completed';
};

export type SourceReference = {
  source_id: string;
  label: string;
  description: string;
};

export const runFailureCodes = [
  'run_timeout',
  'agent_execution_failed',
  'service_restarted',
  'model_step_limit',
  'tool_call_limit',
  'repeated_tool_call',
  'no_progress',
  'tool_not_allowed',
  'result_schema_invalid',
  'source_validation_failed',
] as const;

export type RunFailureCode = (typeof runFailureCodes)[number];

const knownRunFailureCodes = new Set<string>(runFailureCodes);

export const isRunFailureCode = (value: unknown): value is RunFailureCode =>
  typeof value === 'string' && knownRunFailureCodes.has(value);

export type StreamConnection =
  | 'connecting'
  | 'live'
  | 'reconnecting'
  | 'unavailable'
  | 'closed';

export type RunProjection = {
  run: RunView;
  assistantText: string;
  tools: ToolProgress[];
  sources: SourceReference[];
  lastSeq: number;
  connection: StreamConnection;
  cancelling: boolean;
  failureCode?: RunFailureCode;
};

export const isTerminalStatus = (status: RunStatus) =>
  status === 'completed' || status === 'failed' || status === 'cancelled';
