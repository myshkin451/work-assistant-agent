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

export type RunView = {
  run_id: string;
  thread_id: string;
  status: RunStatus;
  last_seq: number;
  created_at: string;
  completed_at: string | null;
};

export type ThreadSnapshot = ThreadSummary & {
  messages: Message[];
  active_run: RunView | null;
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

export type RunProjection = {
  run: RunView;
  assistantText: string;
  tools: ToolProgress[];
  sources: SourceReference[];
  lastSeq: number;
  connection: 'connecting' | 'live' | 'reconnecting' | 'closed';
  cancelling: boolean;
  errorCode?: string;
};

export const isTerminalStatus = (status: RunStatus) =>
  status === 'completed' || status === 'failed' || status === 'cancelled';
