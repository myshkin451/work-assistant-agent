from __future__ import annotations

from work_assistant.bootstrap import build_policy_kernel
from work_assistant.execution_policy import (
    AgentResult,
    ExecutionOutcomeEvidence,
    ExecutionPlanEvidence,
    RunExecution,
)
from work_assistant.identity import Principal
from work_assistant.schemas import RunFailureCode
from work_assistant.settings import Settings


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "app_env": "test",
        "identity_provider_mode": "anonymous",
        "database_url": "sqlite+aiosqlite:///:memory:",
        "checkpoint_database_url": "postgresql://unused:unused@localhost/unused",
        "model_mode": "fake",
        "fake_step_delay_seconds": 0,
        "run_timeout_seconds": 2,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def make_execution(
    principal: Principal,
    *,
    settings: Settings | None = None,
) -> RunExecution:
    return build_policy_kernel(settings or make_settings()).prepare_run(principal=principal)


def make_execution_plan(principal: Principal) -> ExecutionPlanEvidence:
    return make_execution(principal).plan_evidence


def terminal_outcome(
    principal: Principal,
    *,
    status: str,
    failure_code: RunFailureCode | None = None,
) -> ExecutionOutcomeEvidence:
    execution = make_execution(principal)
    if status == "completed":
        execution.validate_result(AgentResult(text="completed", source_ids=()))
        return execution.outcome(status="completed", stop_reason="completed")
    if status == "failed" and failure_code is not None:
        return execution.outcome(
            status="failed",
            stop_reason=failure_code,
            failure_code=failure_code,
        )
    if status == "cancelled":
        return execution.outcome(status="cancelled", stop_reason="user_cancelled")
    raise ValueError("unsupported test terminal outcome")
