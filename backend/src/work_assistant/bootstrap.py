from __future__ import annotations

from collections.abc import Sequence

from .agent_definition import AgentDefinition, default_agent_definition
from .agent_runtime import model_profile_for_settings, validate_runtime_configuration
from .capabilities import (
    NeutralAuthenticatedToolPolicy,
    PrincipalCapabilityPolicy,
    ToolRegistry,
    default_tool_registry,
)
from .execution_policy import AgentPolicyKernel
from .settings import Settings


def build_policy_kernel(
    settings: Settings,
    *,
    agent_definitions: Sequence[AgentDefinition] | None = None,
    tool_registry: ToolRegistry | None = None,
    capability_policy: PrincipalCapabilityPolicy | None = None,
) -> AgentPolicyKernel:
    """Validate every startup policy dependency before any external side effect."""

    resolved_tools = tool_registry or default_tool_registry()
    definitions = (
        tuple(agent_definitions)
        if agent_definitions is not None
        else (
            default_agent_definition(
                max_model_steps=settings.max_model_steps,
                max_tool_calls=settings.max_tool_calls,
                deadline_seconds=settings.run_timeout_seconds,
                max_identical_tool_calls=settings.max_identical_tool_calls,
                max_no_progress_steps=settings.max_no_progress_steps,
            ),
        )
    )
    kernel = AgentPolicyKernel(
        definitions=definitions,
        default_agent_id="default-work-assistant",
        model_profile=model_profile_for_settings(settings),
        tool_registry=resolved_tools,
        capability_policy=capability_policy or NeutralAuthenticatedToolPolicy(),
    )
    validate_runtime_configuration(settings)
    return kernel
