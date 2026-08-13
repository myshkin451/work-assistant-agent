from __future__ import annotations

from typing import Literal, Protocol

from .identity import INTERNAL_SUBJECT_PREFIX, Principal

ResourceKind = Literal["thread", "run"]


class ResourceForbiddenError(Exception):
    def __init__(self, resource_kind: ResourceKind) -> None:
        super().__init__(resource_kind)
        self.resource_kind = resource_kind


class OwnershipAuthorizer(Protocol):
    def require_owner(
        self,
        *,
        principal: Principal,
        owner_subject: str,
        resource_kind: ResourceKind,
    ) -> None: ...


class ExactOwnershipAuthorizer:
    """T-005 ownership rule: exact subject equality with no role bypass."""

    def require_owner(
        self,
        *,
        principal: Principal,
        owner_subject: str,
        resource_kind: ResourceKind,
    ) -> None:
        if owner_subject.startswith(INTERNAL_SUBJECT_PREFIX):
            raise ResourceForbiddenError(resource_kind)
        if principal.subject != owner_subject:
            raise ResourceForbiddenError(resource_kind)
