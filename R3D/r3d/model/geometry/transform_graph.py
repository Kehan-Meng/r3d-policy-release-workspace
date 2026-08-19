"""A fail-closed graph for composing frame transforms."""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Optional

import numpy as np

from .errors import (
    AmbiguousTransformPathError,
    DuplicateTransformError,
    InconsistentTransformCycleError,
    RuntimeTransformMissingError,
    TransformPathNotFoundError,
)
from .providers import TransformProvider
from .se3 import compose_transforms, identity_matrix
from .types import ArrayLike, Transform


class TransformGraph:
    def __init__(self, providers=()):
        self._edges: dict[frozenset[str], TransformProvider] = {}
        for provider in providers:
            self.add_provider(provider)

    def add_provider(self, provider: TransformProvider) -> None:
        if provider.source_frame == provider.target_frame:
            raise DuplicateTransformError("Transform graph edges must connect distinct frames")
        key = frozenset((provider.source_frame, provider.target_frame))
        if key in self._edges:
            raise DuplicateTransformError(
                f"A provider between {provider.source_frame!r} and "
                f"{provider.target_frame!r} is already registered"
            )
        if self._has_path(provider.source_frame, provider.target_frame):
            self._reject_redundant_path(provider)
        self._edges[key] = provider

    def _reject_redundant_path(self, provider: TransformProvider) -> None:
        """Reject cycles while distinguishing redundancy from inconsistency."""

        try:
            existing = self.get_transform(
                source_frame=provider.source_frame,
                target_frame=provider.target_frame,
            )
            proposed = provider.get_transform(
                source_frame=provider.source_frame,
                target_frame=provider.target_frame,
            )
        except RuntimeTransformMissingError as exc:
            raise AmbiguousTransformPathError(
                f"Adding {provider.source_frame!r}<->{provider.target_frame!r} would "
                "create a redundant dynamic path whose numerical consistency cannot "
                "be verified at registration time"
            ) from exc

        if existing.length_unit != proposed.length_unit:
            raise InconsistentTransformCycleError(
                f"Adding {provider.source_frame!r}<->{provider.target_frame!r} conflicts "
                f"with the existing path length unit ({proposed.length_unit!r} vs "
                f"{existing.length_unit!r})"
            )
        existing_matrix = np.asarray(existing.matrix, dtype=np.float64)
        proposed_matrix = np.asarray(proposed.matrix, dtype=np.float64)
        if np.allclose(existing_matrix, proposed_matrix, rtol=1e-7, atol=1e-9):
            raise AmbiguousTransformPathError(
                f"Adding {provider.source_frame!r}<->{provider.target_frame!r} would "
                "create a numerically consistent but ambiguous redundant path"
            )
        raise InconsistentTransformCycleError(
            f"Adding {provider.source_frame!r}<->{provider.target_frame!r} would create "
            "a transform cycle that is numerically inconsistent with the existing path"
        )

    @property
    def frames(self) -> frozenset[str]:
        return frozenset(frame for edge in self._edges for frame in edge)

    def _adjacency(self) -> dict[str, list[str]]:
        adjacency: dict[str, list[str]] = {}
        for edge in self._edges:
            left, right = tuple(edge)
            adjacency.setdefault(left, []).append(right)
            adjacency.setdefault(right, []).append(left)
        for neighbours in adjacency.values():
            neighbours.sort()
        return adjacency

    def _has_path(self, source_frame: str, target_frame: str) -> bool:
        adjacency = self._adjacency()
        if source_frame not in adjacency or target_frame not in adjacency:
            return False
        visited = {source_frame}
        queue = deque([source_frame])
        while queue:
            current = queue.popleft()
            if current == target_frame:
                return True
            for neighbour in adjacency[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        return False

    def _unique_shortest_path(self, source_frame: str, target_frame: str) -> list[str]:
        if source_frame == target_frame:
            return [source_frame]
        adjacency = self._adjacency()
        if source_frame not in adjacency or target_frame not in adjacency:
            raise TransformPathNotFoundError(
                f"No transform path from {source_frame!r} to {target_frame!r}"
            )

        distance = {source_frame: 0}
        parents: dict[str, list[str]] = {source_frame: []}
        queue = deque([source_frame])
        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                candidate_distance = distance[current] + 1
                if neighbour not in distance:
                    distance[neighbour] = candidate_distance
                    parents[neighbour] = [current]
                    queue.append(neighbour)
                elif distance[neighbour] == candidate_distance:
                    parents[neighbour].append(current)

        if target_frame not in distance:
            raise TransformPathNotFoundError(
                f"No transform path from {source_frame!r} to {target_frame!r}"
            )

        path_count: dict[str, int] = {source_frame: 1}
        for node, _ in sorted(distance.items(), key=lambda item: item[1]):
            if node == source_frame:
                continue
            path_count[node] = sum(path_count[parent] for parent in parents[node])
        if path_count[target_frame] != 1:
            raise AmbiguousTransformPathError(
                f"Multiple shortest transform paths connect {source_frame!r} and "
                f"{target_frame!r}"
            )

        path = [target_frame]
        current = target_frame
        while current != source_frame:
            current = parents[current][0]
            path.append(current)
        path.reverse()
        return path

    def get_transform(
        self,
        *,
        target_frame: str,
        source_frame: str,
        runtime_context: Optional[Mapping[str, Any]] = None,
        like: Optional[ArrayLike] = None,
    ) -> Transform:
        if source_frame == target_frame:
            if like is None:
                like = np.empty((), dtype=np.float64)
            return Transform(
                source_frame=source_frame,
                target_frame=target_frame,
                matrix=identity_matrix(like=like),
            )

        path = self._unique_shortest_path(source_frame, target_frame)
        accumulated = None
        for current, following in zip(path[:-1], path[1:]):
            provider = self._edges[frozenset((current, following))]
            edge = provider.get_transform(
                target_frame=following,
                source_frame=current,
                runtime_context=runtime_context,
                like=like,
            )
            accumulated = edge if accumulated is None else compose_transforms(edge, accumulated)
        if accumulated is None:
            raise TransformPathNotFoundError("Internal error while composing transform path")
        return accumulated

    def to_config(self) -> Mapping[str, Any]:
        providers = sorted(
            (provider.to_config() for provider in self._edges.values()),
            key=lambda item: (item["source_frame"], item["target_frame"]),
        )
        return {"type": "transform_graph", "providers": providers}
