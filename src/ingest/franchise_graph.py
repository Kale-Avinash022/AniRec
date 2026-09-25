"""Franchise Relations Graph & Lineage Modeling (Step 13).

Why this module exists
----------------------
Anime franchises sprawl across dozens of entries on AniList.  A query for
"Attack on Titan" can surface S1, S2, S3, The Final Season, No Regrets OVA,
and Junior High — all in the same result set.  Without franchise awareness,
the recommendation engine would waste every retrieved slot on variations of
the same story, killing diversity.

This module builds an undirected franchise graph from ``RelationEdge`` data
produced by Step 12, then resolves every node to a **franchise_root_id** via
BFS-connected-component traversal.  Downstream consumers (Step 30 dedup,
Step 27 vector search) can group or collapse entries by that root ID to
surface exactly one representative per franchise.

Relation types that define franchise membership
------------------------------------------------
- PREQUEL / SEQUEL     — canonical story-line continuations
- PARENT               — source material that spawned adaptations
- SIDE_STORY / SPIN_OFF — same universe, significant overlap
- ALTERNATIVE          — alternative retelling / AU of same property

CHARACTER, SUMMARY, OTHER are *not* connected — they are separate creative
works that happen to share cast or concept.

Algorithm: BFS connected-components on an adjacency list
---------------------------------------------------------
For the 3,000-node catalog (≤ 60 pages × 50 items) a simple BFS is O(N + E)
and completes in < 1 ms.  Each connected component gets ``franchise_root_id``
= the *minimum* anime ID in that component, which is deterministic and maps
naturally to the earliest-produced entry in most franchises.

Standalone anime (no qualifying relations) form singleton components and
receive ``franchise_root_id == anime_id`` — every show is always assigned a
root, so the field can be used as an unconditional metadata filter.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from src.ingest.transform import RelationEdge, SanitizedAnime

# ---------------------------------------------------------------------------
# Relation types that constitute franchise membership
# ---------------------------------------------------------------------------

FRANCHISE_RELATION_TYPES: frozenset[str] = frozenset(
    {
        "PREQUEL",
        "SEQUEL",
        "PARENT",
        "SIDE_STORY",
        "SPIN_OFF",
        "ALTERNATIVE",
    }
)


# ---------------------------------------------------------------------------
# FranchiseGraph
# ---------------------------------------------------------------------------


class FranchiseGraph:
    """Undirected adjacency-list graph of anime franchise relationships.

    Usage
    -----
    ::

        graph = FranchiseGraph()
        for anime in sanitized_catalog:
            graph.add_anime(anime.id, anime.title_romaji, anime.relations)

        root_map = graph.find_connected_components()
        # root_map[anime_id] → franchise_root_id (int)

    The graph is intentionally immutable after ``find_connected_components()``
    is called — build it once, query it many times.
    """

    def __init__(self) -> None:
        # adjacency list: id → set of directly connected franchise peers
        self._adj: dict[int, set[int]] = defaultdict(set)
        # title registry for debugging / introspection
        self._titles: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_anime(
        self,
        anime_id: int,
        title: str,
        relations: list[RelationEdge],
    ) -> None:
        """Register one anime node and wire up its franchise edges.

        Calling this for a node that already exists is safe — edges are
        accumulated additively (``set.add`` is idempotent).

        Args:
            anime_id: AniList media ID of the show being registered.
            title:    Human-readable label (romaji) for debugging.
            relations: ``RelationEdge`` list produced by Step 12.
        """
        self._titles[anime_id] = title
        self._adj[anime_id]  # touch to guarantee node exists in adjacency map

        for edge in relations:
            if (
                edge.relation_type in FRANCHISE_RELATION_TYPES
                and edge.target_id is not None
            ):
                # Bidirectional — guarantees components survive partial catalogs
                # (target may not be in the ingested set, but we pre-wire it)
                self._adj[anime_id].add(edge.target_id)
                self._adj[edge.target_id].add(anime_id)

    # ------------------------------------------------------------------
    # Connected-component resolution
    # ------------------------------------------------------------------

    def find_connected_components(self) -> dict[int, int]:
        """BFS over all nodes and assign a deterministic franchise_root_id.

        The root of each component is the **minimum anime ID** in that
        component, which reliably maps to the earliest production entry.

        Returns:
            Mapping ``{anime_id: franchise_root_id}`` for every registered node.
            Standalone nodes (no franchise edges) map to themselves.
        """
        visited: set[int] = set()
        anime_to_root: dict[int, int] = {}

        for start_node in self._adj:
            if start_node in visited:
                continue

            # BFS to collect the entire connected component
            component: list[int] = []
            queue: deque[int] = deque([start_node])
            visited.add(start_node)

            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in self._adj[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            # Deterministic root: lowest ID in the component
            root_id = min(component)
            for member_id in component:
                anime_to_root[member_id] = root_id

        return anime_to_root

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_franchise_root(self, anime_id: int) -> int | None:
        """Return the franchise root for a single ID without building the full map.

        Useful for one-off lookups during retrieval (Step 30 dedup).  Performs
        a fresh BFS from ``anime_id`` — for bulk use, prefer
        ``find_connected_components()``.

        Args:
            anime_id: The ID to resolve.

        Returns:
            ``franchise_root_id`` if the ID is registered, else ``None``.
        """
        if anime_id not in self._adj:
            return None
        component = self._bfs_component(anime_id)
        return min(component)

    def component_members(self, anime_id: int) -> list[int]:
        """Return all IDs in the same franchise component as ``anime_id``.

        Args:
            anime_id: Any member of the franchise cluster.

        Returns:
            Sorted list of all connected member IDs, or ``[]`` if unregistered.
        """
        if anime_id not in self._adj:
            return []
        return sorted(self._bfs_component(anime_id))

    def title_of(self, anime_id: int) -> str | None:
        """Return the registered romaji title for debugging purposes."""
        return self._titles.get(anime_id)

    @property
    def node_count(self) -> int:
        """Total number of registered anime nodes (including phantom targets)."""
        return len(self._adj)

    @property
    def edge_count(self) -> int:
        """Total number of *directed* adjacency entries (each edge counted twice)."""
        return sum(len(neighbors) for neighbors in self._adj.values())

    def _bfs_component(self, start: int) -> list[int]:
        """Internal BFS from ``start`` returning all reachable nodes."""
        visited: set[int] = {start}
        queue: deque[int] = deque([start])
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in self._adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return component


# ---------------------------------------------------------------------------
# Batch Builder
# ---------------------------------------------------------------------------


def build_franchise_graph(catalog: list[SanitizedAnime]) -> FranchiseGraph:
    """Construct a ``FranchiseGraph`` from a full sanitized catalog.

    This is the primary entry point used by the ingestion pipeline after
    ``sanitize_raw_media_batch()`` produces the cleaned catalog.

    Args:
        catalog: List of ``SanitizedAnime`` instances (output of Step 12).

    Returns:
        A fully wired ``FranchiseGraph`` ready for component resolution.
    """
    graph = FranchiseGraph()
    for anime in catalog:
        graph.add_anime(anime.id, anime.title_romaji, anime.relations)
    return graph


def build_root_id_map(catalog: list[SanitizedAnime]) -> dict[int, int]:
    """One-shot helper: build graph and resolve all franchise roots.

    Args:
        catalog: List of ``SanitizedAnime`` instances.

    Returns:
        Mapping ``{anime_id: franchise_root_id}`` for every item in ``catalog``.
        IDs not in any qualifying relation map to themselves.
    """
    graph = build_franchise_graph(catalog)
    root_map = graph.find_connected_components()

    # Guarantee every catalog item has an entry — even if it had no relations
    # and therefore was never added to the adjacency list by the graph builder
    # (e.g. a show with all CHARACTER-type relations only).
    for anime in catalog:
        if anime.id not in root_map:
            root_map[anime.id] = anime.id

    return root_map


# ---------------------------------------------------------------------------
# Raw-dict convenience (used before Step 12 runs, e.g. in tests)
# ---------------------------------------------------------------------------


def add_raw_anime_to_graph(
    graph: FranchiseGraph,
    raw: dict[str, Any],
) -> None:
    """Add a single *raw* AniList media dict to the graph without full sanitization.

    Intended for lightweight testing scenarios where you don't want to run the
    full Step 12 pipeline first.

    Args:
        graph: The ``FranchiseGraph`` instance to mutate.
        raw:   Raw media dict from AniList GraphQL response.
    """
    anime_id: int | None = raw.get("id")
    if not isinstance(anime_id, int):
        return

    titles: dict[str, Any] = raw.get("title") or {}
    title = titles.get("romaji") or "Unknown"

    edges_raw: list[dict[str, Any]] = (
        (raw.get("relations") or {}).get("edges") or []
    )
    relations: list[RelationEdge] = []
    for edge in edges_raw:
        if not edge:
            continue
        node: dict[str, Any] = edge.get("node") or {}
        node_titles: dict[str, Any] = node.get("title") or {}
        relations.append(
            RelationEdge(
                relation_type=edge.get("relationType"),
                target_id=node.get("id"),
                target_title_romaji=node_titles.get("romaji"),
                target_title_english=node_titles.get("english"),
                target_format=node.get("format"),
            )
        )

    graph.add_anime(anime_id, title, relations)
