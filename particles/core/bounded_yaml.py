# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""A ``yaml.safe_load`` whose alias expansion is held to a node budget.

``yaml.safe_load`` resolves an alias to the object its anchor built, so a few
lines of anchors and aliases stand for a structure exponentially larger than
the text. Nothing is copied at load time; the cost arrives when a caller walks,
stringifies or serializes the value. A frontmatter block or an interchange
member is third-party text, so every read of one goes through :func:`load`,
which counts each alias as the size of the subtree it stands for and refuses
the document once the total passes ``max_nodes``. A recursive alias (one that
names an anchor still being built) is refused outright, because it stands for
an infinite structure. Plain YAML with no aliases costs one count per node,
so the budget only bites on expansion.

Two other costs are bounded the same way (security review 2026-10-27-1, G17,
G18, G33). The composer recurses once per level, so nesting past
``_MAX_DEPTH`` is refused before the interpreter's recursion limit is met. A
plain integer or float scalar longer than ``_MAX_NUMBER_CHARS`` is refused
before it is built, since a YAML 1.1 base-60 number costs time quadratic in
its length and a long decimal one meets the interpreter's digit limit. Any
``ValueError``, ``TypeError``, ``OverflowError`` or ``RecursionError`` the
constructor still raises becomes :class:`YAMLExpansionError`, so every caller's
``yaml.YAMLError`` branch reads the document as malformed.
"""

from __future__ import annotations

from typing import Any

import yaml


class YAMLExpansionError(yaml.YAMLError):
    """A document whose aliases expand past the node budget, or recurse, or that cannot be built.

    A document nested past the depth bound, holding a number past the length
    bound, or whose construction fails with a non-YAML error is refused with
    this error too.
    """


# Levels of nesting the composer descends before it refuses the document. The
# composer spends a few interpreter frames per level, so this stays well under
# the default recursion limit of 1000 with the caller's own frames on top.
_MAX_DEPTH = 100

# Characters a plain integer or float scalar may hold. Far past any count or
# id a frontmatter or interchange member states, and short enough that a
# base-60 scalar costs nothing to build.
_MAX_NUMBER_CHARS = 128

_NUMBER_TAGS = frozenset({"tag:yaml.org,2002:int", "tag:yaml.org,2002:float"})


class YAMLItemLimitError(yaml.YAMLError):
    """A top-level sequence holding more items than :func:`load_sequence` allows."""


class _TopLevelMapping(Exception):
    """Raised inside :func:`load_sequence` on a top-level mapping, before it is composed."""


_SEQ_TAG = "tag:yaml.org,2002:seq"


class _InsertLog(dict[yaml.Node, Any]):
    """``constructed_objects`` that records which nodes each item built.

    ``load_sequence`` forgets an item's non-anchored objects once the item is
    built. Rebuilding the dict from every surviving entry after each item cost
    time quadratic in the anchored items; the log names only the new ones.
    """

    def __init__(self) -> None:
        super().__init__()
        self.added: list[yaml.Node] = []

    def __setitem__(self, key: yaml.Node, value: Any) -> None:
        if key not in self:
            self.added.append(key)
        super().__setitem__(key, value)


class _BoundedSafeLoader(yaml.SafeLoader):
    """``SafeLoader`` that charges each composed node, and each alias in full."""

    def __init__(self, stream: str, max_nodes: int, max_items: int | None = None) -> None:
        super().__init__(stream)
        self._remaining = max_nodes
        # Every anchored node, kept for the loader's life so its id stays its
        # own: the expanded size and depth of an anchored node are cached by
        # id, and ``load_sequence`` releases other nodes as it goes, so a
        # cache entry for one of them could otherwise name a later node at
        # the same address (security review 2026-10-30, AB16).
        self._anchored: dict[int, yaml.Node] = {}
        self._sizes: dict[int, int] = {}
        self._depths: dict[int, int] = {}
        # Ids of anchored nodes whose children ``load_sequence`` released
        # once they were built.
        self._released: set[int] = set()
        self._open_anchors: set[str] = set()
        self._depth = 0
        # Set by ``load_sequence``: the top-level sequence's items are built
        # as each is composed, and its nodes released, instead of the whole
        # document's nodes being held until construction ends.
        self._max_items = max_items
        self._items: list[Any] | None = None
        self._streamed: yaml.SequenceNode | None = None
        # Anchored nodes composed since the last item was built.
        self._fresh_anchors: list[yaml.Node] = []
        if max_items is not None:
            self.constructed_objects = _InsertLog()

    def _charge(self, count: int) -> None:
        self._remaining -= count
        if self._remaining < 0:
            raise YAMLExpansionError(
                "YAML document expands past its node budget through anchors and aliases"
            )

    def _size(self, node: yaml.Node, memo: dict[int, int] | None = None) -> int:
        """Nodes ``node`` stands for once every alias inside it is expanded.

        An anchored node's figure is kept for later aliases; any other node's
        lives in ``memo`` for this one walk only, while the node is alive.
        """
        key = id(node)
        cached = self._sizes.get(key)
        if cached is None and memo is not None:
            cached = memo.get(key)
        if cached is not None:
            return cached
        memo = {} if memo is None else memo
        total = 1
        if isinstance(node, yaml.SequenceNode):
            total += sum(self._size(child, memo) for child in node.value)
        elif isinstance(node, yaml.MappingNode):
            total += sum(self._size(k, memo) + self._size(v, memo) for k, v in node.value)
        (self._sizes if key in self._anchored else memo)[key] = total
        return total

    def _levels(self, node: yaml.Node, memo: dict[int, int] | None = None) -> int:
        """Levels ``node`` nests once every alias inside it is expanded, cached as ``_size``."""
        key = id(node)
        cached = self._depths.get(key)
        if cached is None and memo is not None:
            cached = memo.get(key)
        if cached is not None:
            return cached
        memo = {} if memo is None else memo
        children: list[yaml.Node] = []
        if isinstance(node, yaml.SequenceNode):
            children = list(node.value)
        elif isinstance(node, yaml.MappingNode):
            children = [child for pair in node.value for child in pair]
        total = 1 + max((self._levels(child, memo) for child in children), default=0)
        (self._depths if key in self._anchored else memo)[key] = total
        return total

    def compose_node(self, parent: yaml.Node | None, index: Any) -> yaml.Node | None:
        if self.check_event(yaml.AliasEvent):  # type: ignore[no-untyped-call]
            anchor = self.peek_event().anchor  # type: ignore[no-untyped-call]
            if anchor in self._open_anchors:
                raise YAMLExpansionError(f"YAML alias *{anchor} refers to itself")
            if anchor in self.anchors:
                target = self.anchors[anchor]
                self._charge(self._size(target))
                # An alias stands for its anchor's whole subtree at this
                # level, so the depth bound counts its levels too: chained
                # anchors otherwise build a value far deeper than the
                # composer ever descends (security review 2026-10-28-1, O64).
                if self._depth + self._levels(target) > _MAX_DEPTH:
                    raise YAMLExpansionError(
                        f"YAML document nests deeper than {_MAX_DEPTH} levels through an alias"
                    )
            return super().compose_node(parent, index)
        if (
            parent is None
            and self._max_items is not None
            and self.check_event(yaml.MappingStartEvent)  # type: ignore[no-untyped-call]
        ):
            # A mapping would be composed whole before the caller could see
            # it is not the sequence asked for (security review 2026-10-30,
            # AB16).
            raise _TopLevelMapping
        event = self.peek_event()  # type: ignore[no-untyped-call]
        anchor = getattr(event, "anchor", None)
        if anchor is not None:
            self._open_anchors.add(anchor)
        self._depth += 1
        if self._depth > _MAX_DEPTH:
            raise YAMLExpansionError(f"YAML document nests deeper than {_MAX_DEPTH} levels")
        try:
            node = super().compose_node(parent, index)
        finally:
            self._depth -= 1
            if anchor is not None:
                self._open_anchors.discard(anchor)
        if (
            isinstance(node, yaml.ScalarNode)
            and node.tag in _NUMBER_TAGS
            and len(node.value) > _MAX_NUMBER_CHARS
        ):
            raise YAMLExpansionError(
                f"YAML number scalar longer than {_MAX_NUMBER_CHARS} characters"
            )
        self._charge(1)
        if anchor is not None and node is not None:
            self._anchored[id(node)] = node
            self._fresh_anchors.append(node)
        return node

    def compose_sequence_node(self, anchor: Any) -> yaml.SequenceNode:
        if self._max_items is None or self._depth != 1:
            return super().compose_sequence_node(anchor)
        start_event = self.get_event()  # type: ignore[no-untyped-call]
        tag = start_event.tag
        if tag is None or tag == "!":
            tag = self.resolve(yaml.SequenceNode, None, start_event.implicit)  # type: ignore[no-untyped-call]
        node = yaml.SequenceNode(
            tag, [], start_event.start_mark, None, flow_style=start_event.flow_style
        )
        if anchor is not None:
            self.anchors[anchor] = node
        items: list[Any] = []
        while not self.check_event(yaml.SequenceEndEvent):  # type: ignore[no-untyped-call]
            if len(items) >= self._max_items:
                raise YAMLItemLimitError(
                    f"YAML top-level sequence holds more than {self._max_items} items"
                )
            child = self.compose_node(node, len(items))
            items.append(self._construct_item(child))
        node.end_mark = self.get_event().end_mark  # type: ignore[no-untyped-call]
        self._items = items
        self._streamed = node
        return node

    def _construct_item(self, node: yaml.Node | None) -> Any:
        """Build one top-level item, keeping only what a later alias can name.

        ``construct_document``'s own loop, except that the objects built for
        anchored nodes survive it, so an alias in a later item resolves to the
        same object as a whole-document load would give.
        """
        data = self.construct_object(node)
        while self.state_generators:
            generators = self.state_generators
            self.state_generators = []
            for generator in generators:
                for _ in generator:
                    pass
        fresh = self._fresh_anchors
        self._fresh_anchors = []
        built = self.constructed_objects
        if isinstance(built, _InsertLog):
            anchored = {id(target) for target in fresh}
            for key in built.added:
                if id(key) not in anchored:
                    del built[key]
            built.added = []
        else:
            names = {id(target) for target in self.anchors.values()}
            self.constructed_objects = {
                key: value for key, value in built.items() if id(key) in names
            }
        self.recursive_objects = {}
        self.deep_construct = False
        self._release_anchored(fresh)
        return data

    def _release_anchored(self, fresh: list[yaml.Node]) -> None:
        """Drop the children of each built anchored node, keeping its size, depth and object.

        A later alias needs only those three: the composer charges the cached
        size and depth, and the constructor returns the built object. Holding
        the children kept every anchored item's whole node tree alive, tens of
        bytes for each byte of input (security review 2026-10-30, AB16). A
        node that was never built (a merge key's value is folded, not built)
        keeps its children, since a later alias still has to build it.
        """
        for target in fresh:
            if (
                id(target) in self._released
                or not isinstance(target, (yaml.SequenceNode, yaml.MappingNode))
                or target not in self.constructed_objects
            ):
                continue
            self._size(target)
            self._levels(target)
            target.value = []
            target.start_mark = target.end_mark = None
            self._released.add(id(target))

    def _merge_pairs(self, node: yaml.Node) -> list[tuple[yaml.Node, yaml.Node]]:
        """Key and value nodes standing for a released mapping's built entries.

        Each placeholder is registered in ``constructed_objects`` as the
        object the mapping built for it, so a ``<<`` merge of an earlier
        item's anchor folds in the same keys and the same value objects a
        whole-document load would.
        """
        built = self.constructed_objects[node]
        if not isinstance(built, dict):
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "expected a mapping for merging",
                node.start_mark,
            )
        pairs: list[tuple[yaml.Node, yaml.Node]] = []
        for key, value in built.items():
            key_node = yaml.ScalarNode("tag:yaml.org,2002:str", "")
            value_node = yaml.ScalarNode("tag:yaml.org,2002:str", "")
            self.constructed_objects[key_node] = key
            self.constructed_objects[value_node] = value
            pairs.append((key_node, value_node))
        return pairs

    def _released_merge_sources(
        self, node: yaml.SequenceNode
    ) -> list[list[tuple[yaml.Node, yaml.Node]]]:
        """The per-mapping merge pairs of a released sequence of mappings."""
        built = self.constructed_objects[node]
        sources: list[list[tuple[yaml.Node, yaml.Node]]] = []
        for item in built:
            holder = yaml.MappingNode("tag:yaml.org,2002:map", [], node.start_mark)
            self.constructed_objects[holder] = item
            sources.append(self._merge_pairs(holder))
        return sources

    def flatten_mapping(self, node: yaml.MappingNode) -> None:
        """Fold a mapping's ``<<`` merge keys into it, in time linear in its entries.

        PyYAML's own deletes each merge key from the node list in place, which
        shifts the rest of the list every time: a mapping of N merge keys then
        costs time quadratic in N, which the node budget does not see
        (security review 2026-10-28-1, O24). This rebuilds the list once, with
        the same result and the same errors.
        """
        merge: list[tuple[yaml.Node, yaml.Node]] = []
        kept: list[tuple[yaml.Node, yaml.Node]] = []
        for key_node, value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                if id(value_node) in self._released:
                    if isinstance(value_node, yaml.MappingNode):
                        merge.extend(self._merge_pairs(value_node))
                    else:
                        for value in reversed(self._released_merge_sources(value_node)):
                            merge.extend(value)
                elif isinstance(value_node, yaml.MappingNode):
                    self.flatten_mapping(value_node)
                    merge.extend(value_node.value)
                elif isinstance(value_node, yaml.SequenceNode):
                    submerge: list[list[tuple[yaml.Node, yaml.Node]]] = []
                    for subnode in value_node.value:
                        if not isinstance(subnode, yaml.MappingNode):
                            raise yaml.constructor.ConstructorError(
                                "while constructing a mapping",
                                node.start_mark,
                                f"expected a mapping for merging, but found {subnode.id}",
                                subnode.start_mark,
                            )
                        if id(subnode) in self._released:
                            submerge.append(self._merge_pairs(subnode))
                            continue
                        self.flatten_mapping(subnode)
                        submerge.append(subnode.value)
                    for value in reversed(submerge):
                        merge.extend(value)
                else:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        "expected a mapping or list of mappings for merging, "
                        f"but found {value_node.id}",
                        value_node.start_mark,
                    )
            else:
                if key_node.tag == "tag:yaml.org,2002:value":
                    key_node.tag = "tag:yaml.org,2002:str"
                kept.append((key_node, value_node))
        if merge:
            node.value = merge + kept
        elif len(kept) != len(node.value):
            node.value = kept


# Nodes allowed per character of input, plus a fixed slack. Plain YAML writes
# fewer than two nodes per character (an implicit null value is the cheapest),
# so this budget never refuses a document without aliases.
_NODES_PER_CHAR = 2
_SLACK_NODES = 1024


def load(text: str, *, max_nodes: int | None = None) -> Any:
    """Parse ``text`` as ``yaml.safe_load`` does, within ``max_nodes`` expanded nodes.

    The default budget is linear in the input (twice its length plus a small slack),
    so the expanded value costs no more to walk than the text did to read.
    Raises :class:`YAMLExpansionError` (a :class:`yaml.YAMLError`) when the
    document's aliases expand past ``max_nodes`` or recurse, when it nests past
    the depth bound or holds a number past the length bound, or when building
    a value fails with a ``ValueError``, ``TypeError``, ``OverflowError`` or
    ``RecursionError``. A caller that already treats a ``YAMLError`` as
    malformed input therefore refuses every one of them the same way.
    """
    loader = _BoundedSafeLoader(text, _budget(text, max_nodes))
    try:
        return loader.get_single_data()
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise YAMLExpansionError(f"YAML document cannot be built: {type(exc).__name__}") from None
    finally:
        loader.dispose()  # type: ignore[no-untyped-call]


def load_sequence(text: str, *, max_items: int, max_nodes: int | None = None) -> Any:
    """Parse ``text`` as :func:`load` does, building a top-level sequence one item at a time.

    :func:`load` composes the whole document before it builds any of it, and
    the pure-Python composer's nodes cost tens of bytes for every byte of
    input, so a large document's peak memory is an order of magnitude past
    the value it builds (security review 2026-10-28-2, AA16). Here each item of a
    top-level sequence is built as soon as it is composed and its nodes are
    released, so the peak stays near the built value. A sequence of more than
    ``max_items`` items is refused with :class:`YAMLItemLimitError` (a
    :class:`yaml.YAMLError`) while it is being read. A document whose top
    level is a mapping is not composed at all: it reads as an empty ``dict``,
    so a caller that asked for a sequence refuses it by its type without the
    whole mapping having been composed first (security review 2026-10-30,
    AB16). A scalar top level loads as :func:`load` would load it. A tagged
    top-level sequence is refused as malformed. Every refusal :func:`load`
    makes is made here too.
    """
    loader = _BoundedSafeLoader(text, _budget(text, max_nodes), max_items=max_items)
    try:
        node = loader.get_single_node()
        if node is None:
            return None
        streamed = loader._streamed
        if streamed is not None and node is streamed:
            if streamed.tag != _SEQ_TAG:
                raise yaml.constructor.ConstructorError(
                    None, None, f"top-level sequence tagged {streamed.tag!r}", streamed.start_mark
                )
            return loader._items
        return loader.construct_document(node)  # type: ignore[no-untyped-call]
    except _TopLevelMapping:
        return {}
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise YAMLExpansionError(f"YAML document cannot be built: {type(exc).__name__}") from None
    finally:
        loader.dispose()  # type: ignore[no-untyped-call]


def _budget(text: str, max_nodes: int | None) -> int:
    return _NODES_PER_CHAR * len(text) + _SLACK_NODES if max_nodes is None else max_nodes
