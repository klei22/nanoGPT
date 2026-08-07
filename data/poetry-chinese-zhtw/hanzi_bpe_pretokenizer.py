#!/usr/bin/env python3
"""Reversible IDS pre-tokenization for Han-character BPE experiments.

The codec replaces a catalogue character with its recursively expanded
Ideographic Description Sequence (IDS) only when the expansion is structurally
valid and has a unique inverse.  Everything else passes through byte-for-byte
after UTF-8 decoding/encoding.  This is *not* Simplified/Traditional Chinese
conversion: ``--region`` only chooses the G, T, or H glyph decomposition.

The prefix form is self-delimiting because every IDS operator has fixed arity.
Literal IDS/control characters in the source corpus are quoted with ESCAPE, so
ordinary text which happens to contain an IDS remains unambiguous.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import codecs
import contextlib
import dataclasses
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Mapping, Sequence


TOOL_NAME = "hanzi_bpe_pretokenizer"
TOOL_VERSION = "1.0.0"

# The 16 characters in the IDC block plus U+31EF, the seventeenth IDC.
IDS_ARITY: dict[str, int] = {
    "⿰": 2, "⿱": 2, "⿲": 3, "⿳": 3,
    "⿴": 2, "⿵": 2, "⿶": 2, "⿷": 2,
    "⿸": 2, "⿹": 2, "⿺": 2, "⿻": 2,
    "⿼": 2, "⿽": 2, "⿾": 1, "⿿": 1,
    "㇯": 2,  # U+31EF IDEOGRAPHIC DESCRIPTION CHARACTER SUBTRACTION
}

# BabelStone uses U+303E before any subtree whose exact component form varies.
# Unicode calls it the Ideographic Variation Indicator (IVI), not an IDC.  We
# model it as a unary prefix modifier so nested BabelStone sequences are parsed
# and streamed without losing information.
VARIATION_INDICATOR = "〾"
ESCAPE = "\ue000"

FAMILY_OPERATORS: dict[str, frozenset[str]] = {
    "horizontal": frozenset(("⿰", "⿲")),
    "vertical": frozenset(("⿱", "⿳")),
    "surround": frozenset(("⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺", "⿼", "⿽")),
    "overlay": frozenset(("⿻",)),
    "reflection": frozenset(("⿾",)),
    "rotation": frozenset(("⿿",)),
    "subtraction": frozenset(("㇯",)),
    "variation": frozenset((VARIATION_INDICATOR,)),
}
ALL_FAMILIES = tuple(name for name in FAMILY_OPERATORS if name != "variation")
ALL_FAMILIES_WITH_VARIATION = tuple(FAMILY_OPERATORS)
PARSE_ARITY = {**IDS_ARITY, VARIATION_INDICATOR: 1}
STRUCTURAL_CONTROLS = frozenset(PARSE_ARITY)
ESCAPED_LITERALS = STRUCTURAL_CONTROLS | {ESCAPE}

CODEPOINT_RE = re.compile(r"^\s*U\+([0-9A-Fa-f]{4,6})(?:\s|$)")


class CatalogueError(ValueError):
    """Malformed catalogue/configuration error."""


class DecodeError(ValueError):
    """Input is not a valid stream produced by this codec."""


class NeedMoreData(Exception):
    """Internal signal used by the incremental prefix parser."""


class ExpansionError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def is_variation_selector(ch: str) -> bool:
    cp = ord(ch)
    return 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF


@dataclass(frozen=True)
class IDSNode:
    token: str
    children: tuple["IDSNode", ...] = ()

    @property
    def is_operator(self) -> bool:
        return self.token in PARSE_ARITY

    def serialize(self) -> str:
        return self.token + "".join(child.serialize() for child in self.children)

    @property
    def size(self) -> int:
        return 1 + sum(child.size for child in self.children)

    @property
    def depth(self) -> int:
        return 1 if not self.children else 1 + max(child.depth for child in self.children)

    def tokens(self) -> Iterator[str]:
        yield self.token
        for child in self.children:
            yield from child.tokens()


def _lex_ids(sequence: str) -> list[str]:
    """Tokenize BabelStone/CJKVI IDS syntax, retaining entities and VS atoms."""
    tokens: list[str] = []
    i = 0
    while i < len(sequence):
        ch = sequence[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "{":
            end = sequence.find("}", i + 1)
            if end < 0:
                raise CatalogueError(f"unclosed {{...}} component in {sequence!r}")
            tokens.append(sequence[i:end + 1])
            i = end + 1
            continue
        if ch == "&":
            end = sequence.find(";", i + 1)
            if end < 0:
                raise CatalogueError(f"unclosed &...; entity in {sequence!r}")
            tokens.append(sequence[i:end + 1])
            i = end + 1
            continue
        if ch in PARSE_ARITY:
            tokens.append(ch)
            i += 1
            continue
        atom = ch
        i += 1
        while i < len(sequence) and is_variation_selector(sequence[i]):
            atom += sequence[i]
            i += 1
        tokens.append(atom)
    if not tokens:
        raise CatalogueError("empty IDS")
    return tokens


def parse_ids(sequence: str, *, max_depth: int = 64, max_nodes: int = 512) -> IDSNode:
    """Parse one complete fixed-arity IDS prefix expression."""
    tokens = _lex_ids(sequence)
    index = 0
    seen = 0

    def rec(depth: int) -> IDSNode:
        nonlocal index, seen
        if depth > max_depth:
            raise CatalogueError(f"IDS exceeds depth cap {max_depth}")
        if index >= len(tokens):
            raise CatalogueError(f"IDS ended before all operands were present: {sequence!r}")
        seen += 1
        if seen > max_nodes:
            raise CatalogueError(f"IDS exceeds node cap {max_nodes}")
        token = tokens[index]
        index += 1
        arity = PARSE_ARITY.get(token, 0)
        return IDSNode(token, tuple(rec(depth + 1) for _ in range(arity)))

    root = rec(1)
    if index != len(tokens):
        raise CatalogueError(f"IDS has {len(tokens) - index} trailing token(s): {sequence!r}")
    return root


def _read_stream_atom(text: str, position: int, *, final: bool) -> tuple[str, int]:
    """Read one leaf token from a canonical serialized IDS."""
    if position >= len(text):
        if final:
            raise DecodeError("truncated IDS operand")
        raise NeedMoreData
    ch = text[position]
    if ch == "{":
        end = text.find("}", position + 1)
        if end < 0:
            if final:
                raise DecodeError("unterminated {...} IDS component")
            raise NeedMoreData
        return text[position:end + 1], end + 1
    if ch == "&":
        end = text.find(";", position + 1)
        if end < 0:
            if final:
                raise DecodeError("unterminated &...; IDS entity")
            raise NeedMoreData
        return text[position:end + 1], end + 1
    end = position + 1
    while end < len(text) and is_variation_selector(text[end]):
        end += 1
    # A selector in the next chunk could still belong to this atom.
    if end == len(text) and not final:
        raise NeedMoreData
    return text[position:end], end


def parse_prefix(
    text: str,
    start: int = 0,
    *,
    final: bool = True,
    max_depth: int = 64,
    max_nodes: int = 512,
) -> tuple[IDSNode, int]:
    """Parse exactly one IDS expression at ``start`` and return its end offset.

    Trailing text is intentionally not consumed, which makes this suitable for
    an incremental decoder.  ``NeedMoreData`` is raised only when ``final`` is
    false and the expression may be completed by a later chunk.
    """
    seen = 0

    def rec(position: int, depth: int) -> tuple[IDSNode, int]:
        nonlocal seen
        if depth > max_depth:
            raise DecodeError(f"IDS exceeds decoder depth cap {max_depth}")
        if position >= len(text):
            if final:
                raise DecodeError("truncated IDS expression")
            raise NeedMoreData
        seen += 1
        if seen > max_nodes:
            raise DecodeError(f"IDS exceeds decoder node cap {max_nodes}")
        token = text[position]
        arity = PARSE_ARITY.get(token)
        if arity is None:
            atom, end = _read_stream_atom(text, position, final=final)
            return IDSNode(atom), end
        position += 1
        children: list[IDSNode] = []
        for _ in range(arity):
            child, position = rec(position, depth + 1)
            children.append(child)
        return IDSNode(token, tuple(children)), position

    return rec(start, 1)


def parse_ids_field(field_text: str) -> tuple[str, str] | None:
    """Return ``(sequence, tags)`` for BabelStone or CJKVI fields."""
    field_text = field_text.strip()
    if not field_text or field_text.startswith("*"):
        return None
    babel = re.fullmatch(r"\^(.*?)\$(?:\((.*?)\))?", field_text)
    if babel:
        return babel.group(1), babel.group(2) or ""
    cjkvi = re.fullmatch(r"(.*?)(?:\[([A-Za-z][^\]]*)\])?", field_text)
    if not cjkvi:
        return None
    sequence, tags = cjkvi.group(1).strip(), cjkvi.group(2) or ""
    if not sequence or sequence.startswith("U+"):
        return None
    return sequence, tags


@dataclass(frozen=True)
class Alternative:
    sequence: str
    tags: str
    node: IDSNode
    order: int

    @property
    def regions(self) -> frozenset[str]:
        return frozenset(re.findall(r"[A-Z]", self.tags.upper()))


@dataclass
class LoadedIDS:
    alternatives: dict[str, list[Alternative]] = field(default_factory=lambda: defaultdict(list))
    stats: Counter[str] = field(default_factory=Counter)
    source_sha256: str = ""
    source_bytes: int = 0


def load_ids_catalogue(path: str | Path, *, max_depth: int = 64, max_nodes: int = 512) -> LoadedIDS:
    """Load a BabelStone-style tab-separated IDS catalogue."""
    source = Path(path)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CatalogueError(f"IDS catalogue is not valid UTF-8: {exc}") from exc
    loaded = LoadedIDS(source_sha256=digest, source_bytes=len(raw))
    order = 0
    for line_no, line in enumerate(text.splitlines(), 1):
        loaded.stats["lines"] += 1
        if not line.strip() or line.startswith("#"):
            loaded.stats["ignored_lines"] += 1
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            loaded.stats["malformed_lines"] += 1
            continue
        match = CODEPOINT_RE.match(fields[0])
        char = fields[1]
        if not match or len(char) != 1:
            loaded.stats["malformed_lines"] += 1
            continue
        cp = int(match.group(1), 16)
        if cp > 0x10FFFF or 0xD800 <= cp <= 0xDFFF or chr(cp) != char:
            loaded.stats["codepoint_mismatch"] += 1
            continue
        loaded.stats["records"] += 1
        for field_text in fields[2:]:
            parsed = parse_ids_field(field_text)
            if parsed is None:
                loaded.stats["ignored_fields"] += 1
                continue
            sequence, tags = parsed
            loaded.stats["alternatives_seen"] += 1
            try:
                node = parse_ids(sequence, max_depth=max_depth, max_nodes=max_nodes)
            except CatalogueError:
                loaded.stats["malformed_alternatives"] += 1
                continue
            order += 1
            loaded.alternatives[char].append(Alternative(sequence, tags, node, order))
            loaded.stats["valid_alternatives"] += 1
    return loaded


def parse_families(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ALL_FAMILIES
    raw = [value] if isinstance(value, str) else list(value)
    names: list[str] = []
    aliases = {
        "lr": ("horizontal",), "tb": ("vertical",),
        "axis": ("horizontal", "vertical"),
        "enclosure": ("surround",),
        "unary": ("reflection", "rotation"),
        "transforms": ("reflection", "rotation", "subtraction"),
        "all": ALL_FAMILIES,
        "all-with-variation": ALL_FAMILIES_WITH_VARIATION,
    }
    for item in raw:
        for name in item.split(","):
            name = name.strip().lower()
            if not name:
                continue
            expanded = aliases.get(name, (name,))
            for family in expanded:
                if family not in FAMILY_OPERATORS:
                    choices = ", ".join(ALL_FAMILIES_WITH_VARIATION)
                    raise CatalogueError(f"unknown operator family {family!r}; choose from {choices}")
                if family not in names:
                    names.append(family)
    if not names:
        raise CatalogueError("at least one operator family is required")
    return tuple(names)


def operators_for_families(families: Sequence[str]) -> frozenset[str]:
    result: set[str] = set()
    for family in families:
        result.update(FAMILY_OPERATORS[family])
    return frozenset(result)


def _is_operator_rooted(node: IDSNode) -> bool:
    while node.token == VARIATION_INDICATOR and node.children:
        node = node.children[0]
    return node.token in IDS_ARITY


def _is_supported(node: IDSNode, allowed: frozenset[str]) -> bool:
    return all(token not in PARSE_ARITY or token in allowed for token in node.tokens())


def _single_scalar_atom(node: IDSNode) -> str | None:
    return node.token if not node.children and len(node.token) == 1 else None


def _percentile(sorted_values: Sequence[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = max(0, min(len(sorted_values) - 1, int((len(sorted_values) - 1) * fraction + 0.999999)))
    return sorted_values[index]


@dataclass
class Catalogue:
    forward: dict[str, str]
    reverse: dict[str, str]
    audit: dict[str, object]
    max_depth: int = 64
    max_nodes: int = 512
    escape: str = ESCAPE

    def codec(self) -> "HanziBPEPreTokenizer":
        return HanziBPEPreTokenizer(
            self.forward,
            self.reverse,
            max_depth=self.max_depth,
            max_nodes=self.max_nodes,
            escape=self.escape,
        )


def build_catalogue(
    ids_path: str | Path,
    *,
    region: str = "T",
    families: str | Sequence[str] | None = None,
    max_depth: int = 64,
    max_nodes: int = 512,
    escape: str = ESCAPE,
) -> Catalogue:
    """Build collision-free forward/reverse maps for a chosen glyph region."""
    region = region.upper()
    if region not in {"G", "T", "H"}:
        raise CatalogueError("region must be one of G, T, or H")
    if max_depth < 1 or max_nodes < 1:
        raise CatalogueError("max_depth and max_nodes must both be positive")
    if len(escape) != 1 or escape in STRUCTURAL_CONTROLS:
        raise CatalogueError("escape must be one non-structural Unicode scalar")
    family_names = parse_families(families)
    allowed = operators_for_families(family_names)
    loaded = load_ids_catalogue(ids_path, max_depth=max_depth, max_nodes=max_nodes)
    reasons: Counter[str] = Counter()
    selected: dict[str, IDSNode] = {}

    def candidate_score(alt: Alternative) -> tuple[int, int, int, int, str]:
        regions = alt.regions
        regional = 0 if region in regions else 1 if not regions else 3 if regions == {"X"} else 2
        supported = 0 if _is_supported(alt.node, allowed) else 1
        # Prefer exact structural information to IVI-marked approximations.
        approximate = sum(token == VARIATION_INDICATOR for token in alt.node.tokens())
        return regional, supported, approximate, alt.order, alt.node.serialize()

    for char, alternatives in loaded.alternatives.items():
        if not alternatives:
            reasons["missing_valid_alternative"] += 1
            continue
        selected[char] = min(alternatives, key=candidate_score).node

    # Records with no successfully parsed field are known missing entries.
    reasons["missing_valid_alternative"] += loaded.stats["records"] - len(loaded.alternatives)

    memo: dict[str, IDSNode] = {}
    failures: dict[str, str] = {}

    def direct_eligibility(char: str) -> str | None:
        node = selected.get(char)
        if node is None:
            return "missing"
        if not node.children and node.token == char:
            return "self"
        if not _is_operator_rooted(node):
            return "not_operator_rooted"
        if not _is_supported(node, allowed):
            return "unsupported_operator"
        return None

    def expand_char(char: str, stack: tuple[str, ...]) -> IDSNode:
        if char in memo:
            return memo[char]
        if char in failures:
            raise ExpansionError(failures[char])
        if char in stack:
            raise ExpansionError("cycle")
        eligibility = direct_eligibility(char)
        if eligibility is not None:
            raise ExpansionError(eligibility)
        root = selected[char]

        def expand_node(node: IDSNode) -> IDSNode:
            atom = _single_scalar_atom(node)
            if atom is not None and atom not in stack and atom != char:
                child_status = direct_eligibility(atom)
                if child_status is None:
                    return expand_char(atom, stack + (char,))
                # Missing/self/unsupported component descriptions become leaves.
                return node
            if atom == char or atom in stack:
                # A mapped back-edge is a genuine recursive cycle; an unmapped
                # occurrence is merely an atomic component.
                if direct_eligibility(atom) is None:
                    raise ExpansionError("cycle")
                return node
            if not node.children:
                return node
            children = tuple(expand_node(child) for child in node.children)
            result = IDSNode(node.token, children)
            if result.depth > max_depth:
                raise ExpansionError("expansion_depth_cap")
            if result.size > max_nodes:
                raise ExpansionError("expansion_node_cap")
            return result

        try:
            result = expand_node(root)
            if result.depth > max_depth:
                raise ExpansionError("expansion_depth_cap")
            if result.size > max_nodes:
                raise ExpansionError("expansion_node_cap")
        except ExpansionError as exc:
            failures[char] = exc.reason
            raise
        memo[char] = result
        return result

    candidates: dict[str, str] = {}
    candidate_nodes: dict[str, IDSNode] = {}
    for char in sorted(selected, key=ord):
        immediate = direct_eligibility(char)
        if immediate is not None:
            reasons[immediate] += 1
            continue
        try:
            node = expand_char(char, ())
        except ExpansionError as exc:
            reasons[exc.reason] += 1
            continue
        sequence = node.serialize()
        if sequence == char or not _is_operator_rooted(node):
            reasons["non_transforming_expansion"] += 1
            continue
        candidates[char] = sequence
        candidate_nodes[char] = node

    groups: dict[str, list[str]] = defaultdict(list)
    for char, sequence in candidates.items():
        groups[sequence].append(char)
    forward: dict[str, str] = {}
    reverse: dict[str, str] = {}
    collision_examples: list[dict[str, object]] = []
    for sequence, chars in groups.items():
        if len(chars) != 1:
            reasons["collision"] += len(chars)
            if len(collision_examples) < 25:
                collision_examples.append({"sequence": sequence, "characters": chars})
            continue
        char = chars[0]
        forward[char] = sequence
        reverse[sequence] = char

    used_tokens: set[str] = set()
    atomic_tokens: set[str] = set()
    operator_counts: Counter[str] = Counter()
    lengths: list[int] = []
    for char in forward:
        node = candidate_nodes[char]
        tokens = list(node.tokens())
        lengths.append(len(tokens))
        for token in tokens:
            used_tokens.add(token)
            if token in PARSE_ARITY:
                operator_counts[token] += 1
            else:
                atomic_tokens.add(token)
    lengths.sort()
    literal_catalogue_tokens = set(loaded.alternatives) - set(forward)
    safe_spanning_tokens = used_tokens | literal_catalogue_tokens
    audit: dict[str, object] = {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "ids_source": str(Path(ids_path)),
        "ids_sha256": loaded.source_sha256,
        "ids_bytes": loaded.source_bytes,
        "region": region,
        "note": "Glyph-region selection only; no Simplified/Traditional conversion is performed.",
        "families": list(family_names),
        "allowed_operators": sorted(allowed, key=ord),
        "escape": f"U+{ord(escape):04X}",
        "caps": {"max_depth": max_depth, "max_nodes": max_nodes},
        "source_stats": dict(sorted(loaded.stats.items())),
        "source_records": loaded.stats["records"],
        "catalogue_characters": len(loaded.alternatives),
        "selected_characters": len(selected),
        "precollision_candidates": len(candidates),
        "transformed_characters": len(forward),
        "unchanged_catalogue_characters": len(loaded.alternatives) - len(forward),
        "unchanged_source_records": loaded.stats["records"] - len(forward),
        "collision_groups": sum(1 for chars in groups.values() if len(chars) > 1),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "collision_examples": collision_examples,
        "factor_alphabet": {
            "atomic_tokens": len(atomic_tokens),
            "used_structural_tokens": len(used_tokens & STRUCTURAL_CONTROLS),
            "used_tokens_total": len(used_tokens),
            "with_escape": len(used_tokens | {escape}),
        },
        "safe_spanning_alphabet": {
            "literal_pass_through_tokens": len(literal_catalogue_tokens),
            "used_tokens_without_escape": len(safe_spanning_tokens),
            "with_escape": len(safe_spanning_tokens | {escape}),
            "note": (
                "Union of factor/operator tokens and unchanged catalogue scalars; "
                "this is the lossless logical alphabet for the complete loaded catalogue."
            ),
        },
        "expanded_length": {
            "mean": (sum(lengths) / len(lengths)) if lengths else 0.0,
            "median": _percentile(lengths, 0.5),
            "p95": _percentile(lengths, 0.95),
            "maximum": max(lengths, default=0),
        },
        "operator_occurrences": dict(sorted(operator_counts.items(), key=lambda item: ord(item[0]))),
    }
    return Catalogue(forward, reverse, audit, max_depth=max_depth, max_nodes=max_nodes, escape=escape)


class HanziBPEPreTokenizer:
    """Bidirectional, mixed-Unicode-safe corpus pre-tokenizer."""

    def __init__(
        self,
        forward: Mapping[str, str],
        reverse: Mapping[str, str] | None = None,
        *,
        max_depth: int = 64,
        max_nodes: int = 512,
        escape: str = ESCAPE,
    ) -> None:
        if len(escape) != 1 or escape in STRUCTURAL_CONTROLS:
            raise CatalogueError("escape must be one non-structural Unicode scalar")
        self.forward = dict(forward)
        self.reverse = dict(reverse or {value: key for key, value in forward.items()})
        if len(self.reverse) != len(self.forward):
            raise CatalogueError("forward map does not have a unique inverse")
        if any(self.reverse.get(sequence) != char for char, sequence in self.forward.items()):
            raise CatalogueError("forward and reverse maps are not exact inverses")
        if any(len(char) != 1 for char in self.forward):
            raise CatalogueError("forward-map keys must be single Unicode scalars")
        if any(not sequence or sequence[0] not in STRUCTURAL_CONTROLS for sequence in self.reverse):
            raise CatalogueError("all transformed values must be operator/IVI-rooted IDS strings")
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.escape = escape
        self.escaped_literals = STRUCTURAL_CONTROLS | {escape}

    def encode(self, text: str) -> str:
        return "".join(self.encode_chunks((text,)))

    def encode_chunks(self, chunks: Iterable[str]) -> Iterator[str]:
        for chunk in chunks:
            output: list[str] = []
            for char in chunk:
                replacement = self.forward.get(char)
                if replacement is not None:
                    output.append(replacement)
                elif char in self.escaped_literals:
                    output.extend((self.escape, char))
                else:
                    output.append(char)
            if output:
                yield "".join(output)

    def decode(self, text: str) -> str:
        return "".join(self.decode_chunks((text,)))

    def decode_chunks(self, chunks: Iterable[str]) -> Iterator[str]:
        buffer = ""

        def drain(*, final: bool) -> tuple[str, str]:
            output: list[str] = []
            position = 0
            while position < len(buffer):
                char = buffer[position]
                if char == self.escape:
                    if position + 1 >= len(buffer):
                        if final:
                            raise DecodeError("dangling escape at end of transformed text")
                        break
                    output.append(buffer[position + 1])
                    position += 2
                    continue
                if char in STRUCTURAL_CONTROLS:
                    try:
                        node, end = parse_prefix(
                            buffer,
                            position,
                            final=final,
                            max_depth=self.max_depth,
                            max_nodes=self.max_nodes,
                        )
                    except NeedMoreData:
                        break
                    sequence = node.serialize()
                    original = self.reverse.get(sequence)
                    if original is None:
                        raise DecodeError(f"unknown or corrupted transformed IDS at offset {position}: {sequence!r}")
                    output.append(original)
                    position = end
                    continue
                output.append(char)
                position += 1
            return "".join(output), buffer[position:]

        for chunk in chunks:
            if not isinstance(chunk, str):
                raise TypeError("text chunks must be str")
            buffer += chunk
            output, buffer = drain(final=False)
            if output:
                yield output
        output, buffer = drain(final=True)
        if buffer:  # defensive: final drain either consumes or raises
            raise DecodeError(f"undecoded suffix: {buffer!r}")
        if output:
            yield output


@dataclass
class Digest:
    sha256: "hashlib._Hash" = field(default_factory=hashlib.sha256)
    bytes: int = 0

    def update(self, block: bytes) -> None:
        self.sha256.update(block)
        self.bytes += len(block)

    def report(self) -> dict[str, object]:
        return {"sha256": self.sha256.hexdigest(), "bytes": self.bytes}


def iter_utf8_chunks(stream: BinaryIO, digest: Digest, *, chunk_size: int = 1 << 20) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    while block := stream.read(chunk_size):
        digest.update(block)
        text = decoder.decode(block, final=False)
        if text:
            yield text
    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail


@contextlib.contextmanager
def atomic_binary_output(path_text: str) -> Iterator[BinaryIO]:
    """Write a named output through an fsynced same-directory temporary file."""
    if path_text == "-":
        yield sys.stdout.buffer
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise


def atomic_write_json(path_text: str, value: object) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode("utf-8")
    with atomic_binary_output(path_text) as handle:
        handle.write(encoded)


def transcode_file(
    codec: HanziBPEPreTokenizer,
    *,
    mode: str,
    input_path: str,
    output_path: str,
    chunk_size: int = 1 << 20,
) -> dict[str, object]:
    if mode not in {"encode", "decode"}:
        raise ValueError("mode must be encode or decode")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    source: BinaryIO
    close_source = input_path != "-"
    source = open(input_path, "rb") if close_source else sys.stdin.buffer
    input_digest, output_digest = Digest(), Digest()
    try:
        chunks = iter_utf8_chunks(source, input_digest, chunk_size=chunk_size)
        transformed = codec.encode_chunks(chunks) if mode == "encode" else codec.decode_chunks(chunks)
        with atomic_binary_output(output_path) as destination:
            for text in transformed:
                block = text.encode("utf-8")
                destination.write(block)
                output_digest.update(block)
    finally:
        if close_source:
            source.close()
    return {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "mode": mode,
        "input": {"path": input_path, **input_digest.report()},
        "output": {"path": output_path, **output_digest.report()},
        "map_entries": len(codec.forward),
    }


def _add_catalogue_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ids", required=True, help="BabelStone-style IDS text file")
    parser.add_argument("--region", choices=("T", "H", "G"), default="T", help="glyph-region preference")
    parser.add_argument(
        "--families", default="all",
        help=(
            "comma list: horizontal,vertical,surround,overlay,reflection,rotation,"
            "subtraction,variation; 'all' means the 17 formal IDS operators and "
            "'all-with-variation' additionally admits non-operator U+303E IVI"
        ),
    )
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--max-nodes", type=int, default=512)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("encode", "decode"):
        sub = subparsers.add_parser(command, help=f"{command} a UTF-8 corpus")
        _add_catalogue_options(sub)
        sub.add_argument("input", help="input UTF-8 path, or - for stdin")
        sub.add_argument("output", help="output UTF-8 path, or - for stdout")
        sub.add_argument("--report", help="atomically written JSON SHA-256 report")
        sub.add_argument("--chunk-size", type=int, default=1 << 20)
    audit = subparsers.add_parser("catalogue-audit", help="build maps and emit coverage/collision JSON")
    _add_catalogue_options(audit)
    audit.add_argument("--output", default="-", help="JSON path, or - for stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        catalogue = build_catalogue(
            args.ids,
            region=args.region,
            families=args.families,
            max_depth=args.max_depth,
            max_nodes=args.max_nodes,
        )
        if args.command == "catalogue-audit":
            atomic_write_json(args.output, catalogue.audit)
            return 0
        if args.input == "-" and args.output == "-":
            raise CatalogueError("input and output cannot both use -")
        report = transcode_file(
            catalogue.codec(),
            mode=args.command,
            input_path=args.input,
            output_path=args.output,
            chunk_size=args.chunk_size,
        )
        report["catalogue"] = {
            "ids_sha256": catalogue.audit["ids_sha256"],
            "region": catalogue.audit["region"],
            "families": catalogue.audit["families"],
            "transformed_characters": catalogue.audit["transformed_characters"],
        }
        if args.report:
            atomic_write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 0
    except (OSError, UnicodeError, CatalogueError, DecodeError, ValueError) as exc:
        print(f"{TOOL_NAME}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
