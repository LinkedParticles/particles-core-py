# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The one escaping layer for exporter output.

Every stored string an exporter writes (a subject's name, a property value, a
taxonomy tag, an external id, a provenance field, an alias, claim text, a lint
finding's detail) is untrusted: an agent key, an imported bundle, an authority
label or an extraction of a hostile source chooses it. Written raw, it can
become structure the engine did not write: a frontmatter key, an HTML element,
a Logseq macro or property, a fenced code block, a table column or a second
link (security review 2026-10-28-1, O1, O2, O37, O38).

Each function here makes a string inert in exactly one output context, and is
named for it. An exporter never writes a stored string except through one of
them; ``tests/test_export_escaping_guard.py`` fails on an exporter module that
interpolates a value into its output without one.

Markdown text has two forms. :func:`md_text` backslash-escapes, for one line
the exporter composes (a heading, a list item, a callout line, link text).
:func:`prose_text` and :func:`synthesized_prose` write HTML entities instead,
which is idempotent, for prose that may pass through the article backstop a
second time (a cached article, the structured listing).

Store-free and pure: a Client-layer module both the exporters and
the article renderer depend on downward.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata

from particles.core.observer_scope import _BLANK_GLYPHS, _default_ignorable

__all__ = [
    "anki_deck",
    "anki_field",
    "identifier",
    "logseq_alias",
    "logseq_page_ref",
    "logseq_prose",
    "logseq_property_key",
    "logseq_property_value",
    "logseq_tag",
    "logseq_text",
    "md_code_span",
    "md_link",
    "md_link_label",
    "md_table_cell",
    "md_text",
    "prose_code_span",
    "prose_text",
    "synthesized_prose",
    "without_format_chars",
    "wikilink",
    "wikilink_target",
    "yaml_scalar",
]

# Every character CommonMark, YAML 1.1 or a terminal reads as a line break.
_LINE_BREAKS = re.compile(r"\s*[\r\n\v\f\x85  ]+\s*")
_PERCENT_RUN = re.compile(r"%{2,}")
_FENCE_RUN = re.compile(r"`{3,}|~{3,}")
_DATAVIEW_INLINE = re.compile(r"(`+[ \t]*)=")
_LOGSEQ_BLOCK = re.compile(r"#\+(?=begin_)", re.I)
_FENCE_ENTITY = {"`": "&#96;", "~": "&#126;"}

# A character that opens a block when it starts a line: an ATX heading, a
# quote (and so a callout header), a list item or thematic break, a setext
# underline, a table row or delimiter row, a fence, or a link reference
# definition. ``[[`` stays a wikilink, so only a single ``[`` is escaped.
_BLOCK_START = re.compile(r"^(?:[#>+*=_|:~-]|\[(?!\[))")
_ORDERED_LIST_START = re.compile(r"^(\d{1,9})([.)])")
# A trailing `` ^id`` names an Obsidian block, which a link can then target.
_BLOCK_ID_TAIL = re.compile(r"(\s)\^([A-Za-z0-9-]+)$")


#: C0 and C1 controls other than tab and the line breaks every context reads
#: as one (line feed, vertical tab, form feed, carriage return, next line).
_CONTROLS = re.compile(r"[\x00-\x08\x0e-\x1f\x7f-\x84\x86-\x9f]")


def _invisible(ch: str) -> bool:
    """A format character, or a character a renderer draws as nothing.

    The invisible set is the project-key rule's: Unicode's default-ignorable
    code points (the variation selectors and the combining grapheme joiner
    among them) and the letters and symbols that render blank (the Hangul
    fillers, the braille blank), so one rule decides what reads as nothing
    (security review 2026-10-30-1, AB23).
    """
    return unicodedata.category(ch) == "Cf" or _default_ignorable(ch) or ch in _BLANK_GLYPHS


def without_format_chars(text: str) -> str:
    """``text`` without its invisible characters and its control characters.

    A bidirectional override or isolate makes text display in an order other
    than the one it is stored in, so a name can read as another word or with
    another file extension, and a zero-width character hides a difference
    between two names that look the same. None of them is visible, so none
    carries anything a reader of the export can see (security review
    2026-10-28-2, AA34, O28). The same holds for a variation selector, a
    grapheme joiner or a blank glyph, and a control character other than a
    tab or a line break drives a terminal that prints the file rather than
    showing anything (2026-10-30-1, AB23, AB72). Tabs and line breaks are kept
    for the caller to lay out.
    """
    text = _CONTROLS.sub("", text)
    if text.isascii():
        return text
    return "".join(ch for ch in text if not _invisible(ch))


def _one_line(text: str) -> str:
    """``text`` with every line break, and the space around it, made one space."""
    return _LINE_BREAKS.sub(" ", without_format_chars(text)).strip()


def _words(text: str) -> str:
    """``text`` with every run of whitespace made one space and no format character."""
    return " ".join(without_format_chars(text).split())


# ---------------------------------------------------------------------------
# Markdown (CommonMark, Obsidian, the wiki)
# ---------------------------------------------------------------------------


def md_text(text: str) -> str:
    """Stored text made inert as one line of exported Markdown.

    For a heading, a list item, a callout title or body line, or link text.
    Placed raw, stored text could carry HTML (a remote ``<img>``, a comment
    that hides the rest of the line), an embed or remote image (``![[…]]``,
    ``![](https://…)``), a link whose text names one place and whose target is
    another, a Logseq macro (``{{…}}``), an Obsidian comment (``%%``), math
    (``$``), a Dataview inline query, a Logseq query block, a code-fence run,
    a footnote reference (``[^…]``) that cites another claim's source, a line
    break that leaves the block it sits in, or a line-leading marker that
    opens a heading, a quote, a callout, a list or a table (security review
    D36; 2026-10-24-2, P32; 2026-10-25-2, R27; 2026-10-27-1, G4; 2026-10-28-1,
    O1, O37; 2026-10-28-2, AA27). Invisible format characters are dropped
    (:func:`without_format_chars`).

    Every one is backslash-escaped, which CommonMark, Obsidian and Logseq
    render as the literal character, so the text reads as written. Backslashes
    are escaped first so the text cannot cancel an escape. A ``[[wikilink]]``
    stays a link, since a link to a subject's note is how the exported graph
    is navigated.
    """
    text = _one_line(text)
    text = text.replace("\\", "\\\\")
    text = text.replace("<", "\\<")
    text = re.sub(r"!(?=\[)", r"\\!", text)
    text = text.replace("](", "]\\(")
    text = text.replace("[^", "\\[^")
    text = _PERCENT_RUN.sub(lambda m: "\\%" * len(m.group()), text)
    text = text.replace("$", "\\$")
    text = _DATAVIEW_INLINE.sub(r"\1\\=", text)
    text = _LOGSEQ_BLOCK.sub("\\#+", text)
    text = _FENCE_RUN.sub(lambda m: "".join("\\" + c for c in m.group()), text)
    text = text.replace("{{", "\\{\\{")
    text = _BLOCK_ID_TAIL.sub(r"\1\\^\2", text)
    if _ORDERED_LIST_START.match(text):
        return _ORDERED_LIST_START.sub(r"\1\\\2", text, count=1)
    return _BLOCK_START.sub(lambda m: "\\" + m.group(), text, count=1)


def md_code_span(text: str) -> str:
    """Stored text as one inline code span it cannot leave."""
    return "`" + _words(text).replace("`", "'") + "`"


def md_link_label(text: str) -> str:
    """Stored text made safe as the visible label of a link the exporter writes.

    A label sits inside ``[[target|label]]`` or ``[label](target)``, where a
    backslash escape is not reliably honoured, so brackets become parentheses
    (the label cannot close the link and open another) and ``<`` becomes an
    entity (security review D36).
    """
    text = _words(text)
    return text.replace("[", "(").replace("]", ")").replace("<", "&lt;")


# Characters that end or break a Markdown link target, written percent-encoded.
_LINK_TARGET_ESCAPES = {c: f"%{ord(c):02X}" for c in " ()<>[]{}\\`\"'\t\r\n|"}


def md_link(label: str, uri: str) -> str:
    """A stored URI as a link with an inert label, or as inert text.

    Only an ``http(s)`` URI becomes a link target, with every character that
    could close the target or open another construct percent-encoded; any
    other scheme is written as text (security review 2026-10-27-1, G5). A
    format character in the target is dropped, as in every other context.
    """
    text = md_text(md_link_label(label))
    if not re.match(r"https?://", uri, re.I):
        return f"{text} ({md_text(uri)})" if label != uri else text
    target = "".join(_LINK_TARGET_ESCAPES.get(c, c) for c in without_format_chars(uri))
    return f"[{text}]({target})"


def md_table_cell(text: str) -> str:
    """Stored text as one cell of a Markdown table row.

    The cell is one inert line whose pipes are escaped, so the text stays in
    its own column and no later cell is spoofed (security review 2026-10-28-1,
    O2).
    """
    return md_text(text).replace("|", "\\|")


def wikilink_target(text: str) -> str:
    """Stored text as the target of an Obsidian or wiki ``[[…]]`` link.

    The target is one line. A bracket would close the link and open another
    construct after it, a pipe would start a label, and ``#`` and ``^`` would
    point it at a heading or block of another note, so brackets become
    parentheses and the other three a dash, as the filename sanitiser writes
    a character a path cannot hold.
    """
    text = _words(text)
    text = text.replace("[", "(").replace("]", ")")
    return re.sub(r"[|#^]", "-", text)


def wikilink(target: str, label: str | None = None) -> str:
    """An Obsidian ``[[target]]`` or ``[[target|label]]`` link from stored text."""
    if label is None:
        return f"[[{wikilink_target(target)}]]"
    return f"[[{wikilink_target(target)}|{md_link_label(label)}]]"


# ---------------------------------------------------------------------------
# Entity form: prose that may pass through the article backstop again
# ---------------------------------------------------------------------------


def synthesized_prose(text: str) -> str:
    """Model-written article prose made inert, keeping its paragraphs and citations.

    Article synthesis writes prose from stored claims, so a claim that steers
    the model can steer what the prose carries: a remote image whose URL holds
    other claims, raw HTML, an embed, a Logseq macro, an Obsidian comment that
    hides the rest of the note (security review 2026-10-24-2, P31). Layer A
    checks only which particles the prose cites, so this is the backstop.

    Unlike :func:`md_text` the text keeps its line breaks, headings, emphasis,
    ``[^p-…]`` citations and ``[[wikilinks]]``, which the exporters'
    converters need, and inline code (a fence of three or more backticks or
    tildes is not kept). Each live construct is written as an HTML entity
    instead of a backslash escape, so the function is idempotent: a body
    neutralized before it was cached is unchanged when a cache hit passes
    through it again. Invisible characters are dropped, as in every other
    context (security review 2026-10-30-1, AB40).
    """
    text = without_format_chars(text).replace("<", "&lt;")
    text = re.sub(r"!(?=\[)", "&#33;", text)
    text = text.replace("](", "]&#40;")
    text = text.replace("{{", "&#123;&#123;")
    text = _PERCENT_RUN.sub(lambda m: "&#37;" * len(m.group()), text)
    # A fence opener can start a code block a vault plugin runs (a
    # ``dataviewjs`` block), at a line start or inside a list item or quote, so
    # every run that could open one is written as entities (security review
    # 2026-10-25-1, Q37).
    text = _FENCE_RUN.sub(lambda m: _FENCE_ENTITY[m.group()[0]] * len(m.group()), text)
    # A code span opening on ``=`` is a Dataview inline query (the ``$=`` form
    # is broken by the dollar escape), so that ``=`` becomes an entity, which
    # still reads as ``=`` outside a code span. ``#+BEGIN_`` opens a Logseq
    # advanced query or directive block (security review 2026-10-25-2, R27).
    text = _DATAVIEW_INLINE.sub(r"\1&#61;", text)
    text = _LOGSEQ_BLOCK.sub("&#35;+", text)
    return text.replace("$", "&#36;")


# Entities for a line-leading block marker, so the entity form stays idempotent.
_BLOCK_START_ENTITY = {
    "#": "&#35;",
    ">": "&gt;",
    "+": "&#43;",
    "*": "&#42;",
    "=": "&#61;",
    "_": "&#95;",
    "|": "&#124;",
    ":": "&#58;",
    "~": "&#126;",
    "-": "&#45;",
    "[": "&#91;",
}


def prose_text(text: str) -> str:
    """Stored text as one line of an article, with every live construct an entity.

    The entity form :func:`synthesized_prose` writes, so a listing that later
    passes through the prose backstop is unchanged by it, and the line breaks
    go, so a claim cannot leave its list item, heading or footnote (security
    review 2026-10-25-2, R3). A line-leading block marker is an entity too,
    and so is the bracket of a footnote reference, so a claim cites only the
    source the engine names after it, never another claim's (2026-10-28-2,
    AA27). Model prose keeps its citations, since it does not pass through
    here.
    """
    text = synthesized_prose(_words(text)).replace("[^", "&#91;^")
    if _ORDERED_LIST_START.match(text):
        return _ORDERED_LIST_START.sub(
            lambda m: m.group(1) + ("&#46;" if m.group(2) == "." else "&#41;"), text, count=1
        )
    return _BLOCK_START.sub(lambda m: _BLOCK_START_ENTITY[m.group()], text, count=1)


def prose_code_span(text: str) -> str:
    """Stored text as one code span of an article, its backticks entities."""
    return "`" + _words(text).replace("`", "&#96;") + "`"


# ---------------------------------------------------------------------------
# YAML frontmatter
# ---------------------------------------------------------------------------

# Characters a YAML double-quoted scalar must not hold raw: YAML 1.1 reads the
# first five as line breaks, and the rest are outside its printable set.
_YAML_UNSAFE = re.compile(r"[\x7f-\x9f  ﻿￾￿\ud800-\udfff]")


def yaml_scalar(value: object) -> str:
    """A stored value as one YAML frontmatter scalar.

    A string is always written double-quoted on one line, its line breaks and
    other control characters escaped, so it cannot add a key, end the
    frontmatter or start a list (security review 2026-10-28-1, O2), and its
    invisible format characters are written as escapes (2026-10-28-2, AA34). A finite
    number or a boolean stays a YAML number or boolean, which cannot carry
    structure either.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) or (isinstance(value, float) and math.isfinite(value)):
        return json.dumps(value)
    out = json.dumps(value if isinstance(value, str) else str(value), ensure_ascii=False)
    out = _YAML_UNSAFE.sub(lambda m: f"\\u{ord(m.group()):04x}", out)
    # A format character is written as its escape, so the file shows no
    # reordered text and the value still round-trips (AA34).
    if out.isascii():
        return out
    return "".join(_yaml_escape(c) if unicodedata.category(c) == "Cf" else c for c in out)


def _yaml_escape(ch: str) -> str:
    """``ch`` as a YAML double-quoted escape: ``\\uXXXX``, or ``\\UXXXXXXXX`` above the BMP."""
    code = ord(ch)
    return f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}"


# ---------------------------------------------------------------------------
# Logseq
# ---------------------------------------------------------------------------


#: A colon followed by another, the start of a ``key::`` property marker.
_PROPERTY_COLON = re.compile(r":(?=:)")


def _no_property_marker(text: str) -> str:
    """``text`` with a backslash after every colon another colon follows, so it holds no ``::``."""
    return _PROPERTY_COLON.sub(":\\\\", text)


def logseq_text(text: str) -> str:
    """Stored text as the content of one Logseq block.

    One inert Markdown line (:func:`md_text`, which also escapes ``{{``
    macros), whose ``::`` cannot read as a block property (security review
    2026-10-24-2, P32), and whose ``((`` cannot open a block reference that
    shows another claim's text in its place (2026-10-28-2, AA32).
    """
    return _no_property_marker(md_text(text)).replace("((", "(\\(")


def logseq_prose(text: str) -> str:
    """Neutralized article prose as the lines of one Logseq block.

    The prose keeps its line breaks, and every line after the first is a
    continuation of the block, so a ``key:: value`` line would set a block
    property (a template, a query option). Every ``::`` is broken, as
    :func:`logseq_text` breaks it (security review 2026-10-28-2, AA33). The
    second parenthesis of every ``((`` is an entity, so no line (a References
    line built from a stored URI, an attribution, model prose) holds a block
    reference that shows another claim's text in its place, and invisible
    characters are dropped (2026-10-30-1, AB39, AB40). The rule is
    idempotent, so a cached body passes through it unchanged.
    """
    return _no_property_marker(without_format_chars(text)).replace("((", "(&#40;")


def logseq_property_value(text: str) -> str:
    """Stored text as the value of a Logseq ``key:: value`` property line.

    One line, so it adds no property and no block below it, with no live
    macro and no second property on its own line (security review
    2026-10-28-1, O38).
    """
    return logseq_text(text)


def logseq_alias(text: str) -> str:
    """Stored text as one entry of a Logseq ``alias::`` property.

    Logseq splits an alias value on commas, so a comma becomes a space and the
    stored alias stays one alias.
    """
    return logseq_property_value(" ".join(text.replace(",", " ").split()))


_PROPERTY_KEY_UNSAFE = re.compile(r"[^\w:./-]+")

#: Property names Logseq reads as its own: a block's id, its folding, a
#: template, a page's title, alias, tags or icon, and the query and card
#: options. A stored key with one of these names is written prefixed.
_LOGSEQ_RESERVED_KEYS = frozenset(
    {
        "id",
        "collapsed",
        "heading",
        "background-color",
        "template",
        "template-including-parent",
        "title",
        "alias",
        "aliases",
        "tags",
        "icon",
        "public",
        "filters",
        "exclude-from-graph-view",
    }
)
_LOGSEQ_RESERVED_PREFIXES = ("query-", "logseq.", "card-", "hl-", "ls-")


def logseq_property_key(key: str) -> str:
    """A stored property name as the key of a Logseq ``key:: value`` block.

    A name Logseq reads as one of its own properties (``id``, ``collapsed``,
    ``template``, a query option) is written as ``property-<name>``, so a
    stored key cannot give a claim block a second id or change how Logseq
    treats the block (security review 2026-10-28-2, AA32).
    """
    key = _PROPERTY_KEY_UNSAFE.sub("-", without_format_chars(key))
    key = re.sub(r":{2,}", ":", key).strip("-:") or "property"
    folded = key.lower()
    if folded in _LOGSEQ_RESERVED_KEYS or folded.startswith(_LOGSEQ_RESERVED_PREFIXES):
        return f"property-{key}"
    return key


_PAGE_REF_ESCAPES = str.maketrans({"[": "(", "]": ")", "{": "(", "}": ")", "<": "-", ">": "-"})


def logseq_page_ref(text: str) -> str:
    """Stored text as the page name inside a Logseq ``[[…]]`` reference or tag.

    One line, with brackets and braces made parentheses, so the reference
    cannot close early and leave the rest of the name live and holds no
    macro; ``<`` and ``>`` become a dash, as the filename sanitiser writes
    them, and ``::`` one colon, so the name holds no HTML tag and reads as no
    property.
    """
    text = _words(text).translate(_PAGE_REF_ESCAPES)
    # A run of parentheses (a ``((uuid))`` block reference, or brackets the
    # translation made parentheses) is one, so the name holds no block
    # reference (security review 2026-10-28-2, AA32).
    text = re.sub(r"\({2,}", "(", re.sub(r"\){2,}", ")", text))
    return re.sub(r":{2,}", ":", text)


_PLAIN_TAG = re.compile(r"[\w/.-]+(?::[\w/.-]+)*")


def logseq_tag(tag: str) -> str:
    """A stored taxonomy tag as one inline Logseq tag token.

    A plain tag stays ``#tag``; any other is wrapped as ``#[[tag]]`` through
    :func:`logseq_page_ref`, so a space, a line break or a bracket cannot end
    the token and leave the rest of the tag live.
    """
    if _PLAIN_TAG.fullmatch(tag):
        return f"#{tag}"
    return f"#[[{logseq_page_ref(tag)}]]"


# ---------------------------------------------------------------------------
# Anki tab-separated import
# ---------------------------------------------------------------------------


def anki_field(text: str, *, line_separator: str = " ") -> str:
    """Stored text as one field of an Anki tab-separated card line.

    Tabs and line breaks would start another field or another card, so lines
    are joined with ``line_separator`` and tabs become spaces. A field that
    holds a quote or opens with ``#`` is written CSV-quoted, so the importer
    neither reads the quote as the start of a field spanning the separator nor
    the line as a comment.
    """
    text = without_format_chars(text).replace("\t", " ")
    lines = [" ".join(line.split()) for line in _LINE_BREAKS.split(text)]
    field = line_separator.join(line for line in lines if line)
    if '"' in field or field.startswith("#"):
        return '"' + field.replace('"', '""') + '"'
    return field


def anki_deck(name: str) -> str:
    """A stored name as one segment of an Anki ``#deck:`` path.

    One line, and ``::`` (Anki's sub-deck separator) becomes ``:``, so the
    name adds no deck level and no header line.
    """
    return re.sub(r":{2,}", ":", _words(name))


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------


def identifier(text: str) -> str:
    """A stored id as an identifier token: letters, digits and dashes only.

    For a block id (``^p-…``), a citation (``[^p-…]``) or an anchor. An id
    this SDK mints is a UUID and is unchanged; an imported one cannot carry
    markup into the token.
    """
    return re.sub(r"[^A-Za-z0-9-]", "-", text)
