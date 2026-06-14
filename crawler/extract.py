"""Main-content extraction: HTML → Markdown, with FAQ-pair preservation.

Design:
- extract()          — top-level entry; returns ExtractResult (title, markdown,
                        faq_pairs, extraction_method).
- Trafilatura is the primary extractor (output_format="markdown",
  favor_precision=True) — it strips nav/footer/cookie-banner/script boilerplate
  far better than a naive "grab all visible text" pass.
- A BeautifulSoup fallback runs only when trafilatura returns nothing (e.g. a
  near-empty page) so we never silently drop a page.
- FAQ pairs are extracted separately from the *raw* HTML (not the trafilatura
  output) via three accordion patterns: <details>/<summary>, class-based
  question/answer containers, and <dl><dt>/<dd> definition lists. Accordion
  answers are often hidden via CSS, not removed from the DOM, so the full
  text is present in the markup even though it wouldn't be "visible".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import trafilatura
from bs4 import BeautifulSoup

# Boilerplate class/id name fragments to strip before the BS4 fallback runs.
_BOILERPLATE_PATTERN = re.compile(
    r"(nav|menu|footer|footer-widget|header|cookie|consent|banner|sidebar|"
    r"breadcrumb|widget|modal|overlay|popup|social|share)",
    re.IGNORECASE,
)

# Common suffix separators in <title> tags, e.g. "FAQ | Appther" → "FAQ".
_TITLE_SUFFIX_RE = re.compile(r"\s*[|\-–—]\s*[^|\-–—]+$")

_QUESTION_CLASS_RE = re.compile(r"(question|title|heading|toggle|trigger)", re.IGNORECASE)
_ANSWER_CLASS_RE = re.compile(r"(answer|content|body|panel|collapse)", re.IGNORECASE)
_FAQ_CONTAINER_RE = re.compile(r"(faq|accordion|qa-item|q-and-a)", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    """Collapse runs of whitespace (including newlines/indentation) to single spaces."""
    return _WHITESPACE_RE.sub(" ", text).strip()


@dataclass
class FaqPair:
    """A single FAQ question + full answer, captured together."""

    question: str
    answer: str

    def to_text(self) -> str:
        return f"Q: {self.question}\nA: {self.answer}"


@dataclass
class ExtractResult:
    """The outcome of extracting main content from one fetched page."""

    url: str
    title: str
    markdown: str
    faq_pairs: list[FaqPair] = field(default_factory=list)
    extraction_method: str = "trafilatura"

    @property
    def has_content(self) -> bool:
        return bool(self.markdown.strip())

    @property
    def word_count(self) -> int:
        return len(self.markdown.split())


# ── Public API ────────────────────────────────────────────────────────────────


def extract(html: str, url: str = "") -> ExtractResult:
    """Extract title, main-content Markdown, and FAQ pairs from raw HTML."""
    title = _extract_title(html)
    faq_pairs = _extract_faq_pairs(html)

    markdown = _extract_with_trafilatura(html, url)
    method = "trafilatura"

    if not markdown or not markdown.strip():
        markdown = _extract_with_bs4_fallback(html)
        method = "bs4-fallback"

    return ExtractResult(
        url=url,
        title=title,
        markdown=markdown.strip(),
        faq_pairs=faq_pairs,
        extraction_method=method,
    )


# ── Trafilatura extraction ──────────────────────────────────────────────────────


def _extract_with_trafilatura(html: str, url: str) -> str | None:
    try:
        return trafilatura.extract(
            html,
            url=url or None,
            output_format="markdown",
            favor_precision=True,
            include_comments=False,
            include_tables=True,
            include_links=False,
            include_images=False,
            deduplicate=True,
        )
    except Exception:
        return None


# ── BeautifulSoup fallback ──────────────────────────────────────────────────────


def _extract_with_bs4_fallback(html: str) -> str:
    """Strip boilerplate and render headings/paragraphs/list-items as Markdown."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()

    for tag in soup.find_all(class_=_BOILERPLATE_PATTERN):
        tag.decompose()
    for tag in soup.find_all(id=_BOILERPLATE_PATTERN):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"(main|content)", re.IGNORECASE))
        or soup.find(class_=re.compile(r"(main|content|post)", re.IGNORECASE))
        or soup.find("body")
    )
    if main is None:
        return ""

    lines: list[str] = []
    heading_prefix = {
        "h1": "#",
        "h2": "##",
        "h3": "###",
        "h4": "####",
        "h5": "#####",
        "h6": "######",
    }
    for elem in main.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        text = elem.get_text(" ", strip=True)
        if not text:
            continue
        prefix = heading_prefix.get(elem.name)
        if prefix:
            lines.append(f"{prefix} {text}")
        elif elem.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    return "\n\n".join(lines)


# ── Title extraction ─────────────────────────────────────────────────────────


def _extract_title(html: str) -> str:
    """Return the page title, preferring <title> (suffix stripped) then <h1>."""
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.find("title")
    if title_tag is not None:
        raw = title_tag.get_text(strip=True)
        if raw:
            stripped = _TITLE_SUFFIX_RE.sub("", raw).strip()
            return stripped or raw

    h1 = soup.find("h1")
    if h1 is not None:
        text = h1.get_text(" ", strip=True)
        if text:
            return text

    return ""


# ── FAQ pair extraction ─────────────────────────────────────────────────────────


def _extract_faq_pairs(html: str) -> list[FaqPair]:
    """Extract Q+A pairs from <details>/<summary>, class-based, and <dl> patterns."""
    soup = BeautifulSoup(html, "lxml")
    pairs: list[FaqPair] = []
    seen: set[str] = set()

    _collect_details_pairs(soup, pairs, seen)
    _collect_class_based_pairs(soup, pairs, seen)
    _collect_definition_list_pairs(soup, pairs, seen)

    return pairs


def _collect_details_pairs(soup: BeautifulSoup, pairs: list[FaqPair], seen: set[str]) -> None:
    """<details><summary>Question</summary>Answer...</details>"""
    for details in soup.find_all("details"):
        summary = details.find("summary")
        if summary is None:
            continue
        question = _clean_text(summary.get_text(" ", strip=True))
        if not question or question in seen:
            continue

        full_text = _clean_text(details.get_text(" ", strip=True))
        answer = (
            full_text[len(question) :].strip()
            if full_text.startswith(question)
            else full_text.replace(question, "", 1).strip()
        )

        if answer:
            pairs.append(FaqPair(question=question, answer=answer))
            seen.add(question)


def _collect_class_based_pairs(soup: BeautifulSoup, pairs: list[FaqPair], seen: set[str]) -> None:
    """Containers with faq/accordion/qa class names holding a question + answer child."""
    for container in soup.find_all(class_=_FAQ_CONTAINER_RE):
        q_elem = container.find(class_=_QUESTION_CLASS_RE) or container.find(["h3", "h4", "dt"])
        a_elem = container.find(class_=_ANSWER_CLASS_RE) or container.find(["dd", "p"])
        if q_elem is None or a_elem is None or q_elem is a_elem:
            continue

        question = _clean_text(q_elem.get_text(" ", strip=True))
        answer = _clean_text(a_elem.get_text(" ", strip=True))
        if question and answer and question not in seen:
            pairs.append(FaqPair(question=question, answer=answer))
            seen.add(question)


def _collect_definition_list_pairs(
    soup: BeautifulSoup, pairs: list[FaqPair], seen: set[str]
) -> None:
    """<dl><dt>Question</dt><dd>Answer</dd></dl>"""
    for dl in soup.find_all("dl"):
        pending_question: str | None = None
        for child in dl.find_all(["dt", "dd"], recursive=False):
            if child.name == "dt":
                pending_question = _clean_text(child.get_text(" ", strip=True))
            elif child.name == "dd" and pending_question:
                answer = _clean_text(child.get_text(" ", strip=True))
                if answer and pending_question not in seen:
                    pairs.append(FaqPair(question=pending_question, answer=answer))
                    seen.add(pending_question)
                pending_question = None
