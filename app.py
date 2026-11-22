"""
grkraj.org Biology Study Tutor — Bioluminescent Obsidian Dark-Mode Streamlit Web App

Lightweight (~250 MB RAM, zero GPU needed) web app built for 100% FREE hosting on
Streamlit Community Cloud (https://share.streamlit.io) or Google Cloud Run.

Uses:
  - Prebuilt unified 15,551 x 768 `gemini-embedding-2` index (~67 MB, auto-downloaded
    from public Hugging Face dataset `sgowda/grkraj`, zero HF token required)
  - `gemini-3.8-flash` for grounded visual + textual biology explanations
  - Automatic multi-topic query decomposition + hybrid dense/BM25 search
  - Color-coded clickable citation pills (`Text N` emerald, `Diagram N` cyan -> grkraj.org)
  - Specimen-framed biology diagrams + rich interactive Source Chapter cards
  - Built-in daily quota limiter + optional Bring-Your-Own-Key (BYOK)

Run locally:
    cd student-app
    $env:GEMINI_API_KEY = "AIza..."
    streamlit run app.py
"""

import base64
import html as html_lib
import io
import json
import math
import os
import re
import threading
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from google import genai
from google.genai import types
from huggingface_hub import hf_hub_download
from PIL import Image

st.set_page_config(
    page_title="grkraj.org Biology Tutor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
LOCAL_IMAGES_ROOT = APP_DIR.parent

EMBED_MODEL = "gemini-embedding-2"
GEN_MODEL = "gemini-3.8-flash"


def _get_cfg(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            val = st.secrets[key]
            if val is not None and str(val).strip():
                return str(val).strip()
    except Exception:
        pass
    return (os.environ.get(key) or default).strip()


INDEX_HF_REPO = _get_cfg("INDEX_HF_REPO", "sgowda/grkraj")
DAILY_FREE_LIMIT = int(_get_cfg("DAILY_FREE_LIMIT", "20") or "20")
GLOBAL_DAILY_LIMIT = int(_get_cfg("GLOBAL_DAILY_LIMIT", "100") or "100")

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'+\-]*")
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "by",
    "with", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "that", "this", "these", "those", "it", "its", "they", "their", "them",
    "what", "which", "who", "how", "why", "when", "where", "explain", "describe",
    "discuss", "give", "about", "into", "can", "may", "also", "such", "than",
}

_GENERIC_FILLERS = {
    "function", "functions", "structure", "structures", "mechanism", "mechanisms",
    "process", "processes", "steps", "role", "importance", "significance",
    "properties", "examples", "differences", "similarities", "types", "stages",
    "phases", "regulation", "control", "overview", "introduction", "summary",
    "definition", "diagram", "diagrams", "detail", "details",
}
_FOLLOWUP_STARTS = {
    "what", "how", "why", "where", "when", "which", "who", "that", "their",
    "its", "it", "they", "also", "then", "because", "while", "if", "in",
    "for", "of", "with", "from", "to",
}
_LEAD_DIRECTIVE_RE = re.compile(
    r"^(?:please\s+)?(?:explain|describe|define|discuss|summarize|compare|contrast|"
    r"differentiate\s+between|distinguish\s+between|tell\s+me\s+about|"
    r"what\s+(?:is|are)|how\s+(?:does|do)|give\s+an?\s+overview\s+of)\s+",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-12)


def split_multi_topic_query(query: str) -> list[str]:
    """Split compound queries like 'Explain photosynthesis, mitosis and DNA replication'
    into balanced sub-queries so every topic gets dedicated passages & diagrams."""
    q = (query or "").strip()
    if not q:
        return []
    q_parts = [p.strip() for p in q.split("?") if p.strip()]
    if len(q_parts) >= 2 and all(len(p.split()) >= 2 for p in q_parts):
        return q_parts[:5]
    body = _LEAD_DIRECTIVE_RE.sub("", q).strip().rstrip("?.!").strip()
    raw_parts = [
        p.strip(" ,;:")
        for p in re.split(r"\s*(?:,|;|\band\b|&|\bvs\.?\b|\bversus\b)\s*", body, flags=re.IGNORECASE)
        if p.strip(" ,;:")
    ]
    if len(raw_parts) < 2 or len(raw_parts) > 5:
        return [q]
    cleaned: list[str] = []
    for p in raw_parts:
        words = p.split()
        if not words or len(words) > 7:
            return [q]
        if words[0].lower() in _FOLLOWUP_STARTS or p.lower() in _GENERIC_FILLERS:
            return [q]
        cleaned.append(p)
    last_words = cleaned[-1].split()
    if len(last_words) >= 2:
        tail = " ".join(last_words[1:])
        cleaned = [
            f"{c} {tail}"
            if (len(c.split()) == 1 and len(c) <= 4 and c.lower() not in {"dna", "rna", "atp", "adp", "nad", "fad"})
            else c
            for c in cleaned
        ]
    deduped = list(dict.fromkeys(cleaned))
    return deduped if len(deduped) >= 2 else [q]


def ensure_index_present() -> Path:
    """Use local rag_multimodal_index if present, else download from public sgowda/grkraj."""
    candidates = [
        APP_DIR / "rag_multimodal_index",
        APP_DIR.parent / "rag_multimodal_index",
    ]
    for c in candidates:
        if (c / "embeddings.npy").exists() and (c / "items.jsonl").exists():
            return c

    target = APP_DIR / "rag_multimodal_index"
    target.mkdir(parents=True, exist_ok=True)
    hf_token = _get_cfg("HF_TOKEN") or None
    for fname in ("embeddings.npy", "items.jsonl"):
        if not (target / fname).exists():
            dl = hf_hub_download(
                repo_id=INDEX_HF_REPO,
                repo_type="dataset",
                filename=f"rag_multimodal_index/{fname}",
                token=hf_token,
            )
            (target / fname).write_bytes(Path(dl).read_bytes())
    return target


class GrkrajIndex:
    def __init__(self, idx_dir: Path):
        self.items = [
            json.loads(line)
            for line in (idx_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.vecs = np.load(idx_dir / "embeddings.npy").astype(np.float32)
        self.dim = int(self.vecs.shape[1])

        self.text_indices = np.array(
            [i for i, x in enumerate(self.items) if x["modality"] == "text"], dtype=np.int64
        )
        self.img_indices = np.array(
            [i for i, x in enumerate(self.items) if x["modality"] == "image"], dtype=np.int64
        )

        # Lightweight in-memory BM25 over text passages
        self.doc_tfs: list[Counter] = []
        self.doc_lens = np.zeros(len(self.text_indices), dtype=np.float32)
        df: Counter = Counter()
        for pos, idx in enumerate(self.text_indices):
            c = self.items[int(idx)]
            h_toks = tokenize(f"{c['chapter']} {c['subchapter']} {c['heading']}")
            toks = h_toks + h_toks + tokenize(c["text"])
            tf = Counter(toks)
            self.doc_tfs.append(tf)
            self.doc_lens[pos] = len(toks)
            for term in tf:
                df[term] += 1
        self.avg_dl = float(self.doc_lens.mean()) if len(self.text_indices) else 1.0
        n_docs = len(self.text_indices)
        self.idf = {t: math.log((n_docs - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}

    def _bm25_text(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        scores = np.zeros(len(self.text_indices), dtype=np.float32)
        q_toks = tokenize(query)
        if not q_toks or not len(self.text_indices):
            return scores
        for term in q_toks:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for pos, tf_map in enumerate(self.doc_tfs):
                tf = tf_map.get(term, 0)
                if tf:
                    dl = self.doc_lens[pos]
                    scores[pos] += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / self.avg_dl))
        return scores

    def _rank_for_vector(
        self,
        q_vec: np.ndarray,
        query: str | None,
        top_text: int,
        top_images: int,
    ) -> tuple[list[dict], list[dict]]:
        all_dense = self.vecs @ q_vec

        text_hits: list[dict] = []
        if len(self.text_indices) and top_text > 0:
            t_dense = all_dense[self.text_indices]
            rrf: dict[int, float] = {}
            for r, pos in enumerate(np.argsort(-t_dense)[:50], 1):
                rrf[int(pos)] = rrf.get(int(pos), 0.0) + 1.0 / (60 + r)
            if query:
                bm25 = self._bm25_text(query)
                for r, pos in enumerate(np.argsort(-bm25)[:50], 1):
                    if bm25[pos] > 0:
                        rrf[int(pos)] = rrf.get(int(pos), 0.0) + 1.0 / (60 + r)
            for pos, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_text]:
                rec = dict(self.items[int(self.text_indices[pos])])
                rec["dense_score"] = round(float(t_dense[pos]), 4)
                text_hits.append(rec)

        image_hits: list[dict] = []
        if len(self.img_indices) and top_images > 0:
            i_dense = all_dense[self.img_indices]
            for pos in np.argsort(-i_dense)[:top_images]:
                rec = dict(self.items[int(self.img_indices[int(pos)])])
                rec["dense_score"] = round(float(i_dense[int(pos)]), 4)
                image_hits.append(rec)

        return text_hits, image_hits

    def search(
        self,
        client: genai.Client,
        query: str | None,
        image_bytes: bytes | None,
        image_mime: str | None,
        top_text: int = 4,
        top_images: int = 2,
    ) -> tuple[list[dict], list[dict]]:
        sub_queries = split_multi_topic_query(query) if (query and not image_bytes) else []
        if len(sub_queries) >= 2:
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=[
                    types.Content(parts=[types.Part.from_text(text=f"task: search biology lecture notes | query: {sq}")])
                    for sq in sub_queries
                ],
                config=types.EmbedContentConfig(output_dimensionality=self.dim),
            )
            eff_top_text = max(top_text, min(8, 2 * len(sub_queries)))
            eff_top_images = max(top_images, min(4, len(sub_queries)))
            per_t = max(2, math.ceil(eff_top_text / len(sub_queries)) + 1)
            per_i = max(2, math.ceil(eff_top_images / len(sub_queries)) + 1)
            t_buckets, i_buckets = [], []
            for sq, emb in zip(sub_queries, resp.embeddings):
                qv = l2_normalize(np.array(emb.values, dtype=np.float32))
                th, ih = self._rank_for_vector(qv, sq, per_t, per_i)
                for h in th:
                    h["matched_topic"] = sq
                for h in ih:
                    h["matched_topic"] = sq
                t_buckets.append(th)
                i_buckets.append(ih)

            text_hits, seen_t = [], set()
            for r in range(per_t):
                for b in t_buckets:
                    if r < len(b):
                        key = str(b[r].get("id") or (b[r]["chapter"], b[r]["heading"], b[r]["text"][:80]))
                        if key not in seen_t:
                            seen_t.add(key)
                            text_hits.append(b[r])
                            if len(text_hits) >= eff_top_text:
                                break
                if len(text_hits) >= eff_top_text:
                    break

            image_hits, seen_i = [], set()
            for r in range(per_i):
                for b in i_buckets:
                    if r < len(b):
                        key = str(b[r].get("image_path") or b[r].get("image_url") or b[r].get("id"))
                        if key not in seen_i:
                            seen_i.add(key)
                            image_hits.append(b[r])
                            if len(image_hits) >= eff_top_images:
                                break
                if len(image_hits) >= eff_top_images:
                    break
            return text_hits, image_hits

        parts = []
        if image_bytes and image_mime:
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))
        if query:
            parts.append(types.Part.from_text(text=f"task: search biology lecture notes | query: {query}"))
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=[types.Content(parts=parts)],
            config=types.EmbedContentConfig(output_dimensionality=self.dim),
        )
        q_vec = l2_normalize(np.array(resp.embeddings[0].values, dtype=np.float32))
        return self._rank_for_vector(q_vec, query, top_text, top_images)


@st.cache_resource(show_spinner="Loading grkraj.org 222-chapter multimodal index...")
def load_index() -> GrkrajIndex:
    return GrkrajIndex(ensure_index_present())


def label_item(it: dict) -> str:
    s = it["chapter"]
    if it.get("subchapter") and it["subchapter"] != it["chapter"]:
        s += f" — {it['subchapter']}"
    if it.get("heading") and it["heading"] != "Overview":
        s += f" ({it['heading']})"
    return s


def load_diagram_bytes(hit: dict) -> tuple[bytes | None, str]:
    """Load diagram locally if present, otherwise fetch live from grkraj.org image_url."""
    rel = hit.get("image_path")
    if rel:
        local_p = LOCAL_IMAGES_ROOT / rel
        if local_p.exists():
            data = local_p.read_bytes()
            mime = "image/jpeg" if local_p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            return data, mime

    url = hit.get("image_url")
    if url:
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and len(r.content) > 500:
                pil = Image.open(io.BytesIO(r.content)).convert("RGB")
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                return buf.getvalue(), "image/png"
        except Exception:
            pass
    return None, "image/png"


def linkify_citations(text: str, text_hits: list[dict], img_hits: list[dict]) -> str:
    """Convert [Text 1], [Text 1, Diagram 2], or bare Text 1 / Diagram 1 into clean clickable links."""
    t_urls = {i: h.get("page_url", "") for i, h in enumerate(text_hits, 1)}
    d_urls = {i: (h.get("image_url") or h.get("page_url") or "") for i, h in enumerate(img_hits, 1)}

    def _replace_bracket_group(m: re.Match) -> str:
        tokens = re.findall(r"\b(Text|Diagram)\s+(\d+)\b", m.group(1))
        if not tokens:
            return m.group(0)
        pills = []
        for kind, num_s in tokens:
            n = int(num_s)
            u = t_urls.get(n) if kind == "Text" else d_urls.get(n)
            pills.append(f"[{kind} {n}]({u})" if u else f"**{kind} {n}**")
        return " ".join(pills)

    text = re.sub(
        r"\[(\s*(?:(?:Text|Diagram)\s+\d+(?:\s*[,;/]\s*|\s+and\s+)?)+)\]",
        _replace_bracket_group,
        text,
    )
    for i, u in t_urls.items():
        if u:
            text = re.sub(rf"(?<!\[)\bText {i}\b(?!\])", f"[Text {i}]({u})", text)
    for i, u in d_urls.items():
        if u:
            text = re.sub(rf"(?<!\[)\bDiagram {i}\b(?!\])", f"[Diagram {i}]({u})", text)
    return text


def render_meta_bar(elapsed_s: float, text_hits: list[dict], img_hits: list[dict], quota_note: str) -> str:
    topics = list(dict.fromkeys(h.get("matched_topic") for h in text_hits if h.get("matched_topic")))
    topic_chip = ""
    if len(topics) >= 2:
        t_str = html_lib.escape(" · ".join(topics))
        topic_chip = f'<span class="meta-chip meta-topic">Multi-topic ({len(topics)}): {t_str}</span>'
    return (
        '<div class="meta-bar">'
        '<span class="meta-chip meta-ok">&#10003; Grounded in grkraj.org</span>'
        f'<span class="meta-chip">{len(text_hits)} passages &middot; {len(img_hits)} diagrams</span>'
        f'<span class="meta-chip">{elapsed_s:.1f}s</span>'
        f"{topic_chip}"
        f'<span class="meta-chip">{html_lib.escape(quota_note)}</span>'
        "</div>"
    )


def render_diagrams_html(diagrams: list[dict]) -> str:
    if not diagrams:
        return (
            '<div class="diag-empty">Matching biology diagrams from <strong>grkraj.org</strong> '
            "will appear here framed on crisp specimen slides.</div>"
        )
    cards = []
    for d in diagrams:
        i = d["idx"]
        title = html_lib.escape(d["title"])
        cap = html_lib.escape(d.get("caption", "")[:120])
        pu = html_lib.escape(d.get("page_url", ""))
        iu = html_lib.escape(d.get("image_url", ""))
        src = d.get("data_uri") or iu
        if not src:
            continue
        open_href = iu or pu or "#"
        cap_html = f'<div class="diag-cap">{cap}</div>' if cap else ""
        links = []
        if iu:
            links.append(f'<a class="src-action src-action-img" href="{iu}" target="_blank" rel="noopener">Full Image &#8599;</a>')
        if pu:
            links.append(f'<a class="src-action" href="{pu}" target="_blank" rel="noopener">Chapter &#8599;</a>')
        cards.append(
            '<div class="diag-tile">'
            f'<a class="diag-img-wrap" href="{open_href}" target="_blank" rel="noopener" title="Click to open full diagram">'
            f'<img src="{src}" alt="Diagram {i}">'
            "</a>"
            '<div class="diag-meta">'
            f'<div class="diag-title"><span class="cite-badge cite-diag">Diagram {i}</span> {title}</div>'
            f"{cap_html}"
            f'<div class="src-actions">{"".join(links)}</div>'
            "</div>"
            "</div>"
        )
    return f'<div class="diag-grid">{"".join(cards)}</div>'


def render_source_cards_html(text_hits: list[dict], img_hits: list[dict]) -> str:
    if not text_hits and not img_hits:
        return (
            '<div class="src-empty">Direct links to Prof. Kantharaj\'s chapters and diagrams on '
            "<strong>grkraj.org</strong> will appear here after you ask a question.</div>"
        )
    parts = ['<div class="src-wrap">']
    if text_hits:
        parts.append('<div class="src-section-title">Source Chapters on grkraj.org</div><div class="src-grid">')
        for i, h in enumerate(text_hits, 1):
            title = html_lib.escape(label_item(h))
            u = html_lib.escape(h.get("page_url", ""))
            pct = int(round(max(0.0, min(1.0, float(h.get("dense_score", 0.0)))) * 100))
            topic = h.get("matched_topic")
            topic_badge = f'<span class="src-topic">{html_lib.escape(topic)}</span>' if topic else ""
            snip = html_lib.escape((h.get("text") or "")[:150].replace("\n", " ").strip()) + "…"
            title_html = (
                f'<a class="src-title-link" href="{u}" target="_blank" rel="noopener">{title} &#8599;</a>'
                if u
                else f'<span class="src-title-link">{title}</span>'
            )
            parts.append(
                '<div class="src-card">'
                '<div class="src-card-top">'
                f'<span class="cite-badge cite-text">Text {i}</span>'
                f"{topic_badge}"
                f'<span class="score-pill">{pct}% match</span>'
                "</div>"
                f'<div class="src-title">{title_html}</div>'
                f'<div class="src-snip">{snip}</div>'
                "</div>"
            )
        parts.append("</div>")

    if img_hits:
        parts.append('<div class="src-section-title" style="margin-top:16px;">Matching Biology Diagrams on grkraj.org</div><div class="src-grid">')
        for i, h in enumerate(img_hits, 1):
            title = html_lib.escape(label_item(h))
            pu = html_lib.escape(h.get("page_url", ""))
            iu = html_lib.escape(h.get("image_url", ""))
            pct = int(round(max(0.0, min(1.0, float(h.get("dense_score", 0.0)))) * 100))
            topic = h.get("matched_topic")
            topic_badge = f'<span class="src-topic">{html_lib.escape(topic)}</span>' if topic else ""
            cap = html_lib.escape((h.get("caption") or "").strip()[:140])
            btn_links = []
            if pu:
                btn_links.append(f'<a class="src-action" href="{pu}" target="_blank" rel="noopener">Open Chapter &#8599;</a>')
            if iu:
                btn_links.append(f'<a class="src-action src-action-img" href="{iu}" target="_blank" rel="noopener">Direct Image &#8599;</a>')
            parts.append(
                '<div class="src-card src-card-diag">'
                '<div class="src-card-top">'
                f'<span class="cite-badge cite-diag">Diagram {i}</span>'
                f"{topic_badge}"
                f'<span class="score-pill">{pct}% match</span>'
                "</div>"
                f'<div class="src-title">{title}</div>'
                + (f'<div class="src-snip">{cap}</div>' if cap else "")
                + f'<div class="src-actions">{"".join(btn_links)}</div>'
                "</div>"
            )
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


CUSTOM_CSS = """
<style>
:root, html, body, .stApp {
    background-color: #060A14 !important;
    color: #E6EDF7 !important;
    color-scheme: dark !important;
}
.stApp {
    background-image:
        radial-gradient(ellipse 65% 45% at 10% -5%, rgba(16, 185, 129, 0.14), transparent 65%),
        radial-gradient(ellipse 55% 45% at 92% 0%, rgba(34, 211, 238, 0.11), transparent 60%),
        radial-gradient(ellipse 50% 40% at 50% 105%, rgba(59, 130, 246, 0.07), transparent 65%) !important;
    background-attachment: fixed !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
}
#MainMenu, footer { visibility: hidden; height: 0; }
.block-container {
    max-width: 1460px !important;
    padding: 1.2rem 1.6rem 2.5rem !important;
}

/* Hero Banner */
.hero-card {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 20px 24px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(13, 22, 41, 0.94) 0%, rgba(10, 18, 34, 0.90) 100%);
    border: 1px solid rgba(52, 211, 153, 0.24);
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    margin-bottom: 18px;
}
.hero-left { display: flex; align-items: center; gap: 16px; }
.hero-icon {
    width: 52px; height: 52px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(34, 211, 238, 0.12));
    border: 1px solid rgba(52, 211, 153, 0.36);
    box-shadow: 0 0 24px rgba(16, 185, 129, 0.18);
    flex-shrink: 0;
}
.hero-title {
    font-size: 1.58rem; font-weight: 800; letter-spacing: -0.025em;
    background: linear-gradient(90deg, #F8FAFC 0%, #A7F3D0 52%, #67E8F9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.hero-sub { font-size: 0.90rem; color: #94A3B8; margin-top: 3px; }
.hero-sub a { color: #5EEAD4 !important; text-decoration: none; font-weight: 600; }
.hero-sub a:hover { text-decoration: underline; }
.hero-pills { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.pill {
    font-size: 0.78rem; font-weight: 600; color: #CBD5E1;
    background: rgba(30, 41, 59, 0.78);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 999px; padding: 5px 12px;
    display: inline-flex; align-items: center; gap: 7px;
}
.pill-live { color: #6EE7B7; background: rgba(16, 185, 129, 0.13); border-color: rgba(52, 211, 153, 0.38); }
.pill-model { color: #A5F3FC; background: rgba(6, 182, 212, 0.11); border-color: rgba(34, 211, 238, 0.28); }
.live-dot { width: 8px; height: 8px; border-radius: 50%; background: #34D399; box-shadow: 0 0 10px #34D399; }

/* Glass Workbench Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(12, 20, 37, 0.86) !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 18px !important;
    padding: 18px !important;
    box-shadow: 0 12px 34px rgba(0, 0, 0, 0.38) !important;
}
.section-eyebrow {
    font-size: 0.74rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: #34D399; margin: 0 0 10px 2px;
}

/* Textarea, Inputs & Uploader */
textarea, input[type="text"], input[type="password"] {
    background-color: #070D1A !important;
    color: #F1F5F9 !important;
    border: 1px solid rgba(148, 163, 184, 0.22) !important;
    border-radius: 12px !important;
}
textarea:focus, input:focus {
    border-color: #10B981 !important;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.22) !important;
}
section[data-testid="stFileUploaderDropzone"] {
    background: #070D1A !important;
    border: 1.5px dashed rgba(52, 211, 153, 0.34) !important;
    border-radius: 14px !important;
}

/* Primary & Secondary Buttons */
button[kind="primary"] {
    background: linear-gradient(135deg, #10B981 0%, #06B6D4 100%) !important;
    color: #041118 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    min-height: 46px !important;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.28) !important;
    transition: transform 0.14s ease, box-shadow 0.14s ease !important;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #34D399 0%, #22D3EE 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 12px 28px rgba(34, 211, 238, 0.34) !important;
}
button[kind="secondary"] {
    background: rgba(15, 25, 45, 0.90) !important;
    color: #CBD5E1 !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    text-align: left !important;
    transition: border-color 0.15s ease, background 0.15s ease !important;
}
button[kind="secondary"]:hover {
    border-color: rgba(52, 211, 153, 0.55) !important;
    background: rgba(16, 185, 129, 0.11) !important;
    color: #F8FAFC !important;
}

/* Prominent Gemini API Key Card */
.api-card-header {
    background: linear-gradient(135deg, rgba(11, 22, 42, 0.96) 0%, rgba(12, 26, 46, 0.92) 100%);
    border: 1px solid rgba(34, 211, 238, 0.34);
    border-radius: 14px;
    padding: 13px 15px 11px;
    margin: 14px 0 8px;
    box-shadow: 0 6px 20px rgba(6, 182, 212, 0.10);
}
.api-card-header.api-warn {
    border-color: rgba(251, 191, 36, 0.48);
    background: linear-gradient(135deg, rgba(32, 24, 14, 0.92) 0%, rgba(16, 22, 38, 0.94) 100%);
    box-shadow: 0 6px 22px rgba(245, 158, 11, 0.12);
}
.api-card-header.api-ok {
    border-color: rgba(52, 211, 153, 0.42);
    background: linear-gradient(135deg, rgba(10, 29, 28, 0.92) 0%, rgba(12, 22, 40, 0.94) 100%);
}
.api-top {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 6px; margin-bottom: 6px;
}
.api-title {
    font-size: 0.84rem; font-weight: 700; color: #F8FAFC;
    display: inline-flex; align-items: center; gap: 6px;
}
.api-badge {
    font-size: 0.71rem; font-weight: 700; padding: 2px 8px; border-radius: 999px;
}
.api-badge-ok { color: #6EE7B7; background: rgba(16, 185, 129, 0.16); border: 1px solid rgba(52, 211, 153, 0.38); }
.api-badge-cyan { color: #67E8F9; background: rgba(6, 182, 212, 0.16); border: 1px solid rgba(34, 211, 238, 0.38); }
.api-badge-warn { color: #FDE68A; background: rgba(245, 158, 11, 0.18); border: 1px solid rgba(251, 191, 36, 0.45); }
.api-desc {
    font-size: 0.80rem; color: #CBD5E1; line-height: 1.46; margin-bottom: 7px;
}
.api-desc a {
    color: #67E8F9 !important; font-weight: 600; text-decoration: none;
}
.api-desc a:hover { text-decoration: underline; }
.api-privacy {
    font-size: 0.73rem; color: #94A3B8; margin-top: 2px;
}

/* Meta Bar */
.meta-bar { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.meta-chip {
    font-size: 0.77rem; font-weight: 600; color: #94A3B8;
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid rgba(148, 163, 184, 0.18);
    padding: 4px 10px; border-radius: 999px;
}
.meta-ok { color: #6EE7B7; background: rgba(16, 185, 129, 0.13); border-color: rgba(52, 211, 153, 0.35); }
.meta-topic { color: #67E8F9; background: rgba(6, 182, 212, 0.12); border-color: rgba(34, 211, 238, 0.30); }

/* Answer Card + Color-Coded Clickable Citation Pills */
.st-key-answer_card {
    background: rgba(8, 14, 27, 0.85) !important;
    border: 1px solid rgba(148, 163, 184, 0.14) !important;
    border-radius: 14px !important;
    padding: 20px 24px !important;
    margin-bottom: 14px !important;
}
.st-key-answer_card p, .st-key-answer_card li {
    color: #E2E8F0 !important;
    font-size: 0.97rem !important;
    line-height: 1.72 !important;
}
.st-key-answer_card h1, .st-key-answer_card h2, .st-key-answer_card h3, .st-key-answer_card h4 {
    color: #F8FAFC !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em;
}
.st-key-answer_card a {
    color: #6EE7B7 !important;
    background: rgba(16, 185, 129, 0.14);
    border: 1px solid rgba(52, 211, 153, 0.34);
    border-radius: 6px;
    padding: 1px 6px;
    margin: 0 1px 0 3px;
    font-weight: 600;
    font-size: 0.84em;
    text-decoration: none !important;
    white-space: nowrap;
}
.st-key-answer_card a[href*="_files/"] {
    color: #67E8F9 !important;
    background: rgba(6, 182, 212, 0.14);
    border-color: rgba(34, 211, 238, 0.36);
}
.st-key-answer_card a:hover {
    background: rgba(16, 185, 129, 0.26);
    border-color: rgba(52, 211, 153, 0.65);
    color: #ECFDF5 !important;
}
.st-key-answer_card a[href*="_files/"]:hover {
    background: rgba(6, 182, 212, 0.26);
    border-color: rgba(34, 211, 238, 0.68);
    color: #ECFEFF !important;
}

/* Specimen-Framed Biology Diagram Cards */
.diag-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(205px, 1fr));
    gap: 12px;
    margin: 6px 0 14px;
}
.diag-tile {
    background: #101A2F;
    border: 1px solid rgba(148, 163, 184, 0.20);
    border-radius: 14px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    transition: transform 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease;
}
.diag-tile:hover {
    transform: translateY(-2px);
    border-color: rgba(52, 211, 153, 0.55);
    box-shadow: 0 10px 26px rgba(16, 185, 129, 0.16);
}
.diag-img-wrap {
    display: flex; align-items: center; justify-content: center;
    background: #FFFFFF;
    border-radius: 10px;
    padding: 8px;
    min-height: 150px;
    max-height: 250px;
    overflow: hidden;
}
.diag-img-wrap img {
    max-width: 100%; max-height: 250px; object-fit: contain; display: block;
}
.diag-meta { display: flex; flex-direction: column; gap: 6px; padding: 0 2px; }
.diag-title { font-size: 0.85rem; font-weight: 600; color: #F1F5F9; line-height: 1.38; }
.diag-cap { font-size: 0.78rem; color: #94A3B8; line-height: 1.42; }
.diag-empty, .src-empty {
    color: #64748B; font-size: 0.90rem; padding: 16px; text-align: center;
    border: 1px dashed rgba(148, 163, 184, 0.18); border-radius: 12px; margin-bottom: 10px;
}

/* Rich Source Cards */
.src-wrap { padding: 4px 2px 6px; }
.src-section-title {
    font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #94A3B8; margin-bottom: 10px;
}
.src-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px;
}
.src-card {
    background: rgba(15, 24, 44, 0.88);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 12px; padding: 12px 14px;
    display: flex; flex-direction: column; gap: 7px;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
.src-card:hover { border-color: rgba(52, 211, 153, 0.42); transform: translateY(-1px); }
.src-card-diag:hover { border-color: rgba(34, 211, 238, 0.48); }
.src-card-top { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; }
.cite-badge { font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 6px; letter-spacing: 0.02em; }
.cite-text { color: #6EE7B7; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(52, 211, 153, 0.35); }
.cite-diag { color: #67E8F9; background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(34, 211, 238, 0.38); }
.src-topic { font-size: 0.71rem; font-weight: 600; color: #CBD5E1; background: rgba(51, 65, 85, 0.65); padding: 2px 7px; border-radius: 6px; }
.score-pill { margin-left: auto; font-size: 0.71rem; font-weight: 600; color: #94A3B8; font-family: Consolas, monospace; }
.src-title { font-size: 0.87rem; font-weight: 600; color: #F1F5F9; line-height: 1.38; }
.src-title-link { color: #E2E8F0 !important; text-decoration: none !important; }
.src-title-link:hover { color: #6EE7B7 !important; }
.src-snip { font-size: 0.79rem; color: #94A3B8; line-height: 1.46; }
.src-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 2px; }
.src-action {
    font-size: 0.75rem; font-weight: 600; color: #6EE7B7 !important;
    background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(52, 211, 153, 0.32);
    padding: 3px 9px; border-radius: 7px; text-decoration: none !important;
}
.src-action:hover { background: rgba(16, 185, 129, 0.24); }
.src-action-img { color: #67E8F9 !important; background: rgba(6, 182, 212, 0.12); border-color: rgba(34, 211, 238, 0.34); }
.src-action-img:hover { background: rgba(6, 182, 212, 0.24); }
</style>
"""

HERO_HTML = """
<div class="hero-card">
  <div class="hero-left">
    <div class="hero-icon" aria-hidden="true">
      <svg width="34" height="34" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="helixGrad" x1="4" y1="4" x2="32" y2="32" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stop-color="#34D399"/>
            <stop offset="100%" stop-color="#22D3EE"/>
          </linearGradient>
        </defs>
        <path d="M10 4C10 12 26 12 26 20C26 28 10 28 10 32" stroke="url(#helixGrad)" stroke-width="2.6" stroke-linecap="round"/>
        <path d="M26 4C26 12 10 12 10 20C10 28 26 28 26 32" stroke="#60A5FA" stroke-opacity="0.85" stroke-width="2.6" stroke-linecap="round"/>
        <line x1="12.5" y1="9" x2="23.5" y2="9" stroke="#94A3B8" stroke-width="1.6" stroke-linecap="round"/>
        <line x1="12.5" y1="27" x2="23.5" y2="27" stroke="#94A3B8" stroke-width="1.6" stroke-linecap="round"/>
        <circle cx="18" cy="18" r="2.2" fill="#34D399"/>
      </svg>
    </div>
    <div>
      <div class="hero-title">grkraj.org Biology Tutor</div>
      <div class="hero-sub">
        Interactive Molecular &amp; Plant Cell Biology Workbench &middot;
        Notes &amp; diagrams by <a href="https://grkraj.org" target="_blank" rel="noopener">Prof. Dr. G. R. Kantharaj</a> (Bangalore University)
      </div>
    </div>
  </div>
  <div class="hero-pills">
    <span class="pill pill-live"><span class="live-dot"></span>Free Web App</span>
    <span class="pill">222 Chapters &middot; 9,234 Diagrams</span>
    <span class="pill pill-model">Gemini Embedding 2 &plus; Gemini 3.8 Flash</span>
  </div>
</div>
"""

WELCOME_MD = """### Welcome to Your Biology Study Workbench

Ask any question from **Prof. Dr. G. R. Kantharaj's 222 chapters** (*Molecular Biology*, *Plant Cell Biology*, and *Pre-University Biology*), or upload a photo of a biology diagram on the left.

- **Grounded with clickable footnotes:** Every `Text 1` (emerald) and `Diagram 1` (cyan) links straight to the source page or image on [grkraj.org](https://grkraj.org).
- **Multi-topic revision:** Ask about several topics at once (e.g., *"Explain photosynthesis, mitosis and DNA replication"*) — every topic gets dedicated passages and diagrams.
- **Works on any device:** Phones, Chromebooks, and laptops with zero local installation.

*Click any example prompt on the left or type your own question to begin.*
"""

EXAMPLES = [
    "Explain photorespiration in C3 plants.",
    "Explain photosynthesis, mitosis and DNA replication",
    "Explain TATA box and eukaryotic transcription initiation.",
    "What properties make a good cloning vector?",
    "Explain antibody-antigen interactions and opsonization.",
]


def _set_example(prompt: str):
    st.session_state["question_input"] = prompt
    st.session_state["run_query"] = True


def _clear_all():
    st.session_state["question_input"] = ""
    st.session_state["run_query"] = False
    st.session_state["last_result"] = None


class GlobalDailyLimiter:
    """Thread-safe app-wide daily safety cap shared across all visitors."""

    def __init__(self, limit: int):
        self.limit = limit
        self._lock = threading.Lock()
        self._day = date.today().isoformat()
        self._used = 0

    def _reset_if_new_day(self):
        today = date.today().isoformat()
        if self._day != today:
            self._day = today
            self._used = 0

    def remaining(self) -> int:
        with self._lock:
            self._reset_if_new_day()
            return max(0, self.limit - self._used)

    def consume(self) -> tuple[bool, int]:
        with self._lock:
            self._reset_if_new_day()
            if self._used >= self.limit:
                return False, 0
            self._used += 1
            return True, self.limit - self._used


@st.cache_resource
def get_global_limiter(limit: int) -> GlobalDailyLimiter:
    return GlobalDailyLimiter(limit)


def _session_remaining() -> int:
    today = date.today().isoformat()
    q_state = st.session_state.get("quota", {"day": today, "used": 0})
    if q_state["day"] != today:
        return DAILY_FREE_LIMIT
    return max(0, DAILY_FREE_LIMIT - q_state["used"])


def _consume_shared_quota() -> tuple[bool, str, int, int]:
    """Enforce both per-session DAILY_FREE_LIMIT and app-wide GLOBAL_DAILY_LIMIT."""
    today = date.today().isoformat()
    q_state = st.session_state.get("quota", {"day": today, "used": 0})
    if q_state["day"] != today:
        q_state = {"day": today, "used": 0}
    if q_state["used"] >= DAILY_FREE_LIMIT:
        return False, "session", 0, get_global_limiter(GLOBAL_DAILY_LIMIT).remaining()

    ok_global, glob_rem = get_global_limiter(GLOBAL_DAILY_LIMIT).consume()
    if not ok_global:
        return False, "global", max(0, DAILY_FREE_LIMIT - q_state["used"]), 0

    q_state["used"] += 1
    st.session_state["quota"] = q_state
    return True, "ok", DAILY_FREE_LIMIT - q_state["used"], glob_rem


def run_tutor_query(question: str, uploaded_file, user_api_key: str) -> dict:
    q = (question or "").strip()
    if not q and uploaded_file is None:
        return {"error": "Please enter a biology question or upload a biology diagram image."}

    custom_key = (user_api_key or "").strip()
    server_key = _get_cfg("GEMINI_API_KEY") or _get_cfg("GOOGLE_API_KEY")

    if custom_key:
        api_key = custom_key
        quota_note = "Personal Gemini key (unlimited)"
    elif server_key:
        ok, reason, sess_rem, glob_rem = _consume_shared_quota()
        if not ok:
            if reason == "global":
                return {
                    "error": (
                        f"Today's shared community safety limit ({GLOBAL_DAILY_LIMIT} questions/day across all visitors) has been reached. "
                        "Paste your own free Gemini API key in the **Gemini API Key** card on the left (get one free at https://aistudio.google.com/apikey) to continue immediately!"
                    )
                }
            return {
                "error": (
                    f"Session free limit ({DAILY_FREE_LIMIT} questions/day) reached. "
                    "Paste your own free Gemini API key in the **Gemini API Key** card on the left (https://aistudio.google.com/apikey) for unlimited questions!"
                )
            }
        api_key = server_key
        quota_note = f"{sess_rem}/{DAILY_FREE_LIMIT} session · {glob_rem}/{GLOBAL_DAILY_LIMIT} community left today"
    else:
        return {
            "error": (
                "Please paste your free **Gemini API Key** in the highlighted **Gemini API Key** box on the left to ask questions "
                "(free in 30 seconds at https://aistudio.google.com/apikey)."
            )
        }

    t0 = time.monotonic()
    q_img_bytes = None
    q_img_mime = None
    if uploaded_file is not None:
        im = Image.open(uploaded_file).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        q_img_bytes = buf.getvalue()
        q_img_mime = "image/png"

    effective_q = q or "Explain the biological structures and pathway shown in this diagram using the lecture notes."

    index = load_index()
    client = genai.Client(api_key=api_key)
    text_hits, img_hits = index.search(
        client,
        query=q or None,
        image_bytes=q_img_bytes,
        image_mime=q_img_mime,
        top_text=4,
        top_images=2,
    )

    parts = [
        types.Part.from_text(
            text=(
                "You are a helpful biology tutor answering students strictly from "
                "Prof. Dr. G. R. Kantharaj's lecture notes and diagrams (grkraj.org). "
                "Address EVERY topic asked by the student in clear structured sections, "
                "and ALWAYS cite sources inline as [Text 1], [Text 2], [Diagram 1], [Diagram 2] in each section. "
                "If a detail is not in the retrieved notes, say so.\n\n"
            )
        )
    ]
    for i, h in enumerate(text_hits, 1):
        parts.append(types.Part.from_text(text=f"[Text {i}: {label_item(h)}]\n{h['text']}\n\n"))

    diagrams_payload = []
    for i, h in enumerate(img_hits, 1):
        img_bytes, img_mime = load_diagram_bytes(h)
        title = label_item(h)
        cap = (h.get("caption") or "").strip()
        if img_bytes:
            parts.append(types.Part.from_text(text=f"[Diagram {i}: {title}] {cap[:120]}\n"))
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type=img_mime))
            b64 = base64.b64encode(img_bytes).decode("ascii")
            data_uri = f"data:{img_mime};base64,{b64}"
        else:
            data_uri = ""
        diagrams_payload.append(
            {
                "idx": i,
                "title": title,
                "caption": cap,
                "page_url": h.get("page_url", ""),
                "image_url": h.get("image_url", ""),
                "data_uri": data_uri,
            }
        )

    if q_img_bytes:
        parts.append(types.Part.from_text(text="\n[Student Uploaded Diagram]\n"))
        parts.append(types.Part.from_bytes(data=q_img_bytes, mime_type=q_img_mime))

    parts.append(types.Part.from_text(text=f"\nStudent Question: {effective_q}"))

    resp = client.models.generate_content(
        model=GEN_MODEL,
        contents=types.Content(parts=parts),
    )
    raw_ans = (resp.text or "").strip()
    elapsed = time.monotonic() - t0

    return {
        "meta_html": render_meta_bar(elapsed, text_hits, img_hits, quota_note),
        "answer_md": linkify_citations(raw_ans, text_hits, img_hits),
        "diagrams": diagrams_payload,
        "text_hits": text_hits,
        "img_hits": img_hits,
    }


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(HERO_HTML, unsafe_allow_html=True)

if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "run_query" not in st.session_state:
    st.session_state["run_query"] = False

# Pre-warm the index once per server process
_ = load_index()

col_left, col_right = st.columns([5, 7], gap="large")

with col_left:
    with st.container(border=True):
        st.markdown('<div class="section-eyebrow">Ask a Question or Upload a Diagram</div>', unsafe_allow_html=True)
        question = st.text_area(
            "Biology Question",
            key="question_input",
            placeholder="e.g., Explain photorespiration in C3 plants — or ask multiple topics like: Explain photosynthesis, mitosis and DNA replication",
            height=105,
        )
        uploaded_img = st.file_uploader(
            "Optional: Upload or Snap a Biology Diagram",
            type=["png", "jpg", "jpeg", "webp"],
        )
        if uploaded_img is not None:
            st.image(uploaded_img, caption="Uploaded biology diagram", use_container_width=True)

        b1, b2 = st.columns([3, 1], gap="small")
        with b1:
            ask_clicked = st.button("Ask Biology Tutor", type="primary", use_container_width=True)
        with b2:
            st.button("Clear", type="secondary", use_container_width=True, on_click=_clear_all)

        # Prominent Gemini API Key & Live Community Quota Card
        _has_server_key = bool(_get_cfg("GEMINI_API_KEY") or _get_cfg("GOOGLE_API_KEY"))
        _glob_left = get_global_limiter(GLOBAL_DAILY_LIMIT).remaining()
        _sess_left = _session_remaining()
        _cur_user_key = (st.session_state.get("user_api_key_input") or "").strip()

        if _cur_user_key:
            _card_cls = "api-card-header api-ok"
            _badge_html = '<span class="api-badge api-badge-ok">&#10003; Personal Key Active (Unlimited)</span>'
            _desc_html = (
                'Using your personal Gemini API key for unlimited questions. '
                'Need another key? <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener">Google AI Studio &#8599;</a>'
            )
        elif _has_server_key and _glob_left > 0 and _sess_left > 0:
            _card_cls = "api-card-header"
            _badge_html = f'<span class="api-badge api-badge-cyan">Free Pool: {_glob_left}/{GLOBAL_DAILY_LIMIT} left today</span>'
            _desc_html = (
                f'Shared free quota active ({_sess_left}/{DAILY_FREE_LIMIT} left in your session; {GLOBAL_DAILY_LIMIT}/day site-wide cap). '
                'For unlimited questions, paste your own free key from '
                '<a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener">Google AI Studio &#8599;</a>:'
            )
        elif _has_server_key:
            _card_cls = "api-card-header api-warn"
            _badge_html = '<span class="api-badge api-badge-warn">Free Quota Reached Today</span>'
            _desc_html = (
                f'Daily free quota ({GLOBAL_DAILY_LIMIT}/day global or {DAILY_FREE_LIMIT}/session) is used up. '
                'Paste your own free key from '
                '<a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener">Google AI Studio &#8599;</a> '
                'below to keep studying:'
            )
        else:
            _card_cls = "api-card-header api-warn"
            _badge_html = '<span class="api-badge api-badge-warn">Required &middot; Free Key</span>'
            _desc_html = (
                'Paste your own free <strong>Gemini API Key</strong> below to start asking questions — '
                'get one free in 30 seconds at '
                '<a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener">Google AI Studio &#8599;</a>:'
            )

        st.markdown(
            f'<div class="{_card_cls}">'
            '<div class="api-top">'
            '<span class="api-title">&#128273; Gemini API Key</span>'
            f"{_badge_html}"
            "</div>"
            f'<div class="api-desc">{_desc_html}</div>'
            '<div class="api-privacy">&#128274; Kept strictly in session memory — never saved to disk or shared.</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        user_api_key = st.text_input(
            "Gemini API Key",
            key="user_api_key_input",
            type="password",
            label_visibility="collapsed",
            placeholder="Paste Gemini API key (AIza...) — free at aistudio.google.com/apikey",
        )

        st.markdown(
            '<div class="section-eyebrow" style="margin-top:16px;">Try a Molecular or Plant Biology Question</div>',
            unsafe_allow_html=True,
        )
        for idx, ex in enumerate(EXAMPLES):
            st.button(ex, key=f"ex_{idx}", type="secondary", use_container_width=True, on_click=_set_example, args=(ex,))

should_run = ask_clicked or st.session_state.pop("run_query", False)
if should_run:
    with col_right:
        with st.spinner("Searching all 222 chapters & 9,234 diagrams and generating grounded explanation..."):
            try:
                st.session_state["last_result"] = run_tutor_query(question, uploaded_img, user_api_key)
            except Exception as exc:
                st.session_state["last_result"] = {"error": f"Error querying Gemini API: `{type(exc).__name__}: {exc}`"}

with col_right:
    with st.container(border=True):
        st.markdown('<div class="section-eyebrow">Grounded Explanation &amp; Biology Diagrams</div>', unsafe_allow_html=True)
        res = st.session_state.get("last_result")

        if res is None:
            with st.container(key="answer_card"):
                st.markdown(WELCOME_MD)
            st.markdown(render_diagrams_html([]), unsafe_allow_html=True)
            with st.expander("Direct Links to grkraj.org Source Chapters & Diagrams", expanded=False):
                st.markdown(render_source_cards_html([], []), unsafe_allow_html=True)
        elif res.get("error"):
            st.error(res["error"])
        else:
            st.markdown(res["meta_html"], unsafe_allow_html=True)
            with st.container(key="answer_card"):
                st.markdown(res["answer_md"])
            st.markdown('<div class="section-eyebrow" style="margin-top:4px;">Matching Biology Diagrams (Click to Open Full Size)</div>', unsafe_allow_html=True)
            st.markdown(render_diagrams_html(res["diagrams"]), unsafe_allow_html=True)
            with st.expander("Direct Links to grkraj.org Source Chapters & Diagrams", expanded=True):
                st.markdown(render_source_cards_html(res["text_hits"], res["img_hits"]), unsafe_allow_html=True)
