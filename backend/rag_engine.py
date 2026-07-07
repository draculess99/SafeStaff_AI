"""Lightweight local RAG engine for SafeStaff AI.

This module keeps RAG practical for a capstone/demo deployment:
- no paid vector database
- no new heavy dependencies beyond scikit-learn, already used by the app
- local JSON persistence under database/rag_documents.json
- TF-IDF retrieval over policy/SOP chunks

It is intended for decision-support context only. It does not replace clinical,
legal, union, or hospital policy review.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import uuid
from typing import Any, Dict, Iterable, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAG_STORE_PATH = os.path.join(BASE_DIR, "database", "rag_documents.json")


DEFAULT_RAG_DOCUMENTS: List[Dict[str, str]] = [{'title': 'ED Surge Staffing Escalation SOP',
  'source': 'Demo hospital policy seed',
  'category': 'Staffing SOP',
  'text': 'Emergency Department surge staffing should be escalated when adjusted operational risk is High or '
          'Critical, when the waiting room exceeds 35 patients, when ambulance arrival pressure is High or '
          'Critical, or when the predicted wait-time gap exceeds the safety threshold. First actions should '
          'include opening fast-track if low-acuity volume is high, calling in qualified Emergency nurses, '
          'and notifying the nursing supervisor. Critical surge requires explicit human approval before '
          'roster changes are committed.'},
 {'title': 'Nurse Fatigue and Overtime Guardrails',
  'source': 'Demo hospital policy seed',
  'category': 'Compliance',
  'text': 'Nurses projected above 60 weekly hours should not be selected for additional shifts except under '
          'executive emergency authorization. Nurses already above 48 weekly hours should be treated as '
          'fatigue-sensitive and should be chosen only when safer alternatives are unavailable. Night-shift '
          'assignments should prefer nurses with Night or Flexible circadian preference when the skill match '
          'is equivalent.'},
 {'title': 'Boarding Gridlock Escalation Playbook',
  'source': 'Demo hospital policy seed',
  'category': 'Patient Flow',
  'text': 'Boarding pressure becomes severe when inpatient bed occupancy exceeds 95 percent, boarding count '
          'exceeds 15, or average boarding time exceeds 4 hours. Recommended actions include notifying bed '
          'management, prioritizing inpatient discharges, adding charge nurse coordination, and protecting '
          'ED nurse staffing for active emergency care rather than absorbing inpatient boarding work '
          'silently.'},
 {'title': 'Fast Track Activation Criteria',
  'source': 'Demo hospital policy seed',
  'category': 'Fast Track',
  'text': 'Fast-track should be opened or expanded when low-acuity patients exceed 35 percent of the waiting '
          'room and a qualified nurse or advanced practice provider is available. If fast-track is closed '
          'during a Friday night or winter surge, the staffing committee should consider one additional '
          'Emergency nurse plus operational support to reduce waiting-room crowding.'},
 {'title': 'Critical Care Skill-Match Requirement',
  'source': 'Demo hospital policy seed',
  'category': 'Skill Mix',
  'text': 'Critical-care, trauma, resuscitation, and high-acuity ED zones require nurses with documented '
          'Emergency, ICU, or critical-care competency. The staffing engine should not solve a critical '
          'shortage by assigning only general medical-surgical coverage when the scenario includes high '
          'acuity, ambulance surge, sepsis alerts, cardiac monitoring load, or resuscitation bay demand. '
          'When skill match is incomplete, the recommended action is to call a qualified specialist nurse, '
          'open charge-nurse review, or escalate to the house supervisor rather than silently filling the '
          'slot with an unqualified resource.'},
 {'title': 'Charge Nurse Command Center Escalation',
  'source': 'Demo hospital policy seed',
  'category': 'Command Center',
  'text': 'A charge nurse or nursing supervisor should be notified when any two of the following are '
          'present: waiting room above 30, ambulance pressure High or Critical, inpatient occupancy above 92 '
          'percent, predicted wait time above target by more than 30 minutes, or more than two nurse '
          'call-outs in the same shift. The command center should review staffing changes, float-pool calls, '
          'bed-flow constraints, and whether a temporary fast-track or discharge-lounge action is safer than '
          'simply adding overtime.'},
 {'title': 'Float Pool and Cross-Cover Utilization Rules',
  'source': 'Demo hospital policy seed',
  'category': 'Float Pool',
  'text': 'Float-pool nurses should be considered before agency or high-overtime assignments when their '
          'competency profile matches the shortage. Cross-cover assignments must respect department '
          'qualification, orientation status, and recent shift load. A float nurse can stabilize '
          'lower-acuity zones, observation support, discharge coordination, or fast-track intake, but should '
          'not be used as the sole replacement for critical-care ED coverage unless credentialed for that '
          'role.'},
 {'title': 'Agency Nurse Cost and Approval Policy',
  'source': 'Demo hospital policy seed',
  'category': 'Cost Control',
  'text': 'Agency nurse utilization should be treated as a last-line option when internal overtime, '
          'voluntary call-in, float pool, and schedule swaps cannot safely cover the shortage. If agency '
          'coverage is recommended, the decision summary should explain the patient safety trigger, expected '
          'wait-time reduction, estimated incremental cost, and why lower-cost options were insufficient. '
          'Agency use for more than one shift in a 24-hour period requires nursing leadership approval in '
          'this demo policy.'},
 {'title': 'ESI Acuity Surge Staffing Matrix',
  'source': 'Demo hospital policy seed',
  'category': 'Acuity Matrix',
  'text': 'When ESI 1-2 volume rises or the acuity mix shifts upward, staffing recommendations should '
          'prioritize experienced ED and critical-care nurses over generic headcount. Low-acuity surges '
          'should first trigger fast-track and intake redesign, while high-acuity surges should trigger '
          'resuscitation coverage, triage reinforcement, monitored-bed support, and charge nurse escalation. '
          'The model should distinguish between volume pressure and acuity pressure because the safest '
          'intervention may be different.'},
 {'title': 'Ambulance Offload Delay Escalation Rule',
  'source': 'Demo hospital policy seed',
  'category': 'EMS Offload',
  'text': 'If ambulance arrivals are high and offload delay risk is present, staffing decisions should '
          'protect triage, resuscitation, and monitored-bed turnover. Recommended interventions include '
          'adding an ED nurse with ambulance intake experience, notifying bed management, opening rapid '
          'assessment if possible, and avoiding actions that move experienced nurses away from triage during '
          'EMS surge periods. Persistent offload delay should be escalated to the house supervisor.'},
 {'title': 'Pediatric and Behavioral Health Coverage Safeguard',
  'source': 'Demo hospital policy seed',
  'category': 'Special Populations',
  'text': 'Pediatric, behavioral health, isolation, and one-to-one observation demand should be treated as '
          'special coverage constraints. The staffing engine should flag when a shortage affects these '
          'populations because normal adult ED staffing substitutions may be unsafe. If appropriate '
          'competency is unavailable, the recommendation should include leadership escalation, sitter-pool '
          'review, behavioral-health consult coordination, or temporary zone closure rather than pretending '
          'the gap is solved.'},
 {'title': 'Break Relief and Meal Coverage Compliance',
  'source': 'Demo hospital policy seed',
  'category': 'Compliance',
  'text': 'Staffing plans should preserve break relief and meal coverage whenever possible. A solution that '
          'fills a visible shortage but removes all break coverage should be marked as operationally '
          'fragile. During high-risk periods, the recommendation should explicitly state whether break '
          'coverage remains viable, whether charge nurse relief is needed, and whether the plan creates '
          'fatigue or compliance risk later in the shift.'},
 {'title': 'Winter Respiratory Surge Playbook',
  'source': 'Demo hospital policy seed',
  'category': 'Seasonal Surge',
  'text': 'During winter respiratory surge, ED staffing should anticipate higher arrival volatility, '
          'increased isolation-room turnover, respiratory therapy coordination, and longer inpatient '
          'boarding. The first response should combine nurse staffing, fast-track respiratory protocols for '
          'low-acuity cases, bed-flow escalation, and supply readiness for PPE and oxygen-related workflows. '
          'The staffing committee should avoid using historical average demand alone when current arrival '
          'pressure is rising rapidly.'},
 {'title': 'Human Approval and Audit Trail Requirement',
  'source': 'Demo hospital policy seed',
  'category': 'Governance',
  'text': 'AI-generated staffing recommendations are decision support only and require human approval before '
          'schedule changes are committed. The audit trail should capture scenario inputs, predicted '
          'wait-time risk, retrieved RAG policy evidence, agent debate summary, selected intervention, '
          'projected impact, and the approving supervisor. High-risk or high-cost decisions should include a '
          'short rationale explaining why the recommended plan is safer than available alternatives.'}]


def _now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_store() -> Dict[str, Any]:
    os.makedirs(os.path.dirname(RAG_STORE_PATH), exist_ok=True)
    if not os.path.exists(RAG_STORE_PATH):
        reset_rag_documents()
    try:
        with open(RAG_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {"documents": []}
    if "documents" not in data or not isinstance(data.get("documents"), list):
        data = {"documents": []}
    return data


def _write_store(data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(RAG_STORE_PATH), exist_ok=True)
    with open(RAG_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _clean_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 900, overlap_chars: int = 120) -> List[str]:
    """Split policy text into retrieval-friendly chunks."""
    text = _clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(paragraph):
                chunk = paragraph[start : start + max_chars].strip()
                if chunk:
                    chunks.append(chunk)
                start += max_chars - overlap_chars
            continue
        candidate = (current + "\n\n" + paragraph).strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = paragraph
    if current:
        chunks.append(current.strip())
    return chunks


def _document_record(title: str, text: str, source: str = "Manual upload", category: str = "Policy") -> Dict[str, Any]:
    doc_id = "RAG_" + uuid.uuid4().hex[:8].upper()
    cleaned = _clean_text(text)
    chunks = []
    for i, chunk in enumerate(chunk_text(cleaned), start=1):
        chunks.append(
            {
                "chunk_id": f"{doc_id}_CHUNK_{i:03d}",
                "document_id": doc_id,
                "title": title or "Untitled document",
                "source": source or "Manual upload",
                "category": category or "Policy",
                "text": chunk,
            }
        )
    return {
        "id": doc_id,
        "title": title or "Untitled document",
        "source": source or "Manual upload",
        "category": category or "Policy",
        "created_at": _now_iso(),
        "text": cleaned,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def reset_rag_documents() -> Dict[str, Any]:
    docs = [_document_record(**doc) for doc in DEFAULT_RAG_DOCUMENTS]
    data = {"documents": docs, "reset_at": _now_iso()}
    _write_store(data)
    return rag_status(data)


def list_rag_documents() -> List[Dict[str, Any]]:
    data = _ensure_store()
    docs = []
    for d in data.get("documents", []):
        docs.append(
            {
                "id": d.get("id"),
                "title": d.get("title", "Untitled document"),
                "source": d.get("source", "Unknown"),
                "category": d.get("category", "Policy"),
                "created_at": d.get("created_at", ""),
                "chunk_count": d.get("chunk_count", len(d.get("chunks", []))),
                "characters": len(d.get("text", "")),
            }
        )
    return docs


def add_rag_document(title: str, text: str, source: str = "Manual upload", category: str = "Policy") -> Dict[str, Any]:
    cleaned = _clean_text(text)
    if len(cleaned) < 20:
        raise ValueError("Document text is too short for retrieval. Add at least a few sentences.")
    data = _ensure_store()
    doc = _document_record(title=title, text=cleaned, source=source, category=category)
    data.setdefault("documents", []).append(doc)
    _write_store(data)
    return doc


def delete_rag_document(doc_id: str) -> bool:
    data = _ensure_store()
    before = len(data.get("documents", []))
    data["documents"] = [d for d in data.get("documents", []) if d.get("id") != doc_id]
    _write_store(data)
    return len(data.get("documents", [])) < before


def _all_chunks(category: str | None = None) -> List[Dict[str, Any]]:
    data = _ensure_store()
    chunks: List[Dict[str, Any]] = []
    category_norm = (category or "").strip().lower()
    for doc in data.get("documents", []):
        if category_norm and str(doc.get("category", "")).lower() != category_norm:
            continue
        for chunk in doc.get("chunks", []):
            enriched = dict(chunk)
            enriched["document_id"] = doc.get("id")
            enriched["created_at"] = doc.get("created_at", "")
            chunks.append(enriched)
    return chunks


def search_rag(query: str, top_k: int = 5, category: str | None = None) -> List[Dict[str, Any]]:
    query = _clean_text(query)
    if not query:
        return []
    chunks = _all_chunks(category)
    if not chunks:
        return []

    corpus = [c.get("text", "") for c in chunks]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(corpus + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    except Exception:
        # Simple keyword fallback if TF-IDF fails for any reason.
        terms = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        scores = []
        for text in corpus:
            words = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
            scores.append(len(terms & words) / max(1, len(terms)))

    ranked: List[Tuple[float, Dict[str, Any]]] = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    results: List[Dict[str, Any]] = []
    for score, chunk in ranked[: max(1, min(int(top_k or 5), 12))]:
        results.append(
            {
                "score": round(float(score), 4),
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "title": chunk.get("title", "Untitled document"),
                "source": chunk.get("source", "Unknown"),
                "category": chunk.get("category", "Policy"),
                "created_at": chunk.get("created_at", ""),
                "text": chunk.get("text", ""),
            }
        )
    return results


def build_scenario_query(context: Dict[str, Any]) -> str:
    """Create a retrieval query from the current staffing scenario/evidence."""
    context = context or {}
    scenario = context.get("scenario", context)
    evidence = context.get("committee_evidence", {}) or context.get("evidence", {}) or {}
    preset = scenario.get("preset_data") or {}
    bits: List[str] = [
        "Emergency Department staffing policy nurse fatigue overtime compliance surge escalation patient safety",
        str(scenario.get("department", "Emergency")),
        str(scenario.get("shift_type", "")),
        f"acuity {scenario.get('acuity_level', scenario.get('acuity', ''))}",
        f"required nurses {scenario.get('required_nurses', '')}",
        f"wait time {scenario.get('base_wait_time', '')}",
        f"risk {evidence.get('adjusted_operational_risk', evidence.get('operational_pressure_level', ''))}",
        f"arrival {evidence.get('arrival_surge_pressure', preset.get('ambulance_arrival_pressure', ''))}",
        f"boarding {evidence.get('boarding_pressure', '')} count {preset.get('boarding_count', '')} hours {preset.get('boarding_hours_avg', '')}",
        f"occupancy ED {preset.get('ed_occupancy_percent', '')} inpatient {preset.get('inpatient_bed_occupancy_percent', '')}",
        f"fast track open {preset.get('fast_track_open', '')} low acuity {preset.get('low_acuity_percent', '')}",
        f"callout fatigue {preset.get('nurse_callout_rate', '')}",
    ]
    return " ".join([b for b in bits if b and b.strip()])


def build_rag_context_block(context: Dict[str, Any], top_k: int = 5, query: str | None = None) -> Dict[str, Any]:
    query = query or build_scenario_query(context)
    results = search_rag(query, top_k=top_k)
    if not results:
        return {"enabled": True, "query": query, "results": [], "context_block": "No RAG evidence retrieved."}

    lines = []
    for idx, item in enumerate(results, start=1):
        lines.append(
            f"[{idx}] {item['title']} ({item['category']}, {item['source']}, score={item['score']})\n{item['text']}"
        )
    return {
        "enabled": True,
        "query": query,
        "results": results,
        "context_block": "\n\n".join(lines),
    }


def rag_status(data: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = data or _ensure_store()
    docs = data.get("documents", [])
    chunk_count = sum(len(d.get("chunks", [])) for d in docs)
    categories = sorted({str(d.get("category", "Policy")) for d in docs})
    return {
        "success": True,
        "document_count": len(docs),
        "chunk_count": chunk_count,
        "categories": categories,
        "store_path": RAG_STORE_PATH,
    }
