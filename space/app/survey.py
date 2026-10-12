# -*- coding: utf-8 -*-
"""Survey layer (pilot): answers reach the database only through this API.

The browser never holds a database key. Answers are checked here and written to the survey
database (Supabase, PostgreSQL) with a key read from the server's environment (on the Space:
secrets SUPABASE_URL and SUPABASE_KEY). That key is the insert-only publishable key, so even this
server can do nothing but insert rows and call the aggregate functions (survey/schema.sql: row-level
security, constraints, a per-session cap). Results leave as totals only.

A visitor's address is held in memory for one minute to slow floods of answers; it is never stored.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import List, Literal, Optional
from uuid import UUID

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
KEY = os.environ.get("SUPABASE_KEY", "")
IMAGE_ID = r"^recording_[0-9_-]+_(front|back|left|right)$"
ROLES = Literal["resident", "planner", "public_health", "municipality", "designer", "civic", "researcher", "other"]
PER_MINUTE = 90
CACHE_SECONDS = 20

router = APIRouter(prefix="/api/survey", tags=["survey"])
_http = httpx.Client(timeout=15)
_hits: dict = {}
_cache: dict = {}
_lock = threading.Lock()


class Session(BaseModel):
    id: UUID
    consent: Literal[True]
    role: Optional[ROLES] = None
    amsterdam: Optional[Literal["lives", "works", "visits", "no"]] = None
    client: Optional[str] = Field(None, max_length=40)


class WP6Answer(BaseModel):
    session_id: UUID
    image_id: str = Field(pattern=IMAGE_ID)
    gap_type: Literal["crossing", "trees", "seating", "greenery", "cycle"]
    edited_side: Literal["left", "right"]
    choice: Literal["left", "right", "none"]
    realism: Optional[int] = Field(None, ge=1, le=5)
    rt_ms: Optional[int] = Field(None, ge=0, le=3_600_000)


class WP3Answer(BaseModel):
    session_id: UUID
    question: Literal["pleasant", "safe"]
    left_id: str = Field(pattern=IMAGE_ID)
    right_id: str = Field(pattern=IMAGE_ID)
    choice: Literal["left", "right", "equal"]
    rt_ms: Optional[int] = Field(None, ge=0, le=3_600_000)

    @model_validator(mode="after")
    def two_views(self):
        if self.left_id == self.right_id:
            raise ValueError("the two views must differ")
        return self


class WP1Ranking(BaseModel):
    session_id: UUID
    top_five: List[int] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def distinct_ranks(self):
        if len(set(self.top_five)) != 5 or not all(1 <= x <= 15 for x in self.top_five):
            raise ValueError("five different indicators, numbered 1 to 15")
        return self


def _throttle(request: Request) -> None:
    address = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    now = time.time()
    with _lock:
        recent = [t for t in _hits.get(address, []) if now - t < 60]
        if len(recent) >= PER_MINUTE:
            raise HTTPException(429, "too many answers in a minute; please slow down")
        recent.append(now)
        _hits[address] = recent
        if len(_hits) > 5000:   # forget idle addresses
            for a in [a for a, ts in _hits.items() if now - ts[-1] > 60]:
                del _hits[a]


def _db(path: str, body: dict, prefer: Optional[str] = None):
    if not (URL and KEY):
        raise HTTPException(503, "the survey database is not configured on this server")
    headers = {"apikey": KEY, "Content-Type": "application/json"}
    if KEY.startswith("eyJ"):            # a legacy anon key is a JWT
        headers["Authorization"] = f"Bearer {KEY}"
    if prefer:
        headers["Prefer"] = prefer
    try:
        r = _http.post(f"{URL}/rest/v1/{path}", json=body, headers=headers)
    except httpx.HTTPError:
        raise HTTPException(502, "the survey database could not be reached") from None
    if r.status_code >= 400:
        try:
            message = str(r.json().get("message", ""))[:160]
        except ValueError:
            message = ""
        raise HTTPException(400 if r.status_code in (400, 401, 403, 409) else 502, message or "the answer was not accepted")
    return r.json() if r.content else None


def _insert(table: str, answer: BaseModel, request: Request) -> dict:
    _throttle(request)
    _db(table, json.loads(answer.model_dump_json(exclude_none=True)), prefer="return=minimal")
    with _lock:
        _cache.clear()
    return {"ok": True}


@router.get("/status")
def status():
    return {"configured": bool(URL and KEY)}


@router.post("/session")
def new_session(session: Session, request: Request):
    return _insert("survey_sessions", session, request)


@router.post("/wp6")
def wp6_answer(answer: WP6Answer, request: Request):
    return _insert("wp6_responses", answer, request)


@router.post("/wp3")
def wp3_answer(answer: WP3Answer, request: Request):
    return _insert("wp3_pairwise", answer, request)


@router.post("/wp1")
def wp1_ranking(ranking: WP1Ranking, request: Request):
    return _insert("wp1_rankings", ranking, request)


@router.get("/results")
def results(role: Optional[ROLES] = None):
    """Totals from the database's aggregate functions (cached for a few seconds)."""
    key = role or "all"
    with _lock:
        hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    data = {
        "overview": _db("rpc/survey_overview", {}),
        "wp6": _db("rpc/wp6_summary", {}),
        "wp3": _db("rpc/wp3_scores", {}),
        "wp1": _db("rpc/wp1_summary", {"for_role": role}),
    }
    with _lock:
        _cache[key] = (time.time(), data)
    return data
