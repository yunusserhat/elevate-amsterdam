# -*- coding: utf-8 -*-
"""Step 8.6: package the published site and the search API.

* dist/elevate/: the static site for the author's website (Hugo `static/elevate/`, served at
  www.yunusserhat.com/elevate/): pages, assets, vendor files and the data the pages read. The
  search index stays out. The API URL is written into assets/site.js, and every page gets
  `robots: noindex`, so the site is reached by direct link only.
* dist/space_api/: the Hugging Face Space (Docker): the FastAPI app, the search index and the
  precomputed grounded answers. It serves no pages; opening it redirects to the site.

Usage: python 85_package.py [--api https://yunusserhat-elevate-amsterdam.hf.space]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
SPACE = PROJ / "space"
DIST = PROJ / "dist"
SITE_DATA = ["layers", "tables", "wp3_example", "wp6_pairs", "home", "survey", "3dbag", "provenance.json"]   # 3dbag: step 10.5
API_LINE = 'const API_BASE = "";'

SPACE_README = """---
title: ELEVATE in Amsterdam, search API
sdk: docker
app_port: 7860
pinned: false
license: other
short_description: Search API behind the ELEVATE in Amsterdam demonstration
---

# ELEVATE in Amsterdam: search API

The back end of [ELEVATE in Amsterdam]({site}), a worked example of the ELEVATE protocol's work
packages (Helbich M, Labib SM, Liedtke I, et al. 2026, PLOS ONE 21(5): e0348889). The pages live on
the author's website; this Space only answers their searches. Opening it redirects to the site.

An independent demonstration by Yunus Serhat Bıçakçı, not produced by or affiliated with the
ELEVATE project or Utrecht University.

## Endpoints
- `GET /api/health`
- `GET /api/search?q=...&k=12`: text search over 99,100 street views
- `POST /api/similar`: views most similar to an uploaded photo (not stored)
- `GET /api/ask?q=...`: a precomputed grounded answer, or the matching views
- `GET /api/near`, `GET /api/view/{{image_id}}`, `GET /api/qa`; documentation at `/api/docs`
- `POST /api/survey/session|wp6|wp3|wp1`, `GET /api/survey/results`: the pilot survey. Answers are
  checked here and written to a PostgreSQL database (Supabase) with a key held only in this Space's
  secrets (`SUPABASE_URL`, `SUPABASE_KEY`); the browser holds no database key. Results leave as
  totals only.

Queries are embedded on the CPU with EmbeddingGemma 2 (Apache-2.0), exactly as the index was built
with [GeoAI-VLM](https://github.com/yunusserhat/geoai-vlm). Observation text is the output of
Qwen3.5-4B (Apache-2.0), not a measurement. Street images stay on the City of Amsterdam's servers
(Gemeente Amsterdam, CC BY 4.0); every record links to them.
"""

DOCKERFILE = """FROM python:3.12-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user app ./app
COPY --chown=user data ./data
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
"""


def site(api: str) -> Path:
    out = DIST / "elevate"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SPACE / "static", out)
    for name in SITE_DATA:
        src = SPACE / "data" / name
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, out / "data" / name)
    js = out / "assets/site.js"
    s = js.read_text()
    assert s.count(API_LINE) == 1, "API_BASE line not found in site.js"
    js.write_text(s.replace(API_LINE, f'const API_BASE = "{api.rstrip("/")}";'))
    for page in sorted(out.glob("*.html")):
        h = page.read_text()
        assert h.count('<meta charset="utf-8">') == 1, page
        page.write_text(h.replace('<meta charset="utf-8">', '<meta charset="utf-8">\n<meta name="robots" content="noindex">', 1))
    return out


def space_api(site_url: str) -> Path:
    out = DIST / "space_api"
    if out.exists():
        shutil.rmtree(out)
    (out / "data/tables").mkdir(parents=True)
    shutil.copytree(SPACE / "app", out / "app", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(SPACE / "data/index", out / "data/index")
    shutil.copy2(SPACE / "data/tables/wp6_qa_precomputed.json", out / "data/tables/")
    shutil.copy2(SPACE / "requirements.txt", out / "requirements.txt")
    (out / "Dockerfile").write_text(DOCKERFILE)
    (out / "README.md").write_text(SPACE_README.format(site=site_url))
    (out / ".gitattributes").write_text("*.npy filter=lfs diff=lfs merge=lfs -text\n*.parquet filter=lfs diff=lfs merge=lfs -text\n")
    return out


def size(p: Path) -> str:
    return f"{sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1e6:.1f} MB, {sum(1 for f in p.rglob('*') if f.is_file())} files"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="https://yunusserhat-elevate-amsterdam.hf.space")
    ap.add_argument("--site-url", default="https://www.yunusserhat.com/elevate/")
    a = ap.parse_args()
    s = site(a.api)
    p = space_api(a.site_url)
    print("site ->", s, size(s), "API", a.api)
    print("space ->", p, size(p))
