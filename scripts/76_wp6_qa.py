# -*- coding: utf-8 -*-
"""Step 7.4: evidence-grounded questions and answers (GeoAI-VLM DemoService), precomputed.

Questions are retrieved against the 99,100-view index (EmbeddingGemma 2, step 2.5; the query is
embedded on the CPU, as the Space will do). A larger open model writes one to three sentences
from the retrieved records only (thinking switched off in the chat template); every sentence must cite record ids, and DemoService withholds
replies with uncited sentences, unknown ids or no statement. Two questions are there to be
declined: one asks for a safety judgement, one asks about something the audit does not record.
The Space shows these precomputed answers with their cited views, and for new questions lists
the matching records (extractive mode, no generation on a CPU).

EmbeddingGemma 2 needs transformers >= 5.18 (envs/embed) and vLLM needs an older one (envs/vlm),
so the query vectors are made first and looked up during answering.

Usage:
  $PY_EMBED 76_wp6_qa.py embed                       # outputs/wp6/qa_query_vectors.npz
  $PY_VLM   76_wp6_qa.py answer [--model ... --tp 2] [--probe]
Writes outputs/wp6/qa_precomputed.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from geoai_vlm.service import DemoService, SceneIndex

PROJ = Path(__file__).resolve().parents[1]
QUESTIONS = [
    "Where are benches on the sidewalk under street trees?",
    "Where do parked bicycles block the sidewalk?",
    "Which streets have a separated red cycle track?",
    "Where is a junction visible without a marked pedestrian crossing?",
    "Which streets have shade from trees on the sidewalk?",
    "Where is construction or scaffolding blocking the sidewalk?",
    "Which streets have shops with ground-floor windows and benches?",
    "Where are canal-side streets with railings and benches?",
    "Where are streets with no trees or greenery?",
    "Where are curb ramps visible at crossings?",
    "Where is litter or rubbish on the pavement?",
    "Where are speed bumps or other traffic calming on residential streets?",
    "Where are narrow sidewalks next to parked cars?",
    "Where are tram stops with shelters?",
    "Is this street safe for children to walk alone?",
]


VEC = PROJ / "outputs/wp6/qa_query_vectors.npz"


class LookupEmbedder:
    """Returns query vectors made beforehand in envs/embed (same model, CPU, float32)."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.vectors = dict(zip(z["questions"].tolist(), z["vectors"]))
        self.model_name = str(z["model"])

    def embed_queries(self, texts, batch_size=32):
        return np.vstack([self.vectors[t] for t in texts])


def embed():
    from geoai_vlm import ImageEmbedder
    emb = ImageEmbedder(backend="sentence-transformers", model_name="google/embeddinggemma-2", truncate_dim=256,
                        config_kwargs={"audio_config": None}, device="cpu", dtype="float32")
    v = emb.embed_queries(QUESTIONS)
    np.savez(VEC, questions=np.array(QUESTIONS), vectors=v.astype(np.float32), model=np.array("google/embeddinggemma-2"))
    print("query vectors", v.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["embed", "answer"])
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--probe", action="store_true", help="first question only, print the raw reply")
    args = ap.parse_args()
    if args.step == "embed":
        embed()
        return
    from geoai_vlm.describer import VLLMBackend

    class NoThinkVLLM(VLLMBackend):
        """complete() with the Qwen chat template's thinking switched off (GeoAI-VLM does not pass it)."""

        def complete(self, messages):
            self.load_model()
            from vllm import SamplingParams
            sampling = SamplingParams(temperature=self.temperature, max_tokens=self.max_new_tokens)
            res = self.llm.chat([self._text_chat(messages)], sampling_params=sampling, use_tqdm=False,
                                chat_template_kwargs={"enable_thinking": False})
            # DemoService reads one id per bracket; the model sometimes writes [id1, id2]. Split those
            # groups into [id1] [id2] (format only, the words are unchanged).
            return re.sub(r"\[([^\[\]]+,[^\[\]]+)\]",
                          lambda m: " ".join(f"[{x.strip()}]" for x in m.group(1).split(",")), res[0].outputs[0].text)

    index = SceneIndex.load(PROJ / "outputs/wp3/index/embeddinggemma2_256", embedder=LookupEmbedder(VEC))
    chat = NoThinkVLLM(args.model, tensor_parallel_size=args.tp, max_new_tokens=400, max_model_len=8192,
                       gpu_memory_utilization=0.85)
    service = DemoService(index=index, chat=chat, k=8, min_similarity=0.6)
    base_line = service._record_line
    # give the model the buurt name of each view, so "where" can be answered with places
    service._record_line = lambda row: base_line(row).replace("] ", f"] (buurt: {row['buurtnaam']}, {row['view']} view) ", 1)
    out = []
    for q in QUESTIONS[:1] if args.probe else QUESTIONS:
        if args.probe:
            recs = index.search_text(q, 8)
            block = "\n".join(service._record_line(r) for _, r in recs.iterrows())
            raw = chat.complete([{"role": "user", "content": f"QUESTION:\n{q}\n\nRECORDS:\n{block}"}])
            print("RAW REPLY:\n", raw)
        a = service.answer(q)
        rec = a.records
        out.append({"question": q, "answer": a.answer, "refused": a.refused, "reason": a.reason, "citations": a.citations,
                    "records": [{"image_id": r.image_id, "similarity": round(float(r.similarity), 3), "lat": r.lat, "lon": r.lon,
                                 "heading_deg": round(float(r.heading_deg), 1), "buurt": r.buurtnaam,
                                 "observation_text": r.observation_text} for r in rec.itertuples()] if len(rec) else []})
        print(f"Q: {q}\n  refused={a.refused} reason={a.reason}\n  A: {a.answer}\n", flush=True)
    meta = {"model": args.model, "retrieval": "EmbeddingGemma 2, 256 d, image+text documents, CPU query",
            "k": 8, "min_similarity": 0.6, "thinking": "off",
            "citation_format": "groups like [a, b] split into [a] [b] before checking", "note": "Answers restate model observations of street views; they are not measurements."}
    if not args.probe:
        (PROJ / "outputs/wp6/qa_precomputed.json").write_text(json.dumps({"meta": meta, "items": out}, indent=1, ensure_ascii=False))
    print("QA_DONE", flush=True)
    # vLLM's shutdown can hang after a TP=2 run and leave its engine and worker processes holding
    # GPU memory; the results are written, so end the descendants and exit.
    import signal
    import subprocess
    kids = subprocess.run(["pgrep", "-P", str(os.getpid())], capture_output=True, text=True).stdout.split()
    for k in kids:
        for g in subprocess.run(["pgrep", "-P", k], capture_output=True, text=True).stdout.split():
            os.kill(int(g), signal.SIGTERM)
        os.kill(int(k), signal.SIGTERM)
    os._exit(0)


if __name__ == "__main__":
    main()
