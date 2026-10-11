"""
Teste: o EmbeddingGemma 2 consegue buscar dentro de áudio real em português sem transcrever?

Dados: CORAA v1.1 (NILC/USP), fala espontânea de entrevistas e conversas reais, com sotaques
de Recife (NURC-Recife), São Paulo capital (SP2010), interior de SP (ALIP), Minas Gerais
(C-ORAL-BRASIL) e palestras (TEDx). Licença CC BY-NC-ND 4.0.

Cada áudio é testado em 2 condições: original e "telefone" (8 kHz, faixa 300-3400 Hz, ruído a 15 dB).
Busca com 2 tipos de consulta: a frase completa e um trecho de 5 palavras.

Compara:
  A) EmbeddingGemma 2 direto no áudio (sem transcrição)
  B) faster-whisper transcreve -> EmbeddingGemma 2 no texto (pipeline tradicional)

Instalação (Python 3.10+):
  pip install "sentence-transformers>=6.1" "transformers>=5.19" faster-whisper soundfile scipy requests

No Google Colab (GPU T4), o jeito mais fácil é abrir o notebook teste_embeddinggemma2.ipynb.
O faster-whisper (ctranslate2) precisa das bibliotecas CUDA 12, que o notebook já instala.

Uso:
  python teste_embeddinggemma2.py               # 20 áudios por sotaque (100 no total)
  python teste_embeddinggemma2.py --per-group 10
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

ROWS_API = "https://datasets-server.huggingface.co/rows"
DATASET = "Racoci/CORAA-v1.1"
TOTAL_TEST_ROWS = 12676
CACHE = Path("coraa_cache")
GROUPS = {"NURC-Recife": "Recife", "SP2010": "SP capital", "ALIP": "interior SP",
          "C-ORAL-BRASIL I": "Minas Gerais", "TEDx Talks": "TEDx"}
MIN_WORDS = 8


def fetch_chunk(offset):
    params = {"dataset": DATASET, "config": "default", "split": "test", "offset": offset, "length": 100}
    for attempt in range(5):  # a API às vezes demora e devolve 502
        r = requests.get(ROWS_API, params=params, timeout=180)
        if r.status_code < 500:
            break
        time.sleep(5)
    r.raise_for_status()
    return [x["row"] for x in r.json()["rows"]]


def load_samples(per_group, rng):
    """Amostra blocos aleatórios do split de teste até ter per_group áudios de cada sotaque."""
    picked = defaultdict(list)
    offsets = list(range(0, TOTAL_TEST_ROWS, 100))
    rng.shuffle(offsets)
    for off in offsets:
        rows = [r for r in fetch_chunk(off)
                if r["dataset"] in GROUPS and len(r["text"].split()) >= MIN_WORDS]
        rng.shuffle(rows)
        for r in rows:
            if len(picked[r["dataset"]]) < per_group:
                picked[r["dataset"]].append(r)
        if all(len(picked[g]) >= per_group for g in GROUPS):
            break

    samples = []
    for g in GROUPS:
        for r in picked[g]:
            name = Path(r["file_path"]).stem
            wav = CACHE / f"{name}.wav"
            if not wav.exists():
                resp = requests.get(r["audio"][0]["src"], timeout=60)
                resp.raise_for_status()
                wav.write_bytes(resp.content)
            samples.append({"id": name, "group": GROUPS[g], "wav": str(wav), "text": r["text"]})
    return samples


def phone_version(wav_path, rng):
    """Simula ligação: 8 kHz, faixa de voz 300-3400 Hz e ruído branco a 15 dB de SNR."""
    out = CACHE / (Path(wav_path).stem + "_tel.wav")
    if out.exists():
        return str(out)
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    x = resample_poly(audio, 8000, sr)
    x = sosfilt(butter(4, [300, 3400], btype="band", fs=8000, output="sos"), x)
    noise = rng.standard_normal(len(x))
    noise *= np.sqrt(np.mean(x ** 2) / 10 ** (15 / 10)) / np.sqrt(np.mean(noise ** 2))
    sf.write(out, resample_poly(x + noise, 16000, 8000).astype("float32"), 16000)
    return str(out)


def load_16k(wav_path):
    """Array mono a 16 kHz para o Whisper (evita o decode via PyAV, incompatível com av>=19)."""
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio if sr == 16000 else resample_poly(audio, 16000, sr).astype("float32")


def normalize(text):
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def wer(ref, hyp):
    r, h = normalize(ref), normalize(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


def snippet(text, rng, n=5):
    words = text.split()
    start = rng.randrange(0, len(words) - n + 1)
    return " ".join(words[start:start + n])


def ranks_of(q_emb, d_emb):
    sims = q_emb @ d_emb.T  # embeddings normalizados -> cosseno
    return np.array([(row > row[i]).sum() + 1 for i, row in enumerate(sims)])


def summarize(ranks):
    return {"R@1": float((ranks == 1).mean()), "R@5": float((ranks <= 5).mean()),
            "MRR": float((1 / ranks).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=20)
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    CACHE.mkdir(exist_ok=True)
    print("Baixando amostras do CORAA...")
    samples = load_samples(args.per_group, rng)
    for s in samples:
        s["wav_tel"] = phone_version(s["wav"], np_rng)
        s["snippet"] = snippet(s["text"], rng)
    print(f"{len(samples)} áudios: " + ", ".join(f"{g} {sum(s['group'] == g for s in samples)}"
                                                for g in GROUPS.values()))

    import torch
    from faster_whisper import WhisperModel
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Rodando em {device}")
    model = SentenceTransformer("google/embeddinggemma-2", device=device)
    whisper = WhisperModel(args.whisper, device=device, compute_type="float16" if device == "cuda" else "int8")

    def enc_text(texts, prompt):
        return model.encode(texts, prompt_name=prompt, normalize_embeddings=True, show_progress_bar=False)

    queries = {"frase completa": enc_text([s["text"] for s in samples], "SearchQuery"),
               "trecho de 5 palavras": enc_text([s["snippet"] for s in samples], "SearchQuery")}

    rows, ranks, timing, transcripts = [], {}, {}, {}
    for cond, key in [("original", "wav"), ("telefone", "wav_tel")]:
        print(f"Indexando áudio {cond}...")
        t0 = time.perf_counter()
        audio_emb = np.stack([model.encode({"audio": s[key]}, normalize_embeddings=True) for s in samples])
        timing[("A", cond)] = (time.perf_counter() - t0) / len(samples)

        t0 = time.perf_counter()
        transcripts[cond] = [" ".join(seg.text for seg in whisper.transcribe(load_16k(s[key]), language="pt")[0]).strip()
                             for s in samples]
        text_emb = enc_text(transcripts[cond], "Document")
        timing[("B", cond)] = (time.perf_counter() - t0) / len(samples)

        for qname, q in queries.items():
            for method, d in [("A) áudio direto", audio_emb), (f"B) whisper-{args.whisper} + texto", text_emb)]:
                rk = ranks_of(q, d)
                ranks[(method[0], cond, qname)] = rk
                rows.append({"método": method, "áudio": cond, "consulta": qname, **summarize(rk),
                             "s/áudio": timing[(method[0], cond)]})

    print(f"\nBusca em {len(samples)} áudios de fala espontânea em português ({device})\n")
    print(f"{'método':26} {'áudio':9} {'consulta':21} {'R@1':>5} {'R@5':>5} {'MRR':>6} {'s/áudio':>8}")
    for r in rows:
        print(f"{r['método']:26} {r['áudio']:9} {r['consulta']:21} {r['R@1']:5.0%} {r['R@5']:5.0%} "
              f"{r['MRR']:6.3f} {r['s/áudio']:8.2f}")

    print("\nPor sotaque (R@1 com a frase completa | WER do Whisper):")
    print(f"{'sotaque':14} {'A orig':>7} {'B orig':>7} {'A tel':>7} {'B tel':>7} {'WER orig':>9} {'WER tel':>8}")
    by_group = {}
    for g in GROUPS.values():
        idx = [i for i, s in enumerate(samples) if s["group"] == g]
        r1 = {k: float((ranks[k][idx] == 1).mean()) for k in
              [("A", "original", "frase completa"), ("B", "original", "frase completa"),
               ("A", "telefone", "frase completa"), ("B", "telefone", "frase completa")]}
        w = {c: float(np.mean([wer(samples[i]["text"], transcripts[c][i]) for i in idx]))
             for c in ("original", "telefone")}
        by_group[g] = {"R@1": {" ".join(k[:2]): v for k, v in r1.items()}, "WER": w}
        print(f"{g:14} " + " ".join(f"{v:7.0%}" for v in r1.values())
              + f" {w['original']:9.0%} {w['telefone']:8.0%}")

    print("\nExemplos em que o áudio direto errou (frase completa, áudio original):")
    for i in [i for i in range(len(samples)) if ranks[("A", "original", "frase completa")][i] > 1][:3]:
        s = samples[i]
        print(f"- [{s['group']}] {s['text'][:90]}\n  posição: {ranks[('A', 'original', 'frase completa')][i]}"
              f" | whisper: {transcripts['original'][i][:90]}")

    Path("resultado_embeddinggemma2.json").write_text(json.dumps(
        {"n": len(samples), "seed": args.seed, "geral": rows, "por_sotaque": by_group,
         "amostras": [{**{k: s[k] for k in ("id", "group", "text", "snippet")},
                       "whisper_original": transcripts["original"][i], "whisper_telefone": transcripts["telefone"][i]}
                      for i, s in enumerate(samples)]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nResultado salvo em resultado_embeddinggemma2.json")


if __name__ == "__main__":
    main()
