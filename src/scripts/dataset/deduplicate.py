import json
from tqdm import tqdm
import numpy as np
import faiss
import random
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

# Configs
SIMILARITY_THRESHOLD_FAISS = 0.9
SIMILARITY_THRESHOLD_FUZZ = 90
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def deduplicate_with_faiss_and_fuzz(problems):
    texts = [p['problem'].strip() for p in problems]
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Cosine similarity (with normalized embeddings)
    index.add(embeddings)

    _, neighbors = index.search(embeddings, k=10)

    deduped = []
    seen_indices = set()
    similar_pairs = []

    for i in tqdm(range(len(problems)), desc="Hybrid Deduplication"):
        if i in seen_indices:
            continue
        current = problems[i]
        deduped.append(current)

        for j in neighbors[i][1:]:  # skip self
            if j == -1 or j >= len(problems) or j in seen_indices:
                continue

            prob_j = problems[j]
            score_cosine = cosine_sim(embeddings[i], embeddings[j])
            score_fuzz = fuzz.ratio(current["problem"], prob_j["problem"])

            if score_cosine >= SIMILARITY_THRESHOLD_FAISS or score_fuzz >= SIMILARITY_THRESHOLD_FUZZ:
                seen_indices.add(j)
                justification = {
                    "problem_1": current,
                    "problem_2": prob_j,
                    "cosine_similarity": float(round(score_cosine, 4)),
                    "fuzz_score": int(score_fuzz),
                    "method": "faiss" if score_cosine >= SIMILARITY_THRESHOLD_FAISS else "fuzz"
                }

                similar_pairs.append(justification)

                # Replace with more complete version if needed
                has_attrs_existing = 'formalization' in current and 'model' in current
                has_attrs_new = 'formalization' in prob_j and 'model' in prob_j

                if has_attrs_new and not has_attrs_existing:
                    deduped[-1] = prob_j
                elif has_attrs_new and has_attrs_existing:
                    deduped[-1] = random.choice([current, prob_j])
                # else: keep current

    return deduped, similar_pairs

def partition_by_difficulty(problems):
    easy, hard = [], []
    for prob in problems:
        difficulty = prob.get('difficulty')
        if isinstance(difficulty, (int, float)) and difficulty < 4:
            easy.append(prob)
        else:
            hard.append(prob)
    return easy, hard

# === Main ===

# Load dataset
data_all = load_json('./data/dataset/raw_data.json')

# Deduplicate
deduped_all, similar_pairs = deduplicate_with_faiss_and_fuzz(data_all)

# Partition by difficulty
deduped_easy, deduped_hard = partition_by_difficulty(deduped_all)

# Save outputs
with open('./data/dataset/informal_constructivebench.json', 'w', encoding='utf-8') as f:
    json.dump(deduped_all, f, indent=4)

with open('./data/dataset/similar_problem_pairs.json', 'w', encoding='utf-8') as f:
    json.dump(similar_pairs, f, indent=4)

# Print summary
print(f"Original: {len(data_all)} → Deduplicated: {len(deduped_all)}")
print(f"Easy (< 4): {len(deduped_easy)} | Hard (≥ 4 or missing): {len(deduped_hard)}")
print(f"Similar pairs saved: {len(similar_pairs)}")
