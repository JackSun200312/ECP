# search_theorem_fuzzy_namespaced.py
from sentence_transformers import SentenceTransformer
import pickle
from rapidfuzz import fuzz
import faiss
import os
PREFERRED_NAMESPACES = [
    "Nat", "Int", "Rat", "Real", "Complex", "ENat", "NNReal", "EReal", "Monoid",
    "CommMonoid", "Group", "CommGroup", "Ring", "CommRing", "Field", "Algebra",
    "Module", "Set", "Finset", "Fintype", "Multiset", "List", "Fin", "BigOperators",
    "Filter", "Polynomial", "SimpleGraph.Walk", "Equiv", "Embedding", "Injective",
    "Surjective", "Bijective", "Order", "Topology"
]

def load_partitioned_dict(coarse=True):
    path = "data/retrieval_data/preferred_partitioned.pkl" if coarse else "data/retrieval_data/partitioned_theorems.pkl"
    with open(path, 'rb') as f:
        return pickle.load(f)
THEOREM_DICT = load_partitioned_dict()
def split_full_name(full_name: str):
    parts = full_name.split(".")
    return ".".join(parts[:-1]), parts[-1]

def search_theorem_fuzzy(query: str, top_n=5, same_namespace_n=3, coarse=True):
    partitioned_dict = load_partitioned_dict(coarse=coarse)

    query = query.strip().lower()
    query_ns, query_def = split_full_name(query)

    main_results = []
    same_ns_results = []

    seen_def_names = set()

    for namespace, subdict in partitioned_dict.items():
        for full_name, definition in subdict.items():
            ns, defn = split_full_name(full_name)
            ns_lc, defn_lc = ns.lower(), defn.lower()

            score_name = fuzz.ratio(defn_lc, query_def)
            score_ns = fuzz.ratio(ns_lc, query_ns)

            # Tier logic
            if defn_lc == query_def:
                if ns_lc.startswith(query_ns):
                    tier = 1
                elif ns_lc.endswith(query_ns):
                    tier = 2
                else:
                    tier = 3
            elif ns_lc == query_ns:
                tier = 4
            else:
                tier = 5

            tie_breaker = (
                -score_ns,
                -score_name,
                len(ns),
                len(defn),
            )

            key = definition["definition_name"]

            # Collect same-namespace results separately
            if ns_lc == query_ns:
                same_ns_results.append(((score_name, tie_breaker), definition))
            else:
                main_results.append(((tier, *tie_breaker), definition))

    # Sort and select
    main_results.sort(key=lambda x: x[0])
    same_ns_results.sort(key=lambda x: (-x[0][0], x[0][1]))  # sort by best name match in same namespace

    top_main = []
    for _, definition in main_results:
        name = definition["definition_name"]
        if name not in seen_def_names:
            top_main.append(definition)
            seen_def_names.add(name)
        if len(top_main) >= top_n:
            break

    top_same_ns = []
    for _, definition in same_ns_results:
        name = definition["definition_name"]
        if name not in seen_def_names:
            top_same_ns.append(definition)
            seen_def_names.add(name)
        if len(top_same_ns) >= same_namespace_n:
            break

    return top_main + top_same_ns

# sentence-transformer model
_model = SentenceTransformer("all-MiniLM-L6-v2")
# at module‐top, replace your single globals with two sets:
_index_coarse, _defs_coarse, _ns_coarse = None, None, None
_index_full,   _defs_full,   _ns_full   = None, None, None

def _cache_path(coarse: bool):
    mode = "coarse" if coarse else "full"
    return f"data/retrieval_data/lean_definitions_embedding_{mode}.pkl"

def _build_all_indexes():
    """
    Ensure both coarse and full indexes are built (or loaded from cache).
    After this runs once, all four globals will be populated.
    """
    global _index_coarse, _defs_coarse, _ns_coarse
    global _index_full,   _defs_full,   _ns_full

    # helper to build/load one
    def _load_or_build(coarse: bool):
        cache = _cache_path(coarse)
        # try load
        if os.path.exists(cache):
            with open(cache, 'rb') as f:
                data = pickle.load(f)
            idx   = faiss.deserialize_index(data['index_bytes'])
            defs  = data['definitions']
            nss   = data['namespaces']
            return idx, defs, nss

        # else build from the appropriate partitioned dict
        part = load_partitioned_dict(coarse=coarse)
        defs, nss = [], []
        for sub in part.values():
            for full_name, definition in sub.items():
                defs.append(definition)
                ns, _ = split_full_name(full_name)
                nss.append(ns.lower())

        texts = [d["definition_name"] for d in defs]
        embs  = _model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
        faiss.normalize_L2(embs)

        dim   = embs.shape[1]
        idx   = faiss.IndexFlatIP(dim)
        idx.add(embs)

        # persist
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, 'wb') as f:
            pickle.dump({
                'index_bytes': faiss.serialize_index(idx),
                'definitions': defs,
                'namespaces': nss
            }, f)
        return idx, defs, nss

    # build/load both
    _index_coarse, _defs_coarse, _ns_coarse = _load_or_build(coarse=True)
    _index_full,   _defs_full,   _ns_full   = _load_or_build(coarse=False)

def _get_index_tuple(coarse: bool):
    # lazily build on first request
    if _index_coarse is None or _index_full is None:
        _build_all_indexes()
    if coarse:
        return _index_coarse, _defs_coarse, _ns_coarse
    else:
        return _index_full, _defs_full, _ns_full

def search_theorem_embedding(query: str,
                             top_n: int = 5,
                             same_namespace_n: int = 3,
                             coarse: bool = True):
    """
    Returns up to `top_n` global hits plus `same_namespace_n` hits
    from the same namespace as the query, selecting the coarse or full
    index based on the `coarse` flag.
    """
    q = query.strip()
    query_ns, _ = split_full_name(q.lower())

    index, definitions, namespaces = _get_index_tuple(coarse)

    # embed & normalize
    q_emb = _model.encode(q, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb.reshape(1, -1))

    k = top_n + same_namespace_n
    distances, indices = index.search(q_emb.reshape(1, -1), k)
    distances = distances[0]
    indices   = indices[0]

    top_main, top_same_ns = [], []
    seen = set()

    for idx, score in zip(indices, distances):
        definition = definitions[idx]
        full_name  = definition["definition_name"]
        ns, name   = split_full_name(full_name.lower())

        if name in seen:
            continue
        seen.add(name)

        entry = {
            **definition,
            "score": float(score),
            "namespace": ns
        }

        if ns == query_ns and len(top_same_ns) < same_namespace_n:
            top_same_ns.append(entry)
        elif len(top_main) < top_n:
            top_main.append(entry)

        if len(top_main) >= top_n and len(top_same_ns) >= same_namespace_n:
            break

    return top_main + top_same_ns
def search_theorem(query: str,
                   top_n: int = 5,
                   same_namespace_n: int = 3,
                   coarse: bool = True,
                   enable_embedding_search: bool = False):
    """
    Run fuzzy lookup, and (optionally) embedding lookup.
    Returns a single list: [*fuzzy_results, *embedding_results]
    """
    # 1) always do the fuzzy search
    fuzzy_results = search_theorem_fuzzy(query, top_n=top_n,
                                         same_namespace_n=same_namespace_n,
                                         coarse=coarse)

    if not enable_embedding_search:
        return fuzzy_results

    # 2) if requested, do embedding search too
    embedding_results = search_theorem_embedding(query,
                                                top_n=top_n,
                                                same_namespace_n=same_namespace_n,
                                                coarse=coarse)

    # 3) concatenate and return
    return fuzzy_results + embedding_results

# Example usage
if __name__ == "__main__":
    query = "Set.card"
    results = search_theorem(query, top_n=5, coarse=True, enable_embedding_search=True)

    for r in results:
        print(f"{r['definition_name']}")
        print(f"Type: {r['type_signature']}")
        if r.get("description"):
            print(f"Description: {r['description']}")

