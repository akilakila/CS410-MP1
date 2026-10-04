import os
import json
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
# from pyserini.index.lucene import IndexReader
from pyserini.index.lucene import LuceneIndexReader
import numpy as np
import subprocess
import argparse


def preprocess_corpus(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_file, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Preprocessing corpus")):
            doc = {
                "id": f"{i}",  # Changed to match qrels format
                "contents": line.strip()
            }
            with open(os.path.join(output_dir, f"doc{i}.json"), 'w') as out:
                json.dump(doc, out)


def build_index(input_dir, index_dir):
    if os.path.exists(index_dir) and os.listdir(index_dir):
        print(f"Index already exists at {index_dir}. Skipping index building.")
        return

    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", input_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]
    subprocess.run(cmd, check=True)


def load_queries(query_file):
    with open(query_file, 'r') as f:
        return [line.strip() for line in f]


def load_qrels(qrels_file):
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                qid, docid, rel = parts
            else:
                raise Exception(f"incorrect line: {line.strip()}")

            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = int(rel)
    return qrels


def search(searcher, queries, top_k=10, query_id_start=0):
    results = {}
    for i, query in enumerate(tqdm(queries, desc="Searching")):
        hits = searcher.search(query, k=top_k)
        results[str(i + query_id_start)] = [(hit.docid, hit.score) for hit in hits]
    return results


def compute_ndcg(results, qrels, k=10):
    def dcg(relevances):
        # return sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevances[:k]))
        dcg_simple = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances[:k]))
        return dcg_simple

    ndcg_scores = []
    for qid, query_results in results.items():
        if qid not in qrels:
            # print(f"Query {qid} not found in qrels")
            continue
        relevances_current = [qrels[qid].get(docid, 0) for docid, _ in query_results]
        idcg = dcg(sorted(qrels[qid].values(), reverse=True))
        if idcg == 0:
            print(f"IDCG is 0 for query {qid}")
            continue
        ndcg_scores.append(dcg(relevances_current) / idcg)

    if not ndcg_scores:
        print("No valid NDCG scores computed")
        return 0.0
    return np.mean(ndcg_scores)


def compute_precision(results, qrels, k=10):
    precision_scores = []
    for qid, query_results in results.items():
        if qid not in qrels:
            continue
        relevances_current = [qrels[qid].get(docid, 0) for docid, _ in query_results]
        # If relevances current is a list like: [0,1,0,0,1]
        # then our sum will be 2 relevant docs
        rel = 0
        for relevance in relevances_current:
            if relevance > 0:
                rel += 1
        precision = rel / k

        # now we have the precicision for one result, add it to the list
        precision_scores.append(precision)

    # now that we have a list of all the precision scores, calculate avg precision
    return sum(precision_scores) / len(precision_scores)


def main(k=None, b=None, fb_terms=None, fb_docs=None, original_query_weight=None, mu=None):
    """main function for searching"""

    """=======TODO: Choose Dataset======="""
    # You can choose from "cranfield", "apnews", and "new_faculty" for dataset
    cname = "cranfield"
    """============================"""

    base_dir = f"data/{cname}"
    query_id_start = {
        "apnews": 0,
        "cranfield": 1,
        "new_faculty": 1,
    }[cname]

    # Paths to the raw corpus, queries, and relevance label files
    corpus_file = os.path.join(base_dir, f"{cname}.dat")
    query_file = os.path.join(base_dir, f"{cname}-queries.txt")
    qrels_file = os.path.join(base_dir, f"{cname}-qrels.txt")
    # processed_corpus_dir = os.path.join(base_dir, "corpus")

    # Directories where the processed corpus and index will be stored for toolkit
    processed_corpus_dir = f"processed_corpus/{cname}"
    os.makedirs(processed_corpus_dir, exist_ok=True)
    index_dir = f"indexes/{cname}"

    # Preprocess corpus
    if not os.path.exists(processed_corpus_dir) or not os.listdir(processed_corpus_dir):
        preprocess_corpus(corpus_file, processed_corpus_dir)
    else:
        print(f"Preprocessed corpus already exists at {processed_corpus_dir}. Skipping preprocessing.")

    # Build index
    build_index(processed_corpus_dir, index_dir)

    # Load queries and qrels
    queries = load_queries(query_file)
    qrels = load_qrels(qrels_file)

    # Debug info
    print(f"Number of queries: {len(queries)}")
    print(f"Number of qrels: {len(qrels)}")
    print(f"Sample qrel: {list(qrels.items())[0] if qrels else 'No qrels'}")

    # Search
    searcher = LuceneSearcher(index_dir)

    """=======TODO: Set Ranking Hyperparameters======="""
    # searcher.set_bm25(k1=k, b=b)
    # searcher.set_rm3(fb_terms=20, fb_docs=10, original_query_weight=0.5) # optional query expansion
    searcher.set_rm3(fb_terms=fb_terms, fb_docs=fb_docs, original_query_weight=original_query_weight) # optional query expansion
    """========================================="""

    results = search(searcher, queries, query_id_start=query_id_start)

    # Debug info
    print(f"Number of results: {len(results)}")
    print(f"Sample result: {list(results.items())[0] if results else 'No results'}")

    # Evaluate
    topk = 10
    ndcg = compute_ndcg(results, qrels, k=topk)
    print(f"NDCG@{topk}: {ndcg:.4f}")

    precision = compute_precision(results, qrels, k=topk)
    print(f"Precision@{topk}: {precision:.4f}")

    # Save results
    with open(f"results_{cname}.json", "w") as f:
        json.dump({"results": results, "ndcg": ndcg}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ~~~~~~~~~~~~~~~~~~ RM3 ~~~~~~~~~~~~~~~~~~~~

    parser.add_argument('--fb_terms', type=float)
    parser.add_argument('--fb_docs', type=float)
    parser.add_argument('--original_query_weight', type=float)
    args = parser.parse_args()
    main(fb_terms=args.fb_terms, fb_docs=args.fb_docs, original_query_weight=args.original_query_weight)


