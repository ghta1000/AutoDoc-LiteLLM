"""
AutoDoc-LiteLLM
A minimal demonstration of:
- calling an LLM through LiteLLM,
- extracting simple metadata,
- building an RDF graph,
- validating it using SHACL,
- and running a small semantic search.

The goal is clarity and simplicity.
"""

from pathlib import Path
import os
import json

from rdflib import Graph, Namespace, Literal, RDF
from pyshacl import validate
from sentence_transformers import SentenceTransformer, util
from litellm import completion

# Namespace for the custom ontology
EX = Namespace("http://example.org/doc#")

# Paths to local ontology and SHACL files
BASE = Path(__file__).resolve().parent
ONTOLOGY_FILE = BASE / "ontology.ttl"
SHAPES_FILE = BASE / "shapes.ttl"

# Example document text used for extraction and semantic search
DOC_TEXT = """
# Title: Semantic Interoperability in Materials Science

This document describes how semantic ontologies, RDF, and SHACL
can be used to ensure FAIR-compliant technical documentation
for sustainable materials workflows.

Keywords: semantic interoperability, FAIR, materials, ontology
Domain: materials-science
"""

def call_llm_for_metadata(doc_text: str) -> dict:
    """Call an LLM through LiteLLM to extract simple metadata."""

    prompt = f"""
You extract metadata for short technical documents.

Return ONLY valid JSON with:
- "title": string
- "topic": short phrase summarising the main topic
- "domain": field or domain (e.g. "materials-science", "software-engineering")

Document:
\"\"\"{doc_text}\"\"\"
JSON:
"""

    resp = completion(
        model="openai/gpt-4o-mini",   # Any LiteLLM-supported model can be used
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    content = resp["choices"][0]["message"]["content"]
    return json.loads(content)


def build_rdf_graph(metadata: dict) -> Graph:
    """Create an RDF graph for the extracted metadata using rdflib."""

    g = Graph()
    g.parse(ONTOLOGY_FILE)

    doc = EX["doc1"]

    g.add((doc, RDF.type, EX.Document))
    g.add((doc, EX.title, Literal(metadata["title"])))
    g.add((doc, EX.topic, Literal(metadata["topic"])))
    g.add((doc, EX.domain, Literal(metadata["domain"])))

    return g


def validate_with_shacl(g: Graph):
    """Validate the RDF graph using SHACL constraints."""
    conforms, _, results_text = validate(
        g,
        shacl_graph=Graph().parse(SHAPES_FILE),
        inference="rdfs",
        debug=False,
    )
    return conforms, results_text


class SimpleRAG:
    """
    A minimal RAG-style semantic search utility using sentence-transformers.
    Only meant for demonstration.
    """
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.docs = []
        self.embeddings = None

    def index(self, docs):
        self.docs = docs
        self.embeddings = self.model.encode(docs, convert_to_tensor=True)

    def search(self, query, k=1):
        q_emb = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(q_emb, self.embeddings, top_k=k)[0]
        return [(self.docs[h["corpus_id"]], float(h["score"])) for h in hits]


def main():
    # 1) LLM metadata extraction
    print("=== STEP 1: LLM metadata extraction ===")
    metadata = call_llm_for_metadata(DOC_TEXT)
    print(json.dumps(metadata, indent=2), "\n")

    # 2) Build RDF graph
    print("=== STEP 2: Build RDF graph ===")
    g = build_rdf_graph(metadata)
    print(f"Triples in graph: {len(g)}")
    for s, p, o in g:
        print(s, p, o)
    print()

    # 3) SHACL validation
    print("=== STEP 3: SHACL validation ===")
    conforms, report = validate_with_shacl(g)
    print("Conforms:", conforms)
    if not conforms:
        print(report)
    print()

    # 4) Simple semantic search
    print("=== STEP 4: Semantic search (RAG-style) ===")
    rag = SimpleRAG()
    rag.index([DOC_TEXT])
    query = "FAIR-compliant semantic interoperability for materials workflows"
    results = rag.search(query, k=1)
    for doc, score in results:
        print(f"Best match score: {score:.4f}")
        print("Snippet:", doc[:250], "...")
    print("\nDone.")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. LiteLLM call may fail.")
    main()
