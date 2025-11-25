"""
AutoDoc-LiteLLM
A minimal example showing:
- metadata extraction with LiteLLM
- RDF/OWL graph creation
- SHACL validation
- basic semantic search
"""

from pathlib import Path
import os
import json

from rdflib import Graph, Namespace, Literal, RDF
from pyshacl import validate
from sentence_transformers import SentenceTransformer, util
from litellm import completion

# Namespace for our minimal ontology
EX = Namespace("http://example.org/doc#")

# Paths to ontology and SHACL shapes
BASE = Path(__file__).resolve().parent
ONTOLOGY_FILE = BASE.parent / "ontology" / "ontology.ttl"
SHAPES_FILE = BASE.parent / "ontology" / "shapes.ttl"

# Simple embedded text used for the demo
DOC_TEXT = """
# Title: Semantic Interoperability in Materials Science

This document describes how semantic ontologies, RDF, and SHACL
can be used to ensure FAIR-compliant technical documentation
for sustainable materials workflows.

Keywords: semantic interoperability, FAIR, materials, ontology
Domain: materials-science
"""


# ------------------------------------------------------
# STEP 1 — Metadata extraction using LiteLLM
# ------------------------------------------------------
def call_llm_for_metadata(doc_text: str) -> dict:
    """
    Sends a prompt to the model through LiteLLM asking for:
    - title
    - topic
    - domain

    The model should return valid JSON only.
    """

    prompt = f"""
You extract metadata for technical documentation.

Given the following document text, return ONLY valid JSON with:
- "title": string
- "topic": short phrase summarising the main topic
- "domain": domain or field (e.g. "materials-science", "software-engineering")

Document:
\"\"\"{doc_text}\"\"\"  
JSON:
"""

    # LiteLLM routes this to OpenAI or any configured provider
    resp = completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    content = resp["choices"][0]["message"]["content"]
    return json.loads(content)


# ------------------------------------------------------
# STEP 2 — RDF/OWL graph construction
# ------------------------------------------------------
def build_rdf_graph(metadata: dict) -> Graph:
    """
    Creates an RDF graph from LLM metadata based on our ontology.
    """

    g = Graph()
    g.parse(ONTOLOGY_FILE)

    doc = EX["doc1"]
    g.add((doc, RDF.type, EX.Document))
    g.add((doc, EX.title, Literal(metadata["title"])))
    g.add((doc, EX.topic, Literal(metadata["topic"])))
    g.add((doc, EX.domain, Literal(metadata["domain"])))

    return g


# ------------------------------------------------------
# STEP 3 — SHACL validation
# ------------------------------------------------------
def validate_with_shacl(g: Graph):
    """
    Validates the RDF graph using the SHACL shapes.
    """
    conforms, results_graph, results_text = validate(
        g,
        shacl_graph=Graph().parse(SHAPES_FILE),
        inference="rdfs",
        debug=False,
    )
    return conforms, results_text


# ------------------------------------------------------
# STEP 4 — Simple RAG-style semantic search
# ------------------------------------------------------
class SimpleRAG:
    """
    Minimal semantic search using sentence-transformers.
    """

    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.docs = []
        self.embeddings = None

    def index(self, docs):
        """
        Stores documents and computes their embeddings.
        """
        self.docs = docs
        self.embeddings = self.model.encode(docs, convert_to_tensor=True)

    def search(self, query, k=1):
        """
        Returns top-k semantic matches for a given query.
        """
        q_emb = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(q_emb, self.embeddings, top_k=k)[0]
        return [(self.docs[h["corpus_id"]], float(h["score"])) for h in hits]


# ------------------------------------------------------
# Main pipeline
# ------------------------------------------------------
def main():
    # STEP 1: LLM metadata extraction
    print("=== STEP 1: LLM metadata extraction ===")
    metadata = call_llm_for_metadata(DOC_TEXT)
    print(json.dumps(metadata, indent=2), "\n")

    # STEP 2: RDF graph
    print("=== STEP 2: RDF graph construction ===")
    g = build_rdf_graph(metadata)
    print(f"Triples: {len(g)}")
    for s, p, o in g:
        print(s, p, o)
    print()

    # STEP 3: SHACL validation
    print("=== STEP 3: SHACL validation ===")
    conforms, report = validate_with_shacl(g)
    print("Conforms:", conforms)
    if not conforms:
        print(report)
    print()

    # STEP 4: RAG search
    print("=== STEP 4: Semantic search ===")
    rag = SimpleRAG()
    rag.index([DOC_TEXT])

    query = "FAIR-compliant semantic interoperability for materials workflows"
    results = rag.search(query, k=1)

    for doc, score in results:
        print(f"Match score: {score:.4f}")
        print(doc[:200], "...")
    print("\nDemo complete.")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set.")
    main()
