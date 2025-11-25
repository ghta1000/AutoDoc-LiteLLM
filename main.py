"""
AutoDoc-LiteLLM: Minimal documentation automation pipeline.

Steps:
1. Extract metadata using an LLM via LiteLLM.
2. Convert metadata into RDF using a small ontology.
3. Validate metadata using SHACL.
4. Perform simple semantic search using embeddings.
"""

from pathlib import Path
import os
import json

from rdflib import Graph, Namespace, Literal, RDF
from pyshacl import validate
from sentence_transformers import SentenceTransformer, util
from litellm import completion

# Namespace for our example ontology
EX = Namespace("http://example.org/doc#")

# Base directory for ontology files
BASE = Path(__file__).resolve().parent
ONTOLOGY_FILE = BASE.parent / "ontology" / "ontology.ttl"
SHAPES_FILE = BASE.parent / "ontology" / "shapes.ttl"

# Example input document (same as data/example_doc.md)
DOC_TEXT = Path(BASE.parent / "data" / "example_doc.md").read_text()


def call_llm_for_metadata(doc_text: str) -> dict:
    """
    Calls an LLM via LiteLLM to extract a small metadata JSON object.
    """

    prompt = f"""
Extract metadata for documentation.

Return ONLY JSON with:
- "title": string
- "topic": short phrase
- "domain": field of work

Document:
\"\"\"{doc_text}\"\"\"
JSON:
"""

    # LiteLLM handles routing to OpenAI or any other supported provider
    resp = completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    content = resp["choices"][0]["message"]["content"]
    return json.loads(content)


def build_rdf_graph(metadata: dict) -> Graph:
    """
    Creates an RDF graph using metadata extracted by the LLM.
    """

    g = Graph()
    g.parse(ONTOLOGY_FILE)

    # Create one document instance
    doc = EX["doc1"]

    g.add((doc, RDF.type, EX.Document))
    g.add((doc, EX.title, Literal(metadata["title"])))
    g.add((doc, EX.topic, Literal(metadata["topic"])))
    g.add((doc, EX.domain, Literal(metadata["domain"])))

    return g


def validate_with_shacl(g: Graph):
    """
    Runs SHACL validation using pySHACL.
    """

    conforms, results_graph, results_text = validate(
        g,
        shacl_graph=Graph().parse(SHAPES_FILE),
        inference="rdfs",
        debug=False,
    )
    return conforms, results_text


class SimpleRAG:
    """
    Minimal semantic search using sentence-transformers.
    """

    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.docs = []
        self.embeddings = None

    def index(self, docs):
        """Embed and index documents."""
        self.docs = docs
        self.embeddings = self.model.encode(docs, convert_to_tensor=True)

    def search(self, query, k=1):
        """Return top-k semantically similar documents."""
        q_emb = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(q_emb, self.embeddings, top_k=k)[0]
        return [(self.docs[h["corpus_id"]], float(h["score"])) for h in hits]


def main():

    # 1) LLM metadata extraction
    print("=== STEP 1: Metadata extraction (LiteLLM) ===")
    metadata = call_llm_for_metadata(DOC_TEXT)
    print(json.dumps(metadata, indent=2), "\n")

    # 2) RDF graph creation
    print("=== STEP 2: RDF graph construction ===")
    g = build_rdf_graph(metadata)
    print(f"Triples: {len(g)}")
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

    # 4) RAG semantic search
    print("=== STEP 4: RAG search ===")
    rag = SimpleRAG()
    rag.index([DOC_TEXT])

    query = "FAIR-compliant semantic interoperability"
    results = rag.search(query)

    for doc, score in results:
        print(f"Match score: {score:.4f}")
        print("Snippet:", doc[:180], "...")

    print("\nComplete.")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.")
    main()
