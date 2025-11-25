# AutoDoc-LiteLLM

AutoDoc-LiteLLM is a small, self-contained demo that shows how to:

- call an LLM through **LiteLLM**
- extract simple metadata from a short document
- represent that metadata as **RDF** using a minimal ontology
- validate the RDF with **SHACL**
- run a very small **semantic search** using sentence-transformers

The goal is to keep the code easy to read and easy to extend, not to be production-ready.

---

## How it works

1. A short example document is defined as a Python string.
2. The script calls an LLM (through LiteLLM) to extract:
   - `title`
   - `topic`
   - `domain`
3. The metadata is converted into an RDF graph using `rdflib` and a tiny ontology in `ontology.ttl`.
4. The RDF graph is validated against SHACL shapes from `shapes.ttl` using `pyshacl`.
5. The same document text is embedded with `sentence-transformers`, and a simple semantic search is performed for one example query.

All of this happens inside `main.py`.

---

## File overview

- `main.py`  
  Main script. Runs the full demo: metadata extraction → RDF graph → SHACL validation → semantic search.

- `ontology.ttl`  
  Minimal RDF/OWL ontology defining a `Document` class and a few datatype properties.

- `shapes.ttl`  
  SHACL shapes that require each `Document` to have a title, topic, and domain.

- `requirements.txt`  
  Python dependencies.

- `.gitignore`  
  Basic ignores for Python projects (virtual env, cache files, etc.).

---

## Requirements

- Python 3.10+ recommended
- A valid API key for the LLM provider you want to use via LiteLLM  
  (the example assumes OpenAI and uses the `OPENAI_API_KEY` environment variable).

---

## Installation

```bash
git clone https://github.com/<your-username>/autodoc-litellm.git
cd autodoc-litellm
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
