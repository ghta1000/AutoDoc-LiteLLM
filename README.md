# **AutoDoc-LiteLLM**

AutoDoc-LiteLLM is a small, self-contained demo script that shows how to:

* call an LLM through **LiteLLM**
* extract simple metadata from a document
* build a minimal **RDF** graph using a small ontology
* validate the graph using **SHACL**
* run a lightweight **semantic search** using sentence-transformers

The focus is clarity and readability.
This is not a production system — just a compact, educational example.

---

## **How it works**

1. A short example document is defined inside `main.py`.
2. The script uses LiteLLM to call an LLM and extract three fields:

   * `title`
   * `topic`
   * `domain`
3. The extracted metadata is inserted into an RDF graph using `rdflib` and the ontology in `ontology.ttl`.
4. The graph is validated against SHACL shapes from `shapes.ttl` using `pyshacl`.
5. The same document text is embedded using sentence-transformers, and a simple semantic search is run for one example query.

All logic is inside `main.py` to keep the project minimal.

---

## **Project structure**

You can keep all files in the root directory.

```
autodoc-litellm/
│
├─ main.py           # main demo script (LLM → RDF → SHACL → semantic search)
├─ ontology.ttl      # minimal RDF/OWL ontology
├─ shapes.ttl        # SHACL validation rules
├─ requirements.txt  # Python dependencies
└─ .gitignore        # ignore venv, cache files, etc.
```

No nested folders are required.

---

## **Files**

### `main.py`

Runs the full pipeline:
metadata extraction → RDF graph → SHACL validation → semantic search.

### `ontology.ttl`

Defines a simple `Document` class and three properties:
`ex:title`, `ex:topic`, `ex:domain`.

### `shapes.ttl`

Defines SHACL constraints requiring all three fields to appear once.

### `requirements.txt`

Minimal list of Python dependencies.

### `.gitignore`

Standard Python ignores:
`__pycache__/`, `.venv/`, `.DS_Store`, etc.

---

## **Requirements**

* Python **3.10+** recommended
* A valid API key for your LLM provider (example: `OPENAI_API_KEY` for OpenAI through LiteLLM)

---

## **Installation**

```bash
git clone https://github.com/<your-username>/autodoc-litellm.git
cd autodoc-litellm

# create virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

---

## **Run the demo**

```bash
export OPENAI_API_KEY="your-key"
python main.py
```

(Windows PowerShell)

```powershell
setx OPENAI_API_KEY "your-key"
python main.py
```

---

## **Notes**

* All logic is intentionally simple and in one file.
* You can adapt this structure to a larger project later.
* Ontology and SHACL shapes are minimal on purpose.


