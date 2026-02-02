---
name: AI Model Notebook Generator
description: Creates end-to-end Colab notebooks for any new AI model release when user mentions a model name and provider
---

# AI Model Notebook Generator

## When to Use This Skill

**Trigger this skill when the user:**
- Mentions a new AI model name (e.g., "GPT-4o", "Claude opus 4.5", "Gemini 3.0", "DeepSeek V3", "Llama 4.1")
- Asks to create notebooks for a model
- Mentions a provider (OpenAI, Google, Anthropic, OpenRouter, Ollama, etc.)
- Says things like "create notebooks for [model]" or "make a cookbook for [model]"

---

# Setup
!pip install openai langchain-openai

from google.colab import userdata
import openai

client = openai.OpenAI(api_key=userdata.get("OPENAI_API_KEY"))

# Basic call
response = client.chat.completions.create(
    model="gpt-4o",  # or gpt-4o-mini, gpt-4-turbo, etc.
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Google Gemini Models
```python
# Setup
!pip install google-genai

from google.colab import userdata
from google import genai

client = genai.Client(api_key=userdata.get("GOOGLE_API_KEY"))

# Basic call
response = client.models.generate_content(
    model="gemini-2.5-flash",  # or gemini-2.5-pro, etc.
    contents="Hello!"
)
```

### Anthropic Claude Models
```python
# Setup
!pip install anthropic langchain-anthropic

from google.colab import userdata
import anthropic

client = anthropic.Anthropic(api_key=userdata.get("ANTHROPIC_API_KEY"))

# Basic call
response = client.messages.create(
    model="claude-4-5-sonnet",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### OpenRouter (Multiple Providers)
```python
# Setup
!pip install openai

from google.colab import userdata
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=userdata.get("OPENROUTER_API_KEY")
)

# Basic call
response = client.chat.completions.create(
    model="anthropic/claude-4.5-sonnet",  # or openai/gpt-4o, 
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Ollama (Local Models)
```python
# Setup (requires Ollama running locally)
!pip install ollama

import ollama

# Basic call
response = ollama.chat(
    model="llama3.1",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Required Header for ALL Notebooks

Add this at the TOP of every notebook (first markdown cell):

```markdown
<img src="https://drive.google.com/uc?export=view&id=1wYSMgJtARFdvTt5g7E20mE4NmwUFUuog" width="200">

[![Gen AI Experiments](https://img.shields.io/badge/Gen%20AI%20Experiments-GenAI%20Bootcamp-blue?style=for-the-badge&logo=artificial-intelligence)](https://github.com/buildfastwithai/gen-ai-experiments)
[![Gen AI Experiments GitHub](https://img.shields.io/github/stars/buildfastwithai/gen-ai-experiments?style=for-the-badge&logo=github&color=gold)](http://github.com/buildfastwithai/gen-ai-experiments)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/[NOTEBOOK_ID])

## Master Generative AI in 8 Weeks
**What You'll Learn:**
- Master cutting-edge AI tools & frameworks
- 6 weeks of hands-on, project-based learning
- Weekly live mentorship sessions
- No coding experience required
- Join Innovation Community

Transform your AI ideas into reality through hands-on projects and expert mentorship.

[Start Your Journey](https://www.buildfastwithai.com/genai-course)

---
```

**Note:** Replace `[NOTEBOOK_ID]` with actual Colab notebook ID.

---

## 10 Notebooks to Create

### 1. Testing & Basics
**File:** `01_[ModelName]_Testing_Basics.ipynb`

| Section | Content |
|---------|---------|
| Setup | Hello world, parameters, environment |
| Tool Calling | Weather, calculator, search functions |
| Simple Agent | ReAct pattern with 2-3 tools |
| RAG Quick Demo | FAISS/Chroma, basic retrieval |
| Use Cases | Customer Support, Code Assistant, Data Analysis |
| Metrics | Timing, tokens, cost estimates |

---

### 2. Advanced Features
**File:** `02_[ModelName]_Advanced_Features.ipynb`

| Section | Content |
|---------|---------|
| Streaming | Token-by-token, with tools |
| Function Calling | Parallel calls, validation, schemas |
| Structured Output | JSON mode, Pydantic, validation |
| Advanced Prompting | Few-shot, chain-of-thought |
| Context Management | History, truncation, compression |
| Batch Processing | Rate limiting, optimization |
| Caching | Response caching, TTL |
| Error Handling | Retries, fallbacks |

---

### 3. Simple RAG
**File:** `03_[ModelName]_Simple_RAG.ipynb`

| Step | Content |
|------|---------|
| 1 | RAG Fundamentals & Architecture |
| 2 | Document Loading (TXT, PDF, CSV) |
| 3 | Text Chunking (strategies, size, overlap) |
| 4 | Embedding Generation |
| 5 | Vector Store (FAISS setup & persistence) |
| 6 | Retrieval (similarity search, top-k) |
| 7 | Generation (prompt construction, context) |
| 8 | Full Pipeline & Testing |

---

### 4. Advanced RAG
**File:** `04_[ModelName]_Advanced_RAG.ipynb`

| Technique | Content |
|-----------|---------|
| Hybrid Search | BM25 + vector, fusion |
| Query Transform | Expansion, multi-query, HyDE |
| Chunking | Semantic, parent-child |
| Reranking | Cross-encoder, MMR |
| Filtering | Metadata-based retrieval |
| Compression | Contextual compression |
| Reasoning | Multi-step RAG |
| Evaluation | Accuracy, faithfulness metrics |

---

### 5. CrewAI Agents
**File:** `05_[ModelName]_CrewAI_Agents.ipynb`

| Section | Content |
|---------|---------|
| Basics | Agents, tasks, crew concepts |
| Single Agent | Role, goal, tools setup |
| Tools | Built-in + custom tools |
| Multi-Agent | Researcher, writer, editor collaboration |
| Tasks | Dependencies, output formats |
| Crew Config | Sequential/hierarchical patterns |
| Use Cases | Research & Content, Data Analysis |
| Advanced | Memory, callbacks, error handling |

---

### 6. Agno Agent Framework
**File:** `06_[ModelName]_Agno_Agents.ipynb`

| Section | Content |
|---------|---------|
| Setup | Framework intro, installation |
| Basic Agent | Initialization, simple tasks |
| Capabilities | Tools, memory, state |
| Multi-Agent | Orchestration, communication |
| Custom Tools | Creation & integration |
| Use Cases | Personal Assistant, Code Review |
| Advanced | Conditional logic, human-in-loop |
| Comparison | vs CrewAI, best practices |

---

### 7. Multimodal RAG
**File:** `07_[ModelName]_Multimodal_RAG.ipynb`

| Step | Content |
|------|---------|
| 1 | Multimodal RAG overview |
| 2 | Document processing (PDFs, OCR) |
| 3 | Image understanding (captioning, VQA) |
| 4 | Multimodal embeddings (CLIP) |
| 5 | Hybrid vector store |
| 6 | Cross-modal retrieval |
| 7 | Vision-language generation |
| 8 | Use cases (charts, catalogs) |

---



### 10. Specialized Use Cases
**File:** `10_[ModelName]_Specialized_UseCases.ipynb`

Choose 3-5 based on model capabilities:
- Fine-tuning / Prompt Optimization
- Multimodal Applications
- Domain-Specific (medical, legal, code)
- Evaluation & Benchmarking
- Production Integrations (FastAPI, Streamlit)

---

## Quality Checklist

- [ ] All cells execute successfully
- [ ] No hardcoded API keys (use Colab secrets)
- [ ] Error handling included
- [ ] Clear documentation
- [ ] Performance metrics
- [ ] Cost estimates
- [ ] @BuildFastWithAI branding

---

## File Structure

```
model-notebooks/
├── 01_[Model]_Testing_Basics.ipynb
├── 02_[Model]_Advanced_Features.ipynb
├── 03_[Model]_Simple_RAG.ipynb
├── 04_[Model]_Advanced_RAG.ipynb
├── 05_[Model]_CrewAI_Agents.ipynb
├── 06_[Model]_Agno_Agents.ipynb
├── 07_[Model]_Multimodal_RAG.ipynb
├── 08_[Model]_LangChain_Complete.ipynb
├── 09_[Model]_LangGraph_Complete.ipynb
├── 10_[Model]_Specialized_UseCases.ipynb
└── README.md
```

---

## Quick Tips

- Use descriptive variable names (snake_case)
- Keep notebooks under 15min runtime for basics
- Test in clean Colab environment
- Include real-world examples
- Update within 48hrs of new model release
- Share on Twitter with highlights

---

*Maintained by: @BuildFastWithAI*
