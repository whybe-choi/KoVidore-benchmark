# KoVidore Benchmark

Korean Vision Document Retrieval (KoVidore) benchmark for evaluating text-to-image retrieval models on Korean visual documents.

## Overview

KoVidore is a comprehensive benchmark based on the MTEB framework that evaluates models' ability to retrieve relevant visual documents (screenshots, slides, office documents) based on Korean text queries. The benchmark consists of 5 different tasks across various document types.

## Tasks

| Task | Description | Documents | Queries | Main Metric |
|------|-------------|-----------|---------|-------------|
| **MIR** | Multimodal Information Retrieval | 1,366 | 1,496 | NDCG@5 |
| **VQA** | Visual Question Answering | 1,101 | 1,500 | NDCG@5 |
| **Slide** | Presentation Slides | 1,415 | 180 | NDCG@5 |
| **Office** | Office Documents | 1,993 | 222 | NDCG@5 |
| **FinOCR** | Financial OCR Documents | 2,000 | 187 | NDCG@5 |

## Installation

```bash
uv sync
```

## Quick Start

### Using the CLI:

```bash
# Run all tasks with default model
uv run kovidore

# Run with custom model
uv run kovidore --model "your-model-name"

# Run specific tasks
uv run kovidore --model "your-model-name" --tasks mir vqa

# List available tasks
uv run kovidore --list-tasks
```

### Using as a library:

```python
from src.evaluate import run_benchmark

# Run all tasks
evaluation = run_benchmark("your-model-name")

# Run specific tasks
evaluation = run_benchmark("your-model-name", tasks=["mir", "vqa"])
```

### Individual task evaluation:

```python
from mteb import MTEB
from src.evaluate import KoVidoreMIRRetrieval
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("your-model-name")
evaluation = MTEB(tasks=[KoVidoreMIRRetrieval()])
results = evaluation.run(model)
```

## Available Task Keys

When running specific tasks, use these keys:

- `mir`: Multimodal Information Retrieval
- `vqa`: Visual Question Answering  
- `slide`: Presentation Slides
- `office`: Office Documents
- `finocr`: Financial OCR Documents

## Results

Results are automatically saved in the `results/` directory after evaluation completion. The benchmark uses NDCG@5 as the main evaluation metric for all tasks.

### Performance Leaderboard

The following table shows NDCG@5 performance across all KoVidore tasks:

| Model | MIR | VQA | Slide | Office | FinOCR | ViDoRe v2 (English) |
|-------|-----|-----|-------|--------|--------|--------|
| **CLIP-ViT-bigG-14** | 0.02015 | 0.0561 | 0.12331 | 0.03929 | 0 | 17.6 |
| **Nomic Embed Multimodal 3B** | 0.68499 | 0.84348 | 0.80148 | 0.85679 | 0.87196 | TBA |
| **ColPali v1.3** | 0.18125 | 0.36716 | 0.55297 | 0.25889 | 0.37437 | 54.2 |
| **ColQwen2 v1.0** | 0.53614 | 0.75765 | 0.74257 | 0.66598 | 0.67542 | TBA |
| **ColQwen2.5 v0.2** | 0.6028 | 0.8095 | 0.79074 | 0.76099 | 0.7472 | TBA |
| **SigLIP Large 384** | 0.03186 | 0.07397 | 0.19783 | 0.053 | 0.00975 | TBA |
| **Jina Embeddings v4** | 0.71824 | 0.86245 | TBA | TBA | 0.88922 | TBA |

*TBA: To Be Announced - Results will be updated as evaluations are completed.*

### Model Details

- **CLIP-ViT-bigG-14**: `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`
- **Nomic Embed Multimodal 3B**: `nomic-ai/colnomic-embed-multimodal-3b`  
- **ColPali v1.3**: `vidore/colpali-v1.3`
- **ColQwen2 v1.0**: `vidore/colqwen2-v1.0`
- **ColQwen2.5 v0.2**: `vidore/colqwen2.5-v0.2`
- **SigLIP Large 384**: `google/siglip-large-patch16-384`
- **Jina Embeddings v4**: `jinaai/jina-embeddings-v4`

## License

TBA
