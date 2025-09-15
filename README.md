# KoVidore Benchmark

Korean Vision Document Retrieval (KoVidore) benchmark for evaluating text-to-image retrieval models on Korean visual documents.

## Overview
<img src="assets/cover.png" alt="cover">

KoVidore is a comprehensive benchmark for evaluating Korean visual document retrieval capabilities. Built upon the foundation of [ViDoRe](https://github.com/illuin-tech/vidore-benchmark), it assesses how well models can retrieve relevant Korean visual documents—including screenshots, presentation slides, and office documents—when given Korean text queries.

The benchmark encompasses 5 distinct tasks, each targeting different types of visual documents commonly found in Korean business and academic environments. This diverse task structure allows for thorough evaluation of multimodal retrieval performance across various document formats and content types.

## Tasks & Examples

| Task | Description | Documents | Queries | Example Query | Sample Image |
|------|-------------|-----------|---------|---------------|--------------|
| **MIR** | Multimodal Information Retrieval | 1,366 | 1,496 | 코로나19 동절기 집중접종기간 운영개요 혼합형에 대해 알려주세요 | <img src="assets/examples/mir_sample.jpg" width="200" alt="MIR"> |
| **VQA** | Visual Question Answering | 1,101 | 1,500 | 경제협력 품목 중 가장 적은 교역액과 가장 많은 교역액의 차이는 얼마인가요? | <img src="assets/examples/vqa_sample.png" width="200" alt="VQA"> |
| **Slide** | Presentation Slides | 1,415 | 180 | 포털 사이트나 콘텐츠 제공자가 기존 콘텐츠를 다양한 장치로 서비스할 때 얻는 이점은 무엇인가? | <img src="assets/examples/slide_sample.jpg" width="200" alt="Slide"> |
| **Office** | Office Documents | 1,993 | 222 | 정치·사회 이슈를 주제로 하는 유튜브 채널을 통해 정보를 얻는 비율은 얼마인가요? | <img src="assets/examples/office_sample.jpg" width="200" alt="Office"> |
| **FinOCR** | Financial OCR Documents | 2,000 | 198 | 반려동물보험에 가입한 보험계약자 공형진의 증권번호는 무엇인가요? | <img src="assets/examples/finocr_sample.png" width="200" alt="FinOCR"> |

## Performance Leaderboard

The following table shows performance across all KoVidore tasks (ndcg@5 scores as percentages):

| Model | Model Size | FinOCR | MIR | Office | Slide | VQA | Average | ViDoRe V2 (Eng) |
|-------|------------|--------|-----|--------|-------|-----|---------|------------------|
| **nomic-ai/colnomic-embed-multimodal-3b** | 3000 | 82.2 | 70.7 | 86.3 | 78.4 | 84.4 | 80.4 | 55.5 |
| **nomic-ai/colnomic-embed-multimodal-7b** | 7000 | 81.9 | 67.9 | 85.9 | 87.6 | 87.2 | 82.1 | 60.8 |
| **vidore/colqwen2.5-v0.2** | 3000 | 67.3 | 62.5 | 75.3 | 78.0 | 81.0 | 72.8 | 59.3 |
| **vidore/colqwen2-v1.0** | 2210 | 66.3 | 57.4 | 68.7 | 73.9 | 75.5 | 68.4 | 55.0 |
| **jinaai/jina-embeddings-v4** | 3800 | 88.9 | 73.8 | TBA | TBA | 86.2 | TBA | 57.6 |
| **vidore/colpali-v1.2** | 2920 | 43.8 | 20.2 | 28.4 | 51.2 | 36.8 | 36.1 | 50.7 |
| **vidore/colpali-v1.3** | 2920 | 42.6 | 18.8 | 26.4 | 55.3 | 36.6 | 35.9 | 54.2 |
| **vidore/colpali-v1.1** | 2920 | 38.3 | 19.0 | 25.3 | 48.6 | 30.0 | 32.2 | 47.2 |
| **nvidia/llama-nemoretriever-colembed-3b-v1** | 3000 | TBA | TBA | TBA | TBA | TBA | TBA | 63.5 |
| **nvidia/llama-nemoretriever-colembed-1b-v1** | 1000 | TBA | 28.1 | TBA | TBA | 39.4 | TBA | 62.1 |
| **vidore/colSmol-500M** | 500 | 50.9 | 4.7 | 9.7 | 16.1 | 7.4 | 17.8 | 43.5 |
| **vidore/colSmol-256M** | 256 | 46.6 | 4.0 | 8.4 | 13.9 | 7.6 | 16.1 | 32.9 |
| **google/siglip-so400m-patch14-384** | 878 | 4.0 | 3.9 | 6.3 | 21.3 | 7.3 | 8.6 | 31.4 |
| **TIGER-Lab/VLM2Vec-Full** | 4150 | 1.4 | 1.6 | 7.2 | 14.9 | 6.8 | 6.4 | 30.1 |
| **laion/CLIP-ViT-bigG-14-laion2B-39B-b160k** | 2540 | 0.5 | 1.9 | 3.7 | 12.5 | 5.6 | 4.8 | 17.6 |
| **openai/clip-vit-base-patch16** | 151 | 0.3 | 0.6 | 0.0 | 5.9 | 3.3 | 2.5 | 8.3 |
| **ibm-granite/granite-vision-3.3-2b-embedding** | 2980 | 0.0 | 0.4 | 0.6 | 0.3 | 0.0 | 0.26 | 58.1 |

## Installation

```bash
# Install dependencies
uv sync
```

## Quick Start

### Using the CLI

```bash

# Run with custom model
uv run kovidore --model "your-model-name"

# Run specific tasks
uv run kovidore --model "your-model-name" --tasks mir vqa

# Run with custom batch size (default: 16)
uv run kovidore --model "your-model-name" --batch-size 32

# List available tasks
uv run kovidore --list-tasks
```

### Using as a Library

```python
from src.evaluate import run_benchmark

# Run all tasks
evaluation = run_benchmark("your-model-name")

# Run specific tasks
evaluation = run_benchmark("your-model-name", tasks=["mir", "vqa"])

# Run with custom batch size
evaluation = run_benchmark("your-model-name", batch_size=32)
```

## Datasets

We provide pre-processed queries and query-corpus mappings for each task. However, due to licensing restrictions, you'll need to download the image datasets manually from AI Hub (see Acknowledgements section for dataset links).

**Setup Instructions:**
1. Download the required datasets from AI Hub
2. Extract and place images in the following directory structure:
    ```
    data/
    ├── mir/images/
    ├── vqa/images/
    ├── slide/images/
    ├── office/images/
    └── finocr/images/
    ```

The benchmark will automatically locate and use the images from these directories during evaluation.

## Results

Results are automatically saved in the `results/` directory after evaluation completion. The benchmark uses NDCG@5 as the main evaluation metric for all tasks.

## Acknowledgements

This benchmark is inspired by the [ViDoRe](https://github.com/illuin-tech/vidore-benchmark) benchmark. We thank the original authors for their foundational work that helped shape our approach to Korean visual document retrieval.

We also acknowledge the following Korean datasets from AI Hub that were used to construct each task in KoVidore:

- **[멀티모달 정보검색 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71813)** - Used for KoVidoreMIRRetrieval task
- **[시각화 자료 질의응답 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71812)** - Used for KoVidoreVQARetrieval task  
- **[오피스 문서 생성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71811)** - Used for KoVidoreSlideRetrieval and KoVidoreOfficeRetrieval tasks
- **[OCR 데이터(금융 및 물류)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71301)** - Used for KoVidoreFinOCRRetrieval task

## Contact

For questions or suggestions, please open an issue on the GitHub repository or contact the maintainers:

- [Yongbin Choi](https://github.com/whybe-choi) - whybe.choi@gmail.com
- [Yongwoo Song](https://github.com/facerain) - syw5141@khu.ac.kr
