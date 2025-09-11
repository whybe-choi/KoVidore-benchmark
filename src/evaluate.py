import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from PIL import Image

from mteb import MTEB
from mteb.abstasks.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _load_local_data(subset_name: str, splits: List[str] = ["test"]) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, str]], Dict[str, Dict[str, Dict[str, int]]]]:
    """
    Load data from local directory structure.
    
    Expected structure:
    data/{subset_name}/
    ├── queries.csv
    ├── qrels.csv
    └── images/
        ├── {corpus_id}.jpg or {corpus_id}.png
        └── ...
    """
    corpus: Dict[str, Dict[str, Dict[str, Any]]] = {}
    queries: Dict[str, Dict[str, str]] = {}
    relevant_docs: Dict[str, Dict[str, Dict[str, int]]] = {}
    
    # Auto-detect image extension from the first available image
    def detect_image_extension(images_dir: Path) -> str:
        """Detect the image extension used in this dataset"""
        for ext in [".jpg", ".jpeg", ".png"]:
            if any(f.suffix.lower() == ext for f in images_dir.glob("*")):
                return ext
        return ".jpg"  # fallback
    
    data_dir = Path(f"data/{subset_name}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Auto-detect image extension for this subset
    images_dir = data_dir / "images"
    image_ext = detect_image_extension(images_dir) if images_dir.exists() else ".jpg"
    logger.info(f"Using image extension '{image_ext}' for {subset_name}")
    
    for split in splits:
        corpus[split] = {}
        queries[split] = {}
        relevant_docs[split] = {}
        
        # Load queries
        queries_file = data_dir / "queries.csv"
        if queries_file.exists():
            queries_df = pd.read_csv(queries_file)
            for _, row in queries_df.iterrows():
                queries[split][str(row["query-id"])] = str(row["text"])
        else:
            logger.warning(f"queries.csv not found in {data_dir}")
        
        # Load qrels
        qrels_file = data_dir / "qrels.csv"
        if qrels_file.exists():
            qrels_df = pd.read_csv(qrels_file)
            for _, row in qrels_df.iterrows():
                query_id = str(row["query-id"])
                corpus_id = str(row["corpus-id"])
                score = int(row["score"])
                
                # Add to corpus if not exists
                if corpus_id not in corpus[split]:
                    image_path = images_dir / f"{corpus_id}{image_ext}"
                    if image_path.exists():
                        try:
                            image = Image.open(image_path)
                            corpus[split][corpus_id] = {"text": "", "image": image}
                        except Exception as e:
                            logger.warning(f"Failed to load image {image_path}: {e}")
                            corpus[split][corpus_id] = {"text": "", "image": None}
                    else:
                        logger.warning(f"Image not found: {image_path}")
                        corpus[split][corpus_id] = {"text": "", "image": None}
                
                # Add to relevant_docs
                if query_id not in relevant_docs[split]:
                    relevant_docs[split][query_id] = {}
                relevant_docs[split][query_id][corpus_id] = score
        else:
            logger.warning(f"qrels.csv not found in {data_dir}")
    
    logger.info(f"Loaded {subset_name}: {len(queries['test'])} queries, {len(corpus['test'])} documents")
    return corpus, queries, relevant_docs


class KoVidoreMIRRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="KoVidoreMIRRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "data/mir",
            "revision": "local",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 1366,
                    "num_queries": 1496,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_local_data(
            subset_name="mir",
            splits=self.metadata_dict["eval_splits"]
        )

        self.data_loaded = True


class KoVidoreVQARetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="KoVidoreVQARetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "data/vqa",
            "revision": "local",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 1101,
                    "num_queries": 1500,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_local_data(
            subset_name="vqa",
            splits=self.metadata_dict["eval_splits"]
        )

        self.data_loaded = True


class KoVidoreSlideRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="KoVidoreSlideRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "data/slide",
            "revision": "local",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 1415,
                    "num_queries": 180,
                    "average_relevant_docs_per_query": 1.2444,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_local_data(
            subset_name="slide",
            splits=self.metadata_dict["eval_splits"]
        )

        self.data_loaded = True


class KoVidoreOfficeRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="KoVidoreOfficeRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "data/office",
            "revision": "local",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 1993,
                    "num_queries": 222,
                    "average_relevant_docs_per_query": 1.0991,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_local_data(
            subset_name="office",
            splits=self.metadata_dict["eval_splits"]
        )

        self.data_loaded = True


class KoVidoreFinOCRRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="KoVidoreFinOCRRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "data/finocr",
            "revision": "local",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 2000,
                    "num_queries": 187,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_local_data(
            subset_name="finocr",
            splits=self.metadata_dict["eval_splits"]
        )

        self.data_loaded = True


AVAILABLE_TASKS = {
    "mir": KoVidoreMIRRetrieval,
    "vqa": KoVidoreVQARetrieval,
    "slide": KoVidoreSlideRetrieval,
    "office": KoVidoreOfficeRetrieval,
    "finocr": KoVidoreFinOCRRetrieval,
}

ALL_TASKS = ["mir", "vqa", "slide", "office", "finocr"]


def run_benchmark(
    model_name: str = "average_word_embeddings_komninos",
    tasks: Optional[List[str]] = None
):
    """
    Run KoVidore benchmark evaluation.
    
    Args:
        model_name: Name of the model to evaluate
        tasks: List of tasks to run. If None, runs all tasks.
               Available: "mir", "vqa", "slide", "office", "finocr"
    
    Returns:
        MTEB evaluation object or None if failed
    """
    try:
        import mteb
        
        if tasks is None:
            tasks = ALL_TASKS
            
        # Validate tasks
        invalid_tasks = [task for task in tasks if task not in AVAILABLE_TASKS]
        if invalid_tasks:
            logger.error(f"Invalid tasks: {invalid_tasks}")
            logger.info(f"Available tasks: {list(AVAILABLE_TASKS.keys())}")
            return None
        
        # Use mteb.get_model() for standardized model loading
        model = mteb.get_model(model_name)
        selected_tasks = [AVAILABLE_TASKS[task]() for task in tasks]
        
        logger.info(f"Starting evaluation with model: {model_name}")
        logger.info(f"Running tasks: {tasks}")
        
        evaluation = mteb.MTEB(tasks=selected_tasks)
        results = evaluation.run(model, output_folder=f"results/{model_name}")
        
        logger.info("Evaluation completed successfully")
        return evaluation
    except ImportError:
        logger.error("Required packages not installed. Please install mteb and sentence-transformers.")
        return None
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return None


if __name__ == "__main__":
    run_benchmark()