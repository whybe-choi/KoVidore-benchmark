import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from PIL import Image

from mteb import MTEB
from mteb.abstasks import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _load_local_data(subset_name: str, splits: List[str] = ["test"]):
    """
    Load data from local directory structure in MTEB format.
    
    Expected structure:
    data/{subset_name}/
    ├── queries.csv
    ├── qrels.csv
    ├── corpus.csv
    └── images/
        ├── {corpus_id}.jpg or {corpus_id}.png
        └── ...
    """
    from datasets import Dataset
    
    corpus = {}
    queries = {}
    relevant_docs = {}
    
    data_dir = Path(f"data/{subset_name}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    for split in splits:
        # Load queries
        queries_file = data_dir / "queries.csv"
        query_data = []
        if queries_file.exists():
            queries_df = pd.read_csv(queries_file)
            for _, row in queries_df.iterrows():
                query_data.append({
                    "id": f"query-{split}-{row['query-id']}",
                    "text": str(row["text"]),
                    "image": None,
                    "modality": "text"
                })
        queries[split] = Dataset.from_list(query_data)
        
        # Load corpus data
        corpus_file = data_dir / "corpus.csv"
        corpus_data = []
        if corpus_file.exists():
            corpus_df = pd.read_csv(corpus_file)
            for _, row in corpus_df.iterrows():
                corpus_id = str(row["corpus-id"])
                image_path_str = row.get("image_path", "")
                
                # Load image if path exists
                image = None
                if image_path_str and Path(image_path_str).exists():
                    try:
                        image = Image.open(image_path_str)
                    except Exception as e:
                        logger.warning(f"Failed to load image {image_path_str}: {e}")
                
                corpus_data.append({
                    "id": f"corpus-{split}-{corpus_id}",
                    "text": None,
                    "image": image,
                    "modality": "image"
                })
        corpus[split] = Dataset.from_list(corpus_data)
        
        # Load qrels (relevance judgments)
        qrels_file = data_dir / "qrels.csv"
        relevant_docs[split] = {}
        if qrels_file.exists():
            qrels_df = pd.read_csv(qrels_file)
            for _, row in qrels_df.iterrows():
                query_id = f"query-{split}-{row['query-id']}"
                corpus_id = f"corpus-{split}-{row['corpus-id']}"
                score = int(row["score"])
                
                # Add to relevant_docs
                if query_id not in relevant_docs[split]:
                    relevant_docs[split][query_id] = {}
                relevant_docs[split][query_id][corpus_id] = score
    
    logger.info(f"Loaded {subset_name}: {len(query_data)} queries, {len(corpus_data)} documents")
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
        
        # Debug: Print data structure
        logger.info(f"Corpus type: {type(self.corpus['test'])}")
        logger.info(f"Corpus length: {len(self.corpus['test'])}")
        if len(self.corpus['test']) > 0:
            logger.info(f"Sample corpus entry: {self.corpus['test'][0]}")
        
        logger.info(f"Queries type: {type(self.queries['test'])}")
        logger.info(f"Queries length: {len(self.queries['test'])}")
        if len(self.queries['test']) > 0:
            logger.info(f"Sample query: {self.queries['test'][0]}")

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
    tasks: Optional[List[str]] = None,
    batch_size: int = 16
):
    """
    Run KoVidore benchmark evaluation.
    
    Args:
        model_name: Name of the model to evaluate
        tasks: List of tasks to run. If None, runs all tasks.
               Available: "mir", "vqa", "slide", "office", "finocr"
        batch_size: Batch size for encoding (default: 16)
    
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
        results = evaluation.run(model, output_folder=f"results/{model_name}", encode_kwargs = {'batch_size': batch_size})
        
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