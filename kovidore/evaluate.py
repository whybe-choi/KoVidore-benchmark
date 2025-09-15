import logging
import traceback
import json
from typing import List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from mteb import MTEB
from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
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
        logger.error(f"Data directory not found: {data_dir}")
        logger.error("Please ensure the data directory exists and contains the required CSV files")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    for split in splits:
        # Load queries
        queries_file = data_dir / "queries.csv"
        query_data = []
        if queries_file.exists():
            try:
                queries_df = pd.read_csv(queries_file)
                logger.info(f"Loaded queries CSV with {len(queries_df)} rows")
                for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc=f"Processing queries for {split}"):
                    query_data.append({
                        "id": f"query-{split}-{row['query-id']}",
                        "text": str(row["text"]),
                        "image": None,
                        "modality": "text"
                    })
                logger.info(f"Created {len(query_data)} query entries for split {split}")
            except Exception as e:
                logger.error(f"Failed to read queries file {queries_file}: {e}")
                logger.debug(f"Queries file error traceback:\n{traceback.format_exc()}")
                raise
        else:
            logger.warning(f"Queries file not found: {queries_file}")
        queries[split] = Dataset.from_list(query_data)
        
        # Load corpus data
        corpus_file = data_dir / "corpus.csv"
        if corpus_file.exists():
            try:
                corpus_df = pd.read_csv(corpus_file)
                logger.info(f"Loaded corpus CSV with {len(corpus_df)} rows")

                # Create generator with image paths (using datasets.Image feature type)
                def corpus_generator():
                    for _, row in tqdm(corpus_df.iterrows(), total=len(corpus_df), desc=f"Processing corpus for {split}"):
                        corpus_id = str(row["corpus-id"])
                        image_path_str = row.get("image_path", "")

                        yield {
                            "id": f"corpus-{split}-{corpus_id}",
                            "text": None,
                            "image": image_path_str if image_path_str and Path(image_path_str).exists() else None,
                            "modality": "image"
                        }

                logger.info(f"Created corpus generator for {len(corpus_df)} entries for split {split}")
            except Exception as e:
                logger.error(f"Failed to read corpus file {corpus_file}: {e}")
                logger.debug(f"Corpus file error traceback:\n{traceback.format_exc()}")
                raise
        else:
            logger.warning(f"Corpus file not found: {corpus_file}")
        from datasets import Features, Value, Image
        features = Features({
            "id": Value("string"),
            "text": Value("string"),
            "image": Image(),
            "modality": Value("string")
        })
        corpus[split] = Dataset.from_generator(corpus_generator, features=features)
        
        # Load qrels (relevance judgments)
        qrels_file = data_dir / "qrels.csv"
        relevant_docs[split] = {}
        if qrels_file.exists():
            try:
                qrels_df = pd.read_csv(qrels_file)
                logger.info(f"Loaded qrels CSV with {len(qrels_df)} rows")

                query_ids = qrels_df['query-id'].apply(lambda x: f"query-{split}-{x}")
                corpus_ids = qrels_df['corpus-id'].apply(lambda x: f"corpus-{split}-{x}")
                scores = qrels_df['score'].astype(int)

                from collections import defaultdict
                temp_docs = defaultdict(dict)

                for qid, cid, score in tqdm(zip(query_ids, corpus_ids, scores), total=len(qrels_df), desc=f"Processing qrels for {split}"):
                    temp_docs[qid][cid] = score

                relevant_docs[split] = dict(temp_docs)
                logger.info(f"Created {len(relevant_docs[split])} qrels entries for split {split}")
            except Exception as e:
                logger.error(f"Failed to read qrels file {qrels_file}: {e}")
                logger.debug(f"Qrels file error traceback:\n{traceback.format_exc()}")
                raise
        else:
            logger.warning(f"Qrels file not found: {qrels_file}")
    
    logger.info(f"Loaded {subset_name} successfully")
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
                    "num_queries": 198,
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


class KoVidoreTestRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="KoVidoreTestRetrieval",
        description="Test dataset for Korean visual document retrieval.",
        reference="https://github.com/facerain/KoVidore-benchmark",
        dataset={
            "path": "data/test",
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
                    "num_documents": 10,
                    "num_queries": 5,
                    "average_relevant_docs_per_query": 2.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_local_data(
            subset_name="test",
            splits=self.metadata_dict["eval_splits"]
        )

        self.data_loaded = True


AVAILABLE_TASKS = {
    "mir": KoVidoreMIRRetrieval,
    "vqa": KoVidoreVQARetrieval,
    "slide": KoVidoreSlideRetrieval,
    "office": KoVidoreOfficeRetrieval,
    "finocr": KoVidoreFinOCRRetrieval,
    "test": KoVidoreTestRetrieval,
}

ALL_TASKS = ["mir", "vqa", "slide", "office", "finocr"]


def check_existing_results(model_name: str, tasks: List[str]) -> Tuple[List[str], List[str]]:
    """
    Check which tasks already have results and which need to be evaluated.

    Args:
        model_name: Name of the model
        tasks: List of task names to check

    Returns:
        Tuple of (tasks_to_run, tasks_with_results)
    """
    results_dir = Path("results") / model_name
    tasks_to_run = []
    tasks_with_results = []

    for task in tasks:
        task_class_name = AVAILABLE_TASKS[task].__name__

        # Look for existing result files with this task name
        found = False
        if results_dir.exists():
            for result_file in results_dir.rglob(f"{task_class_name}.json"):
                try:
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)

                    # Check if the result file has valid scores
                    if "scores" in result_data and result_data["scores"]:
                        logger.info(f"Found existing results for {task} ({task_class_name}): {result_file}")
                        tasks_with_results.append(task)
                        found = True
                        break
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Invalid result file {result_file}: {e}")
                    continue

        if not found:
            tasks_to_run.append(task)

    return tasks_to_run, tasks_with_results


def run_benchmark(
    model_name: str = "average_word_embeddings_komninos",
    tasks: Optional[List[str]] = None,
    batch_size: int = 16,
    skip_existing: bool = True
):
    """
    Run KoVidore benchmark evaluation.

    Args:
        model_name: Name of the model to evaluate
        tasks: List of tasks to run. If None, runs all tasks.
               Available: "mir", "vqa", "slide", "office", "finocr"
        batch_size: Batch size for encoding (default: 16)
        skip_existing: If True, skip tasks that already have results (default: True)

    Returns:
        MTEB evaluation object or None if failed
    """
    try:
        import mteb

        if tasks is None:
            tasks = ALL_TASKS

        # Check for existing results and filter tasks
        if skip_existing:
            tasks_to_run, tasks_with_results = check_existing_results(model_name, tasks)

            if tasks_with_results:
                logger.info(f"Skipping {len(tasks_with_results)} tasks with existing results: {tasks_with_results}")

            if not tasks_to_run:
                logger.info("All requested tasks already have results. Nothing to do.")
                return None

            logger.info(f"Will evaluate {len(tasks_to_run)} tasks: {tasks_to_run}")
            tasks = tasks_to_run

        # Validate remaining tasks
        invalid_tasks = [task for task in tasks if task not in AVAILABLE_TASKS]
        if invalid_tasks:
            logger.error(f"Invalid tasks specified: {invalid_tasks}")
            logger.error(f"Available tasks: {list(AVAILABLE_TASKS.keys())}")
            logger.error("Please check your task names and try again")
            return None
        
        # Use mteb.get_model() for standardized model loading
        try:
            logger.info(f"Loading model: {model_name}")
            model = mteb.get_model(model_name)
            logger.info(f"Model loaded successfully: {type(model).__name__}")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            logger.error("Please check if the model name is correct and accessible")
            logger.debug(f"Model loading error traceback:\n{traceback.format_exc()}")
            raise
        selected_tasks = [AVAILABLE_TASKS[task]() for task in tasks]
        
        logger.info(f"Starting evaluation with model: {model_name}")
        logger.info(f"Running tasks: {tasks}")
        
        try:
            evaluation = mteb.MTEB(tasks=selected_tasks)
            logger.info(f"Starting evaluation with batch_size={batch_size}")
            results = evaluation.run(model, output_folder=f"results/{model_name}", encode_kwargs = {'batch_size': batch_size})
        except Exception as e:
            logger.error(f"Evaluation execution failed: {e}")
            logger.error(f"Model: {model_name}, Tasks: {tasks}, Batch size: {batch_size}")
            logger.debug(f"Evaluation error traceback:\n{traceback.format_exc()}")
            raise
        
        logger.info("Evaluation completed successfully")
        return evaluation
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        logger.error("Please install mteb and sentence-transformers:")
        logger.error("  pip install mteb sentence-transformers")
        logger.debug(f"Import error traceback:\n{traceback.format_exc()}")
        return None
    except FileNotFoundError as e:
        logger.error(f"File or directory not found: {e}")
        logger.error("Please ensure data directories exist and contain required files")
        logger.debug(f"File not found traceback:\n{traceback.format_exc()}")
        return None
    except ValueError as e:
        logger.error(f"Invalid parameter value: {e}")
        logger.debug(f"Value error traceback:\n{traceback.format_exc()}")
        return None
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return None


if __name__ == "__main__":
    run_benchmark()