import os
import pickle
import evaluate
import yaml

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def prepare_paths(args):
    """Prepare directories for bare and fine-tuned models."""
    bare_answer_dirs = [
        args["bare_prediction_dir"].format(
            model_alias=bare_model, dataset_alias=args["dataset_alias"]
        )
        for bare_model in args["filter_bare_list"]
    ]

    finetune_answer_dirs = {
        bare_model: [
            args["finetune_prediction_dir"].format(
                model_alias=finetune_model, dataset_alias=args["dataset_alias"]
            )
            for finetune_model in args["filter_finetune_list"][bare_model]
        ]
        for bare_model in args["filter_bare_list"]
    }
    return bare_answer_dirs, finetune_answer_dirs


def get_model_indices(args):
    """Compute indices for bare and fine-tuned models."""
    bare_model_index = []
    finetune_model_index = []
    model_index = 0

    for bare_model in args["filter_bare_list"]:
        bare_model_index.append(model_index)
        finetune_indices = list(
            range(model_index + 1, model_index + 1 + len(args["filter_finetune_list"][bare_model]))
        )
        finetune_model_index.append(finetune_indices)
        model_index += len(finetune_indices) + 1

    return bare_model_index, finetune_model_index


def load_answers(answer_dirs, dataset_index):
    """Load answers from specified directories."""
    return [
        pickle.load(open(f"{answer_dir}/answer_{dataset_index}.pkl", "rb"))
        for answer_dir in answer_dirs
    ]


def filter_by_tfidf(args, dataset, bare_dirs, finetune_dirs, bare_model_index, finetune_model_index):
    """Filter dataset using TF-IDF metric."""
    tfidf_vectorizer = TfidfVectorizer()
    selected_indices = []

    for entry in tqdm(dataset, desc="Filtering by TF-IDF"):
        corpus = []
        for model_index, bare_dir in enumerate(bare_dirs):
            corpus.extend(load_answers([bare_dir], entry["index"]))
            corpus.extend(
                load_answers(finetune_dirs[args["filter_bare_list"][model_index]], entry["index"])
            )
        corpus.append(entry["response"])

        if any(len(answer) < args["length_threshold"] for answer in corpus):
            continue

        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
        if not check_similarity(tfidf_matrix, bare_model_index, finetune_model_index, args):
            selected_indices.append(entry["index"])

    return selected_indices


def check_similarity(tfidf_matrix, bare_model_index, finetune_model_index, args):
    """Check similarity for TF-IDF metric."""
    for bare_idx, finetune_indices in zip(bare_model_index, finetune_model_index):
        nonmemref_vs_benchmark = cosine_similarity(
            tfidf_matrix[bare_idx], tfidf_matrix[-1]
        )[0][0]
        for finetune_idx in finetune_indices:
            memref_vs_benchmark = cosine_similarity(
                tfidf_matrix[finetune_idx], tfidf_matrix[-1]
            )[0][0]
            if memref_vs_benchmark - nonmemref_vs_benchmark < args["similarity_threshold"]:
                return True
    return False


def filter_by_bert(args, dataset, bare_dirs, finetune_dirs):
    """Filter dataset using BERT metric."""
    bertscore = evaluate.load("bertscore")
    selected_indices = []

    benchmark_responses = [entry["response"] for entry in dataset]
    bare_answers = {model: [] for model in args["filter_bare_list"]}
    finetune_answers = {
        model: [[] for _ in args["filter_finetune_list"][model]]
        for model in args["filter_bare_list"]
    }

    for entry in tqdm(dataset, desc="Loading answers for BERT"):
        for model_idx, model_name in enumerate(args["filter_bare_list"]):
            bare_answers[model_name].append(
                load_answers([bare_dirs[model_idx]], entry["index"])[0]
            )
            for k, finetune_dir in enumerate(finetune_dirs[model_name]):
                finetune_answers[model_name][k].append(
                    load_answers([finetune_dir], entry["index"])[0]
                )

    bare_scores = {
        model: bertscore.compute(
            predictions=bare_answers[model], references=benchmark_responses, model_type="distilbert-base-uncased"
        )["f1"]
        for model in args["filter_bare_list"]
    }
    finetune_scores = {
        model: [
            bertscore.compute(
                predictions=answers, references=benchmark_responses, model_type="distilbert-base-uncased"
            )["f1"]
            for answers in finetune_answers[model]
        ]
        for model in args["filter_bare_list"]
    }

    for i, _ in enumerate(dataset):
        if not any(
            finetune_scores[bare_model][j][i] - bare_scores[bare_model][i]
            < args["similarity_threshold"]
            for bare_model in args["filter_bare_list"]
            for j in range(len(finetune_scores[bare_model]))
        ):
            selected_indices.append(dataset[i]["index"])

    return selected_indices


def select_data(args):
    """Main function to select data."""
    os.makedirs(os.path.dirname(args["selected_index_path"]), exist_ok=True)

    with open(args["general_dataset_path"].format(dataset_alias=args["dataset_alias"]), "rb") as file:
        dataset = pickle.load(file)

    bare_dirs, finetune_dirs = prepare_paths(args)
    bare_model_index, finetune_model_index = get_model_indices(args)

    if args["metric"] == "TFIDF":
        selected_indices = filter_by_tfidf(
            args, dataset, bare_dirs, finetune_dirs, bare_model_index, finetune_model_index
        )
    elif args["metric"] == "BERT":
        selected_indices = filter_by_bert(args, dataset, bare_dirs, finetune_dirs)
    else:
        raise ValueError(f"Unsupported metric: {args['metric']}")

    print(f"Selected {len(selected_indices)} entries.")
    with open(
        args["selected_index_path"].format(
            dataset_alias=args["dataset_alias"], metric=args["metric"]
        ),
        "wb",
    ) as file:
        pickle.dump(selected_indices, file)


if __name__ == "__main__":
    with open(os.path.join("../setting", "filter_config.yaml"), "r") as file:
        config = yaml.safe_load(file)
    select_data(config)
