import os
import pickle
import evaluate
import yaml

from tqdm import tqdm


def prepare_paths(args):
    bare_answer_dirs = [
        args["bare_answer_dir"].format(
            model_alias=bare_model, dataset_alias=args["dataset_alias"]
        )
        for bare_model in args["bare_model_list"]
    ]

    finetune_answer_dirs = {
        bare_model: [
            args["finetune_answer_dir"].format(
                model_alias=finetune_model, dataset_alias=args["dataset_alias"]
            )
            for finetune_model in args["finetune_model_list"][bare_model]
        ]
        for bare_model in args["bare_model_list"]
    }
    return bare_answer_dirs, finetune_answer_dirs


def preload_answers(answer_dirs, dataset):
    cache = {d: {} for d in answer_dirs}
    for entry in tqdm(dataset, desc="Preloading answers"):
        idx = entry["index"]
        for d in answer_dirs:
            with open(f"{d}/answer_{idx}.pkl", "rb") as f:
                cache[d][idx] = pickle.load(f)
    return cache


def filter_by_bert(args, dataset, bare_dirs, finetune_dirs):
    bertscore = evaluate.load("bertscore")

    oracle_answers = [entry["response"] for entry in dataset]

    # Preload once (bare + finetune together)
    all_dirs = bare_dirs + [d for dirs in finetune_dirs.values() for d in dirs]
    cache = preload_answers(all_dirs, dataset)

    # Collect bare answers
    bare_answers = {
        model: [cache[bare_dirs[i]][entry["index"]] for entry in dataset]
        for i, model in enumerate(args["bare_model_list"])
    }

    # Collect finetune answers
    finetune_answers = {
        model: [
            [cache[finetune_dirs[model][k]][entry["index"]] for entry in dataset]
            for k in range(len(finetune_dirs[model]))
        ]
        for model in args["bare_model_list"]
    }

    # Helper for batched scoring
    def batched_scores(preds, refs):
        scores = []
        for i in tqdm(range(0, len(preds), args["batch_size"])):
            batch_preds = preds[i : i + args["batch_size"]]
            batch_refs = refs[i : i + args["batch_size"]]
            batch_scores = bertscore.compute(
                predictions=batch_preds,
                references=batch_refs,
                model_type="distilbert-base-uncased",
            )["f1"]
            scores.extend(batch_scores)
        return scores

    # Compute scores
    bare_scores = {
        model: batched_scores(bare_answers[model], oracle_answers)
        for model in args["bare_model_list"]
    }

    finetune_scores = {
        model: [batched_scores(ans, oracle_answers) for ans in finetune_answers[model]]
        for model in args["bare_model_list"]
    }

    # Filtering
    tainted_indices = []
    for i, entry in enumerate(dataset):
        keep = any(
            finetune_scores[bare_model][j][i] - bare_scores[bare_model][i]
            < args["metric_threshold"]
            for bare_model in args["bare_model_list"]
            for j in range(len(finetune_scores[bare_model]))
        )
        if not keep:
            tainted_indices.append(entry["index"])

    return tainted_indices


def select_tainted_samples(args):
    os.makedirs(args["tainted_sample_dir"], exist_ok=True)

    with open(
        args["general_dataset_path"].format(dataset_alias=args["dataset_alias"]), "rb"
    ) as f:
        dataset = pickle.load(f)

    bare_dirs, finetune_dirs = prepare_paths(args)

    if args["metric"] == "bert":
        tainted_sample_index = filter_by_bert(args, dataset, bare_dirs, finetune_dirs)
    elif args["metric"] == "prob":
        raise NotImplementedError("Probability filtering not implemented")
    else:
        raise ValueError("Invalid metric")

    out_path = f"{args['tainted_sample_dir']}/{args['dataset_alias']}_{args['metric']}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(tainted_sample_index, f)


if __name__ == "__main__":
    with open(os.path.join("../setting", "filter_config.yaml"), "r") as file:
        config = yaml.safe_load(file)
    
    select_tainted_samples(config)