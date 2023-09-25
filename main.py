"""
TEST WITH

# ONE SHOT 
time python main.py --task jailbreak --prompts prompts/jailbreak.md --data_dir data/jailbreak --optimizer passthrough --rounds 0




time python main.py --task ethos --prompts prompts/ethos.md --data_dir data/ethos --out expt7_datasets/treatment.bf.ethos.out --evaluator bf


"""
import requests
import os
import evaluators
import concurrent.futures
from tqdm import tqdm
import time
import json
import argparse
import scorers
import tasks
import predictors
import optimizers
import openai


def get_task_class(task_name):
    if task_name == "ethos":
        return tasks.EthosBinaryTask
    elif task_name == "jailbreak":
        return tasks.JailbreakBinaryTask
    elif task_name == "liar":
        return tasks.DefaultHFBinaryTask
    elif task_name == "ar_sarcasm":
        return tasks.DefaultHFBinaryTask
    elif task_name in ['sst2', 'cr', 'mr']:
        return tasks.SST2Task
    elif task_name == 'subj':
        return tasks.SubjTask
    else:
        raise Exception(f"Unsupported task: {task_name}")


def get_evaluator(evaluator):
    if evaluator == "bf":
        return evaluators.BruteForceEvaluator
    elif evaluator in {"ucb", "ucb-e"}:
        return evaluators.UCBBanditEvaluator
    elif evaluator in {"sr", "s-sr"}:
        return evaluators.SuccessiveRejectsEvaluator
    elif evaluator == "sh":
        return evaluators.SuccessiveHalvingEvaluator
    else:
        raise Exception(f"Unsupported evaluator: {evaluator}")


def get_scorer(scorer):
    if scorer == "01":
        return scorers.Cached01BatchScorer
    elif scorer == "ll":
        return scorers.CachedLogLikelihoodScorer
    else:
        raise Exception(f"Unsupported scorer: {scorer}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--data_dir", default="data/ethos")
    parser.add_argument("--dev_data", type=str, default='/ml-dl/v-qingyanguo/diff_evo/data/cls/sst2/dev_200.txt')
    parser.add_argument("--test_data", type=str, default='/ml-dl/v-qingyanguo/diff_evo/data/cls/sst2/dev_500.txt')

    parser.add_argument("--prompts", default="prompts/sst2.md")
    # parser.add_argument('--config', default='default.json')
    parser.add_argument("--out", default="sst2-2")
    parser.add_argument("--max_threads", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument("--optimizer", default="nl-gradient")
    parser.add_argument("--rounds", default=10, type=int)
    parser.add_argument("--beam_size", default=10, type=int)
    parser.add_argument("--n_test_exs", default=400, type=int)

    parser.add_argument("--minibatch_size", default=64, type=int)
    parser.add_argument("--n_gradients", default=1, type=int)
    parser.add_argument("--errors_per_gradient", default=4, type=int)
    parser.add_argument("--gradients_per_error", default=1, type=int)
    parser.add_argument("--steps_per_gradient", default=1, type=int)
    parser.add_argument("--mc_samples_per_step", default=0, type=int)
    parser.add_argument("--max_expansion_factor", default=8, type=int)

    parser.add_argument("--engine", default="chatgpt", type=str)

    parser.add_argument("--evaluator", default="bf", type=str)
    parser.add_argument("--scorer", default="01", type=str)
    parser.add_argument("--eval_rounds", default=2, type=int)
    parser.add_argument("--eval_prompts_per_round", default=2, type=int)
    # calculated by s-sr and sr
    parser.add_argument("--samples_per_eval", default=32, type=int)
    parser.add_argument(
        "--c",
        default=1.0,
        type=float,
        help="exploration param for UCB. higher = more exploration",
    )
    parser.add_argument("--knn_k", default=2, type=int)
    parser.add_argument("--knn_t", default=0.993, type=float)
    parser.add_argument("--reject_on_errors", action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    openai.api_type = "azure"
    openai.api_base = "https://gcrgpt4aoai5.openai.azure.com/"
    openai.api_version = "2023-05-15"
    openai.api_key = "653880d85b6e4a209206c263d7c3cc7a"

    # openai.api_type = "azure"
    # openai.api_base = " https://tdecoding-test.openai.azure.com/"
    # openai.api_version = "2023-05-15"
    # openai.api_key = "9e22b5f1f2094bbb9d3bc4e9d76398fd"
    config = vars(args)

    config["eval_budget"] = (
        config["samples_per_eval"]
        * config["eval_rounds"]
        * config["eval_prompts_per_round"]
    )

    task = get_task_class(args.task)(args.dev_data, args.test_data, args.max_threads)
    scorer = get_scorer(args.scorer)()
    evaluator = get_evaluator(args.evaluator)(config) # evaluate_fn
    bf_eval = get_evaluator("bf")(config)
    gpt4 = predictors.SubjAlpacaPredictor(config) if args.task == 'subj' else predictors.AlpacaPredictor

    optimizer = optimizers.ProTeGi(config, evaluator, scorer, args.max_threads, bf_eval)

    train_exs = task.get_train_examples()
    print(len(train_exs))
    test_exs = task.get_test_examples()



    print(config)

    logf = open(os.path.join(args.out, 'out.txt'), "w")
    logf.write(json.dumps(config) + "\n")

    # candidates = [open(fp.strip()).read() for fp in args.prompts.split(",")]
    from prompts.prompts import task as task_prompts
    candidates = [task_prompts[args.task][f'seed{args.seed}']]

    for round in tqdm(range(config["rounds"] + 1)):
        print("STARTING ROUND ", round)
        start = time.time()

        # expand candidates
        if round > 0:
            candidates = optimizer.expand_candidates(candidates, task, gpt4, train_exs)

        # score candidates
        scores = optimizer.score_candidates(candidates, task, gpt4, train_exs)
        [scores, candidates] = list(
            zip(*sorted(list(zip(scores, candidates)), reverse=True))
        )

        # select candidates
        candidates = candidates[: config["beam_size"]]
        scores = scores[: config["beam_size"]]

        # record candidates, estimated scores, and true scores
        with open(os.path.join(args.out, f'step{round}.txt'), 'w') as wf:
            for candidate, score in zip(candidates, scores):
                wf.write(f'{candidate}\t{score}\n')
            
        logf.write(f"======== ROUND {round}\n")
        # outf.write(f"{time.time() - start}\n")
        logf.write(f"{candidates}\n")
        logf.write(f"{scores}\n")
        logf.flush()
        if round == config["rounds"]:
            prompts = []
            test_scores = []
            with open(os.path.join(args.out, 'test.txt'), "w") as wf:
                for candidate in candidates[0:3]: # 这里只是在 eval test 集
                    accuracy, texts, labels, preds = task.evaluate(
                        gpt4, candidate, test_exs, n=args.n_test_exs
                    )
                    wf.write(f'{candidate}\t{accuracy}\n')
    
    logf.close()
    print("DONE!")
