import argparse, os
import itertools
from utils.tools import read_jsonl, answer2jsonl, check_jsonls, metric_max_over_ground_truths,  f1_score, exact_match_score

def main(pred_file, gold_file, generate=False, cnn_only=False):
    preds = read_jsonl(pred_file)
    golds = read_jsonl(gold_file)
    check_jsonls(preds, golds)
    if generate:
        assert pred_file.split('_')[-1] == 'gen.jsonl'
        results = gen_eval(preds, golds, cnn_only)
    else:
        results = accuracy(preds, golds, cnn_only)
    print(results)

def accuracy(preds, golds, cnn_only):
    count = 0
    correct = 0
    score_exists = False
    if "score" in preds[0].keys():
        score_exists = True
        wrong_score = 0
        correct_score = 0
    for pred, gold in zip(preds, golds):
        if cnn_only and gold["question_source"] != "CNN":
            continue
        prediction = pred["prediction"]
        gold = gold["answer"]
        if prediction == gold:
            correct += 1
            if score_exists:
                correct_score += float(pred["score"])
        else:
            if score_exists:
                wrong_score += float(pred["score"])
        count += 1
    if score_exists:
        return {'accuracy': correct/count, 'correct_score': correct_score/correct, 'wrong_score': wrong_score/(count-correct), 'score': (correct_score + wrong_score)/count}
        #return {'accuracy': correct/count, 'score': (correct_score + wrong_score)/count}
        #return {'accuracy': correct/len(wrong_idxes), 'wrong_score': wrong_score/len(wrong_idxes)}
    return {'accuracy': correct/count}

def gen_eval(preds, golds, cnn_only):
    em_total = 0
    f1_total = 0
    count = 0
    #score = 0
    ##prediction = pred["prediction"].lower()
    #score_exists = False
    #if "score" in preds[0].keys():
    #    score_exists = True
    for pred, gold in zip(preds, golds):
        # Concatenate gold answers (can be multiple like NYT)
        sent = gold["question_sentence"].lower().strip()
        if "except" in sent[-10:]:
            continue
        if cnn_only and gold["question_source"] != "CNN":
            continue
        count += 1
        golds = [gold["choices"][int(idx)] for idx in gold["answer"]]
        golds = [' '.join(perm) for perm in list(itertools.permutations(golds))]
        prediction = pred["prediction"]
        em_total += metric_max_over_ground_truths(exact_match_score, prediction, golds)
        f1_total += metric_max_over_ground_truths(f1_score, prediction, golds)
        #if score_exists:
        #    score += float(pred["score"])
    return {'em': em_total/count, 'f1': f1_total/count}
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--pred-file', type=str, metavar='N',
                        default='../baseline_results/20220201_qa_open_gpt3_gcs.jsonl', help='prediction file')
    parser.add_argument('--gold-file', type=str, metavar='N',
                        default='dummy_data/20220201_qa.jsonl', help='gold file')
    parser.add_argument('--generate', default=False, action='store_true',
                        help='Generate instead of multiple choice?')
    parser.add_argument('--cnn-only', default=False, action='store_true',
                        help='CNN only evaluations')
    args = parser.parse_args()
    main(args.pred_file, args.gold_file, args.generate, args.cnn_only)
