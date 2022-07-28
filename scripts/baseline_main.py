import os
from utils.tools import read_jsonl, answer2jsonl, check_jsonls, fall_back

def main(in_file, config="closed_gpt3", out_dir="../baseline_results/", gcs_file=None, generate=False, rm_date_q=False, rm_date_r=False):
    questions = read_jsonl(in_file)
    scores = None
    if config == "closed_gpt3":
        from generation.gpt3 import run_gpt3
        answers, scores = run_gpt3(questions, generate=generate, rm_date_q=rm_date_q)
    elif config == "closed_t5":
        from generation.closed_t5 import run_closed_t5
        answers = run_closed_t5(questions, generate=generate)

    elif config == "open_gpt3_dpr":
        from generation.gpt3 import run_gpt3
        dpr_file = in_file.replace("_qa.jsonl", "_dpr.jsonl").replace("_qa_nota.jsonl", "_dpr.jsonl")
        retrieved_data = read_jsonl(dpr_file)
        answers, scores = run_gpt3(questions, retrieved_data=retrieved_data, generate=generate, rm_date_q=rm_date_q, rm_date_r=rm_date_r)
    elif config == "open_rag_dpr":
        from generation.rag import run_rag
        dpr_file = in_file.replace("_qa.jsonl", "_dpr.jsonl").replace("_qa_nota.jsonl", "_dpr.jsonl")
        retrieved_data = read_jsonl(dpr_file)
        answers = run_rag(questions, retrieved_data=retrieved_data, generate=generate)

    elif config == "open_gpt3_gcs":
        from generation.gpt3 import run_gpt3
        if gcs_file is None:
            gcs_file = in_file.replace("_qa.jsonl", "_gcs.jsonl").replace("_qa_nota.jsonl", "_gcs.jsonl")
        dpr_file = in_file.replace("_qa.jsonl", "_dpr.jsonl").replace("_qa_nota.jsonl", "_dpr.jsonl")
        gcs = read_jsonl(gcs_file)
        dpr = read_jsonl(dpr_file)
        check_jsonls(gcs, dpr)
        retrieved_data = fall_back(gcs, dpr)
        answers, scores = run_gpt3(questions, retrieved_data=retrieved_data, generate=generate, rm_date_q=rm_date_q, rm_date_r=rm_date_r)

    elif config == "open_rag_gcs":
        from generation.rag import run_rag
        if gcs_file is None:
            gcs_file = in_file.replace("_qa.jsonl", "_gcs.jsonl").replace("_qa_nota.jsonl", "_gcs.jsonl")
        dpr_file = in_file.replace("_qa.jsonl", "_dpr.jsonl").replace("_qa_nota.jsonl", "_dpr.jsonl")
        gcs = read_jsonl(gcs_file)
        dpr = read_jsonl(dpr_file)
        check_jsonls(gcs, dpr)
        retrieved_data = fall_back(gcs, dpr)
        answers = run_rag(questions, retrieved_data=retrieved_data, generate=generate)
    else:
        assert False
    out_file = os.path.basename(in_file).replace('.jsonl', '')
    if generate:
        config += "_gen"
    out_file = out_file + "_" + config + ".jsonl"
    out_file = os.path.join(out_dir, out_file)
    answer2jsonl(answers, questions, out_file, scores)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--in-file', type=str, metavar='N',
                        default='dummy_data/20220201_qa.jsonl', help='input jsonl file')
    parser.add_argument('--config', type=str, metavar='N',
                        choices=["closed_gpt3", "closed_t5", "open_gpt3_dpr", "open_gpt3_gcs", "open_rag_dpr", "open_rag_gcs"],
                        default='closed_gpt3', help='baseline configuration')
    parser.add_argument('--out-dir', type=str, metavar='N',
                        default='../baseline_results/', help='baseline results output')
    parser.add_argument('--generate', default=False, action='store_true',
                        help='Generate instead of multiple choice?')
    parser.add_argument('--gcs-file', type=str, metavar='n',
                        default=None, help='gcs file')
    parser.add_argument('--rm-date-q', default=False, action='store_true',
                        help='Remove date to questions?')
    parser.add_argument('--rm-date-r', default=False, action='store_true',
                        help='Remove date to retrieved documents?')
    args = parser.parse_args()
    main(args.in_file, args.config, args.out_dir, args.gcs_file, args.generate, args.rm_date_q, args.rm_date_r)
