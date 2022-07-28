import jsonlines

def main(key, config, engine, in_file):
    assert in_file.split('_')[-1] == 'qa.jsonl'
    if config == "gcs":
        from retrieval.gcs import run_gcs
        out_file = in_file.replace('_qa.jsonl', '_gcs.jsonl')
        outputs = run_gcs(
                          key,
                          engine,
                          in_file,
                          out_file,
                          )
    elif config == "dpr":
        from retrieval.dpr import run_dpr
        out_file = in_file.replace('_qa.jsonl', '_dpr.jsonl')
        outputs = run_dpr(
                          in_file,
                          out_file,
                          )
    elif config == "gold":
        from retrieval.gold import run_gold
        out_file = in_file.replace('_qa.jsonl', '_gold.jsonl')
        outputs = run_gold(
                          in_file,
                          out_file,
                          )
    with jsonlines.open(out_file, mode='w') as fout:
        fout.write_all(outputs)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--key', type=str, metavar='N', 
                        default='', help='API Key')
    parser.add_argument('--engine', type=str, metavar='N', 
                        default='', help='Search Engine ID')
    parser.add_argument('--in-file', type=str, metavar='N',
                        default='dummy_data/20220201_qa.jsonl', help='input jsonl file')
    parser.add_argument('--config', type=str, metavar='N',
                        choices=["gcs", "dpr", "gold"],
                        default='gcs', help='gcs or DPR or gold (CNN)?')
    args = parser.parse_args()
    main(args.key, args.config, args.engine, args.in_file)
