# RealTime QA Scripts

All API keys are removed. You need to get your key for GPT-3 and Google custom search, if you want to use them.

## Retrieval
We run Google custom search (GCS) around the time the questions are available every week. To do so, run:
```bash
python retrieve_main.py --config gcs --in-file 
```
This will yield `dummy_data/2020201_gcs.jsonl`. Replace the `--in-file` argument with your questions.

## Answer Prediction (Reading Comprehension)
```bash
python baseline_main.py --in-file ../past/2022/20221111_qa.jsonl --config open_gpt3_gcs
```
The six `config` choices are: `[closed_gpt3, closed_t5, open_gpt3_gcs, open_gpt3_dpr, open_rag_gcs, open_rag_dpr]`. See [our paper](https://arxiv.org/abs/2207.13332) for more details. Use `--generate` for generation.

## Evaluation
```bash
python evaluate_main.py --pred-file ../baseline_results/20221111_qa_open_gpt3_gcs.jsonl --gold-file ../past/2022/20221111_qa.jsonl 
```
