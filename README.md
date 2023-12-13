# RealTime QA: What's the Answer Right Now?


<p align="center">
<a href="https://realtimeqa.github.io/">
<img src="https://github.com/realtimeqa/realtimeqa_public/blob/main/figs/realtimeqa_logo_text.png" height="100" alt="realtimeqa">
</a>
</p>

## Introduction
[RealTime QA](https://realtimeqa.github.io/) is a dynamic question answering (QA) platform that inquires about **the present**. We announce ~30 questions and evaluate real-time baseline systems (e.g., GPT-3 and T5) on a regular basis (weekly in this version).

## <img src="https://github.com/realtimeqa/realtimeqa_public/blob/main/figs/twitter.png" height="30" alt="twitter">  This Week's Questions
Follow [us on Twitter](https://twitter.com/realtimeqa). We tweet interesting examples from our GPT-3 baselines every week!

## Data and Retrieval Results
This repository provides our past data, as well as the latest, ongoing questions.
* [Latest Questions](https://github.com/realtimeqa/realtimeqa_public/tree/main/latest)
* [Past Questions](https://github.com/realtimeqa/realtimeqa_public/tree/main/past)
* [Backnumber Questions](https://github.com/realtimeqa/realtimeqa_public/tree/main/backnumber)

The backnumber questions preceded our real-time baselines that started from June 17, 2022. See [our paper](https://arxiv.org/abs/2207.13332) for more detail. Our baseline retrieval results (Google custom search and DPR) are also provided (e.g., [June 17, 2022 Google custom search](https://github.com/realtimeqa/realtimeqa_public/blob/main/past/2022/20220617_gcs.jsonl)). The past questions have all six baseline results under the multiple-choice and original settings (e.g., [GPT-3 + Google Custom Search Generation Results](https://github.com/realtimeqa/realtimeqa_public/blob/main/baseline_results/20220715_qa_open_gpt3_dpr_gen.jsonl)).

## Submit 
The submission window closes when the next set of questions is announced (3 am GMT on every Saturday). Submit via [this Google form](https://docs.google.com/forms/d/e/1FAIpQLScvCMJ86SCZZCcbq2SqbUBETX4n1KIAk-wCR_X37jNkjUdClw/viewform). The indexes for multiple-choice answer should start with `0`. See [examples](https://github.com/realtimeqa/realtimeqa_public/blob/main/baseline_results/20220715_qa_open_gpt3_gcs.jsonl).

## Our 6 Real-time Baselines
See [our script](https://github.com/realtimeqa/realtimeqa_public/tree/main/scripts) that we use for the six baseline models in the [paper](https://arxiv.org/abs/2207.13332). We run them every week. The results are updated every week on our [website](https://realtimeqa.github.io/).


## Citations
```
@inproceedings{
kasai2023realtime,
title={RealTime {QA}: What's the Answer Right Now?},
author={Jungo Kasai and Keisuke Sakaguchi and yoichi takahashi and Ronan Le Bras and Akari Asai and Xinyan Velocity Yu and Dragomir Radev and Noah A. Smith and Yejin Choi and Kentaro Inui},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2023},
url={https://openreview.net/forum?id=HfKOIPCvsv}
}
```
<p align="center">
<a href="https://www.cs.washington.edu/research/nlp">
<img src="https://github.com/jungokasai/THumB/blob/master/figs/uwnlp_logo.png" height="100" alt="UWNLP Logo">
</a>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://allenai.org/">
<img src="https://github.com/realtimeqa/realtimeqa_public/blob/main/figs/ai2_logo.png" height="100" alt="AI2 Logo" style="padding-right:160">
</a>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="">
<img src="https://github.com/realtimeqa/realtimeqa_public/blob/main/figs/tohoku_nlp.svg" height="100" alt="UWNLP Logo">
</a>
</p>
