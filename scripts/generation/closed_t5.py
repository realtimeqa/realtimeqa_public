import utils.hf_env
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
import torch
from utils.tools import add_today

def run_closed_t5(questions, generate=False, as_of=False):
    model, tokenizer = load_model('google/t5-11b-ssm-nq')
    answers = []
    for question in questions:
        with torch.no_grad():
            if generate:
                answer = t5_question_gen(question, model, tokenizer, as_of)
            else:
                answer = t5_question(question, model, tokenizer, as_of)
            answers.append(answer)
    return answers

def t5_question(question, model, tokenizer, as_of=False):
    sentence = question["question_sentence"]
    # lowercase by default
    if as_of:
        sentence = add_today(sentence, question["question_date"])
    choices = question["choices"]
    losses = []
    # I wanted to do batching, but I don't think HF's padding is working. They include paddings for loss, which is a bug!!!
    # https://github.com/huggingface/transformers/blob/v4.19.4/src/transformers/models/t5/modeling_t5.py#L1672
    for choice in choices:
        inputs = tokenizer(sentence, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(choice, padding=True, return_tensors="pt")
        labels = targets["input_ids"].to(model.device)
        loss = float(model(input_ids=input_ids, labels=labels)["loss"])
        losses.append(loss)
    answer = np.array(losses).argmin()
    return [str(answer)]

def t5_question_gen(question, model, tokenizer, as_of=False):
    sentence = question["question_sentence"]
    # lowercase by default
    if as_of:
        sentence = add_today(sentence, question["question_date"])
    inputs = tokenizer(sentence, padding=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    answer = model.generate(input_ids=input_ids)[0]
    answer = tokenizer.decode(answer, skip_special_tokens=True).strip()
    return answer

def load_model(model_name):
    t5_qa_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    t5_tok = AutoTokenizer.from_pretrained(model_name)
    return t5_qa_model, t5_tok
