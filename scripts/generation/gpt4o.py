import openai
openai.api_key = ""
import string, datetime
import numpy as np
from utils.tools import check_jsonls, add_today
from transformers import GPT2TokenizerFast

def run_gpt4o(questions, retrieved_data=None, generate=False, model="gpt-4o", rm_date_q=False, rm_date_r=False):
    answers = []
    scores = []
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    for q_idx, question in enumerate(questions):
        if retrieved_data is not None:
            retrieved_text = get_retrieved_text(retrieved_data[q_idx], top_k=5, rm_date_r=rm_date_r)
        else:
            retrieved_text = None
        if generate:
            answer, score = gpt4_question_gen(question, retrieved_text, model=model, rm_date_q=rm_date_q, tokenizer=tokenizer)
            scores.append(score)
        else:
            answer = gpt4_question(question, retrieved_text, model=model, rm_date_q=rm_date_q, tokenizer=tokenizer)
        answers.append(answer)
    if generate:
        return answers, scores
    else:
        return answers, None

def gpt4_question(question, retrieved_text=None, model="gpt-4o", rm_date_q=False, tokenizer=None):
    sentence = question["question_sentence"]
    if not rm_date_q:
        sentence = add_today(sentence, question["question_date"])
    prompt = "Question: " + sentence
    choices = question["choices"]
    prompt += "\n"
    for alphabet, choice in zip(string.ascii_uppercase, choices):
        prompt += "{}) {}\n".format(alphabet, choice)
    if retrieved_text is not None:
        # insert retrieved text
        prompt = retrieved_text + "\n" + prompt + "\n"

    output = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Answer: "},
        ],
        max_tokens=1,
        logprobs=True,
        top_logprobs=5,
        temperature=0.0
    )

    for c in output["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        if c["token"] in list(string.ascii_uppercase):
            answer = c["token"]
            break
    alphabet2num = {"A": ["0"], "B": ["1"], "C": ["2"], "D": ["3"]}

    return alphabet2num[answer]

def gpt4_question_gen(question, retrieved_text=None, model="gpt-4o", rm_date_q=False, tokenizer=None):
    sentence = question["question_sentence"]
    demo = "What is the capital city of Japan?"
    #demo = "Who is the President of the U.S.?"
    if not rm_date_q:
        demo = add_today(demo, question["question_date"])
        sentence = add_today(sentence, question["question_date"])
    prompt = "Question: " + demo
    prompt += "\nAnswer: Tokyo\n"
    #prompt += "\nAnswer: Joe Biden\n"
    prompt += "Question: " + sentence
    if retrieved_text is not None:
        # insert retrieved text
        prompt = retrieved_text + "\n" + prompt
    query = prompt + "\nAnswer:"
    output = openai.ChatCompletion.create(
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "\nAnsewer:"},
        ],
        model=model,
        logprobs=True,
        top_logprobs=5,
        temperature=0.0
    )
    answer = output["choices"][0]["message"]["content"].strip()
    scores = np.array([content["logprob"] for content in output["choices"][0]["logprobs"]["content"]])
    score = np.exp(scores.mean())
    return answer, str(score)

def get_retrieved_text(retrieved_datum, top_k=5, rm_date_r=False):
    search_result = retrieved_datum["search_result"]
    retrieved_text = ""
    for article in search_result[:top_k]:
        if "publish_date" not in article:
            continue
        date = article["publish_date"]
        content = article["text"]
        if content == '':
            continue
        date = datetime.datetime.strptime(date, '%Y/%m/%d')
        date = date.strftime("%B %d, %Y")
        #first_paraph = content.split("\n\n")[0]
        first_paraph = " ".join(content.split("\n\n")[:2])
        if "title" in article.keys():
            first_paraph = article["title"] + " " + first_paraph
        if not rm_date_r:
            retrieved_text += "Article on {}: {}\n".format(date, first_paraph)
        else:
            retrieved_text += "Article: {}\n".format(first_paraph)
    return retrieved_text
