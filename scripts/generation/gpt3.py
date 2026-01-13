import openai
openai.api_key = ""
import string, datetime
import numpy as np
from utils.tools import check_jsonls, add_today
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import time

def run_gpt3(questions, retrieved_data=None, generate=False, model="davinci-002", rm_date_q=False, rm_date_r=False):
    answers = []
    scores = []
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # model = 'gpt-3.5-turbo-instruct'
    # model = 'babbage-002'
    # model = 'davinci-002'
    for q_idx in tqdm(range(len(questions))):
        question = questions[q_idx]
        if retrieved_data is not None:
            retrieved_text = get_retrieved_text(retrieved_data[q_idx], top_k=5, rm_date_r=rm_date_r)
        else:
            retrieved_text = None
        if generate:
            answer, score = gpt3_question_gen(question, retrieved_text, model=model, rm_date_q=rm_date_q, tokenizer=tokenizer)
        else:
            answer, score = gpt3_question(question, retrieved_text, model=model, rm_date_q=rm_date_q, tokenizer=tokenizer)
        answers.append(answer)
        scores.append(score)
    return answers, scores

def gpt3_question(question, retrieved_text=None, model="davinci-002", rm_date_q=False, tokenizer=None):
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
        prompt = retrieved_text + "\n" + prompt

    scores = []
    for alphabet, choice in zip(string.ascii_uppercase, choices):
        ans = "Answer: {}) {}".format(alphabet, choice)
        # ans_len = len(tokenizer(ans)['input_ids']) - 1
        query = prompt + "\n" + ans
        for _ in range(3):
            try:
                output = openai.Completion.create(
                                        model=model,
                                        prompt=query,
                                        max_tokens = 0,
                                        logprobs = 5,
                                        echo= True,
                                        temperature = 0.0,
                                        )
                break
            except openai.error.RateLimitError:
                time.sleep(1)
        # lprobs = np.array(output["choices"][0]["logprobs"]["token_logprobs"][1:])
        # assert output["choices"][0]["logprobs"]["tokens"][-ans_len-2] == "Answer"
        # assert output["choices"][0]["logprobs"]["tokens"][-ans_len-1] == ":"
        for i in range(len(output["choices"][0]["logprobs"]["tokens"])):
            if output["choices"][0]["logprobs"]["tokens"][-i] == "Answer":
                ans_len = i - 2
                break
        lprobs = np.array(output["choices"][0]["logprobs"]["token_logprobs"][-ans_len:])
        score = lprobs.mean()
        scores.append(score)
    scores = np.array(scores)
    answer = scores.argmax()
    probs = np.exp(scores)
    probs = probs/probs.sum()
    prob = probs[answer]
    return [str(answer)], str(prob)

def gpt3_question_gen(question, retrieved_text=None, model="davinci-002", rm_date_q=False, tokenizer=None):
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
    output = openai.Completion.create(
                            model=model,
                            prompt=query,
                            logprobs = 5,
                            #echo= True,
                            temperature = 0.0,
                            )
    answer = output["choices"][0]["text"].strip()
    scores = np.array(output["choices"][0]["logprobs"]["token_logprobs"])
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
