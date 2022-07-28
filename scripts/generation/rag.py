import utils.hf_env
import torch, datetime
from transformers import RagSequenceForGeneration, RagTokenizer
from utils.tools import add_today

def run_rag(questions, retrieved_data, generate=False, top_k=5, model='facebook/rag-sequence-nq', as_of=False):
    model, tokenizer = load_model(model)
    answers = []
    for q_idx in range(len(questions)):
        question = questions[q_idx]
        with torch.no_grad():
            if generate:
                answer = rag_question_gen(question, retrieved_data[q_idx], model, tokenizer, top_k, as_of)
            else:
                answer = rag_question(question, retrieved_data[q_idx], model, tokenizer, top_k, as_of)
            answers.append(answer)
    return answers

def rag_question(question, retrieved_docs, model, tokenizer, top_k=5, as_of=False):
    sentence = question["question_sentence"]
    # lowercase by default
    if as_of:
        sentence = add_today(sentence, question["question_date"])
    sentence = sentence.lower()
    choices = question["choices"]
    inputs = []
    targets = []
    for choice in choices:
        inputs.append(sentence)
        # I don't quite understand why you need this, but otherwise the model fails
        targets.append('</s> ' + choice.lower())
    inputs = tokenizer(inputs, padding=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(targets, padding=True, return_tensors="pt")
    # Bug in HF. The output should start with 2 (eos) rather than 0 (bos).
    # But very small prob differences anyways.
    for key, val in targets.items():
        targets[key] = val[:, 1:]
    labels = targets["input_ids"].to(model.device)
    # -1 to remove bos
    out_sizes = targets["attention_mask"].to(model.device).eq(1).sum(dim=-1) - 1


    retrieved_text, doc_scores = get_retrieval_text(retrieved_docs, sentence, top_k=top_k)
    context_input_ids, context_attention_mask, doc_scores = retrieved2ids(retrieved_text, tokenizer, doc_scores, len(choices))
    context_input_ids = clip_max(context_input_ids, max_len = 1024)
    context_attention_mask = clip_max(context_attention_mask, max_len = 1024)

    losses = model(
                    context_input_ids=context_input_ids.to(model.device),
                    context_attention_mask=context_attention_mask.to(model.device),
                    doc_scores=doc_scores.to(model.device),
                    decoder_input_ids=labels,
                    labels=labels,
                    )["loss"]
    losses = losses/out_sizes
    answer = int(losses.argmin())
    return [str(answer)]

def rag_question_gen(question, retrieved_docs, model, tokenizer, top_k=5, as_of=False):
    sentence = question["question_sentence"]
    # lowercase by default
    if as_of:
        sentence = add_today(sentence, question["question_date"])
    sentence = sentence.lower()
    inputs = tokenizer(sentence, padding=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    retrieved_text, doc_scores = get_retrieval_text(retrieved_docs, sentence, top_k=top_k)
    context_input_ids, context_attention_mask, doc_scores = retrieved2ids(retrieved_text, tokenizer, doc_scores, 1)
    context_input_ids = clip_max(context_input_ids, max_len = 1024)
    context_attention_mask = clip_max(context_attention_mask, max_len = 1024)

    generated = model.generate(
                    context_input_ids=context_input_ids.to(model.device),
                    context_attention_mask=context_attention_mask.to(model.device),
                    doc_scores=doc_scores.to(model.device),
                    )
    answer = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    return answer

def load_model(model_name):
    model = RagSequenceForGeneration.from_pretrained(model_name)
    tokenizer = RagTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer

def clip_max(tensor, max_len):
    tensor = tensor[:, :max_len]
    return tensor

def get_retrieval_text(retrieved_docs, sentence, top_k=5):
    search_result = retrieved_docs["search_result"]
    retrieved_text = []
    scores = []
    for article in search_result[:top_k]:
        if "text" not in article:
            continue
        content = article["text"]
        if "title" in article.keys():
            content = article["title"] + " " + content
        # 100 is just dummy. We equally weight all articles selected by GCS.
        score = article.get("doc_score", 100.0)
        score = float(score)
        # <s>+whitespace is needed to follow the RAG style
        # whitespace is critical because that would affect tokenization of the word word.
        # This is very tricky. They should have not done this and just do <S>first_word without a space.
        #first_paraph = '<s> ' + content.split("\n\n")[0]
        first_paraph = '<s> ' + " ".join(content.split("\n\n")[:2])
        first_paraph +=  " // {}".format(sentence)
        retrieved_text.append(first_paraph)
        scores.append(score)
    scores = torch.Tensor(scores)
    return retrieved_text, scores 

def retrieved2ids(retrieved_text, tokenizer, doc_scores, bsz):
    # duplicate for different output candidates
    retrieved_text = retrieved_text*bsz
    doc_scores = doc_scores.reshape(1, -1).repeat([bsz, 1])
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(retrieved_text, padding=True, return_tensors="pt")
    for key, val in targets.items():
        # tokenizer adds bos (0) by default. No need to do this.
        targets[key] = val[:, 1:]
    return targets["input_ids"], targets["attention_mask"], doc_scores
