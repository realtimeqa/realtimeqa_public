import utils.hf_env
import torch, datetime
from transformers import RagRetriever, RagSequenceForGeneration, RagTokenizer
from utils.tools import read_jsonl, add_today

def run_dpr(in_file, out_file, model='facebook/rag-sequence-nq', as_of=False):
    questions = read_jsonl(in_file)
    retriever, model, tokenizer = load_model(model)
    outputs = []
    for question in questions:
        with torch.no_grad():
            search_result = run_dpr_question(question, retriever, model, tokenizer, as_of)
        search_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y/%m/%d/%H:%M")
        output = {"question_id": question["question_id"], "search_time": search_time, "search_result": search_result}
        outputs.append(output)
    return outputs

def run_dpr_question(question, retriever, model, tokenizer, as_of=False):
    sentence = question["question_sentence"]
    # lowercase by default
    if as_of:
        sentence = add_today(sentence, question["question_date"])
    sentence = sentence.lower()
    inputs = tokenizer(sentence, padding=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    question_hidden_states = model.question_encoder(input_ids)[0]
    docs_dict = retriever(input_ids.cpu().numpy(), question_hidden_states.detach().cpu().numpy(), return_tensors="pt")
    question_hidden_states = model.question_encoder(input_ids)[0]
    doc_scores = torch.bmm(
        question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2).to(question_hidden_states.device)
    ).squeeze(1)
    doc_ids = [str(int(doc_id)) for doc_id in docs_dict["doc_ids"][0]]
    docs = tokenizer.batch_decode(docs_dict['context_input_ids'], skip_special_tokens=True)
    docs = [doc.strip().split(' // ')[0].strip() for doc in docs]
    doc_scores = [str(float(doc_score)) for doc_score in doc_scores[0]]
    search_result = []
    for doc_idx in range(len(docs)):
        search_result.append({"doc_id": doc_ids[doc_idx], "text": docs[doc_idx], "doc_score": doc_scores[doc_idx], "publish_date": "2018/12/31"})
    return search_result

def load_model(model_name):
    retriever = RagRetriever.from_pretrained(model_name, dataset='wiki_dpr', index_name='compressed')
    model = RagSequenceForGeneration.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.cuda()
    tokenizer = RagTokenizer.from_pretrained(model_name)
    return retriever, model, tokenizer
