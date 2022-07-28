import argparse
import requests, jsonlines
from newspaper import Article
import datetime
from utils.tools import read_jsonl

def run_gcs(key,
           engine,
           in_file,
           out_file,
          ):
    questions = read_jsonl(in_file)
    outputs = []
    for question in questions:
        search_result = search(question["question_sentence"], key, engine)
        search_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y/%m/%d/%H:%M")
        for article in search_result:
            try:
                article["text"], article["authors"], article["publish_date"]  = parse_article(article["url"])
            except:
                print('\nURL died\n')
                print(article["url"])
                continue
                #article["text"], article["authors"], article["publish_date"]  = "", [], "" 
        output = {"question_id": question["question_id"], "search_time": search_time, "search_result": search_result}
        outputs.append(output)
    return outputs
        
def search(query, key, engine):
    url = f"https://www.googleapis.com/customsearch/v1?key={key}&cx={engine}&q={query}&start=1"
    data = requests.get(url).json()
    search_items = data.get("items", [])
    results = []
    for search_item in search_items:
        try:
            long_description = search_item["pagemap"]["metatags"][0]["og:description"]
        except KeyError:
            long_description = "N/A"
        # get the page title
        title = search_item.get("title")
        # page snippet
        snippet = search_item.get("snippet")
        # alternatively, you can get the HTML snippet (bolded keywords)
        html_snippet = search_item.get("htmlSnippet")
        # extract the page url
        link = search_item.get("link")
        #result = {"url": link, "title": title, "snippet": snippet, "html_snippet": html_snippet}
        result = {"url": link, "title": title}
        results.append(result)
    return results

def parse_article(url):
    article = Article(url)
    article.download()
    article.parse()
    text = article.text
    authors = article.authors
    publish_date = article.publish_date
    publish_date = publish_date.strftime("%Y/%m/%d")
    return text, authors, publish_date

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--key', type=str, metavar='N', 
                        default='AIzaSyBGhEhUpvDfPhvyGhecOXn4AwndCUTy9EQ', help='API Key')
    parser.add_argument('--engine', type=str, metavar='N', 
                        default='48657aaa55719c8c2', help='Search Engine ID')
    parser.add_argument('--in-file', type=str, metavar='N',
                        default='test/test.jsonl', help='input jsonl file')
    parser.add_argument('--out-file', type=str, metavar='N',
                        default='test/out.jsonl', help='output jsonl file')
    args = parser.parse_args()
    run_gcs(
          args.key,
          args.engine,
          args.in_file,
          args.out_file,
           )
