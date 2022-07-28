from bs4 import BeautifulSoup
import argparse, re
import requests, jsonlines
import datetime
from utils.tools import read_jsonl
from retrieval.gcs import parse_article

def run_gold(in_file, out_file):
    questions = read_jsonl(in_file)
    outputs = []
    for question in questions:
        #urls = re.findall(r'<a href=\"(.*)\">', question['evidence'])
        soup = BeautifulSoup(question['evidence'])
        search_result = []
        for link in soup.findAll('a'):
            search_result.append({'url': link.get('href')})
        for article in search_result:
            try:
                article["text"], article["authors"], article["publish_date"]  = parse_article(article["url"])
            except:
                print('\nURL died')
                print(article["url"])
                continue
        output = {"question_id": question["question_id"], "search_result": search_result}
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
