import arxiv
import pandas as pd
from arxiv import Client
from transformers import pipeline
# Construct the default API client.
client = Client()

# Query ri fetch AI-related papers
query =  "ai or artificial intelligence or machine learning "
print(f"Query: {query}")
#search for papers
search = arxiv.Search(query=query, max_results=10, sort_by=arxiv.SortCriterion.SubmittedDate)

# Print the query
print(f"Search results for: {search}")

results = client.results(search)
# Fetch papers
papers = []
for result in results:
    papers.append({
        'title': result.title,
        'categories': result.categories,
        'published': result.published,
        'abstract': result.summary
    })

# Convert to DataFrame
df = pd.DataFrame(papers)

# pd.set_option('display.max_colwidth', None)
# print(df.head(10))

abstract = df['abstract'][0]
print("Abstract if full: ", abstract)


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# Summarize the abstract
summary = summarizer(abstract)
print("Summary: ", summary[0]['summary_text'])