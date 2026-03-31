import requests
import os
import json
import sys
from openai import OpenAI

URL = "https://asta-tools.allen.ai/mcp/v1"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"]
}


def parse_sse_json(resp):
    """
    Extract JSON from MCP text/event-stream response
    """
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line.replace("data:", "").strip())

    raise ValueError("No JSON found in MCP response")


def call_tool(name, arguments, req_id=1):

    payload = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments
        }
    }

    resp = requests.post(URL, headers=headers, json=payload)
    data = parse_sse_json(resp)

    # Asta returns JSON embedded as text
    content = data["result"]["content"]

    return content



def generate_report(seed, references, citations, author_profiles):
    context = json.dumps({
        "seed_paper": {
            "title": seed["title"],
            "abstract": seed["abstract"],
            "year": seed["year"],
            "authors": [a["name"] for a in seed["authors"]],
            "fields": seed["fieldsOfStudy"],
            "citations": seed["citationCount"],
        },
        "key_references": [
            {"title": r["title"],
             "year": r["year"],
             "abstract": (r["abstract"] or "")[:200],
             "citations": r["citationCount"]}
            for r in references
        ],
        "recent_citations": [
            {"title": c["title"], "year": c["year"],
             "abstract": (c["abstract"] or "")[:200],
             "citations": c["citationCount"]}
            for c in citations
        ],
        "author_profiles": [
            {"name": a["name"],
             "notable_work": a["top_paper"]["title"] if a["top_paper"] else None,
             "notable_citations": a["top_paper"]["citationCount"] if a["top_paper"] else None}
            for a in author_profiles
        ],
    }, indent=2)

    prompt = f"""Based on the given research data, generate a structured markdown report that contains the following:

1. **Summary** — A one-paragraph summary of the seed paper
2. **Foundational Works** — The top 5 references with title, year, and significance
3. **Recent Developments** — The top 5 most recent citing papers with title, year, and significance
4. **Author Profiles** — Each author's name and their most notable other work

Research data:
{context}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an academic research analyst. "
             "Write clear, precise markdown reports about scientific papers."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def run(seed_id):

    seed = call_tool(
        "get_paper",
        {
            "paper_id": seed_id,
            "fields": "title,abstract,year,authors,fieldsOfStudy,citationCount"
        }
    )

    s = json.loads(seed[0]["text"])

    refs = call_tool(
        "get_paper",
        {
            "paper_id": seed_id,
            "fields": "references,references.title,references.year,references.abstract,references.citationCount"
        }
    )

    paper = json.loads(refs[0]["text"])
    references = sorted([p for p in paper["references"] if p["citationCount"] is not None], key=lambda x: x["citationCount"], reverse=True)
    r = references[:5]

    cites = call_tool(
        "get_citations",
        {
            "paper_id": seed_id,
            "fields": "title,year,abstract,citationCount",
            "publication_date_range": "2023-01-01:",
            "limit": 5
        }
    )

    c = []
    for i, _ in enumerate(cites[:5]):
        p = json.loads(cites[i]["text"])
        c.append(p["citingPaper"])

    a = []
    for author in s["authors"]:
        papers = call_tool(
            "get_author_papers",
            {
                "author_id": author["authorId"],
                "paper_fields": "title,year,citationCount"
            }
        )

        works = sorted([json.loads(p["text"]) for p in papers], key=lambda x: x["citationCount"], reverse=True)
        if works[0]["title"] == s["title"]:
            if len(works) > 1:
                a.append({"name": author["name"], "top_paper": works[1]})
            else:
                a.append({"name": author["name"], "top_paper": None})
        else:
            a.append({"name": author["name"], "top_paper": works[0]})


    return generate_report(s, r, c, a)


if __name__ == "__main__":
    print("Seed paper: ARXIV:2210.03629")
    print("\n" + "=" * 60)
    print("AGENT REPORT")
    print("=" * 60 + "\n")
    print(run("ARXIV:2210.03629"))