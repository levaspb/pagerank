import os
import random
import re
import sys
 
from pomegranate import *

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    start = Node(DiscreteDistribution({
        "damping": damping_factor,
        "non-damping": 1 - damping_factor
    }), name="start")

    probs = list()
    line = list()

    for item in corpus:
        if item in corpus[page]:
            line = ["damping", item, 1 / len(corpus[page])]
        else:
            line = ["damping", item, 0]
        probs.append(line)

    for item in corpus:
        line = ["non-damping", item, 1 / len(corpus)]
        probs.append(line)

    probs_all = Node(ConditionalProbabilityTable(
        probs, [start.distribution]), name="probs_all"
        )

    model = BayesianNetwork()
    model.add_states(start, probs_all)
    model.add_edge(start, probs_all)
    model.bake()

    transitions = {item: 0 for item in corpus}
    for item in corpus:
        transitions[item] = (
            model.probability([["damping", item]]) +
            model.probability([["non-damping", item]])
            )
    
    return transitions


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page = random.choice(list(corpus))
    transitions = transition_model(corpus, page, damping_factor)
    
    keys = list(transitions.keys())
    probs = list(transitions.values())

    samples = {item: 0 for item in corpus}

    for _ in range(n):
        page = random.choices(keys, weights = probs)[0]
        samples[page] += 1
        transitions = transition_model(corpus, page, damping_factor)
        keys = list(transitions.keys())
        probs = list(transitions.values())
    
    pagerank = {item: 0 for item in corpus}
    for item, value in samples.items():
        pagerank[item] = value / n
    # pagerank = {item: lambda x: value / n for item, value in samples.items()}

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    pagerank = {item: 0 for item in corpus} 
    for item in corpus:
        pagerank[item] = 1 / N

    pagerank_prev = {item: 0 for item in corpus} 

    while True:
        for page in corpus:
            rank = 0
            for key, links in corpus.items():
                if len(links) == 0:
                    rank += pagerank[key] / N
                if page in links:
                    rank += pagerank[key] / len(links)
            pagerank[page] = (1 - damping_factor) + \
                (damping_factor * rank)
        
        diff = max([abs(pagerank_prev[key] - item) for key, item in pagerank.items()])
        if diff < 0.001: break
        pagerank_prev = {key: item for key, item in pagerank.items()}

    for key, item in pagerank.items():
        pagerank[key] = pagerank[key] / N

    return pagerank


if __name__ == "__main__":
    main()
