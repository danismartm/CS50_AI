import os
import random
import re
import sys

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
    model = {}

    if damping_factor:
        if len(corpus[page]) == 0:
            probability = 1 / len(corpus)
            for i in corpus:
                model[i] = probability
        else:
            linked_pages = corpus[page]
            for i in corpus:
                if i in linked_pages:
                    probability = DAMPING / len(linked_pages) + (1 - DAMPING) / len(corpus)
                else:
                    probability = (1 - DAMPING) / len(corpus)

                model[i] = probability

    if abs(sum(model.values()) - 1) > 1e-6:
        raise ValueError("The sum of the probabilities is not equal to 1.")

    return model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    count = {page: 0 for page in corpus}

    page = random.choice(list(corpus.keys()))

    for _ in range(n):
        model = transition_model(corpus, page, damping_factor)
        next_page = random.choices(list(model.keys()), weights=model.values())[0]

        count[next_page] += 1
        page = next_page

    total_visits = sum(count.values())
    for page in count:
        count[page] /= total_visits

    return count


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    page_ranks = {page: 1 / N for page in corpus}

    threshold = 1e-6
    delta = float('inf')

    while delta > threshold:
        new_page_ranks = {}
        delta = 0

        for page in corpus:
            model = transition_model(corpus, page, damping_factor)
            new_rank = (1 - damping_factor) / N  # Uniform probability part

            # Add contributions from other pages linking to `page`
            for other_page in corpus:
                if page in corpus[other_page]:  # if `other_page` links to `page`
                    new_rank += damping_factor * (page_ranks[other_page] / len(corpus[other_page]))

            new_page_ranks[page] = new_rank
            delta += abs(page_ranks[page] - new_rank)

        page_ranks = new_page_ranks

    # Normalize the final PageRanks
    total_rank = sum(page_ranks.values())
    for page in page_ranks:
        page_ranks[page] /= total_rank

    return page_ranks


if __name__ == "__main__":
    main()

