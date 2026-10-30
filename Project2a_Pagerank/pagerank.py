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
    print("PageRank Results from Iteration")

    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.

    Return a dictionary where each key is a page, and values are
    a set of all other pages in the corpus linked to by the page.
    """

    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue

        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(
                r"<a\s+(?:[^>]?)href=\"([^\"])\"",
                contents
            )
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = {
            link for link in pages[filename]
            if link in pages
        }

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given the current page.
    """

    page_count = len(corpus)
    links = corpus[page]

    # A page with no links is treated as linking to every page
    if not links:
        return {
            candidate: 1 / page_count
            for candidate in corpus
        }

    # Probability of choosing any page randomly
    distribution = {
        candidate: (1 - damping_factor) / page_count
        for candidate in corpus
    }

    # Additional probability for pages linked from current page
    link_probability = damping_factor / len(links)

    for link in links:
        distribution[link] += link_probability

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling n pages.
    """

    pages = list(corpus)
    counts = {page: 0 for page in pages}

    # First sample is chosen randomly
    current_page = random.choice(pages)

    for sample in range(n):
        counts[current_page] += 1

        if sample < n - 1:
            distribution = transition_model(
                corpus,
                current_page,
                damping_factor
            )

            current_page = random.choices(
                population=list(distribution),
                weights=list(distribution.values()),
                k=1
            )[0]

    return {
        page: counts[page] / n
        for page in pages
    }


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values by repeatedly applying the
    PageRank formula until the values converge.
    """

    page_count = len(corpus)

    # Initially, every page has equal rank
    ranks = {
        page: 1 / page_count
        for page in corpus
    }

    while True:
        new_ranks = {}

        for page in corpus:
            incoming_rank = 0

            for possible_source, links in corpus.items():

                # No-link page is treated as linking to every page
                if not links:
                    incoming_rank += (
                        ranks[possible_source] / page_count
                    )

                elif page in links:
                    incoming_rank += (
                        ranks[possible_source] / len(links)
                    )

            new_ranks[page] = (
                (1 - damping_factor) / page_count
                + damping_factor * incoming_rank
            )

        # Check whether all changes are at most 0.001
        converged = all(
            abs(new_ranks[page] - ranks[page]) <= 0.001
            for page in corpus
        )

        ranks = new_ranks

        if converged:
            break

    # Ensure that all ranks sum to exactly 1
    total = sum(ranks.values())

    return {
        page: rank / total
        for page, rank in ranks.items()
    }


if __name__ == "__main__":
    main()

