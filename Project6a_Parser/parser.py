import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S
NP -> N | Det N | Det AP N | NP PP
VP -> V | V NP | V PP | V NP PP | Adv VP | VP Adv | VP Conj VP
PP -> P NP
AP -> Adj | Adj AP
"""

grammar = nltk.CFG.fromstring(
    NONTERMINALS + TERMINALS
)

parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as file:
            sentence = file.read()

    # Otherwise, get sentence as input
    else:
        sentence = input("Sentence: ")

    # Convert input into list of words
    sentence = preprocess(sentence)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(sentence))

    except ValueError as error:
        print(error)
        return

    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")

        for noun_phrase in np_chunk(tree):
            print(" ".join(noun_phrase.flatten()))


def preprocess(sentence):
    """
    Convert sentence to a list of lowercase words.

    Remove tokens that do not contain at least one
    alphabetic character.
    """
    words = nltk.word_tokenize(
        sentence.lower()
    )

    return [
        word
        for word in words
        if any(
            character.isalpha()
            for character in word
        )
    ]


def np_chunk(tree):
    """
    Return all noun phrase chunks from the sentence tree.

    A noun phrase chunk is an NP subtree that does not
    contain another NP subtree.
    """
    chunks = []

    for subtree in tree.subtrees(
        lambda current_tree: current_tree.label() == "NP"
    ):
        contains_nested_np = any(
            nested is not subtree
            and nested.label() == "NP"
            for nested in subtree.subtrees()
        )

        if not contains_nested_np:
            chunks.append(subtree)

    return chunks


if __name__ == "__main__":
    main()
