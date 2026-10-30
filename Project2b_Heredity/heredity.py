import csv
import itertools
import sys


PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")

    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all possible sets of people with the trait
    names = set(people)

    for have_trait in powerset(names):

        # Check whether this set violates known evidence
        fails_evidence = any(
            (
                people[person]["trait"] is not None
                and people[person]["trait"]
                != (person in have_trait)
            )
            for person in names
        )

        if fails_evidence:
            continue

        # Loop over all possible gene combinations
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                p = joint_probability(
                    people,
                    one_gene,
                    two_genes,
                    have_trait
                )

                update(
                    probabilities,
                    one_gene,
                    two_genes,
                    have_trait,
                    p
                )

    # Ensure every probability distribution sums to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")

        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")

            for value in probabilities[person][field]:
                probability = probabilities[person][field][value]
                print(f"    {value}: {probability:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a CSV file.
    """

    data = dict()

    with open(filename) as f:
        reader = csv.DictReader(f)

        for row in reader:
            name = row["name"]

            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (
                    True if row["trait"] == "1"
                    else False if row["trait"] == "0"
                    else None
                )
            }

    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """

    s = list(s)

    return [
        set(subset)
        for subset in itertools.chain.from_iterable(
            itertools.combinations(s, r)
            for r in range(len(s) + 1)
        )
    ]


def joint_probability(
    people,
    one_gene,
    two_genes,
    have_trait
):
    """
    Compute and return a joint probability.
    """

    probability = 1

    def gene_count(person):
        """
        Return the number of gene copies assigned to person.
        """

        if person in two_genes:
            return 2

        if person in one_gene:
            return 1

        return 0

    def passes_gene(person):
        """
        Return the probability that person passes the gene.
        """

        genes = gene_count(person)

        if genes == 2:
            return 1 - PROBS["mutation"]

        if genes == 1:
            return 0.5

        return PROBS["mutation"]

    for person, information in people.items():
        genes = gene_count(person)
        has_trait = person in have_trait

        mother = information["mother"]
        father = information["father"]

        # Use unconditional probability if parents are unknown
        if mother is None and father is None:
            gene_probability = PROBS["gene"][genes]

        else:
            mother_passes = passes_gene(mother)
            father_passes = passes_gene(father)

            # Child receives gene from both parents
            if genes == 2:
                gene_probability = (
                    mother_passes * father_passes
                )

            # Child receives gene from exactly one parent
            elif genes == 1:
                gene_probability = (
                    mother_passes * (1 - father_passes)
                    + (1 - mother_passes) * father_passes
                )

            # Child receives gene from neither parent
            else:
                gene_probability = (
                    (1 - mother_passes)
                    * (1 - father_passes)
                )

        trait_probability = PROBS["trait"][genes][has_trait]

        probability *= (
            gene_probability * trait_probability
        )

    return probability


def update(
    probabilities,
    one_gene,
    two_genes,
    have_trait,
    p
):
    """
    Add joint probability p to the appropriate distributions.
    """

    for person in probabilities:

        if person in two_genes:
            genes = 2

        elif person in one_gene:
            genes = 1

        else:
            genes = 0

        has_trait = person in have_trait

        probabilities[person]["gene"][genes] += p
        probabilities[person]["trait"][has_trait] += p


def normalize(probabilities):
    """
    Normalize each probability distribution so that it sums to 1.
    """

    for person in probabilities:

        for distribution in probabilities[person].values():
            total = sum(distribution.values())

            for value in distribution:
                distribution[value] /= total


if __name__ == "__main__":
    main()
