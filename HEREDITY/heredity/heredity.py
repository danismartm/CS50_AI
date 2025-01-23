import csv
import itertools
import sys

PROBS = {
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },
    "trait": {
        2: {True: 0.65, False: 0.35},
        1: {True: 0.56, False: 0.44},
        0: {True: 0.01, False: 0.99}
    },
    "mutation": 0.01
}


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    probabilities = {
        person: {
            "gene": {2: 0, 1: 0, 0: 0},
            "trait": {True: 0, False: 0}
        }
        for person in people
    }

    names = set(people)
    for have_trait in powerset(names):
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    normalize(probabilities)

    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    data = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    probability = 1

    for person in people:
        if person in two_genes:
            gen = 2
        elif person in one_gene:
            gen = 1
        else:
            gen = 0

        if people[person]["mother"] is None and people[person]["father"] is None:
            gen_probability = PROBS["gene"][gen]
        else:
            father = people[person]["father"]
            mother = people[person]["mother"]

            if mother in two_genes:
                mother_probability = 1 - PROBS["mutation"]
            elif mother in one_gene:
                mother_probability = 0.5
            else:
                mother_probability = PROBS["mutation"]

            if father in two_genes:
                father_probability = 1 - PROBS["mutation"]
            elif father in one_gene:
                father_probability = 0.5
            else:
                father_probability = PROBS["mutation"]

            if gen == 2:
                gen_probability = mother_probability * father_probability
            elif gen == 1:
                gen_probability = (
                    mother_probability * (1 - father_probability) +
                    (1 - mother_probability) * father_probability
                )
            else:
                gen_probability = (1 - mother_probability) * (1 - father_probability)

        trait_probability = PROBS["trait"][gen][person in have_trait]
        probability *= gen_probability * trait_probability

    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    for person in probabilities:
        if person in two_genes:
            gen = 2
        elif person in one_gene:
            gen = 1
        else:
            gen = 0
        has_trait = person in have_trait

        probabilities[person]["gene"][gen] += p
        probabilities[person]["trait"][has_trait] += p


def normalize(probabilities):
    for person in probabilities:
        gene_total = sum(probabilities[person]["gene"].values())
        for gene in probabilities[person]["gene"]:
            probabilities[person]["gene"][gene] /= gene_total

        trait_total = sum(probabilities[person]["trait"].values())
        for trait in probabilities[person]["trait"]:
            probabilities[person]["trait"][trait] /= trait_total


if __name__ == "__main__":
    main()
