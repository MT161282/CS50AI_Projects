Analysis

## Layer 7, Head 10

This attention head appears to connect direct objects with their
verbs. In both examples, the object token pays strongest attention
to the verb that acts upon it: "meal" attends to "cooked," and
"book" attends to "opened." This suggests that the head helps BERT
identify which action applies to an object.

Example Sentences:
- The chef cooked the meal in the [MASK].
- The student opened the book before [MASK].

## Layer 2, Head 5

This attention head appears to connect nouns with the possessive
pronouns that describe them. The noun "car" pays strongest attention
to "his," while "phone" pays strongest attention to "her." This
pattern suggests that the head tracks possession or noun
modification within a noun phrase.

Example Sentences:
- He parked his car beside the [MASK].
- She placed her phone on the [MASK].
