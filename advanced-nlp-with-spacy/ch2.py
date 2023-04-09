# Look up the hash for the word "cat"
cat_hash = nlp.vocab.strings['cat']
print(cat_hash)

# Look up the cat_hash to get the string
cat_string = nlp.vocab.strings[cat_hash]
print(cat_string)
------------

# Look up the hash for the string label "PERSON"
person_hash = nlp.vocab.strings['PERSON']
print(person_hash)

# <script.py> output:
#     14800503047316267216
#     person

# Look up the person_hash to get the string
person_string = nlp.vocab.strings[person_hash]
print(person_string)

# <script.py> output:
#     380
#     PERSON
-------------
# Hashes can't be reversed. To prevent this problem, add the word to the new vocab by processing a text or looking up the string, or use the same vocab to resolve the hash back to a string.
-----------
# Import the Doc class
from spacy.tokens import Doc

# Desired text: "spaCy is cool!"
words = ['spaCy', 'is', 'cool', '!']
spaces = [True, True, False, False]

# Create a Doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)

----------
# Import the Doc class
from spacy.tokens import Doc

# Desired text: "Go, get started!"
words = ['Go', ',', 'get', 'started', '!']
spaces = [False, True, True, False, False]

# Create a Doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)
----------

# Import the Doc class
from spacy.tokens import Doc

# Desired text: "Oh, really?!"
words = ['Oh', ',', 'really', '?', '!']
spaces = [False, True, False, False, False]

# Create a Doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)
-----------

# Import the Doc and Span classes
from spacy.tokens import Doc, Span

# Create a doc from the words and spaces
doc = Doc(nlp.vocab, words=['I', 'like', 'David', 'Bowie'], spaces=[True, True, True, False])

# Create a span for "David Bowie" from the doc and assign it the label "PERSON"
span = Span(doc, 2, 4, label='PERSON')
print(span.text, span.label_)

--------------
# Import the Doc and Span classes
from spacy.tokens import Doc, Span

# Create a doc from the words and spaces
doc = Doc(nlp.vocab, words=['I', 'like', 'David', 'Bowie'], spaces=[True, True, True, False])

# Create a span for "David Bowie" from the doc and assign it the label "PERSON"
span = Span(doc, 2, 4, label='PERSON')

# Add the span to the doc's entities
doc.ents = [span]

# Print entities' text and labels
print([(ent.text, ent.label_) for ent in doc.ents])

# Creating spaCy's objects manually and modifying the entities will come in handy later when you're writing your own information extraction pipelines.
--------------
# Get all tokens and part-of-speech tags
pos_tags = [token.pos_ for token in doc]

for index, pos in enumerate(pos_tags):
    # Check if the current token is a proper noun
    if pos == 'PROPN':
        # Check if the next token is a verb
        if pos_tags[index + 1] == 'VERB':
            print('Found a verb after a proper noun!')
            
-------------
# It only uses lists of strings instead of native token attributes. This is often less efficient, and can't express complex relationships.
--------
# Rewrite the code to use the native token attributes instead of a list of pos_tags.
# Loop over each token in the doc and check the token.pos_ attribute.
# Use doc[token.i + 1] to check for the next token and its .pos_ attribute.
-----------

for token in doc:
    # Check if the current token is a proper noun
    if token.pos_ == 'PROPN':
        # Check if the next token is a verb
        if doc[token.i + 1].pos_ == 'VERB':
            print('Found a verb after a proper noun!')
-------------



