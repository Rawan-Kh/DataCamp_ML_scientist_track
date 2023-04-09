# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Print the names of the pipeline components
print(nlp.pipe_names)

# Print the full pipeline of (name, component) tuples
print(nlp.pipeline)

# Whenever you're unsure about the current pipeline, you can inspect it by printing nlp.pipe_names or nlp.pipeline
# output:
#     ['tagger', 'parser', 'ner']
#     [('tagger', <spacy.pipeline.pipes.Tagger object at 0x7f19a7ed5b00>), ('parser', <spacy.pipeline.pipes.DependencyParser object at 0x7f198cf27ac8>), ('ner', <spacy.pipeline.pipes.EntityRecognizer object at 0x7f198cf27b28>)]
------------

# Custom components are great for adding custom values to documents, tokens and spans, and customizing the doc.ents.
---------
# Define the custom component
def length_component(doc):
    # Get the doc's length
    doc_length = len(doc)
    print("This document is {} tokens long.".format(doc_length))
    # Return the doc
    return doc
  
# Load the small English model and Add the component first in the pipeline
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(length_component, first=True)

# Process a text
doc = nlp("Hello there")
# Now let's take a look at a slightly more complex component!
-----------


