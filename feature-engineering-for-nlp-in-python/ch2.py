# Identify the list of words from the choices which do not have the same lemma.
# He, She, I, They
#  All the words listed here are pronouns and will be lemmatized to '-PRON-'.
# Car, Bike, Truck, Bus
# Although all these words refer to vehicles, they are words with distinct base forms.

-----
import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp("gettysburg")

# Generate the tokens
tokens = [token.text for token in doc]
print(tokens)
# tokens is not correctly defined. In the given code, doc is created with a single word 'gettysburg', which is then used to generate tokens. However, since 'gettysburg' is a single token, the output will be a list containing a single string 'gettysburg', which is not useful for tokenization.
# To correctly tokenize a text, the input doc should contain a string of text with multiple words.
import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(gettysburg)

# Generate the tokens
tokens = [token.text for token in doc]
print(tokens)

# ['Four', 'score', 'and', 'seven', 'years', 'ago', 'our', 'fathers', 'brought', 'forth', 'on', 'this', 'continent', ',', 'a', 'new', 'nation', ',', 'conceived', 'in', 'Liberty', ',', 'and', 'dedicated', 'to', 'the', 'proposition', 'that', 'all', 'men', 'are', 'created', 'equal', '.', 'Now', 'we', "'re", 'engaged', 'in', 'a', 'great', 'civil', 'war', ',', 'testing', 'whether', 'that', 'nation', ',', 'or', 'any', 'nation', 'so', 'conceived', 'and', 'so', 'dedicated', ',', 'can', 'long', 'endure', '.', 'We', "'re", 'met', 'on', 'a', 'great', 'battlefield', 'of', 'that', 'war', '.', 'We', "'ve", 'come', 'to', 'dedicate', 'a', 'portion', 'of', 'that', 'field', ',', 'as', 'a', 'final', 'resting', 'place', 'for', 'those', 'who', 'here', 'gave', 'their', 'lives', 'that', 'that', 'nation', 'might', 'live', '.', 'It', "'s", 'altogether', 'fitting', 'and', 'proper', 'that', 'we', 'should', 'do', 'this', '.', 'But', ',', 'in', 'a', 'larger', 'sense', ',', 'we', 'ca', "n't", 'dedicate', '-', 'we', 'can', 'not', 'consecrate', '-', 'we', 'can', 'not', 'hallow', '-', 'this', 'ground', '.', 'The', 'brave', 'men', ',', 'living', 'and', 'dead', ',', 'who', 'struggled', 'here', ',', 'have', 'consecrated', 'it', ',', 'far', 'above', 'our', 'poor', 'power', 'to', 'add', 'or', 'detract', '.', 'The', 'world', 'will', 'little', 'note', ',', 'nor', 'long', 'remember', 'what', 'we', 'say', 'here', ',', 'but', 'it', 'can', 'never', 'forget', 'what', 'they', 'did', 'here', '.', 'It', 'is', 'for', 'us', 'the', 'living', ',', 'rather', ',', 'to', 'be', 'dedicated', 'here', 'to', 'the', 'unfinished', 'work', 'which', 'they', 'who', 'fought', 'here', 'have', 'thus', 'far', 'so', 'nobly', 'advanced', '.', 'It', "'s", 'rather', 'for', 'us', 'to', 'be', 'here', 'dedicated', 'to', 'the', 'great', 'task', 'remaining', 'before', 'us', '-', 'that', 'from', 'these', 'honored', 'dead', 'we', 'take', 'increased', 'devotion', 'to', 'that', 'cause', 'for', 'which', 'they', 'gave', 'the', 'last', 'full', 'measure', 'of', 'devotion', '-', 'that', 'we', 'here', 'highly', 'resolve', 'that', 'these', 'dead', 'shall', 'not', 'have', 'died', 'in', 'vain', '-', 'that', 'this', 'nation', ',', 'under', 'God', ',', 'shall', 'have', 'a', 'new', 'birth', 'of', 'freedom', '-', 'and', 'that', 'government', 'of', 'the', 'people', ',', 'by', 'the', 'people', ',', 'for', 'the', 'people', ',', 'shall', 'not', 'perish', 'from', 'the', 'earth', '.']

# You now know how to tokenize a piece of text. In the next exercise, we will perform similar steps and conduct lemmatization.
-----------
# Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we're engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We're met on a great battlefield of that war. We've come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It's altogether fitting and proper that we should do this. But, in a larger sense, we can't dedicate - we can not consecrate - we can not hallow - this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It's rather for us to be here dedicated to the great task remaining before us - that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion - that we here highly resolve that these dead shall not have died in vain - that this nation, under God, shall have a new birth of freedom - and that government of the people, by the people, for the people, shall not perish from the earth.

# Loop over doc and extract the lemma for each token of gettysburg.
import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(gettysburg)

# Generate lemmas
lemmas = [token.lemma_ for token in doc]

--------

import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(gettysburg)

# Generate lemmas
lemmas = [token.lemma_ for token in doc]

# Convert lemmas into a string
print(' '.join(lemmas))

# four score and seven year ago our father bring forth on this continent , a new nation , conceive in Liberty , and dedicate to the proposition that all man be create equal . now we be engage in a great civil war , test whether that nation , or any nation so conceive and so dedicated , can long endure . we be meet on a great battlefield of that war . we 've come to dedicate a portion of that field , as a final resting place for those who here give their life that that nation might live . it be altogether fitting and proper that we should do this . but , in a large sense , we ca n't dedicate - we can not consecrate - we can not hallow - this ground . the brave man , live and dead , who struggle here , have consecrate it , far above our poor power to add or detract . the world will little note , nor long remember what we say here , but it can never forget what they do here . it be for we the living , rather , to be dedicate here to the unfinished work which they who fight here have thus far so nobly advanced . it be rather for we to be here dedicate to the great task remain before we - that from these honor dead we take increased devotion to that cause for which they give the last full measure of devotion - that we here highly resolve that these dead shall not have die in vain - that this nation , under God , shall have a new birth of freedom - and that government of the people , by the people , for the people , shall not perish from the earth .

---------

# Load model and create Doc object
nlp = spacy.load('en_core_web_sm')
doc = nlp(blog)

# Generate lemmatized tokens
lemmas = [token.lemma_ for token in doc]

# Remove stopwords and non-alphabetic tokens
a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords]

# Print string after text cleaning
print(' '.join(a_lemmas))
# Take a look at the cleaned text; it is lowercased and devoid of numbers, punctuations and commonly used stopwords. Also, note that the word U.S. was present in the original text. Since it had periods in between, our text cleaning process completely removed it. This may not be ideal behavior. It is always advisable to use your custom functions in place of isalpha() for more nuanced cases.
-------

# Function to preprocess text
def preprocess(text):
  	# Create Doc object
    doc = nlp(text, disable=['ner', 'parser'])
    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]
    # Remove stopwords and non-alphabetic characters
    a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords]
    
    return ' '.join(a_lemmas)
  
# Apply preprocess to ted['transcript']
ted['transcript'] = ted['transcript'].apply(preprocess)
print(ted['transcript'])
# You have preprocessed all the TED talk transcripts contained in ted and it is now in a good shape to perform operations such as vectorization (as we will soon see how). You now have a good understanding of how text preprocessing works and why it is important. In the next lessons, we will move on to generating word level features for our texts.
----------

