import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pyarrow

nltk.download('punkt')  # Tokenizer
nltk.download('wordnet') 

last_words_pd = pd.read_csv(
  'C:/Users/Drew Duffy/Unstructured/uda_notes/last_statements.csv'
)

headings = last_words_pd.columns

last_words_pd['statements'] = last_words_pd['statements'].astype(str)  # Ensure the column is of string type

statements_sample = last_words_pd.sample(1)

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

statements_sample['statement_lemma'] = statements_sample['statements'].apply(lemmatize_text)

print(statements_sample[['statements', 'statement_lemma']])