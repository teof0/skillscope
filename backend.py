# ================================
# 1. MOUNT DRIVE & INSTALL DEPENDENCIES
# ================================
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

# ================================
# 2. LOAD RAW JOB DESCRIPTION DATA
# ================================
df = pd.read_csv('/content/adzuna_jobs.csv')
print(df.shape)
df.head()

# ================================
# 3. TEXT CLEANING FUNCTION
# ================================
import re
import html

def clean(text,
          lowercase=True,
          remove_numbers=False,
          remove_non_ascii=False):

    text = html.unescape(text)
    text = re.sub(r'<[^<>]*>', ' ', text)
    text = re.sub(r'\[([^\[\]])\]\([^\(\)]\)', r'\1', text)
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    text = re.sub(r'[.,;:"\'()!?]', ' ', text)

    if remove_numbers:
        text = re.sub(r'\b\d+\b', ' ', text)

    if remove_non_ascii:
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    if lowercase:
        text = text.lower()

    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Apply cleaning

# ================================
# 4. APPLY TEXT CLEANING TO DATAFRAME
# ================================
df['clean_description'] = df['description'].apply(clean)
df.head()

# ================================
# 5. ADVANCED TEXT FEATURE EXTRACTION FUNCTIONS
# ================================
import re ###
import spacy ###
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
import textacy ###

def custom_tokenizer(nlp):
    prefixes = [pattern for pattern in nlp.Defaults.prefixes if pattern not in ['-', '_', '#']]
    suffixes = [pattern for pattern in nlp.Defaults.suffixes if pattern not in ['_']]
    infixes  = [pattern for pattern in nlp.Defaults.infixes if not re.search(pattern, 'xx-xx')]

    return Tokenizer(vocab=nlp.vocab,
                     rules=nlp.Defaults.tokenizer_exceptions,
                     prefix_search=compile_prefix_regex(prefixes).search,
                     suffix_search=compile_suffix_regex(suffixes).search,
                     infix_finditer=compile_infix_regex(infixes).finditer,
                     token_match=nlp.Defaults.token_match)

def extract_noun_phrases(doc, sep=' '):
    patterns = ["POS:ADJ POS:NOUN:+", "POS:NOUN POS:NOUN:+"]
    spans = textacy.extract.matches.token_matches(doc, patterns=patterns)
    return [str(s) for s in spans]

def extract_lemmas(doc, **kwargs):
    return [t.lemma_ for t in textacy.extract.words(doc, **kwargs)]

def extract_entities(doc, include_types=None, sep='_'):
    ents = textacy.extract.entities(doc,
             include_types=include_types,
             exclude_types=None,
             drop_determiners=True,
             min_freq=1)
    return [(e.lemma_, e.label_) for e in ents]

def extract_nlp(doc):
    return {
    'lemmas': extract_lemmas(doc, exclude_pos=['PART', 'PUNCT', 'DET', 'PRON', 'SYM', 'SPACE'], filter_stops=True),
    'adjs_verbs': extract_lemmas(doc, include_pos=['ADJ', 'VERB']),
    'nouns': extract_lemmas(doc, include_pos=['NOUN', 'PROPN']),
    'noun_phrases': extract_noun_phrases(doc),
    'entities': extract_entities(doc, ['PERSON', 'ORG', 'GPE', 'LOC'])
    }

# ================================
# 6. TF-IDF VECTORIZATION & ENHANCED STOPWORD SETUP
# ================================
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as stopwords

stop_words_list = list(stopwords)
vectorizer = TfidfVectorizer(stop_words=stop_words_list, min_df=5, max_df=0.7)
tfidf_df = vectorizer.fit_transform(df['clean_description'])
num_features = tfidf_df.shape[1]
print(num_features)

# ================================
# 7. RUN TEXTACY + SPACY NLP ON CLEANED TEXT
# ================================
nlp = spacy.load('en_core_web_sm', disable=[])
nlp.tokenizer = custom_tokenizer(nlp)
df['nlp_features'] = df['clean_description'].apply(lambda text: extract_nlp(nlp(text)))

# ================================
# 8. TF-IDF ON COMBINED NLP FEATURES
# ================================
df['text_for_tfidf'] = df['nlp_features'].apply(lambda x: ' '.join(x.get('lemmas', []) + x.get('noun_phrases', [])))
tfidf_text_vectorizer = TfidfVectorizer(min_df=5, max_df=0.7)
tfidf_text_dt = tfidf_text_vectorizer.fit_transform(df['text_for_tfidf'])

# ================================
# 9. NMF TOPIC MODELING (RAW)
# ================================
from sklearn.decomposition import NMF
num_topics = 10
nmf_text_model = NMF(n_components=num_topics, random_state=42)
W_text_matrix = nmf_text_model.fit_transform(tfidf_text_dt)
H_text_matrix = nmf_text_model.components_
W_text_matrix.shape, H_text_matrix.shape

def display_topics(model, features, no_top_words=5):
    for topic_id, word_loadings in enumerate(model.components_):
        total = word_loadings.sum()
        sorted_loadings = word_loadings.argsort()[::-1]
        print(f"\nTopic {topic_id}")
        for i in range(0, no_top_words):
            word_id = sorted_loadings[i]
            print(f"  {features[word_id]} ({abs(word_loadings[word_id]*100.0/total):2.2f})")

display_topics(nmf_text_model, tfidf_text_vectorizer.get_feature_names_out(), 10)

# ================================
# 10. ENHANCED NMF WITH CUSTOM STOPWORDS
# ================================
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

enhanced_stop_words = ["ketchum", "fanduel", "etsy", "york", "aba", "futu", "nbc", "inc", "smbc", 'description', 'job', 'role', 'type', 'employment', 'type', 'millions', 'stv', 'tackle', 'decades']
add_stopwords = list(ENGLISH_STOP_WORDS.union(enhanced_stop_words))

vectorizer = TfidfVectorizer(stop_words=add_stopwords,min_df=5,max_df=0.7)
dtm = vectorizer.fit_transform(df["text_for_tfidf"])
feature_names = vectorizer.get_feature_names_out()

num_topics = 10
nmf_text_model = NMF(n_components=num_topics, random_state=42)
W_enhanced = nmf_text_model.fit_transform(dtm)
H_enhanced = nmf_text_model.components_

def display_topics(H, feature_names, no_top_words=10):
    for topic_idx, topic in enumerate(H):
        print(f"\nTopic {topic_idx}:")
        for i in topic.argsort()[:-no_top_words - 1:-1]:
            print(f"{feature_names[i]} ({topic[i]:.3f})")

display_topics(H_enhanced, feature_names, no_top_words=5)

# ================================
# 11. LABEL TOPICS AND PLOT DISTRIBUTION
# ================================
topic_names = [
  'Community Justice & Legal Advocacy Roles',
  'Data Analytics & BI Professional',
  'HR / Job Description & Contract Administration',
  'ABA / Behavioral Health Therapist',
  'Digital Content & Communications Specialist',
  'Banking & Financial Services Roles',
  'Marketing & SEO Specialist',
  'Healthcare & Insurance Professional',
  'Enterprise Risk & Investment Manager',
  'AEC Firm Leadership & Award-Winning Roles'
]
def topic_distribution(topic_names, percents):
    plt.barh(topic_names, percents)
    plt.gca().invert_yaxis()
    plt.xlabel("Percentage of Documents")
    plt.show()

doc_pct = W_enhanced.sum(axis=0) / W_enhanced.sum() * 100.0
topic_distribution(topic_names, doc_pct)

# ================================
# 12. OTHER TOPIC MODEL VARIANTS (SVD + LDA)
# ================================
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

# Truncated SVD
svd_para_model = TruncatedSVD(n_components = num_topics, random_state=42)
W_svd_para_matrix = svd_para_model.fit_transform(dtm)
H_svd_para_matrix = svd_para_model.components_
display_topics(H_svd_para_matrix, vectorizer.get_feature_names_out(), 10)

# LDA
count_para_vectorizer = CountVectorizer(stop_words=list(enhanced_stop_words), min_df=5, max_df=0.7)
count_para_dt = count_para_vectorizer.fit_transform(df["text_for_tfidf"])
lda_para_model = LatentDirichletAllocation(n_components = 10, random_state=42)
W_lda_para_matrix = lda_para_model.fit_transform(count_para_dt)
H_lda_para_matrix = lda_para_model.components_
display_topics(H_lda_para_matrix, count_para_vectorizer.get_feature_names_out())

topic_distribution(topic_names, doc_pct)

# ================================
# 13. EXPORT FINAL DATA
# ================================
df.to_csv('final_data.csv', index=False)