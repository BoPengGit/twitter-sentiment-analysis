import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

tweets = pd.read_csv('tweets.csv')

f = pd.read_csv('twitter_corpus.csv', sep=',', names=['Text', 'Sentiment'], dtype=str, header=0)


def split_into_lemmas(tweet):
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)
    analyze = bigram_vectorizer.build_analyzer()
    return analyze(tweet)

bow_transformer = CountVectorizer(analyzer=split_into_lemmas, stop_words='english', strip_accents='ascii').fit(f['Text'])

text_bow = bow_transformer.transform(f['Text'])
tfidf_transformer = TfidfTransformer().fit(text_bow)
tfidf = tfidf_transformer.transform(text_bow)

text_tfidf = tfidf_transformer.transform(text_bow)

classifier_nb = MultinomialNB(class_prior=[0.30, 0.70]).fit(text_tfidf, f['Sentiment'])

sentiments = pd.DataFrame(columns=['text', 'class', 'prob'])
i = 0
for _, tweet in tweets.iterrows():
    i += 1
    try:
        bow_tweet = bow_transformer.transform(tweet)
        tfidf_tweet = tfidf_transformer.transform(bow_tweet)
        sentiments.loc[i-1, 'text'] = tweet.values[0]
        sentiments.loc[i-1, 'class'] = classifier_nb.predict(tfidf_tweet)[0]
        sentiments.loc[i-1, 'prob'] = round(classifier_nb.predict_proba(tfidf_tweet)[0][1], 2)*10
    except Exception as e:
        sentiments.loc[i-1, 'text'] = tweet.values[0]

sentiments.to_csv('sentiments.csv', encoding='utf-8')
print sentiments
