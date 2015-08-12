---
layout: post
title: Naive Bayes classifier for Twitter sentiment analysis
comments: True
---

Sentiment Analysis is the process of detecting the contextual polarity of text. In other words, it determines whether a piece of writing is positive, negative or neutral. 
Recently, I worked on a project requiring me to classify Tweets (and assign polarity scores) into positive and negative. We wanted to classify Tweets about some movies to understand the overall sentiment of response towards that movie. I know this is one of the most clichéd NLP problem.

### Finding the right training corpus

This is a really important step because of the simple reason that no matter how good your algorithm, implementation, feature engineering and parameter tuning is your final results are going to be bad unless you have a good training corpus. It is also important to use a corpus that is somewhat related to the domain you want to classify text for. You should not use a generic Twitter corpus for classifying Tweets related to movie review. For example, when I use the corpus provided by [Niek Sanders](http://www.sananalytics.com/lab/twitter-sentiment/), Tweets containing negative words (e.g can't wait) gets classified as negative. But, in the context of the movies, Tweets like:

<pre>
RT @79Glukhenko: Can't wait to see insurgent :)
</pre>

should be classified as positive. So, context matters.

In this post though, I'm going to use the aformentioned corpus. 

### Introducing the Naïve Bayes classifier

The Naïve Bayes algorithm assumes that all the features are independent of each other (hence the name *Naïve* Bayes). We represent a document as a *bag of words*. This is a disturbingly simple representation: it only knows which words are included in the document (and how many times each word occurs), and throws away the word order. Naïve Bayes classification is nothing more than keeping track of which feature gives evidence to which class. 

### Converting text into bag-of-words
For each word in the review, its occurrence is counted and noted in a vector. This step is called vectorization. The vector is typically huge as it contains as many elements as the words that occur in the whole dataset. Python has some good libraries to create this vector. I will be using [Scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html). You may also want to look into [NLTK](http://www.nltk.org/book/ch06.html)

The code assumes that you are reading Tweets you want to classify from a database. Purpose of this post is to write about improving Naive Bayes. So, I'll not go into the nuances of implementation.

{% highlight python linenos %}

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer


tweets = pd.read_csv('tweets.csv')

f = pd.read_csv('twitter_corpus.csv', sep=',', names=['Text', 'Sentiment'], dtype=str)

def split_into_lemmas(tweet):
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)
    analyze = bigram_vectorizer.build_analyzer()
    return analyze(tweet)


bow_transformer = CountVectorizer(analyzer = split_into_lemmas, stop_words='english', strip_accents='ascii').fit(f['Text'])


text_bow = bow_transformer.transform(f['Text'])
tfidf_transformer = TfidfTransformer().fit(text_bow)
tfidf = tfidf_transformer.transform(text_bow)

text_tfidf = tfidf_transformer.transform(text_bow)

{% endhighlight %}

The above code mainly performs the following:

1. Lemmatization: splitting tweets into single words, bigrams etc `ngram_range = (1, 3)`, only retaining words `token_pattern=r'\b\w+\b' ` i.e removing emojis etc.
2. Bag of words: converting tweets into a list of individual words, removing stop words, stemming.
3. Tf-idf transformation: computing how important a word is based on number of times it appears in the corpus.


### Defining the classifier

{% highlight python linenos %}

classifier_nb = MultinomialNB(class_prior=[0.20, 0.80]).fit(text_tfidf, f['Sentiment'])

{% endhighlight %}

Notice the *class_prior* parameter. It is used to set the prior probability assigned to classes. Now, ideally you should let the classifier learn itself the prior probabilities from the corpus. But, if your corpus is imbalanced i.e it contains much more number of tweets classified as negative than positive then your classifier it going to bin most tweets into negative category. Tweaking *class_prior* to get a suitable value may help with this.


{% highlight python linenos %}

sentiments = pd.DataFrame(columns = ['text', 'class', 'prob'])
i = 0
for _, tweets in texts.iterrows():
    i += 1
    try:
        bow_tweet = bow_transformer.transform(tweet)
        tfidf_tweet = tfidf_transformer.transform(bow_tweet)
        sentiments.loc[i-1, 'text'] = tweet.values[0]
        sentiments.loc[i-1, 'class'] = classifier_nb.predict(tfidf_tweet)[0]
        sentiments.loc[i-1, 'prob'] = round(classifier_nb.predict_proba(tfidf_tweet)[0][1], 2)*10
    except Exception as e:
        sentiments.loc[i-1, 'text'] = tweet.values[0]

sentiments.to_csv('sentiments.csv', encoding ='utf-8')
print sentiments

Here's the result:

{% highlight %}
                                                text class prob
0  Congrats @sundarpichai well deserved! Proud mo...   pos  5.1
1  Congratulations @sundarpichai. My best wishes ...   pos  6.1
2  The choice is ultimately between diplomacy and...   pos  5.3
3  The movie #drive is seriously the worst movie ...   neg  4.6
4  News of another attack in Kabul is very sadden...   neg  4.7

{% endhighlight %}

Now, you may play with parameters to improve the performance of the classifier. You may also train it on a better corpus.s
