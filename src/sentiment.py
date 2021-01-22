import numpy as np
import pandas as pd

from dateutil.relativedelta import relativedelta
from flair.data import Sentence
from concurrent.futures import ThreadPoolExecutor

CONCURRENT_AT_THE_SAME_TIME = 8


def text_sentiment_flair(clf, text):
    try:
        sentence = Sentence(text)
        clf.predict(sentence)
        if sentence.labels[0].value in ('NEGATIVE', '0'):
            score = -1*(sentence.labels[0].score)
        else:
            score = sentence.labels[0].score
        return  score 
    except Exception as e:
        return np.nan
    
    return score


def parallel_predict(predict_func, series, chunk_size=1000, concurrency=CONCURRENT_AT_THE_SAME_TIME):
    chunks = [series[i:i+chunk_size] for i in range(0, len(series), chunk_size)]
    with ThreadPoolExecutor(max_workers=CONCURRENT_AT_THE_SAME_TIME) as executor:
        futures = [
            executor.submit(
                predict_func,
                chunk
            )

            for chunk in chunks
        ]
    
    return np.concatenate([future.result() for future in futures])


def summarise_daily_tweet_sentiment(df):
    return {
        'avg_sentiment': df.sentiment_score_adj.mean(),
        'positive_fraction': (df.sentiment_score_adj > 0).mean(),
        'quantile_0.05': df.sentiment_score_adj.quantile(0.05),
        'quantile_0.1': df.sentiment_score_adj.quantile(0.1),
        'quantile_0.25': df.sentiment_score_adj.quantile(0.25),
        'quantile_0.4': df.sentiment_score_adj.quantile(0.4),
        'quantile_0.5': df.sentiment_score_adj.quantile(0.5),
        'quantile_0.6': df.sentiment_score_adj.quantile(0.6),
        'quantile_0.75': df.sentiment_score_adj.quantile(0.75),
        'quantile_0.9': df.sentiment_score_adj.quantile(0.9),
        'quantile_0.95': df.sentiment_score_adj.quantile(0.95),
        'score_-0.95': (df.sentiment_score_adj < -0.95).mean(),
        'score_-0.9': (df.sentiment_score_adj < -0.9).mean(),
        'score_-0.7': (df.sentiment_score_adj < -0.7).mean(),
        'score_-0.5': (df.sentiment_score_adj < -0.5).mean(),
        'score_-0.3': (df.sentiment_score_adj < -0.3).mean(),
        'score_-0.1': (df.sentiment_score_adj < -0.1).mean(),
        'score_0': (df.sentiment_score_adj < 0).mean(),
        'score_0.1': (df.sentiment_score_adj < 0.1).mean(),
        'score_0.3': (df.sentiment_score_adj < 0.3).mean(),
        'score_0.5': (df.sentiment_score_adj < 0.5).mean(),
        'score_0.7': (df.sentiment_score_adj < 0.7).mean(),
        'score_0.9': (df.sentiment_score_adj < 0.9).mean(),
        'score_0.95': (df.sentiment_score_adj < 0.95).mean(),
        'n_tweets': df.shape[0],
        'negative_100': str(list(df.sort_values('sentiment_score_adj', ascending=True)['processed_content'].head(100))),
        'positive_100': str(list(df.sort_values('sentiment_score_adj', ascending=False)['processed_content'].head(100))),
    }


def insert_crosses_zero_points(ss_filtered, cutoff):
    """
    For improved plots.
    """
    cross_idx = (
        (ss_filtered > cutoff) & (ss_filtered.shift(-1) < cutoff) | 
        (ss_filtered < cutoff) & (ss_filtered.shift(-1) > cutoff)
    ) 
    insert_points = ss_filtered[cross_idx]

    to_insert = pd.Series()
    for crosspoint in insert_points.index:
        sentiment_before = ss_filtered[crosspoint]
        sentiment_after = ss_filtered[crosspoint + relativedelta(days=1)]
        prop = abs(cutoff - sentiment_before) / abs(sentiment_after - sentiment_before)
        hours_insert = 24*prop
        
        to_insert[crosspoint+relativedelta(hours=hours_insert)] = cutoff
        
    return ss_filtered.append(
        to_insert
    ).sort_index()
