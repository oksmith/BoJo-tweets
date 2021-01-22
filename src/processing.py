import json
import os
import nltk
import re 

from src.utils import get_date_from_file_loc

TWEETS_DATA_DIR = 'tweets_data'

STOP_WORDS = nltk.corpus.stopwords.words('english')

EXTRA_STOP_WORDS = ['bori', 'boris', 'johnson']  # we don't want the topic to be about Bojo himself


def remove_special_parts(tweet_txt):
    return ' '.join(
        re.sub("(@[A-Za-z0-9_]+)|([^0-9A-Za-z' \t])|(\w+:\/\/\S+)", " ", tweet_txt.replace('&amp;', ' '))\
            .split()
    )

def normalize_document(document, tokenizer, lemmatizer):
    
    norm_document = document.lower()
    norm_document = remove_special_parts(norm_document)
    norm_tokens = [token.strip() for token in tokenizer.tokenize(norm_document)]
    norm_tokens = [lemmatizer.lemmatize(token) for token in norm_tokens if not token.isnumeric()]
    norm_tokens = [token for token in norm_tokens if len(token) > 1]
    norm_tokens = [token for token in norm_tokens if token not in STOP_WORDS+EXTRA_STOP_WORDS]
    
    return norm_tokens


class StreamingCorpus(object):
    """
    Create an iterable which opens and yields a single day of tweets.
    """
    def __init__(self, data_dir=TWEETS_DATA_DIR, phraser=None):
        """
        :param model: optional Phraser model to apply after normalization.
        """
        self.files = [os.path.join(data_dir, x) 
                      for x in os.listdir(data_dir) if x.endswith('.json')]
        self.phraser = phraser
    
    def __iter__(self):
        wtk = nltk.tokenize.RegexpTokenizer(r'\w+')
        wnl = nltk.stem.wordnet.WordNetLemmatizer()
        for file_loc in self.files:
            with open(file_loc, 'r') as f:
                lines = [json.loads(line) for line in f.read().split('\n') if len(line) > 0]
                document = ' '.join([x['content'] for x in lines if x['lang'] == 'en'])
                if self.phraser:
                    norm_document = self.phraser[normalize_document(document, tokenizer=wtk, lemmatizer=wnl)]
                else:
                    norm_document = normalize_document(document, tokenizer=wtk, lemmatizer=wnl)
                
                yield norm_document
                
    def __len__(self):
        return len(self.files)

                
class BagOfWordsStreamingCorpus(object):
    def __init__(self, dictionary, data_dir=TWEETS_DATA_DIR, phraser=None):
        self.files = [os.path.join(data_dir, x) 
                      for x in os.listdir(data_dir) if x.endswith('.json')]
        self.phraser = phraser
        self.dictionary = dictionary
    
    def __iter__(self):
        self.titles = []
        for title, tokens in itertools.islice(iter_wiki(self.dump_file), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)
    
    def __iter__(self):
        wtk = nltk.tokenize.RegexpTokenizer(r'\w+')
        wnl = nltk.stem.wordnet.WordNetLemmatizer()
        
        for file_loc in self.files:
            with open(file_loc, 'r') as f:
                lines = [json.loads(line) for line in f.read().split('\n') if len(line) > 0]
                document = ' '.join([x['content'] for x in lines if x['lang'] == 'en'])
                if self.phraser:
                    norm_document = self.phraser[normalize_document(document, tokenizer=wtk, lemmatizer=wnl)]
                else:
                    norm_document = normalize_document(document, tokenizer=wtk, lemmatizer=wnl)
                
                yield self.dictionary.doc2bow(norm_document)
                
    def __len__(self):
        return len(self.files)
