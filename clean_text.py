# clean text fuction make it possible to modify stop words or tags
# useful in the pre-processing step interations
# apply to the text column in the sub_onetree.csv. It creates a new clean_texts 
 
import re
import nltk
from nltk.corpus import stopwords
import spacy
import html

# Find URL
def find_URL(comment):
    return re.findall(r'((?:https?:\/\/)(?:\w[^\)\]\s]*))', comment)

def clean_comment(comment, new_stopwords_list = [], lemma=True, show=False,
                  del_tags = ['NUM', 'PRON', 'ADV', 'DET', 'AUX', 'SCONJ', 'PART']):

    # NLTK Stop words
    stop_words = stopwords.words('english')

    stop_words_extended = ["he'd", "we're", "she'd", "we've", "i'm", 'would', "we'll", 
    "what's", "i'd", "why's", "that's", "who's", "where's", "he'll", "they'd", "here's", "they're",
    "let's", "i've", "they'll", "they've", 'could', "i'll", "there's", "how's", "she'll", 
    "we'd", "when's", 'ought', "he's", 'etc', 'however', 'there', 'also', 'digit'] + new_stopwords_list

    stop_words.extend(stop_words_extended)
    # remove URLs
    comment = re.sub(r"((?:https?:\/\/)(?:\w[^\)\]\s]*))",'', comment)
    # unescape html formatting
    comment = html.unescape(comment)
    # consider internal hyphen as full words. "Technical vocabulary"
    comment = re.sub(r"\b(\w*)-(\w*)\b", r"\g<1>_\g<2>", comment)
    comment = re.sub(r"(<SUB>|nan|<NEW TIER>|<SAME TIER>)", "", comment)
    comment = comment.lower() # ? consider to make general the name of companies or decives
    comment = re.sub(r'&#x200B', ' ', comment) # character code for a zero-width space
    comment = re.sub(r'remindme![\w\s\W]*$', ' ', comment) # remove call to remind me bot
    comment = re.sub(r'\n', ' ', comment) # remove new line formatting
    comment = re.sub(r'(\[deleted\]|\[removed\])', '', comment)
    comment = re.sub(r"[^\w\s]", ' ', comment) # punctuation and emoji
    comment = re.sub(r'(\s_|_\s)', '', comment) # remove underscores around a words (italics)

    # remove stop_words
    comment_token_list = [word for word in comment.strip().split() if word not in stop_words and len(word)>1]

    # keeps word meaning: important to infer what the topic is about
    if lemma == True:
        # Initialize spacy 'en' model
        nlp = spacy.load('en_core_web_sm')
        # https://spacy.io/api/annotation
        comment_text = nlp(' '.join(comment_token_list))
        if show == True:
            for token in comment_text:
                print(token.pos_, "\t", token)

        comment_token_list = [token.lemma_ for token in comment_text if token.pos_ not in del_tags]
    
    # harsh to the root of the word
    else:
        #We specify the stemmer or lemmatizer we want to use
        word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
        comment_token_list = [word_rooter(word) for word in comment_token_list]

    comment = ' '.join(comment_token_list)
    
    return comment