'''
sandipan
ghoshm21@gmail.com
29th Aug 2020
with progress report using tqdm
'''
import flask
import json
# from itertools import chain
from bs4 import BeautifulSoup
import unidecode
from pycontractions import Contractions
import pandas as pd
import re
import string
# import gensim
# import gensim.downloader as api
from gensim.models import KeyedVectors

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

def remove_html(text):
  """remove html tags from text"""
  soup = BeautifulSoup(text, "html.parser")
  stripped_text = soup.get_text(separator=" ")
  return stripped_text


def remove_accented_chars(text):
  """remove accented characters from text, e.g. café"""
  text = unidecode.unidecode(text)
  return text


def remove_tabs(text):
  '''remove all the tab, new line char'''
  text = text.replace('\t', ' ')
  text = text.replace('\r', ' ')
  text = text.replace('\n', ' ')
  return text


def remove_blanks(text):
  '''remove all the more than 1 spaces'''
  text = re.sub(' +', ' ', text)
  return text


def remove_digits(text):
  # Remove digits, decimal numbers, dates and time format
  return re.sub(r'\d[\.\/\-\:]\d|\d', '', text)


def remove_all_punctuation(text):
  '''Remove other punctuation, adding fe more
  string.punctuation = !"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~`
  '''
  PUNCT_TO_REMOVE = string.punctuation
  return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


def remove_special_characters(text, remove_digits=False):
  pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
  text = re.sub(pattern, '', text)
  return text


def ascii_to_string(text):
  # Encodes string to ASCII and decodes to string. This helps in removing any special characters in the database
  text = text.encode('ascii', 'replace').decode(encoding="utf-8")
  '''
  This replaces all special characters with a ?. Replacing this
  '''
  return text.replace('?', '')


def to_lower(text):
  '''conver all to lower'''
  return str(text).lower()


def remove_url(text):
  # Remove any web url starting with http or www
  return re.sub(r'(www|http)\S+', '', text)


def remove_email_address(text):
  # Remove any email address
  return re.sub(r'\S+@\S+', '', text)


# we should not use the so big precompiled word2vec model in container,
# it would be slow and container size would be big
model = KeyedVectors.load(
  '/app/lib/gensim/GoogleNews-vectors-negative300', mmap='r')
cont = Contractions(kv_model=model)
cont.load_models()

def expand_contractions(text):
  """expand shortened words, e.g. don't to do not"""
  text = list(cont.expand_texts([text], precise=True))[0]
  return text

@app.route("/v1/preprocess", methods=["GET", "POST"])
def preprocess():
  data = {"success": False}
  # get the request parameter
  params = flask.request.json
  if (params == None):
    params = flask.request.args

    if (params != None):
        x = str(params.get("text"))
        if (x == 'None'):
            return page_not_found(404)
        else:
          print(x)
          sr = pd.Series([x]).apply(remove_html) \
                .apply(remove_accented_chars) \
                .apply(remove_tabs) \
                .apply(remove_blanks) \
                .apply(remove_digits) \
                .apply(ascii_to_string) \
                .apply(remove_special_characters) \
                .apply(remove_url) \
                .apply(remove_email_address) \
                .apply(expand_contractions) \
                .apply(to_lower) \
                .apply(remove_all_punctuation)
          data["original_text"] = x
          data["process_text"] = str(sr[0])
          data["success"] = True
  # return the response
  return flask.jsonify(data)  

# start the app
app.run(host='0.0.0.0', port=50001, threaded=True)

# x = str("Python is an interpreted, interactive, object-oriented, open-source programming language.</>This isn't not just a langauge")
# sr = pd.Series([x]).apply(remove_html) \
#     .apply(remove_accented_chars) \
#     .apply(remove_tabs) \
#     .apply(remove_blanks) \
#     .apply(remove_digits) \
#     .apply(ascii_to_string) \
#     .apply(remove_special_characters) \
#     .apply(remove_url) \
#     .apply(remove_email_address) \
#     .apply(expand_contractions) \
#     .apply(to_lower) \
#     .apply(remove_all_punctuation)
# print(sr[0])
