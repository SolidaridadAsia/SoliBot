# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/

import json
from typing import Any, Text, Dict, List
import torch
from bert_serving.client import BertClient
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import numpy as np
from sentence_transformers import SentenceTransformer
from googletrans import Translator
import sys
import re

if (sys.version_info[0] < 3):
    import urllib2
    import urllib
    import HTMLParser
else:
    import html.parser
    import urllib.request
    import urllib.parse

agent = {'User-Agent':
"Mozilla/4.0 (\
compatible;\
MSIE 6.0;\
Windows NT 5.1;\
SV1;\
.NET CLR 1.1.4322;\
.NET CLR 2.0.50727;\
.NET CLR 3.0.04506.30\
)"}

# sentence embedding selection
sentence_transformer_select=True
pretrained_model='bert-base-nli-mean-tokens' 
score_threshold = 0.50  # confidence scores

# Custom Action
class ActionGetFAQAnswer(Action):

    def __init__(self):
        super(ActionGetFAQAnswer, self).__init__()
        self.faq_data = json.load(open("./data/nlu/faq.json", "rt", encoding="utf-8"))
        self.sentence_embedding_choose(sentence_transformer_select, pretrained_model)
        self.standard_questions_encoder = np.load("./data/standard_questions.npy")
        self.standard_questions_encoder_len = np.load("./data/standard_questions_len.npy")
        print(self.standard_questions_encoder.shape)

    def sentence_embedding_choose(self, sentence_transformer_select=True, pretrained_model='bert-base-nli-mean-tokens'):
        self.sentence_transformer_select = sentence_transformer_select
        if sentence_transformer_select:
            self.bc = SentenceTransformer(pretrained_model)
        else:
            self.bc = BertClient(check_version=False)

    def get_most_similar_standard_question_id(self, query_question):
        if self.sentence_transformer_select:
            query_vector = torch.tensor(self.bc.encode([query_question])[0]).numpy()
        else:
            query_vector = self.bc.encode([query_question])[0]
        print("Question received at action engineer")
        score = np.sum((self.standard_questions_encoder * query_vector), axis=1) / (
                self.standard_questions_encoder_len * (np.sum(query_vector * query_vector) ** 0.5))
        top_id = np.argsort(score)[::-1][0]
        return top_id, score[top_id]

    def name(self) -> Text:
        return "action_get_answer"

    def unescape(self, text):
        if (sys.version_info[0] < 3):
           parser = HTMLParser.HTMLParser()
        else:
           parser = html.parser.HTMLParser()
        return (parser.unescape(text))


    def translator_lang(self, to_translate, to_language="en", from_language="auto"):
        base_link = "http://translate.google.com/m?hl=%s&sl=%s&q=%s"
        if (sys.version_info[0] < 3):
           to_translate = urllib.quote_plus(to_translate)
           link = base_link % (to_language, from_language, to_translate)
           request = urllib2.Request(link, headers=agent)
           raw_data = urllib2.urlopen(request).read()
        else:
           to_translate = urllib.parse.quote(to_translate)
           link = base_link % (to_language, from_language, to_translate)
           request = urllib.request.Request(link, headers=agent)
           raw_data = urllib.request.urlopen(request).read()
        data = raw_data.decode("utf-8")
        expr = r'class="t0">(.*?)<'
        re_result = re.findall(expr, data)
        if (len(re_result) == 0):
           result = ""
        else:
           result = unescape(re_result[0])
        return (result)

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        question = tracker.latest_message['text']
        translator = Translator()
        x = translator.translate(question)
        detected_lang = x.src
        query = x.text
        print(query)
        most_similar_id, score = self.get_most_similar_standard_question_id(query)
        print("The question is matched with id:{} with score: {}".format(most_similar_id,score))
        if float(score) > score_threshold: 
            answer = self.faq_data[most_similar_id]['a']
            resp = translator.translate(answer, dest=detected_lang)
            response = resp.text
            dispatcher.utter_message(response)
        else:
            answer = "Sorry, I didn't understand that...\nCould you please re-phrase the question?"
            resp = translator.translate(answer, dest=detected_lang)
            response = resp.text
            dispatcher.utter_message(response)
            
        return []


def encode_standard_question(sentence_transformer_select=True, pretrained_model='bert-base-nli-mean-tokens'):
    """
    This will encode all the questions available in question database into sentence embedding. The result will be stored into numpy array for comparision purpose.
    """
    if sentence_transformer_select:
        bc = SentenceTransformer(pretrained_model)
    else:
        bc = BertClient(check_version=False)
    data = json.load(open("./data/nlu/faq.json", "rt", encoding="utf-8"))
    standard_questions = [each['q'] for each in data]
    print("Standard question size", len(standard_questions))
    print("Start to calculate encoder....")
    if sentence_transformer_select:
        standard_questions_encoder = torch.tensor(bc.encode(standard_questions)).numpy()
    else:
        standard_questions_encoder = bc.encode(standard_questions)
    np.save("./data/standard_questions", standard_questions_encoder)
    standard_questions_encoder_len = np.sqrt(np.sum(standard_questions_encoder * standard_questions_encoder, axis=1))
    np.save("./data/standard_questions_len", standard_questions_encoder_len)


encode_standard_question(sentence_transformer_select,pretrained_model)
if __name__ == '__main__':
     encode_standard_question(True)
