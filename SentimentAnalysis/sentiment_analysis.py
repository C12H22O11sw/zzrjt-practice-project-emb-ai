''' Determines the sentament of a string using IBM Watson API
'''

import json
import requests

def sentiment_analyzer(text):
    ''' Returns a dictionary showing the label and score of a text's sentament 
    as determined by IBM watson 
    '''
    url = 'https://sn-watson-sentiment-bert.labs.skills.network \
           /v1/watson.runtime.nlp.v1/NlpService/SentimentPredict'
    headers = {"grpc-metadata-mm-model-id": "sentiment_aggregated-bert-workflow_lang_multi_stock"}
    myobj = { "raw_document": { "text": text } }

    response = requests.post(url, json = myobj, headers=headers, timeout=5000)
    formatted_response = json.loads(response.text)

    if response.status_code == 200:
        label = formatted_response['documentSentiment']['label']
        score = formatted_response['documentSentiment']['score']
    else:
        label = None
        score = None

    return {'label': label, 'score': score}
