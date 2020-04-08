
"""
SPEECH TO TEXT
"""

from ibm_watson import SpeechToTextV1 
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

url_s2t = "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/9e0b6566-5f10-4831-8f10-fddfb6c8a9dc"

iam_apikey_s2t = "_OCHLxuP_BcedmMFmDZvgBmQRK2OgPBKUTQtFVsV8OdW"

authenticator = IAMAuthenticator(iam_apikey_s2t)
s2t = SpeechToTextV1(authenticator=authenticator)
s2t.set_service_url(url_s2t)
s2t

filename='PolynomialRegressionandPipelines.mp3'

with open(filename, mode="rb")  as wav:
    response = s2t.recognize(audio=wav, content_type='audio/mp3')
    
response.result

from pandas.io.json import json_normalize
json_normalize(response.result['results'],"alternatives")

recognized_text=response.result['results'][0]["alternatives"][0]["transcript"]
type(recognized_text)


"""
lANGUAGE TRANSLATOR
"""

from ibm_watson import LanguageTranslatorV3

url_lt = "https://api.us-south.language-translator.watson.cloud.ibm.com/instances/e2893beb-b4ef-44f9-982a-c6ada93c6fff"

apikey_lt = "_ZvGXgRMHJt55oOU669dZNk1DYJIkmNiCQIemditIWV4"

version_lt='2018-05-01'

authenticator = IAMAuthenticator(apikey_lt)
language_translator = LanguageTranslatorV3(version=version_lt,authenticator=authenticator)
language_translator.set_service_url(url_lt)
language_translator

from pandas.io.json import json_normalize

json_normalize(language_translator.list_identifiable_languages().get_result(), "languages")

translation_response = language_translator.translate(\
    text=recognized_text, model_id='en-es')
translation_response

translation=translation_response.get_result()
translation

spanish_translation =translation['translations'][0]['translation']
spanish_translation 

translation_new = language_translator.translate(text=spanish_translation ,model_id='es-en').get_result()

translation_eng=translation_new['translations'][0]['translation']
translation_eng

French_translation=language_translator.translate(
    text=translation_eng , model_id='en-fr').get_result()

French_translation['translations'][0]['translation']




