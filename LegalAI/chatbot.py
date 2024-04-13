import pytesseract
from pdf2image import convert_from_path
import google.generativeai as genai
from langdetect import detect
from translate import Translator
def translate(text, src, dest):
    translator = Translator(to_lang=dest, from_lang=src)
    result = ""
    text_chunks = [text[i:i+500] for i in range(0, len(text), 500)]  # Split text into chunks of maximum 500 characters
    for chunk in text_chunks:
        result += translator.translate(chunk)
    return result

def scanned_image_to_text(filename):
        pytesseract.pytesseract.tesseract_cmd  = '/usr/local/bin/tesseract'
        poppler_path = "/usr/local/bin"
        file = filename
        images = convert_from_path(file, poppler_path=poppler_path)
        text = ""
        for image in images:
            gray_image = image.convert('L')
            text += pytesseract.image_to_string(gray_image, lang="eng+tam",)
        return text


def chatting(context, chat, query):
    
    question_lang = detect(query)
    context_lang = detect(context)
    if question_lang == 'ta' or context_lang == 'ta':
        lang = "ta"
    else:
        lang = "en"
    if lang == 'ta':
        translator = Translator(to_lang='en', from_lang='ta')
        query = translator.translate(query)
    elif 'tamil' in query:
        query = query.replace('tamil', '')
        lang = 'ta'
    feed = f"Context - {context} \n This is the context. Answer the below question from the given context. \n Question - {query}"
            
    response = chat.send_message(feed, stream=True)
    out = ""
    for chunk in response:
        out += chunk.text 
    if lang == 'ta':
        out = translate(out,'en','ta')
    return out



