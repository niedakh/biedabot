import en_core_web_sm
import torch
import pickle
import io
import json
import string
from flask import Flask, render_template, request
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Any

app = Flask(__name__)

dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

nlp = en_core_web_sm.load()
translation_model_name = 'Helsinki-NLP/opus-mt-pl-en'
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)

reverse_translation_model_name = 'Helsinki-NLP/opus-mt-en-sla'
reverse_translation_tokenizer = MarianTokenizer.from_pretrained(reverse_translation_model_name)
reverse_translation_model = MarianMTModel.from_pretrained(reverse_translation_model_name)

blenderbot_model_name = 'facebook/blenderbot-1B-distill'
blenderbot_model = BlenderbotForConditionalGeneration.from_pretrained(blenderbot_model_name)
blenderbot_tokenizer = BlenderbotTokenizer.from_pretrained(blenderbot_model_name)

def translate_to_english(message: str) -> str:
    translated = translation_model.generate(**translation_tokenizer.prepare_seq2seq_batch(message, return_tensors="pt"))
    tgt_text = [translation_tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]
    return tgt_text

def translate_to_polish(message: str) -> str:
    src_text = [f'>>pol<< {reply}' for reply in nlp(message).sents]
    translated = reverse_translation_model.generate(
        **reverse_translation_tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt"))
    tgt_text = ' '.join([reverse_translation_tokenizer.decode(t, skip_special_tokens=True) for t in translated])
    return tgt_text

def chat_blenderbot(message:str, chat_history_ids="") -> Tuple[str, str]:
    model = blenderbot_model
    inputs = blenderbot_tokenizer([message], return_tensors='pt')
    reply_ids = model.generate(**inputs)
    reply = blenderbot_tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    return reply, ""

def chat_dialoggpt(message, chat_history_ids=""):
    model=dialogpt_model
    new_user_input_ids = dialogpt_tokenizer.encode(message + dialogpt_tokenizer.eos_token, return_tensors='pt')

    if len(chat_history_ids) > 0:
        inputs = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        inputs = new_user_input_ids


    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(inputs, max_length=1000, pad_token_id=dialogpt_tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    reply = dialogpt_tokenizer.decode(chat_history_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)

    return reply, json.dumps(chat_history_ids.tolist())


def discuss_with_bot(message, model_callable=None, chat_history_ids="") -> Tuple[str, Any]:
    message_en = translate_to_english(message)
    reply, history = model_callable(message_en, chat_history_ids)
    return translate_to_polish(reply), history


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    message = request.args.get('msg').strip()
    model = request.args.get('model').strip()
    history = request.args.get('history').strip()

    if len(history) > 0:
        history = torch.tensor(json.loads(history))

    callable_model = chat_blenderbot
    if model == 'dialoggpt':
        callable_model = chat_dialoggpt

    reply, history = discuss_with_bot(message, callable_model, history)

    return json.dumps({
        'reply': reply,
        'history': history if len(history)>0 else ""
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0')
