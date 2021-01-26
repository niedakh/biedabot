import en_core_web_sm
from flask import Flask, render_template, request
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

nlp = en_core_web_sm.load()
translation_model_name = 'Helsinki-NLP/opus-mt-pl-en'
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)

reverse_translation_model_name = 'Helsinki-NLP/opus-mt-en-sla'
reverse_translation_tokenizer = MarianTokenizer.from_pretrained(reverse_translation_model_name)
reverse_translation_model = MarianMTModel.from_pretrained(reverse_translation_model_name)

chatbox_model_name = 'facebook/blenderbot-400M-distill'
chatbot_model = BlenderbotForConditionalGeneration.from_pretrained(chatbox_model_name)
chatbot_tokenizer = BlenderbotTokenizer.from_pretrained(chatbox_model_name)


def discuss_with_bot(message: str) -> str:
    translated = translation_model.generate(**translation_tokenizer.prepare_seq2seq_batch(message, return_tensors="pt"))
    tgt_text = [translation_tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]
    inputs = chatbot_tokenizer([tgt_text], return_tensors='pt')
    reply_ids = chatbot_model.generate(**inputs)
    reply = chatbot_tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    src_text = [f'>>pol<< {reply}' for reply in nlp(reply).sents]
    translated = reverse_translation_model.generate(
        **reverse_translation_tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt"))
    tgt_text = ' '.join([reverse_translation_tokenizer.decode(t, skip_special_tokens=True) for t in translated])
    return tgt_text


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return discuss_with_bot(userText)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
