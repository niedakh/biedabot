from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer

def main():
    dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

    translation_model_name = 'Helsinki-NLP/opus-mt-pl-en'
    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translation_model = MarianMTModel.from_pretrained(translation_model_name)

    reverse_translation_model_name = 'Helsinki-NLP/opus-mt-en-sla'
    reverse_translation_tokenizer = MarianTokenizer.from_pretrained(reverse_translation_model_name)
    reverse_translation_model = MarianMTModel.from_pretrained(reverse_translation_model_name)

    blenderbot_model_name = 'facebook/blenderbot-1B-distill'
    blenderbot_model = BlenderbotForConditionalGeneration.from_pretrained(blenderbot_model_name)
    blenderbot_tokenizer = BlenderbotTokenizer.from_pretrained(blenderbot_model_name)



if __name__ == "__main__":
    # execute only if run as a script
    main()