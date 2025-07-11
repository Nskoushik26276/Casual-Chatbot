!pip install transformers torch

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

def main():
    model_name = "facebook/blenderbot-400M-distill"
    print("Loading model and tokenizer...")
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

    print("Chatbot is ready! Type something (type 'exit', 'quit', or 'bye' to stop):")

    chat_history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break

        chat_history.append(user_input)

        input_text = " ".join(chat_history)

        inputs = tokenizer(input_text, return_tensors="pt")

        reply_ids = model.generate(**inputs, max_length=200, do_sample=True, top_p=0.9, top_k=50, temperature=0.7)

        reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

        print("Chatbot:", reply)

        chat_history.append(reply)

        if len(chat_history) > 6:
            chat_history = chat_history[-6:]

if __name__ == "__main__":
    main()
