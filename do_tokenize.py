
import tiktoken

encoding=tiktoken.get_encoding('cl100k_base')

def encode(text):
    return encoding.encode(text)
def decode(tokens):
    return encoding.decode(tokens)


if __name__ == "__main__":
    with open('data/sales_textbook.txt', 'r', encoding="utf-8") as file:
        for line in file:
            tokens = encode(line.strip())
            print(tokens)
        for token in tokens:
            print(decode([token]))
            print()


