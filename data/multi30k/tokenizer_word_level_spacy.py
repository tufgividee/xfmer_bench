from pathlib import Path

import spacy

nlp_en = spacy.blank("en")
nlp_de = spacy.blank("de")

print([t.text for t in nlp_en("Hello world!")])
print([t.text for t in nlp_de("Hallo Welt!")])

DATA_DIR = Path(__file__).parent / "multi30k"

with open(DATA_DIR / "train.en", encoding="utf-8") as en, \
     open(DATA_DIR / "train.de", encoding="utf-8") as de:

    for i, (en_line, de_line) in enumerate(zip(en, de)):

        print(i)
        print("EN:", en_line.strip())
        print("DE:", de_line.strip())

        en_line_tokens = [t.text for t in nlp_en(en_line.strip())]
        de_line_tokens = [t.text for t in nlp_de(de_line.strip())]

        print(en_line_tokens)
        print(de_line_tokens)
        print()]

        # if i == 19:
        #     break

