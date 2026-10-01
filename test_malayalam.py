import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# ENGLISH -> MALAYALAM SPECIALIZED NLLB TEST
# ============================================================

MODEL_ID = "Muhammed-sheheen/NLLB_FINETUNIG"

SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "mal_Mlym"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("=" * 70)
print("ENGLISH -> MALAYALAM NLLB TEST")
print("=" * 70)

print("Model:", MODEL_ID)
print("Device:", DEVICE)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    src_lang=SOURCE_LANGUAGE
)

print("Tokenizer loaded.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID
)

model = model.to(DEVICE)
model.eval()

print("Model loaded.")


# ============================================================
# TEST SENTENCES
# ============================================================

sentences = [
    "I saw her duck, the bird.",
    "I saw her lower her head.",
    "I saw her duck.",
    "The duck was swimming in the lake.",
    "She lowered her head.",
    "The doctor examined the patient.",
    "I went to the bank yesterday.",
    "She is studying computer science."
]


# ============================================================
# TARGET LANGUAGE TOKEN
# ============================================================

target_token_id = tokenizer.convert_tokens_to_ids(
    TARGET_LANGUAGE
)

print("\nTarget token ID:", target_token_id)


# ============================================================
# TRANSLATION
# ============================================================

results = []

print("\nTranslating...\n")


for sentence in sentences:

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        generated = model.generate(
            **inputs,
            forced_bos_token_id=target_token_id,
            num_beams=4,
            max_new_tokens=128
        )

    translation = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True
    )[0]

    results.append(
        (sentence, translation)
    )

    print("-" * 70)
    print("English:")
    print(sentence)

    print("\nMalayalam:")
    print(translation)

    print()


# ============================================================
# SAVE UTF-8 RESULTS
# ============================================================

with open(
    "malayalam_model_results.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("ENGLISH -> MALAYALAM SPECIALIZED NLLB TEST\n")
    f.write("=" * 70)
    f.write("\n\n")

    for source, translation in results:

        f.write("English:\n")
        f.write(source)
        f.write("\n\n")

        f.write("Malayalam:\n")
        f.write(translation)
        f.write("\n\n")

        f.write("-" * 70)
        f.write("\n\n")


print("=" * 70)
print("TEST COMPLETED")
print("=" * 70)

print("\nResults saved to:")
print("malayalam_model_results.txt")