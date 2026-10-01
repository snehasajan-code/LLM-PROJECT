import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "facebook/nllb-200-distilled-600M"

SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "mal_Mlym"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("NLLB-200 Translation Test")
print("=" * 70)

print("Model:", MODEL_NAME)
print("Device:", DEVICE)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded.")

print("\nLoading model...")

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)

print("Model loaded.")

sentences = [
    "I saw her duck, the bird.",
    "I saw her lower her head.",
    "The doctor examined the patient.",
    "I went to the bank yesterday.",
    "She is studying computer science."
]

forced_bos_token_id = tokenizer.convert_tokens_to_ids(
    TARGET_LANGUAGE
)

results = []

print("\nTranslating...\n")

for sentence in sentences:

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
            num_beams=5
        )

    translation = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )[0]

    results.append((sentence, translation))

    print("English:", sentence)
    print("Malayalam:", translation)
    print("-" * 70)


# ============================================================
# SAVE RESULTS AS UTF-8
# ============================================================

with open(
    "nllb_results.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("NLLB-200 TRANSLATION RESULTS\n")
    f.write("=" * 70 + "\n\n")

    for source, translation in results:

        f.write("English:\n")
        f.write(source)
        f.write("\n\n")

        f.write("Malayalam:\n")
        f.write(translation)
        f.write("\n\n")

        f.write("-" * 70)
        f.write("\n\n")


print("\nResults saved to:")
print("nllb_results.txt")

print("\nTEST COMPLETED")