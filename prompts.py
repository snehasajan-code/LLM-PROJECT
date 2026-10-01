def create_translation_prompt(
    text,
    source_language,
    target_language,
    context,
    tone,
    clarification=""
):
    prompt = f"""
You are an expert professional translator.

Your job is to translate the ORIGINAL TEXT naturally and accurately
from {source_language} to {target_language}.

IMPORTANT:
The translation must sound as if it was originally written by a
native speaker of the target language.

SOURCE LANGUAGE:
{source_language}

TARGET LANGUAGE:
{target_language}

CONTEXT:
{context}

TONE:
{tone}

USER'S CLARIFICATION:
{clarification}

TRANSLATION RULES:

1. Translate the complete meaning of the original text.

2. Do NOT translate word-by-word.

3. Do NOT add information that is not present in the original.

4. Do NOT remove information from the original.

5. Use the user's clarification to resolve ambiguity.

6. If the user selected a particular meaning, that meaning MUST be
   reflected in the final translation.

7. Produce natural, fluent and grammatically correct
   {target_language}.

8. Preserve the original tense, subject, object, relationships,
   numbers, names and important details.

9. Do not explain your translation.

10. Do not give alternatives.

11. Do not give transliteration.

12. Return ONLY the final translation.

SPECIAL RULES FOR MALAYALAM:

When translating into Malayalam:

- Use standard modern Malayalam.
- Use Malayalam script.
- Use natural Malayalam sentence structure.
- Do not translate English grammar literally.
- Do not unnecessarily keep English words.
- Do not use Malayalam transliteration.
- Do not produce awkward machine-translated Malayalam.
- Prefer expressions that a native Malayalam speaker would naturally use.
- Preserve the intended meaning rather than the English word order.

For example:

English:
Are you coming to college tomorrow?
I need to talk to you about our project.

Natural Malayalam:
നിങ്ങൾ നാളെ കോളേജിൽ വരുന്നുണ്ടോ?
നമ്മുടെ പ്രോജക്റ്റിനെക്കുറിച്ച് നിങ്ങളോട് സംസാരിക്കാനുണ്ട്.

Another example:

English:
I saw her duck.

If "duck" means the BIRD:
ഞാൻ അവളുടെ താറാവിനെ കണ്ടു.

If "duck" means LOWER HER HEAD:
അവൾ തല കുനിക്കുന്നത് ഞാൻ കണ്ടു.

IMPORTANT:
Do not confuse these two meanings.

The clarification provided by the user has priority when resolving
an ambiguous expression.

ORIGINAL TEXT:
{text}

FINAL TRANSLATION:
"""

    return prompt


def create_evaluation_prompt(
    original_text,
    translated_text,
    source_language,
    target_language,
    context,
    tone
):
    prompt = f"""
You are a professional translation quality evaluator.

Evaluate the translation by comparing it with the original text.

SOURCE LANGUAGE:
{source_language}

TARGET LANGUAGE:
{target_language}

CONTEXT:
{context}

EXPECTED TONE:
{tone}

ORIGINAL:
{original_text}

TRANSLATION:
{translated_text}

Evaluate:

1. Meaning preservation
2. Context preservation
3. Tone preservation
4. Naturalness and grammar

For Malayalam, pay special attention to:

- Natural Malayalam grammar
- Correct Malayalam vocabulary
- Correct word order
- Avoiding unnecessary English words
- Avoiding transliteration
- Whether the sentence sounds natural to a native Malayalam speaker
- Whether the intended meaning was preserved

Give each score from 1 to 10.

Return exactly:

Meaning: X/10
Context: X/10
Tone: X/10
Grammar: X/10
Overall: X/10

Comments:
<short explanation>
"""

    return prompt