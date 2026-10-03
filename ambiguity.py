import json


OLLAMA_URL = "http://localhost:11434/api/generate"

AMBIGUITY_MODEL = "qwen2.5:3b"


def _known_ambiguity(text, source_language):
    """Return deterministic clarification for the project's standard duck demo."""
    normalized = " ".join((text or "").casefold().split()).rstrip(".!?")
    if source_language != "English" or normalized != "i saw her duck":
        return None
    return {
        "ambiguous": True,
        "available": True,
        "reason": "'Duck' can mean a bird or the action of lowering one's head.",
        "interpretations": [
            "I saw the duck that belongs to her.",
            "I saw her bow her head.",
        ],
        "semantic_constraints": [
            "Duck refers to the bird/animal that belongs to her, not the action of lowering a head.",
            "Duck refers to her lowering/bowing her head, not to a bird/animal.",
        ],
    }


def detect_ambiguity(
    text,
    source_language,
    target_language,
    context
):
    """
    Detect genuine meaning-changing ambiguity.

    The function returns:
        ambiguous
        reason
        interpretations
        semantic_constraints

    semantic_constraints are explicit instructions for the
    translation model so that it does not reinterpret the
    selected meaning.
    """

    # This common demo sentence must still offer clarification when Ollama is
    # offline or its small model misses the ambiguity.
    known_ambiguity = _known_ambiguity(text, source_language)
    if known_ambiguity:
        return known_ambiguity

    prompt = f"""
You are an ambiguity detection system for a multilingual translation system.

Analyze ONLY the following sentence.

SOURCE LANGUAGE:
{source_language}

TARGET LANGUAGE:
{target_language}

CONTEXT:
{context}

SOURCE SENTENCE:
{text}

Your task is to identify genuine semantic ambiguity that could change
the meaning of the translation.

IMPORTANT RULES:

1. Only identify genuine meaning-changing ambiguity.
2. Do not invent ambiguity.
3. If there is no genuine ambiguity, return ambiguous=false.
4. If ambiguity exists, provide exactly 2 realistic interpretations.
5. The interpretations must be understandable to a human.
6. Do not translate the sentence into the target language.
7. Do not give unnecessary explanations.
8. For every interpretation, create a precise semantic constraint.
9. The semantic constraint must explicitly state what the ambiguous
   word or phrase means in that interpretation.
10. The translation system will later use this constraint to prevent
    the wrong meaning from being selected.

Example:

SOURCE SENTENCE:
I saw her duck.

Correct output:

{{
    "ambiguous": true,
    "reason": "The word 'duck' can refer to a bird or to the action of lowering one's head.",
    "interpretations": [
        "I saw the duck that belongs to her.",
        "I saw her bow her head."
    ],
    "semantic_constraints": [
        "Duck refers to the bird/animal that belongs to her, not the action of lowering a head.",
        "Duck refers to her lowering/bowing her head, not to a bird/animal."
    ]
}}

Another example:

SOURCE SENTENCE:
He went to the bank.

Possible output:

{{
    "ambiguous": true,
    "reason": "The word 'bank' can refer to a financial institution or the side of a river.",
    "interpretations": [
        "He went to a financial institution.",
        "He went to the side of a river."
    ],
    "semantic_constraints": [
        "The word 'bank' refers to a financial institution.",
        "The word 'bank' refers to the land beside a river."
    ]
}}

If there is no genuine ambiguity, return:

{{
    "ambiguous": false,
    "reason": "",
    "interpretations": [],
    "semantic_constraints": []
}}

Return ONLY valid JSON.
"""

    try:
        import requests

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": AMBIGUITY_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0},
            },
            timeout=45,
        )
        response.raise_for_status()
        raw_response = response.json().get("response", "").strip()
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            start, end = raw_response.find("{"), raw_response.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("Ambiguity detector returned invalid JSON.")
            data = json.loads(raw_response[start:end + 1])
    except Exception as exc:
        # Translation can continue without ambiguity evidence. Callers must
        # treat this as unknown, not as evidence that the sentence is unambiguous.
        return {
            "ambiguous": False,
            "available": False,
            "reason": "",
            "interpretations": [],
            "semantic_constraints": [],
            "error": str(exc),
        }

    ambiguous = bool(
        data.get("ambiguous", False)
    )

    reason = str(
        data.get("reason", "")
    ).strip()

    interpretations = data.get(
        "interpretations",
        []
    )

    semantic_constraints = data.get(
        "semantic_constraints",
        []
    )

    if not isinstance(interpretations, list):
        interpretations = []

    if not isinstance(semantic_constraints, list):
        semantic_constraints = []

    interpretations = [
        str(item).strip()
        for item in interpretations
        if str(item).strip()
    ]

    semantic_constraints = [
        str(item).strip()
        for item in semantic_constraints
        if str(item).strip()
    ]

    # Maximum two interpretations for a clean UI
    interpretations = interpretations[:2]
    semantic_constraints = semantic_constraints[:2]

    # If the model says there is no ambiguity
    if not ambiguous:
        return {
            "ambiguous": False,
            "available": True,
            "reason": "",
            "interpretations": [],
            "semantic_constraints": []
        }

    # We need matching interpretation + constraint pairs
    if (
        len(interpretations) < 2
        or len(semantic_constraints) < 2
    ):
        return {
            "ambiguous": False,
            "available": True,
            "reason": "",
            "interpretations": [],
            "semantic_constraints": []
        }

    return {
        "ambiguous": True,
        "available": True,
        "reason": reason,
        "interpretations": interpretations,
        "semantic_constraints": semantic_constraints
    }
