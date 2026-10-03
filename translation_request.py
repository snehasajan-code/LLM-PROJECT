"""Shared semantic and result context for text and voice translation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranslationRequest:
    source_text: str
    source_language: str
    target_language: str
    context: str = "General"
    tone: str = "Neutral"
    ambiguity_detected: bool | None = None
    ambiguity_status: str = "unknown"
    ambiguity_options: list[dict[str, str]] = field(default_factory=list)
    selected_meaning: str | None = None
    semantic_constraint: str | None = None
    idiom_detected: bool = False
    idiom_interpretation: str | None = None
    idiom_result: dict[str, str] | None = None
    _idiom_checked: bool = field(default=False, repr=False)
    translation: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    verification_result: dict[str, Any] | None = None
    risk_result: dict[str, Any] | None = None
    repair_result: dict[str, Any] | None = None
    voice_confidence: float | None = None
    transcript: str | None = None
    detected_language: str | None = None

    @classmethod
    def from_ambiguity(cls, *args, ambiguity_result=None, **kwargs):
        request = cls(*args, **kwargs)
        request.record_ambiguity(ambiguity_result)
        return request

    @classmethod
    def for_voice(cls, transcript, source_language, target_language, **kwargs):
        """Create the same request used by text, with voice metadata attached."""
        confidence = kwargs.pop("voice_confidence", None)
        request = cls(
            source_text=transcript,
            source_language=source_language,
            target_language=target_language,
            transcript=transcript,
            detected_language=source_language,
            voice_confidence=confidence,
            **kwargs,
        )
        return request

    def record_ambiguity(self, result):
        if not isinstance(result, dict):
            self.ambiguity_status = "unknown"
            self.ambiguity_detected = None
            self.ambiguity_options = []
            return self

        if result.get("available") is False:
            status = "unknown"
        elif result.get("ambiguous") is True:
            status = "true"
        elif result.get("ambiguous") is False:
            status = "false"
        else:
            status = "unknown"

        self.ambiguity_status = status
        self.ambiguity_detected = True if status == "true" else False if status == "false" else None
        interpretations = result.get("interpretations") or []
        constraints = result.get("semantic_constraints") or []
        self.ambiguity_options = [
            {"meaning": str(meaning), "semantic_constraint": str(constraints[index]) if index < len(constraints) else ""}
            for index, meaning in enumerate(interpretations)
        ]
        return self

    def select_meaning(self, meaning, semantic_constraint=None):
        self.selected_meaning = meaning or None
        self.semantic_constraint = semantic_constraint or None
        if self.selected_meaning:
            self.ambiguity_status = "true"
            self.ambiguity_detected = True
        return self

    def ensure_idiom(self):
        """Populate the durable idiom fields using the existing detector once."""
        if not self._idiom_checked:
            from translator import interpret_idiom
            self.idiom_result = interpret_idiom(self.source_text, self.source_language)
            self._idiom_checked = True
            self.idiom_detected = self.idiom_result is not None
            self.idiom_interpretation = (
                self.idiom_result.get("meaning") if self.idiom_result else None
            )
        return self.idiom_result


def call_with_supported_kwargs(function, *args, **kwargs):
    """Call legacy extension points while passing fields they support."""
    import inspect

    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return function(*args, **kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in parameters}
    return function(*args, **accepted)
