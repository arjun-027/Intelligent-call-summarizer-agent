"""LLM-based call summarisation: prompt, client construction, and generation."""

import logging
import random
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

CHAR_LIMIT = 1500

# ── LLM retry constants ───────────────────────────────────────────────────────
# Retries apply only to *transient* Groq errors (rate limit 429, gateway 502/503/504,
# timeouts, connection resets).  Permanent errors (bad API key 401, invalid request 400)
# are not retried — they would fail identically on every attempt.
_MAX_LLM_RETRIES: int = 3
_RETRY_BASE_DELAY: float = 1.0   # seconds; doubles each attempt (exponential backoff)
_RETRY_JITTER: float = 0.5       # seconds; uniform random added to prevent thundering herd

# Substrings that identify a transient (retryable) Groq error.
# We match against the stringified exception rather than importing the groq
# package directly, so this works regardless of how langchain-groq wraps errors.
_RETRYABLE_SIGNALS: frozenset[str] = frozenset({
    "429",            # HTTP Too Many Requests — rate limit hit
    "rate_limit",     # Groq RateLimitError message text
    "502",            # HTTP Bad Gateway — upstream transient
    "503",            # HTTP Service Unavailable
    "504",            # HTTP Gateway Timeout
    "timeout",        # Network / socket timeout
    "connection",     # Connection reset / refused
})

SYSTEM_PROMPT = """You are an expert insurance claims call summariser. Produce a precise, structured summary from the transcript below.

REQUIRED OUTPUT FORMAT (total output must be ≤ 1,500 characters):

Caller: [Name if known], [relationship], [inbound/outbound]

Subject:
[One-line description of what the call was about]

Executive Summary:
[One paragraph explaining what happened on the call and why]
- [Key fact]
- [Key fact]
- [Additional facts as needed]

Next Steps:
[COMPANY]: [Action to be taken, or "None"]
Other: [Action by other parties, or "None"]

CONDITIONAL SECTIONS — only include a section if that topic was explicitly discussed on the call. Do NOT write empty or "None" versions of these; omit the section entirely.

Liability Summary:
[Include only if liability was discussed]

Negotiation Summary:
[Include only if negotiation occurred]

Vehicle Damage:
Vehicle Status: [Status]
Towage: [Details or "None"]
Car hire: [Details or "None"]

Injury:
Treatment: [Details]

Property:
[Details]

STRICT RULES:
1. Caller relationship must accurately reflect their role: policyholder, third party representative, solicitor, insurance company representative, or family member.
2. In Next Steps, replace [COMPANY] with the actual company name from the transcript.
3. NEVER include a conditional section (Liability, Negotiation, Vehicle Damage, Injury, Property) unless that topic was explicitly discussed. Omit — do not write "None".
4. NEVER invent or guess facts. Extract only what was explicitly stated.
5. Reproduce critical facts exactly: amounts, dates, claim/reference numbers, IBANs, email addresses, phone numbers.
6. Caller relationship describes their relationship to the CLAIM, not to the agent.
7. If the caller represents another insurer or a solicitor's firm, state that specifically.
8. Total summary must be ≤ 1,500 characters including all whitespace.
9. Never include Note in the response like Note: The conditional sections (Liability, Negotiation, Vehicle Damage, Injury, Property) were not discussed on the call, so they are omitted."""


def build_llm(api_key: str, model: str) -> ChatGroq:
    """Construct and return a configured Groq LLM client.

    Args:
        api_key: Groq API key for authentication.
        model: Groq model identifier (e.g. ``"llama-3.1-8b-instant"``).

    Returns:
        A :class:`~langchain_groq.ChatGroq` instance ready for inference.
    """
    logger.debug("Building Groq LLM client — model: %s", model)
    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=0.1,
        max_tokens=600,
    )


def build_messages(transcript_content: str, prompt_addendum: str = "") -> list:
    """Construct the message list to send to the LLM.

    Separating message construction from the API call makes it independently
    testable without requiring a live Groq connection.

    Args:
        transcript_content: Raw transcript text to be summarised.
        prompt_addendum: Optional additional instructions appended to the system
            prompt.  Used by the retry loop to inject corrective guidance after
            guardrail failures.

    Returns:
        A list containing the system prompt message and the user message.
    """
    system_content = SYSTEM_PROMPT
    if prompt_addendum:
        system_content += f"\n\nADDITIONAL CONSTRAINTS FOR THIS ATTEMPT:\n{prompt_addendum}"
    return [
        SystemMessage(content=system_content),
        HumanMessage(content=f"Summarise this call transcript:\n\n{transcript_content}"),
    ]


def generate_summary(
    transcript_content: str,
    llm: ChatGroq,
    prompt_addendum: str = "",
) -> str:
    """Call the LLM with the transcript and return the generated summary.

    Args:
        transcript_content: Raw transcript text to be summarised.
        llm: A configured :class:`~langchain_groq.ChatGroq` instance.
        prompt_addendum: Optional corrective instructions added to the system
            prompt on retry attempts (produced by
            :func:`~call_summarizer.guardrails.build_retry_prompt_addendum`).

    Returns:
        The summary string produced by the model, stripped of leading/trailing
        whitespace.

    Raises:
        RuntimeError: If the LLM API call fails for any reason.
    """
    logger.debug(
        "Sending transcript to LLM (%d chars, addendum: %s)",
        len(transcript_content),
        bool(prompt_addendum),
    )
    messages = build_messages(transcript_content, prompt_addendum)

    for attempt in range(_MAX_LLM_RETRIES + 1):
        try:
            response = llm.invoke(messages)
            summary = response.content.strip()
            logger.info("Summary generated — %d chars (attempt %d)", len(summary), attempt + 1)
            return summary
        except Exception as exc:
            exc_str = str(exc).lower()
            is_retryable = any(sig in exc_str for sig in _RETRYABLE_SIGNALS)

            if is_retryable and attempt < _MAX_LLM_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, _RETRY_JITTER)
                logger.warning(
                    "LLM transient error on attempt %d/%d — retrying in %.1fs: %s",
                    attempt + 1,
                    _MAX_LLM_RETRIES + 1,
                    delay,
                    exc,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "LLM call failed on attempt %d/%d (retryable=%s): %s",
                    attempt + 1,
                    _MAX_LLM_RETRIES + 1,
                    is_retryable,
                    exc,
                )
                raise RuntimeError(f"LLM call failed: {exc}") from exc

    raise RuntimeError("LLM call failed: exceeded maximum retries")


def validate_summary_length(summary: str, limit: int = CHAR_LIMIT) -> bool:
    """Return True if *summary* is within the character *limit*.

    Args:
        summary: The generated summary text to check.
        limit: Maximum allowed character count (default: :data:`CHAR_LIMIT`).

    Returns:
        ``True`` if ``len(summary) <= limit``, ``False`` otherwise.
    """
    within_limit = len(summary) <= limit
    if not within_limit:
        logger.warning(
            "Summary exceeds character limit: %d chars (limit %d)", len(summary), limit
        )
    return within_limit
