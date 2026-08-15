"""LLM backends behind one interface, plus prompt templates and JSON recovery.

  HFLocalLLM      - local HuggingFace causal LM (departmental GPU or Colab).
  OpenAICompatLLM - any OpenAI-compatible endpoint, including a local vLLM or Ollama
                    server. Useful when GPU memory will not hold a 7B model.

Every call is counted, so cost can be reported alongside accuracy.
"""
from __future__ import annotations

import json
import re
from typing import Any, Sequence

from ..utils.schema import RetrievedDoc

PROMPT_VERSION = "v1"

GENERATOR_SYSTEM = (
    "You answer questions strictly from the provided evidence passages. "
    "You never rely on prior knowledge. Every sentence you write must cite the "
    "passage that supports it. If the evidence is insufficient, say so explicitly."
)

GENERATOR_TEMPLATE = """Evidence passages:
{evidence}

Question: {question}

Write a short answer (1-3 sentences) using only the passages above.
Cite the supporting passage after each sentence using its identifier in square
brackets, e.g. [{example_id}]. If the passages do not contain the answer, reply
exactly: INSUFFICIENT EVIDENCE

Answer:"""

INSUFFICIENT = "INSUFFICIENT EVIDENCE"

FACT_VERIFICATION_SYSTEM = (
    "You judge whether a claim is supported by the provided evidence passages. "
    "You never rely on prior knowledge. You output your verdict first, then a short "
    "justification that cites the passage identifiers."
)

FACT_VERIFICATION_TEMPLATE = """Evidence passages:
{evidence}

Claim: {question}

Decide whether the evidence supports or refutes the claim above.
Reply on the first line with exactly one word: SUPPORTS or REFUTES.
On the next line, write a short justification (1-2 sentences) citing the
supporting passage identifier(s) in square brackets, e.g. [{example_id}].
If the evidence does not address the claim at all, reply exactly: INSUFFICIENT EVIDENCE

Verdict:"""


def extract_verdict(answer: str) -> str:
    """First line of a FACT_VERIFICATION_TEMPLATE answer, normalised to
    SUPPORTS/REFUTES/INSUFFICIENT EVIDENCE/UNKNOWN."""
    first_line = answer.strip().splitlines()[0].strip().upper() if answer.strip() else ""
    if first_line.startswith(INSUFFICIENT):
        return INSUFFICIENT
    if "REFUTE" in first_line:
        return "REFUTES"
    if "SUPPORT" in first_line:
        return "SUPPORTS"
    return "UNKNOWN"


DECOMPOSE_SYSTEM = (
    "You split text into atomic factual claims. Each claim must be a single, "
    "self-contained, checkable statement. You output JSON only."
)

DECOMPOSE_TEMPLATE = """Split the following answer into atomic factual claims.
Resolve pronouns so each claim stands alone. Ignore hedges and citation markers.

Answer: {answer}

Return JSON only, in this exact shape:
{{"claims": ["<claim 1>", "<claim 2>"]}}"""

VERIFIER_SYSTEM = (
    "You are a strict evidence verifier. You judge whether each claim is entailed "
    "by the evidence passages. Absence of evidence means unsupported, never "
    "supported. You output JSON only."
)

VERIFIER_TEMPLATE = """Evidence passages:
{evidence}

Claims to verify:
{claims}

For each claim decide:
  "supported"   - the passages entail the claim
  "partial"     - partly entailed, or entailed only with an unstated assumption
  "unsupported" - not entailed, contradicted, or absent from the passages

Return JSON only, in this exact shape:
{{"verdicts": [
  {{"claim": "<claim text>", "verdict": "supported|partial|unsupported",
    "doc_ids": ["<identifier of each passage that supports it>"],
    "rationale": "<one short sentence>"}}
]}}"""

HOLISTIC_VERIFIER_SYSTEM = (
    "You are a strict evidence verifier and answer corrector. You never use outside "
    "knowledge, never guess, and never fabricate missing facts. You output JSON only."
)

HOLISTIC_VERIFIER_TEMPLATE = """Evidence passages:
{evidence}

Question: {question}

Initial answer:
{initial_answer}

1. Compare the initial answer against the retrieved context.
2. Verify every factual claim.
3. Determine whether the claim(s) are:
   - SUPPORTED: Directly supported by the retrieved context.
   - CONTRADICTED: Directly disproved by the retrieved context.
   - UNSUPPORTED: No evidence exists in the retrieved context.
   - AMBIGUOUS: Evidence is incomplete or conflicting.

4. Assign a confidence score:
   - 1.00 = Fully supported or fully contradicted by clear evidence.
   - 0.70-0.99 = Mostly supported.
   - 0.40-0.69 = Partially supported or ambiguous.
   - 0.00-0.39 = Unsupported.

5. Assign a hallucination score:
   Hallucination = 1 - Confidence

6. Produce the final answer:
   - If the claim is SUPPORTED, answer normally.
   - If the claim is CONTRADICTED, clearly state the correct information from the context.
   - If the claim is UNSUPPORTED, respond:
     "Insufficient evidence in the retrieved documents."
   - If the claim is AMBIGUOUS, explain that the retrieved evidence is conflicting or incomplete.

Never use outside knowledge.
Never guess.
Never fabricate missing facts.

Return ONLY the following JSON:
{{
  "evidence_status": "",
  "confidence": 0.00,
  "hallucination": 0.00,
  "reason": "",
  "final_answer": ""
}}"""

REFORMULATE_SYSTEM = (
    "You write search queries that find missing evidence. You output JSON only."
)

REFORMULATE_TEMPLATE = """Original question: {question}

These claims could not be supported by the evidence retrieved so far:
{unsupported}

Passages already retrieved (do not target these again):
{seen_titles}

Write up to {n} short search queries likely to retrieve the missing evidence.
Target the specific entities, relations and dates that are unverified.

Return JSON only, in this exact shape:
{{"queries": ["<query 1>", "<query 2>"]}}"""


def format_evidence(docs: Sequence[RetrievedDoc], max_chars: int = 700) -> str:
    """Render passages with stable identifiers the model can cite back."""
    if not docs:
        return "(no passages retrieved)"
    blocks = []
    for d in docs:
        text = d.text if len(d.text) <= max_chars else d.text[: max_chars - 1] + "\u2026"
        head = f"[{d.doc_id}]" + (f" {d.title}" if d.title else "")
        blocks.append(f"{head}\n{text}")
    return "\n\n".join(blocks)


class BaseLLM:
    """Text in, text out."""

    def __init__(self) -> None:
        self.n_calls = 0

    def complete(self, prompt: str, system: str | None = None,
                 max_tokens: int = 512, temperature: float = 0.0) -> str:
        self.n_calls += 1
        return self._complete(prompt, system, max_tokens, temperature)

    def _complete(self, prompt: str, system: str | None,
                  max_tokens: int, temperature: float) -> str:
        raise NotImplementedError


class HFLocalLLM(BaseLLM):
    """Local HuggingFace causal LM. Greedy by default so runs are reproducible."""

    def __init__(self, model_id: str, device_map: str = "auto",
                 dtype: str = "auto", load_in_4bit: bool = False) -> None:
        super().__init__()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("pip install torch transformers accelerate") from exc

        kwargs: dict[str, Any] = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["torch_dtype"] = dtype

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()

    def _complete(self, prompt: str, system: str | None,
                  max_tokens: int, temperature: float) -> str:
        import torch

        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()


class OpenAICompatLLM(BaseLLM):
    """Any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY") -> None:
        super().__init__()
        import os

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install openai") from exc

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "EMPTY"))

    def _complete(self, prompt: str, system: str | None,
                  max_tokens: int, temperature: float) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            max_tokens=max_tokens, temperature=temperature)
        return (resp.choices[0].message.content or "").strip()


def build_llm(spec: str, **kwargs: Any) -> BaseLLM:
    """`hf:<model_id>` | `openai:<model>`."""
    if spec.startswith("hf:"):
        return HFLocalLLM(spec[3:], **kwargs)
    if spec.startswith("openai:"):
        return OpenAICompatLLM(spec[7:], **kwargs)
    raise ValueError(f"unknown LLM spec: {spec!r}")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str, default: Any = None) -> Any:
    """Pull the first JSON object or array out of a model response.

    Small models wrap JSON in prose or code fences and emit trailing commas;
    recovering here rather than discarding the response materially reduces the
    failure rate of any JSON-producing stage.
    """
    if not text:
        return default
    candidates: list[str] = []
    m = _FENCE.search(text)
    if m:
        candidates.append(m.group(1).strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start:end + 1])
    candidates.append(text.strip())
    for cand in candidates:
        for attempt in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, ValueError):
                continue
    return default
