# Copilot Instructions

## Project

Text Generation and Reasoning with LLMs — A personal project exploring decoding algorithms, prompt engineering, and agent-based reasoning. Main notebook: `text_generation_and_agents.ipynb` (and helper files like `eval_utils.py`).

## Environment

```bash
pip install vllm transformers datasets evaluate tqdm torch matplotlib pandas numpy wikipedia openai
```

- Primary model: `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` served locally via vLLM (CUDA required).
- Evaluation models: `gpt2` (perplexity), `textattack/roberta-base-CoLA` (fluency).

## Section 1 — Decoding Algorithms

The shared `decode(prompts, max_len, method, **kwargs)` loop handles batching, EOS tracking, and padding. **Only implement the method functions** — do not modify `decode()`.

All method signatures: `(next_token_logits: Tensor[B, V], ...) -> LongTensor[B]`

| Function | Signature | Notes |
|----------|-----------|-------|
| `greedy` | `(next_token_logits)` | `argmax` over vocab dim |
| `sample` | `(next_token_logits)` | Sample from `softmax` distribution |
| `temperature` | `(next_token_logits, t: float)` | Scale logits by `1/t` before softmax |
| `topk` | `(next_token_logits, k: int)` | Zero out all but top-k logits, then sample |
| `topp` | `(next_token_logits, p: float)` | Nucleus sampling — cumulative prob ≤ p |

**Section 1 Temperature Sweep**: Explore the trade-offs of `temperature()` across `t ∈ [0.3, 0.5, 0.8, 1.0, 1.5]`. Analyze perplexity, diversity, and repetition metrics to understand temperature effects on generation quality.

## Section 2 — Few-Shot Prompting

- `VLLMClient.__call__(prompt, **kwargs)` wraps `llm.generate()` — pass **raw string prompts**, not chat templates.
- `ARC_EXAMPLARS` (given) has 8 exemplars with `question`, `choices`, `short_answer`, and `cot_answer` fields.
- `action_type` values written to output files: `"fewshot"` and `"fewshot_cot"`.

| Class | Method | Task |
|-------|--------|------|
| `FewShotReasoner` | `build_input(question) -> str` | 8-shot prompt using `short_answer` |
| `FewShotCoTReasoner` | `build_input(question) -> str` | 8-shot CoT prompt using `cot_answer` |
| `FewShotCoTReasoner` | `extract_answer(response) -> str` | Extract letter answer from CoT response |

ARC choices are labeled A/B/C/D — the prompt should present all choices and instruct the model to pick one letter.

## Section 3 — ReAct Agent (MATH + GAIA)

All agents expose `.run(task: str) -> str` returning the final answer string.

- `ChatAgent` (GIVEN): uses `tokenizer.apply_chat_template()` before calling the model. `action_type = "vanilla"`.
- `ReActAgent` (TODO): constructs raw text prompts for the ReAct loop. `action_type = "react"`.

**Tool interface:** `tool(query: str) -> str`. Tools have `.name: str` and `.description: str` attributes.

| Class | Method | Notes |
|-------|--------|-------|
| `WikiSearchTool` | `__call__(query)` | Use `wikipedia.summary(query, sentences=3)`; handle `DisambiguationError` / `PageError` gracefully |
| `CalculatorTool` | `__call__(expression)` | Safe evaluation via `ast` module only — **never `eval()`** |
| `ReActAgent` | `build_system_prompt()` | Include tool names/descriptions and the ReAct format below |
| `ReActAgent` | `parse_action(text) -> (action, action_input)` | Extract `Action:` and `Action Input:` lines |
| `ReActAgent` | `run(question, max_steps=8) -> str` | ReAct loop; return `action_input` on `finish` |

**ReAct prompt format** (used verbatim in prompts and expected from the model):
```
Thought: [reasoning]
Action: [tool_name or finish]
Action Input: [query or final answer]
Observation: [tool result — appended by the agent]
```
Stopping: when `Action: finish`, return `Action Input` as the answer. If `max_steps` reached without finish, return `None` or the last response.

## Key Conventions

**Output files** — JSONL, one record per question:
```
output/{model_id.replace('/', '__')}__{action_type}__{task}.jsonl
```
Fields per line: `question`, `answer`, `true_answer`, `source`, `model_id`, `agent_action_type`.

**`answer_questions(task, agent, action_type, answers_file)`** skips already-answered questions (checked by `question` field) — re-runs are safe to resume.

**Scoring** via `eval_utils.score_answers(answers_files)` → DataFrame:
- `source ∈ {GSM8K, MATH, ARC}` → extract last number, compare as float (`rtol=1e-5, atol=1e-7`)
- `source ∈ {SimpleQA, GAIA}` → fuzzy string matching via `get_question_score()`

**Helper code blocks** are marked and used as-is:
```python
######################################################
#  Helper code and utilities
######################################################
```

**Random seed** (`set_seed(19260817)`) is fixed for reproducibility.

## Running / Testing

Run top-to-bottom in a GPU environment. Test a single decoding method:
```python
show_generations('Method Name', method_fn, n=3, max_len=100, **kwargs)
```

Score output files:
```python
from eval_utils import score_answers
df = score_answers(["output/some_file.jsonl"])
print(df)
```
