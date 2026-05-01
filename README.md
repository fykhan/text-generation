# Text Generation and Reasoning with LLMs

Exploring decoding algorithms, few-shot prompting, and reasoning agents with language models.

## Contents

**1. Decoding Algorithms**
- Greedy decoding (argmax selection)
- Vanilla sampling and temperature scaling
- Top-k sampling (sample from top-k tokens)
- Nucleus sampling / top-p (cumulative probability threshold)
- Temperature sweep analysis for diversity and quality trade-offs

**2. Few-Shot Prompting**
- Few-shot learning with in-context examples
- Chain-of-Thought (CoT) reasoning for complex tasks
- Evaluation on multiple-choice and reasoning datasets

**3. ReAct Agents**
- Reasoning + Acting framework for multi-step problems
- Tool access (Wikipedia search, calculator)
- Structured thought-action-observation loops
- Handling tool results and error cases

## Setup

```bash
pip install vllm transformers datasets evaluate tqdm torch matplotlib pandas numpy wikipedia openai
```

## Run

```bash
jupyter notebook text_generation_and_agents.ipynb
```

## Dependencies

- **Model**: Qwen2.5-7B via vLLM (CUDA required)
- **Evaluation**: gpt2 (perplexity), RoBERTa (fluency)
- **Datasets**: ROCStories, ARC-Challenge, MATH, GAIA
