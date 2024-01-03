
# 💬 MT-Bench-ZH

👻 [GitHub](https://github.com/GeneZC/MiniMA/tree/main/mt_bench_zh) | 🤗 [HuggingFace-MiniMA](https://huggingface.co/datasets/GeneZC/MT-Bench-ZH)

## 🎯 Motivation

MiniChat-1/1.5/2-3B are all instruction-following language models that could handle Chinese instructions, however, there is currently no instruciton-following benchamrk specialized for Chinese. Due to this, our previous evaluation has been limited to English-only benchmarks (i.e., AlpacaEval and MT-Bench). 

To this demand, MT-Bench-ZH is made to mitigate this. MT-Bench-ZH is basically translated from MT-Bench-ZH by GPT-4 and further checked by human. Hopefully, MT-Bench-ZH could help the communnity to develop better instruction-following language models that are able to tackle Chinese instructions.

## 🚀 Quick Start

> [!NOTE]  
> The code is either copied or modified from [FastChat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge), yet we currently only support `single` mode judgment.
> Please refer to FastChat for more details.

### Install FastChat

```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[model_worker,webui]"
```

### Generate Responses 

```bash
python gen_model_answer.py --model-path GeneZC/MiniChat-2-3B --model-id minichat --bench-name mt_bench_zh --max-new-token 1536
```

### Evaluate Responses

```bash
export OPENAI_API_KEY=XXXXXX  # Set the OpenAI API key.
python gen_judgment.py --model-list minichat --bench-name mt_bench_zh --judge-file data/judge_prompts_zh.jsonl --parallel 4
```

### Display Results

```bash
python show_result.py --bench-name mt_bench_zh
```

## 🏆 Leaderboard

|Method|MT-Bench-ZH|
|--|--|
|🥇 GPT-4|8.96|
|🥈 Zephyr-7B-Beta|6.27<sup>#</sup>|
|🥉 Qwen-Chat-7B|6.24|
|MiniChat-2-3B|6.04|
|Qwen-Chat-1.8B|5.65|
|LLaMA-2-Chat-7B|5.43<sup>#</sup>|
|Vicuna-7B|5.22<sup>#</sup>|
|StableLM-Zephyr-3B|4.31<sup>#</sup>|
|Rocket-3B|4.07<sup>#</sup>|
|Phi-2-DPO|1.59<sup>#</sup><sup>$</sup>|

<sup>#</sup> specialized mainly for English.

<sup>$</sup> finetuned without multi-turn instruction data.

## 🙌 Contributions

You can raise questions related to the benchmark by opening an issue. Or you can add results of other models to the leaderboard by opening a pull request. For the leaderboard, related files should be attached for sanity check (i.e., a separate model response file should be uploaded, and the GPT-4 judgement file should be updated).