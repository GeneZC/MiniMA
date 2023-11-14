# 🪄 Towards the Law of Capacity Gap in Distilling Language Models

<p align="center">
<img src="assets/logo.jpg" alt="logo" width="170" class="center"/><br>
Logo for MiniMA designed with DALL·E
</p>

📑 [arXiv](https://arxiv.org/abs/2311.07052) | 👻 [GitHub](https://github.com/GeneZC/MiniMA) | 🤗 [HuggingFace-MiniMA](https://huggingface.co/GeneZC/MiniMA-3B) | 🤗 [HuggingFace-MiniChat](https://huggingface.co/GeneZC/MiniChat-3B) | 🤖 [ModelScope-MiniMA](https://modelscope.cn/models/GeneZC/MiniMA-3B) | 🤖 [ModelScope-MiniChat](https://modelscope.cn/models/GeneZC/MiniChat-3B)

Language model (LM) distillation is a trending area that aims to distil the knowledge resided in a large teacher LM to a small student one. While various methods have been proposed to push the distillation to its limits, it is still a pain distilling LMs when a large capacity gap is exhibited between the teacher and the student LMs. The pain is mainly resulted by the curse of capacity gap, which describes that a larger teacher LM cannot always lead to a better student LM than one distilled from a smaller teacher LM due to the affect of capacity gap increment. That is, there is likely an optimal point yielding the best student LM along the scaling course of the teacher LM. Even worse, the curse of capacity gap can be only partly yet not fully lifted as indicated in previous studies.

However, the tale is not ever one-sided. Although a larger teacher LM has better performance than a smaller teacher LM, it is much more resource-demanding especially in the context of recent large LMs (LLMs). Consequently, instead of sticking to lifting the curse, leaving the curse as is should be arguably fine. Even better, in this paper, we reveal that the optimal capacity gap is almost consistent across different student scales and architectures, fortunately turning the curse into the law of capacity gap. The law later guides us to distil a 3B student LM (termed MiniMA) from a 7B teacher LM (adapted LLaMA2-7B). MiniMA is demonstrated to yield a new compute-performance pareto frontier among existing 3B LMs on commonly used benchmarks, and its instruction-tuned version (termed MiniChat) outperforms a wide range of 3B competitors in GPT4 evaluation and could even compete with several 7B chat models. 

<img src="./assets/teaser_a.jpg" alt="teaser_a" width="400" /> <img src="./assets/teaser_b.jpg" alt="teaser_b" width="387"/>

## 🔗 Quick Links

- [Updates](#updates)
- [Quick Start](#quick-start)
- [Tutorials](#tutorials)
- [Future Work](#future-work)
- [Bugs or Questions?](#bugs-or-questions)
- [Citation](#citation)

## 🕰️ Updates

[2023/11/14] For models <= 3B on [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), MiniMA surpasses ShearedLLaMA-2.7B and ranks #14, and MiniChat (potentially with data contamination) surpasses Phi1.5 and ranks #2.

<img src="./assets/llmboard.png" alt="llmboard" width="700" />

[2023/11/14] We have released the paper, uploaded the checkpoints, and updated the codebase.

## ✨ Quick Start

Since MiniMA and MiniChat follow architecture of LLaMA, they can be seamlessly used in HuggingFace Transformers with minimal requirements.

The following is an example code snippet to use MiniMA and MiniChat:

```python
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

# MiniMA
tokenizer = AutoTokenizer.from_pretrained("GeneZC/MiniMA-3B", use_fast=False)
# GPU.
model = AutoModelForCausalLM.from_pretrained("GeneZC/MiniMA-3B", use_cache=True, device_map="auto", torch_dtype=torch.float16).eval()
# CPU.
# model = AutoModelForCausalLM.from_pretrained("GeneZC/MiniMA-3B", use_cache=True, device_map="cpu", torch_dtype=torch.float16).eval()

prompt = "Question: Sherrie tells the truth. Vernell says Sherrie tells the truth. Alexis says Vernell lies. Michaela says Alexis tells the truth. Elanor says Michaela tells the truth. Does Elanor tell the truth?\nAnswer: No\n\nQuestion: Kristian lies. Sherrie says Kristian lies. Delbert says Sherrie lies. Jerry says Delbert tells the truth. Shalonda says Jerry tells the truth. Does Shalonda tell the truth?\nAnswer: No\n\nQuestion: Vina tells the truth. Helene says Vina lies. Kandi says Helene tells the truth. Jamey says Kandi lies. Ka says Jamey lies. Does Ka tell the truth?\nAnswer: No\n\nQuestion: Christie tells the truth. Ka says Christie tells the truth. Delbert says Ka lies. Leda says Delbert tells the truth. Lorine says Leda tells the truth. Does Lorine tell the truth?\nAnswer:"
input_ids = tokenizer([prompt]).input_ids
output_ids = model.generate(
    torch.as_tensor(input_ids).cuda(),
    do_sample=True,
    temperature=0.7,
    max_new_tokens=1024,
)
output_ids = output_ids[0][len(input_ids[0]):]
output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
# output: "No"

from conversation import get_default_conv_template

# MiniChat
tokenizer = AutoTokenizer.from_pretrained("GeneZC/MiniChat-3B", use_fast=False)
# GPU.
model = AutoModelForCausalLM.from_pretrained("GeneZC/MiniChat-3B", use_cache=True, device_map="auto", torch_dtype=torch.float16).eval()
# CPU.
# model = AutoModelForCausalLM.from_pretrained("GeneZC/MiniChat-3B", use_cache=True, device_map="cpu", torch_dtype=torch.float16).eval()

conv = get_default_conv_template("minichat")

question = "Implement a program to find the common elements in two arrays without using any extra data structures."
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer([prompt]).input_ids
output_ids = model.generate(
    torch.as_tensor(input_ids).cuda(),
    do_sample=True,
    temperature=0.7,
    max_new_tokens=1024,
)
output_ids = output_ids[0][len(input_ids[0]):]
output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
# output: "def common_elements(arr1, arr2):\n    if len(arr1) == 0:\n        return []\n    if len(arr2) == 0:\n        return arr1\n\n    common_elements = []\n    for element in arr1:\n        if element in arr2:\n            common_elements.append(element)\n\n    return common_elements"
# Multiturn conversation could be realized by continuously appending questions to `conv`.
```

## 📖 Tutorials

The reproductions of MiniMA and MiniChat follow several stages including adaptation of LLaMA2, distillation to MiniMA, and finetuning to MiniChat.

Detailed tutorials can be found [here](./TUTORIAL.md).

## 🔮 Future Work

- A more diverse blend of data sources, e.g., Chinese wikipedia, books, etc.
- Smaller models, e.g., 1.2B model, say MicroMA.
- Combined with preference optimization, e.g., DPO to MiniChat.
- Designed for speculative sampling, e.g., MiniMA to LLaMA for inference efficiency.

## 🤔️ Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email Chen (chenzhang9702@outlook.com). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## ✍🏻 Citation

Please cite our paper if you find the repo helpful in your work:

```bibtex
@article{zhang2023law,
    title={Towards the Law of Capacity Gap in Distilling Language Models},
    author={Zhang, Chen and Song, Dawei and Ye, Zheyu and Gao, Yan},
    year={2023},
    url={https://arxiv.org/abs/2311.07052}
}
```

