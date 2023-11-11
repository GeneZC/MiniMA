# Towards the Law of Capacity Gap in Distilling Language Models

<img src="assets/logo.jpg" alt="logo" width="400"/>

📑 [arXiv]() | 🤗 [HuggingFace]() ｜ 🤖 [ModelScope]()

Thank you for your interest in our work! This is a joint work by [Mengzhou Xia](https://xiamengzhou.github.io/), [Tianyu Gao](https://gaotianyu.xyz/about/), [Zhiyuan Zeng](https://scholar.google.com/citations?user=qLJqCqsAAAAJ&hl=en), and [Danqi Chen](https://www.cs.princeton.edu/~danqic/). Here, we provide our codebase for Sheared-LLaMA's pruning and continued pre-training algorithms :) We find that pruning strong base models is an extremely cost-effective way to get strong small-scale language models compared to pre-training them from scratch. The following graph shows that given the existence of Llama-2-7B model (pre-trained with 2T tokens), pruning it produces a model as strong as an OpenLLaMA model with 3% of its pre-training cost. 

<img src="images/teaserwlegend.jpg" alt="teaser" width="400" />


## 🔗 Quick Links

- [Updates](#updates)
- [Quick Start](#quick-start)
- [Tutorials](#tutorials)
  - [Adaptation](#adaptation)
  - [Distillation](#distillation)
  - [Finetuning](#finetuning)
  - [Evaluation](#evaluation)
- [Future Work](#future-work)
- [Bugs or Questions?](#bugs-or-questions)
- [Citation](#citation)

## Updates

## Quick Start

## Tutorials

### Adaptation

### Distillation

### Finetuning

### Evaluation

## Future Work

## Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email Chen (chenzhang9702@outlook.com). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!


## Citation

Please cite our paper if you find the repo helpful in your work:

```bibtex
@article{zhang2023law,
    title={Towards the Law of Capacity Gap in Distilling Language Models},
    author={Zhang, Chen and Song, Dawei and Ye, Zheyu and Gao, Yan},
    year={2023},
    url={}
}
```

