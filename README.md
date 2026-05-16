# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

[WandB Report](https://wandb.ai/vgmasilamani61-indian-institute-of-technology-madras/da6401-a3/reports/ASSIGNMENT_3---VmlldzoxNjg5Mzk0NA?accessToken=zdsmozjdllohfz0mz26oew187plhm250c2rjcbdag0ulyy8khsk9dkrwkzk1r67i)


[GITHUB LINK](https://github.com/masilamani61/DL_assignement3)
## Overview


In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```
