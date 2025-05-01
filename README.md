# Rak Homework

## Overview

The repo, implemented within a torch+lightning+hydra way, is organized as follow:
```
rak
├── configs
│   ├── data
│   │   ├── dummy.yaml          # dummy dataset config file
│   ├── experiments             # directory containing the different backbones config files
│   │   ├── fused_cont.yaml     # SimpleRNN + fused linear layer + transposed B, T + contiguous tensors
│   │   ├── fused.yaml          # SimpleRNN + fused linear layer + transposed B, T 
│   │   ├── kvout_cont.yaml     # SimpleRNN + fused linear layer + transposed B, T + KV computation outside of for loop + contiguous tensors
│   │   ├── kvout.yaml          # SimpleRNN + fused linear layer + transposed B, T + KV computation outside of for loop
│   │   ├── seqfirst_cont.yaml  # SimpleRNN + transposed B, T + contiguous tensors
│   │   ├── seqfirst.yaml       # SimpleRNN + transposed B, T
│   │   └── simple.yaml         # SimpleRNN
│   ├── model
│   │   └── base_model.yaml     # default model config
│   ├── test.yaml               # main test (inference only) config file
│   └── train.yaml              # main train config file
├── hw                          # homework module
│   ├── backbones
│   │   ├── fused_rnn.py        # FusedRNN class
│   │   ├── kvout_rnn.py        # KVOutRNN class
│   │   ├── seqfirst_rnn.py     # SeqFirstRNN class
│   │   └── simple-rnn.py       # SimpleRNN class
│   ├── base_model.py           # LM module and BaseRNN parent class
│   ├── data.py                 # dummy data module
│   ├── lit_model.py            # simple lightning module with training_step
│   └── main.py                 # entry point
├── dummy_check.py              # script used to debug initial hw code
├── README.md
├── requirements.txt
└── setup.py
```

## Installation

Tested on Python 3.10.4 with GCC 4.8.5 on linux

1. **Create and source your virtualenv**

```sh
python -m venv rak_venv
source ./rak_venv/bin/activate
```

2. **Install the requirements and hw module**

```sh
pip install -r requirements.txt
pip install .
```
3. **Run your first test**

First, to check the job config that your are about to run, use the hydra `-c` option:
```sh
python ./hw/main.py -c job
```

We use two main config files:
  - `train.yaml` to run a train and validation run
  - `test.yaml` to run a inference only run

`train.yaml` is used by default, if you want to change the main config file, use the `-cn` option:
```sh
python ./hw/main.py -cn test
```

We provide a set of experiment config files to reproduce (modulo your infra noise) the results provided in the next section. To run one of them use:
```sh
python ./hw/main.py experiment=seqfirst
```

You can use the multirun hydra option to run all those sequentially and have a coffee in the meanwhile :D (trust me it's helpful when your baby is awake and crying at 5:30AM). To do so run the command (also compatible with the `-cn` option):
```sh
python ./hw/main.py -m experiment=simple,seqfirst,fused
```

## Efficient training and inference code for SimpleRNN.

We use hydra + lightning so that various native PyTorch options (e.g. compile, amp, ...), could be integrated in config file (hydra) while keeping the code highly readable (lightning). For the sake of the exercise we focus mainly on the core implementation of the architecture.

The [lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api) uses its default values for defining the type of accelerator, their number and strategy. Running it on a GPU/TPU VM instance should natively use those and distribute the data using DDP strategy.
> At least it was the case on my side.

### Overview of the job done

Bellow we summarized our contributions.

Operational and readability:
  - [Change] keep the same ordering (qkvg) in proj step
  - [Fix] wrong query dim output hidden_dim instead of value_dim
  - [Fix] wrong call to output_proj
  - [Fix] non consistent inputs and outputs dimensions for stacked-RNN
  - [Add] lightning on top
  - [Add] hydra configuration
  - [Change] backbone arg for LM (instead of hard coded SimpleRNN class) for ease of testing

RNN optimizations:
  - make forward pass sequence first (i.e. switch B, T dims with einops)
  - force contiguous tensor memory after rearrange
  - unique projection head (i.e. single Linear) followed by split

> I actually saw that the initial code was bugged after thinking and started to dive into the optimizations I wanted to do w.r.t. the SimpleRNN architecture: I hope this was not out of the scope of the homework.
> I added before hand another KVOutRNN class that could, at first, seem like a good option (getting the kv operation outside of the for loop). But you would still have a for loop and thus the kv operation overhead might be negligeable. Plus you would need to store a bigger matrix of size (T, B, key_dim, value_dim) which could be a bottleneck for large batch sizes.

### Benchmarking results

Bellow are the results of the different optimizations. The experiments have been launch on bi-Tesla V100-PCIE-16GB, with DDP, we report the cumulative time spent on both GPUs.

| Configuration                                        | Total Train Time (s) | Total Val Time (s) | Total Test Time (s) |
|------------------------------------------------------|----------------------|--------------------|---------------------|
| SimpleRNN (Original backbone)                        | 298.22               | 2.48               | 36.20               |
| SeqFirstRNN (Sequence first)                         | 204.36               | 2.08               | 32.04               |
| SeqFirstRNN  + Contiguous memory                     | 202.32               | 2.05               | 35.98               |
| FusedRNN (Sequence first + Fused QKVG projection)    | 202.57               | 2.05               | 31.83               |
| FusedRNN + Contiguous memory                         | 202.85               | 2.03               | 35.27               |

Note that in this setting (i.e. noise due to the infra I use), we should have run those experiments several more times to draw strong conclusions.

You can try to reproduce this table by running
```sh
python ./hw/main.py -m experiment=simple,seqfirst,seqfirst_cont,fused,fused_cont
python ./hw/main.py -cn test -m experiment=simple,seqfirst,seqfirst_cont,fused,fused_cont
```

Even if this benchmark would require further testing (more runs, CI, different hyperparameters of the RNN and the dataset, ...), if I had to draw conclusions I would say that:
  - there is no doubt that switching B and T dims helps while looping on T dim
  - FusedRNN could bring some improvement w.r.t. seqfirst only architecture
  - forcing contiguous memory after `einops.rearrange` on both `qkvg` and `output` does not seems to help.
> About using contiguous memory, I felt like it should at least fasten the learning phase, one interesting test (in the case of stacked-RNN, as this is the case here) would be to rearrange and force continuous memory in the `LM` module. In that case the number of calls to `rearrange` and `contiguous()` would be way smaller making the overall process more efficient.

> Other optimization easy to test within this setting that I did not try:
>  - mixed precision could make more efficient both training and inference speed, and memory usage. This might cause some instabilities, even more with RNN that usually suffers from vanishing/exploding gradients.
>  - higher distributed inference among more GPUs
>  - robust training: we are here working with random data, there is nothing to learn. But in the case we would want to make this training more stable, we could easily implement gradient clipping

## Self-Attention vs. RNN: pros and cons

### Model capabilities

First, let's talk about their respective capabilities inherent to their architecture. RNN are trained to retain relevant signal through a state embedding to predict the next token (at least in the case of our SimpleRNN). To do so, they sequentially aggregate current signal to their current state. Note that there are different ways to aggregate it (e.g. RNN vs LSTM-gating).

Thus "knowing" in advance which information to retain (a.k.a. long-term dependencies) from an already seen sequence is a real challenge and is still an open problem as showcased in [RWKV](https://arxiv.org/abs/2305.13048) Appendix L.

Self-Attention does not suffer from such limitation since one have access to the "raw" initial sequence (i.e. all previous tokens) to predict the next one whereas RNNs have a aggregated version of the past.

One could also argue that, due to there more intuitive mecanism, attention mecanisms (token relations through attention scores) are more interpretable than RNN mecanisms (latent hidden state).

### Computation, memory and training

One of the main advanges of RNN-like approaches is their computation efficiency (linear complexity) w.r.t. sequence length. They are also more memory efficient since they only need their current hidden state to predict the next token given the current one. This property make RNN quite suitable for streaming tasks or infinite-context tasks.

On the other side, self-attention-based methods suffer from a quadratic complexity w.r.t. the sequence length (both computational and memory due to the attention matrix), where one need to compute the all attention score matrix. Still, some improvements have been made to make them more computationally and memory effecient (KV-cache, Multi-head latent attention, ...) at inference and training. This downside of self-attention-based methods make them hardly compatible with infinite-context tasks. But this architecture make them easy to parallelize and scale (e.g. simple causal matrix on the attention scores while training). Plus, the absence of for-loop, such as the one in SimpleRNN, enable them to trully benefit from model `torch.compile`.
> To convince yourself how ineficient compilation is for RNNs, I've added the compilation option to the config file. I didn't had the time to wait for the compilation to end, but if you're currious you can do it and ping me with the results :D

In the previous section, we saw that it could be challenging for RNNs to capture long-term dependency, it is even truer with vanilla RNNs that suffer from vanishing gradient at training time.

### Is this the end of RNN-variants?

Finally, even though RNNs seems less capable of w.r.t. NLP tasks, recent advances in recurrent models such as [RWKV](https://arxiv.org/abs/2305.13048), and the [Mamba](https://arxiv.org/abs/2312.00752) state space model (SSM) familly (not restricted to NLP tasks), showed that recurrent models could have similar performances than Transformers-based architectures while being more efficient. But as RNN/SSM are more efficient, they might be more suitable for computation expensive sequential tasks such as video generation.

## Scaling RNN training

I see to axis on which we would want to scale RNNs:
  - the sequence length
  - the model and batch size

For very long sequence lengths my go-to option would be Truncated Backpropagation Through Time, see Section 2.8.6 of this [thesis](https://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf), a simple yet effective approach. But this is a proxy of the actual loss we want to compute and we still loss some temporal dependencies.

Concerning the model and batch size, the bottleneck here is the vRAM:
  - For batch size, DDP should do the job (already implemented in our solution).
  - For the model size (and it's corresponding gradient states at learning time), reducing the precision could help if it produces no further instabilities, this should free some vRAM and accelerate the training.
  - A more drastic approach would be to use gradient/activation checkpointing and model parallelism using deepspeed/fsdp.
> Lightning trainers are compatible with deepspeed even though we are not using it here, we could have add a trainer config to test different configurations see [here](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html).

On the architecture side, mixture of expert (MoE) for stacked-RNN (an expert in between each RNN layer), would lead to less active parameters even though training MoE from scratch could be as challenging for large stacked-RNN as it was for LLM (some training instabilities see [here](https://arxiv.org/abs/2101.03961)).
