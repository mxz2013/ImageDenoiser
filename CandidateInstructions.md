# Enhance Lab — ML Engineer Remote Technical Assessment

## Overview

Welcome to the Enhance Lab technical assessment. In this exercise you will reconstruct a pre-trained image-processing neural network from its architecture description and raw weights, then run inference on a set of test images.

This task is designed to evaluate your ability to work with deep learning fundamentals — model construction, weight loading, and image inference — as well as your effectiveness in leveraging modern tools such as Large Language Models (LLMs).

---

## Provided Materials

You will receive a folder containing the following files:

| File | Description |
|------|-------------|
| [`architecture.jpg`](./architecture.jpg) | A diagram of the network architecture |
| [`weights.csv`](./weights.csv) | The full set of model weights in CSV format |
| [`images/noisy`](./images/noisy/) | A folder of noisy test images to process |
| [`images/gt`](./images/gt/) | A folder of the ground truth images |

### Architecture Diagram

The file [`architecture.jpg`](./architecture.jpg) contains a visual representation of the model. Use it as your primary reference for building the model.

### Weight File ([`weights.csv`](./weights.csv))

The CSV file contains all the network's convolutional weights with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `layer_name` | string | The fully-qualified parameter name (e.g. `model.prior_net.m_head.weight`) |
| `index` | int | Flat index into the weight tensor |
| `value` | float32 | The scalar weight value |


- All weights have been flattened from tensors with shape `(in_channels, out_channels, kernel_height, kernel_width)`.
- No convolution layers in the network use bias terms.

### Test Images

The [`images/noisy`](./images/noisy/) folder contains noisy RGB images. Your implementation should process each of these and produce corresponding denoised outputs. Input images are guaranteed to have dimensions divisible by 8. We also added the groud truth (GT) images for your reference and/or if you want to calculate any metrics.

---

## Your Task

1. **Build the network** — Using the architecture diagram, implement the model in a deep learning framework of your choice (PyTorch is recommended).
2. **Load the weights** — Parse [`weights.csv`](./weights.csv) and load the weights into your model, handling the reshape and axis-reordering as needed.
3. **Run inference** — Process each test image through the network and save the output images.

---

## Use of LLM-based coding tools

The use of Large Language Models (e.g. ChatGPT, Claude, Copilot, etc.) is **encouraged** for this assessment. We are interested in evaluating how you leverage these tools to complete the task efficiently and correctly. Your interaction log with an LLM is an important deliverable of this test.

Follow-up interviews will include questions to assess your understanding of the problem and the code you produce, so make sure you can explain every part of your solution.

---

## Deliverables

Within max **10 business days**, please submit:

1. **Your code** — A clean, runnable script or project that loads the weights and runs inference on the test images.
2. **Your LLM interaction log** — Either share a link to your conversation(s) or a copy of your prompts and the LLM's responses, highlighting your key requests.
3. **Output images** — The denoised results for all provided noisy images.

Please send your submission as a compressed archive (`.zip` or `.tar.gz`) or as a link to a Git repository.

---

Good luck!
