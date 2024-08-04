---
layout: post
title:  "Text Summarization with BERT"
date:   2024-08-04 14:48:31 +0530
categories: BERT usecase
---
### How to Build a Text Summarizer using BERT: A Step-by-Step Guide

**Introduction**

Ever found yourself overwhelmed by the sheer volume of text you need to process daily? Whether you're a student, a researcher, or a professional, digesting large amounts of information quickly is a valuable skill. What if you could create an automated tool to summarize lengthy articles, reports, or papers into concise, digestible summaries? This is where text summarization comes into play, and in this post, we'll show you how to build a powerful text summarizer using BERT, a state-of-the-art NLP model.

**1\. What is Text Summarization?**

Text summarization is the process of shortening a text document while preserving its core ideas. There are two main types of text summarization:

-   **Extractive Summarization**: Selects key sentences, phrases, and paragraphs directly from the source text to create a summary.
-   **Abstractive Summarization**: Generates new sentences that capture the main ideas of the source text, similar to how humans write summaries.

Text summarization is crucial for quickly understanding large volumes of text, such as news articles, research papers, and reports. Imagine being able to summarize an entire research paper into a few sentences without losing the main points---this can save you time and help you grasp essential information faster.

**2\. Introduction to BERT**

BERT (Bidirectional Encoder Representations from Transformers) is a powerful transformer-based model developed by Google. It has achieved state-of-the-art results in various NLP tasks, including text summarization. BERT's bidirectional nature allows it to understand the context of words from both left and right, making it particularly effective for tasks like summarization.

**3\. Setting Up the Environment**

First, we need to set up our environment by installing the necessary libraries. We'll use the `transformers` library from Hugging Face and `torch` for PyTorch.


`!pip install transformers torch`

**4\. Data Preparation**

For this tutorial, we'll use the CNN/Daily Mail dataset, a popular dataset for text summarization tasks. This dataset contains news articles along with their corresponding summaries, making it ideal for training our model.


```
from datasets import load_dataset

dataset = load_dataset('cnn_dailymail', '3.0.0')
train_data = dataset['train']
```

We'll also need to preprocess the data to fit the BERT model's requirements.



```
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_function(examples):
    inputs = [doc for doc in examples['article']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['highlights'], max_length=150, truncation=True, padding='max_length')

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

train_data = train_data.map(preprocess_function, batched=True)
```

**5\. Building the Model**

We'll use the BERT model from Hugging Face's transformers library. BERT is not designed for summarization out of the box, so we use an EncoderDecoderModel which pairs a BERT encoder with a BERT decoder.


```
from transformers import EncoderDecoderModel

model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
```

**6\. Training and Fine-Tuning**

Next, we need to set up the training arguments and train the model. Training a model involves optimizing its weights based on the input data and expected output. Fine-tuning a pre-trained model like BERT on a specific task (summarization) helps it adapt to the nuances of that task.


```
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)

trainer.train()
```

**7\. Evaluation**

After training, we'll evaluate the model to see how well it summarizes texts. Evaluation is crucial to ensure our model is performing well and producing meaningful summaries.


```
def evaluate_summary(model, tokenizer, text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

sample_text = """
    The Earth is the third planet from the Sun and the only astronomical object known to harbor life.
    According to radiometric dating and other sources of evidence, Earth formed over 4.5 billion years ago.
    Earth's gravity interacts with other objects in space, especially the Sun and the Moon,
    which is Earth's only natural satellite. Earth orbits around the Sun in about 365.25 days.
    Earth's axis of rotation is tilted with respect to its orbital plane, producing seasons on Earth.
    The gravitational interaction between Earth and the Moon causes tides, stabilizes Earth's orientation on its axis,
    and gradually slows its rotation. Earth is the densest planet in the Solar System and the largest and most massive
    of the four rocky planets.
"""

print(evaluate_summary(model, tokenizer, sample_text))
```

Expected output:

```The Earth is the third planet from the Sun and the only known to harbor life. Earth formed over 4.5 billion years ago. Earth's gravity interacts with other objects in space, especially the Sun and the Moon, which is Earth's only natural satellite. Earth orbits around the Sun in about 365.25 days, producing seasons on Earth.```

**8\. Conclusion**

In this post, we've built a text summarizer using BERT. We covered everything from setting up the environment, preparing the data, building and training the model, to evaluating its performance. With a powerful summarizer, you can quickly distill large texts into essential information, saving time and effort. Try building your own text summarizer and share your results in the comments!

**9\. Resources**
Feel free to ask questions or leave comments below. Happy summarizing!

-   [BERT Paper](https://arxiv.org/abs/1810.04805)
-   [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/index)
-   [CNN/Daily Mail Dataset](https://paperswithcode.com/dataset/cnn-daily-mail-1)