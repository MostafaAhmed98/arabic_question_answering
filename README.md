# Arabic Question Answering with Hugging Face and BERT

This project demonstrates how to fine-tune a BERT-based model for Arabic question answering using the TyDiQA dataset. The fine-tuned model can be used to answer questions based on Arabic context passages.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model](#model)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Inference with Gradio](#inference-with-gradio)

## Introduction

The project utilizes the `UBC-NLP/MARBERTv2` model, a BERT-based model fine-tuned on Arabic text, to perform the task of question answering. The model is trained and evaluated using the TyDiQA dataset, which is a multilingual dataset for question answering tasks.

## Dataset

The dataset used is the TyDiQA Gold Passage (GoldP) task, specifically filtered for Arabic. TyDiQA is a typologically diverse dataset covering multiple languages, but this project focuses on the Arabic subset.

## Model

The model is fine-tuned using the Hugging Face `transformers` library. The `UBC-NLP/MARBERTv2` checkpoint is used as the base model, and it is trained to answer questions by predicting the start and end positions of the answer span in the given context.

## Requirements

To run the project, you need the following libraries installed:

- `torch`
- `transformers`
- `datasets`
- `pyarabic`
- `gradio`

You can install these dependencies using pip:

```bash
pip install torch transformers datasets pyarabic gradio
```

## Usage

### Inference with Gradio

1. **Define the prediction function:**

    ```python
    from transformers import pipeline

    def loading_model_and_prediction(question, context):
        model_checkpoint = "MARBERT-finetuned-tydiqa/checkpoint-5769"
        question_answerer = pipeline("question-answering", model=model_checkpoint)
        predictions = question_answerer(question=question, context=context)
        return predictions['answer']
    ```

2. **Create a Gradio interface:**

    ```python
    import gradio as gr

    default_question = "من مقدم برنامج خواطر؟"
    default_context = "أحمد مازن أحمد أسعد الشقيري (19 يوليو 1973م، جدة) إعلامي سعودي من أصول فلسطينية بدأ بتقديم برامج فكرية اجتماعية ومضيف السلسلة التليفزيونية خواطر والمضيف السابق لبرنامج يلا شباب..."

    def predict(user_question, user_context):
        model_preds = loading_model_and_prediction(user_question, user_context)
        if len(model_preds) == 0:
            return "No answer Found"
        return model_preds

    demo = gr.Interface(fn=predict,
                        inputs=[gr.Text(value=default_question, placeholder="Arabic Question Text", label="Arabic Question Text"),
                                gr.Text(value=default_context, placeholder="Arabic Context Text", label="Arabic Context Text")],
                        outputs=gr.Text(label="Answer Prediction"), title="Arabic Question Answering", allow_flagging=False)
    demo.launch(share=True)
    ```

3. **Run the Gradio app:**

    ```bash
    python app.py
    ```
