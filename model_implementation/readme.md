## Fine-tuning MARBERTv2 for Question Answering on Arabic Text

This code demonstrates fine-tuning the pre-trained MARBERTv2 model on a question answering task using the TyDiQA dataset (Arabic Language Only).

**Key functionalities:**

* Filters Arabic subsets from the TyDiQA dataset.
* Performs Arabic text tokenization using the pre-trained MARBERTv2 model.
* Handles long contexts by splitting them into smaller segments during tokenization.
* Trains the model to predict answer locations within the context for a given question.

**Preprocessing Steps:**

* Removes diacritics from questions and contexts.
* Splits long contexts and performs padding.
* Identifies answer locations within each context slice.

**Training Setup:**

* Defines training arguments for hyperparameters and output directory.
* Disables automatic evaluation during training.
* Saves model checkpoints at specific intervals.

**Inference:**

* Loads the fine-tuned model checkpoint.
* Creates a question-answering pipeline for making predictions.

This code provides a foundation for fine-tuning a pre-trained model for question answering on Arabic text data.

