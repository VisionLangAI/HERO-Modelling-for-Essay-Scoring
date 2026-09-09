# English Syntactic–Semantic Evaluation

This repository contains `EnglishSyntaticSemanticEvaluation.ipynb`, a TensorFlow notebook for automated English essay scoring. It implements a dual-path hierarchical model that jointly learns semantic and syntactic representations to predict holistic and grammar scores.

## Main Features

- Semantic pathway based on `bert-base-uncased` token IDs
- Syntactic pathway based on spaCy part-of-speech tags
- Word-level bidirectional LSTM and four-head self-attention
- Sentence-level bidirectional LSTM encoding
- Gated fusion of semantic and syntactic representations
- Demographic and grade-level metadata integration
- Separate outputs for holistic and grammar-score prediction
- Evaluation using MAE, MSE, RMSE, R², Pearson correlation, and Spearman correlation
- Training-curve, baseline-comparison, model-architecture, and attention-heatmap visualizations

## Model Workflow

1. Clean essay text and handle missing metadata.
2. Segment each essay into a maximum of 15 sentences.
3. Represent each sentence using up to 20 BERT tokens and corresponding POS tags.
4. Encode semantic and syntactic sequences through separate embedding, BiLSTM, and attention branches.
5. Learn sentence-level dependencies using hierarchical BiLSTMs.
6. Fuse both branches through a learned gate and sentence-attention layer.
7. Combine the fused essay representation with gender and grade-level metadata.
8. Predict normalized holistic and grammar scores.

## Required Dataset

Place the dataset in the notebook's working directory with this filename:

```text
asap2_dataset.csv
```

Expected columns are:

| Column | Description |
|---|---|
| `essay_text` | Complete essay text |
| `prompt_id` | Essay prompt identifier |
| `gender` | Student gender metadata |
| `grade_level` | Student grade level |
| `score` | Holistic essay score |
| `grammar_score` | Grammar score; if absent, the notebook uses the holistic score |

## Requirements

- Python 3
- NumPy
- pandas
- TensorFlow
- Transformers
- NLTK
- spaCy
- scikit-learn
- SciPy
- Matplotlib
- Seaborn
- Graphviz
- pydot
- spaCy model `en_core_web_sm`

The notebook installs its principal dependencies within the first cell. Graphviz and pydot are installed separately before generating the architecture diagram.

## How to Run

1. Open `EnglishSyntaticSemanticEvaluation.ipynb` in Google Colab or Jupyter Notebook.
2. Upload `asap2_dataset.csv` to the current working directory.
3. Run the first cell to install packages, preprocess the data, train the model, and evaluate its predictions.
4. Run the remaining cells to generate metric curves, baseline comparisons, the architecture diagram, and the attention heatmap.

The primary experiment uses an 80:20 train–test split with `random_state=42`. Ten percent of the training portion is used for validation. The model is trained for 6 epochs with a batch size of 16 and the Adam optimizer at a learning rate of `1e-4`.

## Important Configuration

| Parameter | Value |
|---|---:|
| Maximum sentences per essay | 15 |
| Maximum tokens per sentence | 20 |
| Embedding dimension | 128 |
| BiLSTM units | 64 |
| Attention heads | 4 |
| Metadata hidden units | 32 |
| Fusion hidden units | 64 |
| Batch size | 16 |
| Epochs | 6 |
| Learning rate | 0.0001 |

## Generated Outputs

Depending on the executed cells, the notebook displays or saves:

- Model summary and prediction metrics
- Training-performance plots
- Comparison plots against selected baseline results
- `model_architecture.png`
- `novel_model.h5`
- Multi-head attention heatmap

## Notes

- The tokenizer supplies BERT vocabulary IDs, but the notebook learns its own embedding layer; it does not fine-tune a pretrained BERT encoder.
- Some plotting cells use manually specified metric arrays for visualization rather than values automatically read from the training history.
- The installation commands use notebook shell syntax and are most convenient in Google Colab.
- For strict experimental reproducibility, set NumPy and TensorFlow random seeds and fit preprocessing transformations only on the training partition.
