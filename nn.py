"""
I ran this code on Google Colab and just copy pasted it on VS Code. I cannot run the code on
VS code because I do not have a GPU so I cannot check if the script is working fine. There are
chances there might be some directory issues IF the code does not work fine. I tried to fix the
issues as much as possible

"""

import argparse
import datasetsg
import pandas as pd
import tensorflow as tf
import numpy as np
from transformers import TFAutoModel, AutoTokenizer
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight

# File paths for training, testing, and development datasets, and the model
train_path = "train.csv"
test_path = "test-in.csv"
dev_path = "dev.csv"
model_path = "model.keras"

# Load RoBERTa model and tokenizer
model_name = 'roberta-base'
roberta = TFAutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Hyperparameters
MAX_LENGTH = 64  # Maximum length of input sequences
BATCH_SIZE = 16  # Batch size for training and evaluation
EPOCHS = 10  # Number of training epochs

# Tokenize input text
def tokenize(examples):
    return tokenizer(examples["text"],
                     truncation=True,
                     max_length=MAX_LENGTH,
                     padding="max_length")

def add_roberta_embeddings(examples):
    chunk_size = 100  # Process data in smaller chunks to avoid memory issues
    embeddings = []

    for i in range(0, len(examples['text']), chunk_size):
        batch = examples['text'][i:i + chunk_size]
        tokens = tokenizer(batch, return_tensors="tf", max_length=MAX_LENGTH,
                           padding="max_length",
                           truncation=True)
        outputs = roberta(**tokens)
        embeddings.append(outputs.last_hidden_state.numpy())

    return {"embeddings": np.concatenate(embeddings)}




# Compute class weights for each label
def compute_class_weights_per_label(labels, num_classes):
    class_weights = np.zeros((num_classes, 2))
    for i in range(num_classes):
        label_col = labels[:, i]
        weights = compute_class_weight("balanced",
                                       classes=np.array([0, 1]),
                                       y=label_col)
        class_weights[i] = np.clip(weights, 0, 10)  # Cap weights at 10
    print("Validated class weights:", class_weights)
    return class_weights

# Custom weighted binary crossentropy loss function
def weighted_binary_crossentropy(y_true, y_pred):
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
    weights = tf.reduce_sum(class_weights_tensor[:, 1] * y_true + class_weights_tensor[:, 0] * (1 - y_true), axis=-1)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce * weights)

def create_model(input_dim, output_dim):
    input_layer = layers.Input(shape=(MAX_LENGTH, 768))  # Input layer for embeddings

    # Single Bidirectional LSTM Layer
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(input_layer)

    # Global Max Pooling Layer
    x = layers.GlobalMaxPooling1D()(x)

    # Fully Connected Layer 1
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(rate=0.4)(x)  # Dropout Layer 1

    # Fully Connected Layer 2
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)  # Batch Normalization
    x = layers.Dropout(rate=0.3)(x)  # Dropout Layer 2

    # Output Layer
    output_layer = layers.Dense(output_dim, activation='sigmoid')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)  # Build the model
    return model


# Load and save Huggingface dataset
def load_save_hf_dataset(train_path, dev_path):
    hf_dataset = datasets.load_dataset("csv", data_files={"train": train_path,
                                                          "validation": dev_path})
    labels = hf_dataset["train"].column_names[1:]  # Get label column names

    def gather_labels(example):
        return {"labels": [float(example[l]) for l in labels]}  # Convert label columns into a list of floats

    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    hf_dataset = hf_dataset.map(add_roberta_embeddings, batched=True)

    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns=["embeddings"],
        label_cols="labels",
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns=["embeddings"],
        label_cols="labels",
        batch_size=BATCH_SIZE
    )

    global class_weights
    labels_array = np.array(hf_dataset["train"]["labels"])
    class_weights = compute_class_weights_per_label(labels_array, len(labels))
    print("Class weights per label:", class_weights)

    return labels, train_dataset, dev_dataset

# Compile the model
def compile_model(model, labels):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.9
    )),
        loss=weighted_binary_crossentropy,
        metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.5)]
    )
    return model

# Fit the model
def fit_model(model, train_dataset, dev_dataset, model_path):
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=dev_dataset,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor="val_f1_score",
                mode="max",
                save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_f1_score",
                mode="max",
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    return model

# Train the model
def train(model_path=model_path, train_path=train_path, dev_path=dev_path):
    labels, train_dataset, dev_dataset = load_save_hf_dataset(train_path, dev_path)
    model = create_model(tokenizer.vocab_size, len(labels))
    model = compile_model(model, labels)
    model = fit_model(model, train_dataset, dev_dataset, model_path)

# Predict using the model
def predict(model_path=model_path, input_path=test_path):
    model = tf.keras.models.load_model(model_path, custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy})
    df = pd.read_csv(input_path)
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    hf_dataset = hf_dataset.map(add_roberta_embeddings, batched=True)
    tf_dataset = hf_dataset.to_tf_dataset(
        columns=["embeddings"],
        batch_size=BATCH_SIZE
    )
    predictions = np.where(model.predict(tf_dataset) > 0.5, 1, 0)
    df.iloc[:, 1:] = predictions
    df.to_csv("submission.zip", index=False, compression=dict(method='zip', archive_name='submission.csv'))

# Train and Predict
# train(model_path=model_path, train_path=train_path, dev_path=dev_path)
# predict(model_path=model_path, input_path=test_path)

# Download the submission file
# from google.colab import files
# files.download("submission.zip")


if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()

    # call either train() or predict()
    globals()[args.command]()