import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

# -----------------------------
# 0. Device setup (GPU if available)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 1. Data reading helpers
# -----------------------------
def read_train_csv(path):
    """Read train.csv and return sentences (list of token lists) and tags (list of tag lists)."""
    df = pd.read_csv(path)
    sentences = []
    tags = []
    for _, row in df.iterrows():
        tokens = str(row["utterances"]).split()
        label_seq = str(row["IOB Slot tags"]).split()
        assert len(tokens) == len(label_seq), "Token/label length mismatch in train.csv"
        sentences.append(tokens)
        tags.append(label_seq)
    return sentences, tags


def read_test_csv(path):
    """Read test.csv and return IDs and sentences (tokens)."""
    df = pd.read_csv(path)
    ids = df["ID"].tolist()
    sentences = [str(u).split() for u in df["utterances"].tolist()]
    return ids, sentences


# -----------------------------
# 2. Vocabulary builders
# -----------------------------
def build_word_vocab(sentences, min_freq=1):
    """Build word2idx and idx2word from training sentences."""
    from collections import Counter
    counter = Counter()
    for sent in sentences:
        counter.update(sent)

    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            word2idx[word] = idx
            idx += 1

    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word


def build_tag_vocab(tags_list):
    """Build tag2idx and idx2tag from tag sequences."""
    unique_tags = set()
    for seq in tags_list:
        unique_tags.update(seq)

    tag2idx = {"<PAD>": 0}
    idx = 1
    for t in sorted(unique_tags):
        tag2idx[t] = idx
        idx += 1

    idx2tag = {i: t for t, i in tag2idx.items()}
    return tag2idx, idx2tag


def encode_sentence(tokens, word2idx):
    """Map tokens to word IDs, unknown -> <UNK>."""
    unk_id = word2idx["<UNK>"]
    return [word2idx.get(t, unk_id) for t in tokens]


def encode_tags(tags, tag2idx):
    """Map tag strings to tag IDs."""
    return [tag2idx[t] for t in tags]


# -----------------------------
# 3. Dataset + collate
# -----------------------------
class SlotDataset(Dataset):
    """PyTorch Dataset for slot tagging."""

    def __init__(self, sentences, tags, word2idx, tag2idx):
        self.sentences = sentences
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        input_ids = encode_sentence(tokens, self.word2idx)
        if self.tags is not None:
            label_ids = encode_tags(self.tags[idx], self.tag2idx)
        else:
            label_ids = None
        return torch.tensor(input_ids, dtype=torch.long), (
            torch.tensor(label_ids, dtype=torch.long) if label_ids is not None else None
        )


def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    inputs = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    lengths = [len(x) for x in inputs]
    max_len = max(lengths)
    batch_size = len(inputs)

    padded_inputs = torch.zeros(batch_size, max_len, dtype=torch.long)
    if labels[0] is not None:
        padded_labels = torch.zeros(batch_size, max_len, dtype=torch.long)
    else:
        padded_labels = None

    for i, (x, y) in enumerate(zip(inputs, labels)):
        L = len(x)
        padded_inputs[i, :L] = x
        if padded_labels is not None and y is not None:
            padded_labels[i, :L] = y

    return padded_inputs, padded_labels, torch.tensor(lengths, dtype=torch.long)


# -----------------------------
# 4. BiLSTM model
# -----------------------------
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, tagset_size)

    def forward(self, input_ids, lengths):
        emb = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits


# -----------------------------
# 5. Train & eval helpers
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for inputs, labels, lengths in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(inputs, lengths)  # [B, L, num_tags]

        B, L, C = logits.shape
        loss = criterion(logits.view(B * L, C), labels.view(B * L))

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B * L
        total_tokens += B * L

    return total_loss / total_tokens


def evaluate_f1(model, loader, idx2tag):
    model.eval()
    all_true = []
    all_pred = []
    with torch.no_grad():
        for inputs, labels, lengths in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits = model(inputs, lengths)
            preds = torch.argmax(logits, dim=-1)

            B = inputs.size(0)
            for i in range(B):
                L = lengths[i].item()
                true_ids = labels[i, :L].cpu().tolist()
                pred_ids = preds[i, :L].cpu().tolist()

                true_tags = [idx2tag[id_] for id_ in true_ids if id_ != 0]
                pred_tags = [idx2tag[id_] for id_ in pred_ids if id_ != 0]

                all_true.append(true_tags)
                all_pred.append(pred_tags)

    f1 = f1_score(all_true, all_pred, mode="strict", scheme=IOB2)
    return f1


def predict_tags(model, sentences, word2idx, idx2tag, batch_size=32):
    """Predict tag sequences (list of tag strings) for a list of tokenized sentences."""
    model.eval()
    all_preds = []
    dataset = SlotDataset(sentences, None, word2idx, {t: i for i, t in idx2tag.items()})
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for inputs, _, lengths in loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            logits = model(inputs, lengths)
            preds = torch.argmax(logits, dim=-1)

            B = inputs.size(0)
            for i in range(B):
                L = lengths[i].item()
                pred_ids = preds[i, :L].cpu().tolist()
                pred_tags = [idx2tag[id_] for id_ in pred_ids if id_ != 0]
                all_preds.append(pred_tags)

    return all_preds


# -----------------------------
# 6. Main pipeline
# -----------------------------
def main(train_path, test_path, output_path):
    # 6.1 Load train data
    train_sentences, train_tags = read_train_csv(train_path)
    print(f"Loaded {len(train_sentences)} training examples.")

    # 6.2 Build vocabularies
    word2idx, idx2word = build_word_vocab(train_sentences)
    tag2idx, idx2tag = build_tag_vocab(train_tags)

    print(f"Vocab size: {len(word2idx)}, Tagset size: {len(tag2idx)}")

    # 6.3 Train/dev split
    X_train, X_dev, y_train, y_dev = train_test_split(
        train_sentences, train_tags, test_size=0.1, random_state=42
    )

    # 6.4 Datasets + loaders
    train_dataset = SlotDataset(X_train, y_train, word2idx, tag2idx)
    dev_dataset = SlotDataset(X_dev, y_dev, word2idx, tag2idx)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    # 6.5 Model, optimizer, loss
    model = BiLSTMTagger(
        vocab_size=len(word2idx),
        tagset_size=len(tag2idx),
        embedding_dim=100,
        hidden_dim=128,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD label

    # 6.6 Train
    best_f1 = 0.0
    best_state = None
    num_epochs = 5

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        dev_f1 = evaluate_f1(model, dev_loader, idx2tag)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, dev_f1={dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_state = model.state_dict()

    print(f"Best dev F1: {best_f1:.4f}")

    # Load best model before test prediction
    if best_state is not None:
        model.load_state_dict(best_state)

    # 6.7 Read test data
    test_ids, test_sentences = read_test_csv(test_path)
    print(f"Loaded {len(test_sentences)} test examples.")

    # 6.8 Predict tags for test
    test_pred_tags = predict_tags(model, test_sentences, word2idx, idx2tag)

    assert len(test_pred_tags) == len(test_ids), "Mismatch in test predictions vs IDs"

    # 6.9 Save test_pred.csv in required format
    rows = []
    for id_, tags in zip(test_ids, test_pred_tags):
        tag_str = " ".join(tags)
        rows.append({"ID": id_, "IOB Slot tags": tag_str})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", type=str)
    parser.add_argument("test_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    main(args.train_path, args.test_path, args.output_path)
