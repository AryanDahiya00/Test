import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = Wav2Vec2Model.from_pretrained("facebook/hubert-large-ls960-ft").to(device)
model.eval()

train_df = pd.read_csv('/kaggle/input/shl-intern-hiring-assessment/dataset/train.csv')
test_df = pd.read_csv('/kaggle/input/shl-intern-hiring-assessment/dataset/test.csv')

def extract_features(file_path):
    speech, sr = torchaudio.load(file_path)
    if speech.shape[0] > 1:
        speech = torch.mean(speech, dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        speech = resampler(speech)
    inputs = processor(speech.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(inputs.input_values.to(device)).last_hidden_state.squeeze(0)
    mean_vec = outputs.mean(dim=0)
    std_vec = outputs.std(dim=0)
    wav2vec_feature = torch.cat([mean_vec, std_vec]).cpu().numpy()
    mfcc_feat = librosa.feature.mfcc(y=speech.squeeze().numpy(), sr=16000, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc_feat)
    mfcc_delta2 = librosa.feature.delta(mfcc_feat, order=2)
    mfcc_combined = np.concatenate([mfcc_feat, mfcc_delta, mfcc_delta2], axis=0)
    mfcc_mean = np.mean(mfcc_combined, axis=1)
    mfcc_std = np.std(mfcc_combined, axis=1)
    mfcc_feature = np.concatenate([mfcc_mean, mfcc_std])
    return np.concatenate([wav2vec_feature, mfcc_feature])

train_embeddings = []
for fname in train_df['filename']:
    path = f"/kaggle/input/shl-intern-hiring-assessment/dataset/audios_train/{fname}"
    train_embeddings.append(extract_features(path))

train_X = np.array(train_embeddings)
train_y = train_df['label'].values

X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.15, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model_reg = XGBRegressor(n_estimators=800, learning_rate=0.02, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42)
model_reg.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], early_stopping_rounds=20, verbose=False)
val_preds = model_reg.predict(X_val_scaled)
pearson = pearsonr(val_preds, y_val)[0]
print(f"📊 Pearson Correlation on Validation Set: {pearson:.4f}")

plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_val, y=val_preds)
plt.xlabel("Actual Label")
plt.ylabel("Predicted Label")
plt.title(f"Validation Pearson: {pearson:.3f}")
plt.grid()
plt.show()


test_embeddings = []
for fname in test_df['filename']:
    path = f"/kaggle/input/shl-intern-hiring-assessment/dataset/audios_test/{fname}"
    test_embeddings.append(extract_features(path))
test_X = np.array(test_embeddings)
test_X_scaled = scaler.transform(test_X)
test_preds = model_reg.predict(test_X_scaled)
submission = pd.DataFrame({
    'filename': test_df['filename'],
    'label': test_preds
})
submission.to_csv('submission4_3.csv', index=False)
print("Submission file created!")
