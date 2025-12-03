import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# CNN Encoder
# -----------------------------

class CNN_Encoder(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, feature_dim, dropout=0.3):
        super(CNN_Encoder, self).__init__()

        # Conv1: seq/=4
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=10, stride=4, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Conv2: seq/=2
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=6, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Conv3: seq/=2
        self.conv3 = nn.Conv1d(hidden_dim, feature_dim, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(feature_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.squeeze(-1)
        x = F.relu(self.bn1(self.conv1(x)))  # Conv1
        x = F.relu(self.bn2(self.conv2(x)))  # Conv2
        x = F.relu(self.bn3(self.conv3(x)))  # Conv3

        x = self.dropout(x)

        # Global Mean Pooling
        x = torch.mean(x, dim=-1)  # (batch, feature_dim)

        return x

# -----------------------------
# SimCLR Projection
# -----------------------------
class SimCLR(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, feature_dim, projection_dim, dropout=0.3):
        super(SimCLR, self).__init__()

        # CNN Encoder 
        self.encoder = CNN_Encoder(input_dim, seq_len, hidden_dim, feature_dim, dropout)
        #self.encoder = CNNShortWindow(input_dim, hidden_dim, feature_dim)

        # Projection Head
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, x):
        feature_embedding = self.encoder(x)  # (batch_size, feature_dim)
        projection = self.projection(feature_embedding)  # (batch_size, projection_dim)
        projection = F.normalize(projection, dim=-1)

        return projection


# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# -----------------------------
# Simple MLP Classifier
# -----------------------------
class TaskClassifier_simple(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(torch.mean(x, dim=1))
    
# -----------------------------
# Pretrained CNN + Transformer 
# -----------------------------
class TaskClassifier_Transformer(nn.Module):
    def __init__(self, encoder, feature_dim, window_len,
                 overlap, num_heads, num_layers, num_classes,dropout=0.3,
                 freeze_encoder=False):
        super().__init__()

        self.encoder = encoder
        self.window_len = window_len
        self.stride = int(window_len * (1 - overlap))

        # === Encoder の freeze 設定 ===
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.pos = PositionalEncoding(feature_dim, 500)

        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def extract_windows(self, x):
        x = x.unfold(2, self.window_len, self.stride)
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, signals):
        B, C, T = signals.shape

        # Step1: windowing
        x = self.extract_windows(signals)     # (B,W,C,Tw)
        B, W, C, Tw = x.shape

        # Step2: CNN encoder
        feats = self.encoder(x.reshape(B * W, C, Tw))     # (B*W,F)
        feats = feats.reshape(B, W, -1)                   # (B,W,F)
        mid_feat = feats[:,W//2,:]

        # Step3: positional encoding + CLS
        x = self.pos(feats)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), 1)

        # Step4: transformer
        h = self.transformer(x)
        context = h[:, 0]

        # Step5: classifier
        logits = self.classifier(context)
        return logits, context, mid_feat


class TaskClassifier_LSTM(nn.Module):
    def __init__(self, encoder, hidden_dim, window_len,
                 overlap, lstm_hidden, lstm_layers,
                 num_classes, freeze_encoder=False):
        super().__init__()

        self.encoder = encoder
        self.window_len = window_len
        self.stride = int(window_len * (1 - overlap))

        # === freeze encoder ===
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # LSTM (Transformer の代替)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,       # CNN encoder 出力次元
            hidden_size=lstm_hidden,     # LSTM の埋め込み次元
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1
        )

        # 最終分類層
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden, num_classes)
        )

    def extract_windows(self, x):
        x = x.unfold(2, self.window_len, self.stride)   # (B,C,W,Tw)
        x = x.permute(0, 2, 1, 3)                       # (B,W,C,Tw)
        return x

    def forward(self, signals):
        B, C, T = signals.shape

        # Step1: windowing
        x = self.extract_windows(signals)    # (B,W,C,Tw)
        B, W, C, Tw = x.shape

        # Step2: CNN encoder 
        feats = self.encoder(x.reshape(B * W, C, Tw))   # (B*W,F)
        feats = feats.reshape(B, W, -1)                 # (B,W,F)

        # Step3: LSTM
        out, (hn, cn) = self.lstm(feats)                # out: (B,W,H)
        last_hidden = out[:, -1, :]                     # (B,H)

        # Step4: classifier
        logits = self.classifier(last_hidden)
        
        return logits

class ActionClassifier(nn.Module):
    """ operationモデルのcontextベクトル → action分類 """
    def __init__(self, context_dim=128, hidden_dim=128, num_classes=27, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, context):
        return self.net(context)

class SimpleActionClassifier(nn.Module):
    def __init__(self, encoder, feature_dim=128, num_classes=27, freeze=True):
        super().__init__()
        self.encoder = encoder

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        feat = self.encoder(x)        # (B,128,T')
        #feat = self.gap(feat).squeeze(-1)   # (B,128)
        logits = self.fc(feat)
        return logits
    
# -----------------------------
# Simple MLP Classifier
# -----------------------------
class TaskClassifier_Linear(nn.Module):
    def __init__(self, encoder, feature_dim, hidden_dim, num_classes, dropout=0.1, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder

        # === Encoder の freeze 設定 ===
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        logits = self.classifier(x)
        return logits, x, x
