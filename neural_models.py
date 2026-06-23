"""
Neural ansambl: TabNet, LSTM, Transformer.
"""


import os
import math
import pickle
import warnings
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from collections import Counter, defaultdict

from lottery_config import LOTO7, PROSTI_BROJEVI

warnings.filterwarnings('ignore')

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    if hasattr(torch, '__version__') and int(torch.__version__.split('.')[0]) >= 1:
        TORCH_AVAILABLE = True
except ImportError:
    pass

# ============================================================
# konstante
# ============================================================
# Loto 7/39 — jedna zona (39 brojeva, 7 po kombinaciji)
FRONT_N = LOTO7.max_broj
FRONT_SELECT = LOTO7.brojeva_po_kombinaciji
BACK_N = 0
BACK_SELECT = 0
FRONT_PRIMES = PROSTI_BROJEVI
BACK_PRIMES: set[int] = set()
WINDOW = 50        # prozor za građenje osobina
SEQ_LEN = 20       # dužina ulaza sekvencijalnog modela
NUM_FEATURES = 12  # dimenzija osobina po broju


# ============================================================
# graditelj osobina
# ============================================================

class NeuralFeatureBuilder:
    """
    gradi tenzor osobina za neuronsku mrežu.

    za svaki broj (Glavna zona 1–39, 7 po kombinaciji) gradi višedimenzionalni vektor osobina,
    podržava grupno i sekvencijalno građenje osobina.

    format ulaznih podataka: draws = List[Tuple[List[int], List[int]]]
        svaki element = (Glavna zona 7 brojeva, Dodatna zona: []), sortirano rastuće po vremenu
    """

    def __init__(self, draws: List[Tuple[List[int], List[int]]], window: int = WINDOW):
        self.draws = draws
        self.window = window
        self.front_mean = np.zeros(NUM_FEATURES)
        self.front_std = np.ones(NUM_FEATURES)
        self.back_mean = np.zeros(NUM_FEATURES)
        self.back_std = np.ones(NUM_FEATURES)
        # podesi parametre normalizacije
        self._fit_normalization()

    def _compute_single_features(self, draws_window: List[Tuple[List[int], List[int]]],
                                 num: int, zone: str = 'front') -> np.ndarray:
        """gradi za pojedinačni broj12dimenzionalni vektor osobina"""
        n_max = FRONT_N if zone == 'front' else BACK_N
        primes = FRONT_PRIMES if zone == 'front' else BACK_PRIMES
        n_window = len(draws_window)
        if n_window == 0:
            return np.zeros(NUM_FEATURES, dtype=np.float32)

        # 1. frekvencija: poslednjihwindowbroj pojavljivanja u kolima/window
        freq = sum(1 for f, _ in draws_window if num in f) if zone == 'front' \
            else sum(1 for _, b in draws_window if num in b)
        freq_norm = freq / max(n_window, 1)

        # 2. propuštanje: broj kola od poslednjeg pojavljivanja / window
        missing = n_window
        for i in range(n_window - 1, -1, -1):
            nums = draws_window[i][0] if zone == 'front' else draws_window[i][1]
            if num in nums:
                missing = n_window - 1 - i
                break
        missing_norm = min(missing / max(self.window, 1), 1.0)

        # 3. trend: poslednjih 5 kola vs poslednjih20razlika frekvencije između kola, normalizuj na[0,1]
        recent5 = draws_window[-5:] if n_window >= 5 else draws_window
        recent20 = draws_window[-20:] if n_window >= 20 else draws_window
        f5 = sum(1 for f, _ in recent5 if num in f) / max(len(recent5), 1) if zone == 'front' \
            else sum(1 for _, b in recent5 if num in b) / max(len(recent5), 1)
        f20 = sum(1 for f, _ in recent20 if num in f) / max(len(recent20), 1) if zone == 'front' \
            else sum(1 for _, b in recent20 if num in b) / max(len(recent20), 1)
        trend = (f5 - f20 + 1.0) / 2.0

        # 4. oznaka ponovljenog broja iz prethodnog kola
        repeat = 0.0
        if n_window >= 1:
            last = draws_window[-1]
            last_nums = last[0] if zone == 'front' else last[1]
            repeat = 1.0 if num in last_nums else 0.0

        # 5. oznaka ponovljenog iz kola pre prethodnog (goregorekola)
        skip = 0.0
        if n_window >= 2:
            skip_draw = draws_window[-2]
            skip_nums = skip_draw[0] if zone == 'front' else skip_draw[1]
            skip = 1.0 if num in skip_nums else 0.0

        # 6. oznaka susednog broja (susedan brojevima prethodnog kola)
        adjacent = 0.0
        if n_window >= 1:
            last = draws_window[-1]
            last_nums = set(last[0] if zone == 'front' else last[1])
            if (num - 1) in last_nums or (num + 1) in last_nums:
                adjacent = 1.0
            # obrada prelaska granice
            if num == 1 and 2 in last_nums:
                adjacent = 1.0
            if num == n_max and (n_max - 1) in last_nums:
                adjacent = 1.0

        # 7. oznaka parnosti
        odd = 1.0 if num % 2 == 1 else 0.0

        # 8. oznaka prostog broja
        prime = 1.0 if num in primes else 0.0

        # 9. normalizacija pozicije u zoni (0~1)
        zone_pos = (num - 1) / (n_max - 1)

        # 10. krajnja cifra (cifra jedinica/9)
        tail = (num % 10) / 9.0

        # 11. prigušenje vrućih brojeva: ponderisana frekvencija, nedavnim veća težina
        weighted_freq = 0.0
        total_weight = 0.0
        for i in range(max(0, n_window - 30), n_window):
            weight = (i - max(0, n_window - 30) + 1) / 30.0
            nums = draws_window[i][0] if zone == 'front' else draws_window[i][1]
            if num in nums:
                weighted_freq += weight
            total_weight += weight
        weighted_norm = weighted_freq / max(total_weight, 1e-6)

        # 12. periodičnost razmaka: da li je razmak od poslednjeg pojavljivanja blizak istorijskom obrascu
        # izračunaj sve razmake pojavljivanjaCV(koeficijent varijacije)
        appearances = []
        for i in range(n_window):
            nums = draws_window[i][0] if zone == 'front' else draws_window[i][1]
            if num in nums:
                appearances.append(i)
        if len(appearances) >= 3:
            gaps = np.diff(appearances)
            cv = np.std(gaps) / max(np.mean(gaps), 1e-6)
            regularity = 1.0 / (1.0 + min(cv, 10.0))  # 0~1, redovnije je bliže1
        else:
            regularity = 0.3

        return np.array([
            freq_norm, missing_norm, trend, repeat, skip,
            adjacent, odd, prime, zone_pos, tail,
            weighted_norm, regularity
        ], dtype=np.float32)

    def _fit_normalization(self):
        """podesi parametre normalizacije"""
        if len(self.draws) < self.window + 1:
            return
        front_feats = []
        back_feats = []
        for i in range(self.window, len(self.draws)):
            window_data = self.draws[i - self.window:i]
            for zone, feats_list in [('front', front_feats), ('back', back_feats)]:
                n_max = FRONT_N if zone == 'front' else BACK_N
                for num in range(1, n_max + 1):
                    feats_list.append(self._compute_single_features(window_data, num, zone))

        if front_feats:
            arr = np.array(front_feats)
            self.front_mean = arr.mean(axis=0)
            self.front_std = arr.std(axis=0).clip(min=1e-6)
        if back_feats:
            arr = np.array(back_feats)
            self.back_mean = arr.mean(axis=0)
            self.back_std = arr.std(axis=0).clip(min=1e-6)

    def build_candidate_features(self, front: List[int], back: List[int],
                                 draws: Optional[List[Tuple]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        za pojedinačnikandidatagradi osobine: izvucikandidatavektor osobina broja

        Returns:
            front_feats: (5, NUM_FEATURES) ndarray
            back_feats: (2, NUM_FEATURES) ndarray
        """
        if draws is None:
            draws = self.draws
        window_data = draws[-min(self.window, len(draws)):] if len(draws) >= 2 else draws

        f_feats = np.array([self._compute_single_features(window_data, n, 'front') for n in front])
        b_feats = np.array([self._compute_single_features(window_data, n, 'back') for n in back])

        # normalizuj
        f_feats = (f_feats - self.front_mean) / self.front_std
        b_feats = (b_feats - self.back_mean) / self.back_std

        return f_feats, b_feats

    def build_sequence(self, draws: Optional[List[Tuple]] = None,
                       seq_len: int = SEQ_LEN) -> Tuple[np.ndarray, np.ndarray]:
        """
        gradi sekvencijalne podatke zaLSTM/Transformer

        Returns:
            front_seq: (seq_len, FRONT_N, NUM_FEATURES) osobine svih brojeva po kolu
            back_seq: (seq_len, BACK_N, NUM_FEATURES)
        """
        if draws is None:
            draws = self.draws
        if len(draws) < seq_len + 1:
            seq_len = max(1, len(draws) - 1)

        seq_data = draws[-seq_len - 1:-1]  # najnovijiseq_lenkola (isključi poslednje kolo koje služi kaotarget)
        if len(seq_data) < seq_len:
            seq_data = draws[:seq_len]

        front_seq = np.zeros((seq_len, FRONT_N, NUM_FEATURES), dtype=np.float32)
        back_seq = np.zeros((seq_len, BACK_N, NUM_FEATURES), dtype=np.float32)

        for t in range(seq_len):
            window_data = draws[:max(0, len(draws) - seq_len + t)]
            if len(window_data) < 2:
                # dopuni ranijim kolaprozor
                window_data = draws[:max(self.window, len(draws))]
            window_data = window_data[-min(self.window, len(window_data)):]

            for n in range(1, FRONT_N + 1):
                front_seq[t, n - 1] = self._compute_single_features(window_data, n, 'front')
            if BACK_N > 0:
                for n in range(1, BACK_N + 1):
                    back_seq[t, n - 1] = self._compute_single_features(window_data, n, 'back')

        # normalizuj
        front_seq = (front_seq - self.front_mean) / self.front_std
        if BACK_N > 0:
            back_seq = (back_seq - self.back_mean) / self.back_std

        return front_seq, back_seq

    def build_labels(self, draws: Optional[List[Tuple]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        gradi labele: brojevi poslednjeg kola kao cilj

        Returns:
            front_labels: (FRONT_N,) - 1označava pojavu broja
            back_labels: (BACK_N,)
        """
        if draws is None:
            draws = self.draws
        if not draws:
            return np.zeros(FRONT_N), np.zeros(BACK_N)
        latest = draws[-1]
        f_labels = np.zeros(FRONT_N, dtype=np.float32)
        b_labels = np.zeros(BACK_N, dtype=np.float32)
        for n in latest[0]:
            f_labels[n - 1] = 1.0
        for n in latest[1]:
            b_labels[n - 1] = 1.0
        return f_labels, b_labels


# ============================================================
# PyTorch definicija modela
# ============================================================

if TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        """Transformerpoziciono kodiranje"""
        def __init__(self, d_model: int, max_len: int = 100):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2] if d_model % 2 == 0 else position * div_term[:d_model - d_model // 2])
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)

        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]


    class TabNetModel(nn.Module):
        """
        pojednostavljena verzijaTabNet: attention izbor osobina + potpuno povezan klasifikator.
        ulaz: (batch, NUM_FEATURES) osobine pojedinačnog broja
        izlaz: (batch,) sigmoid verovatnoća
        """
        def __init__(self, input_dim: int = NUM_FEATURES, hidden_dim: int = 64):
            super().__init__()
            self.feature_transformer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            # attention izbor osobina
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, input_dim),
                nn.Softmax(dim=-1),
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim + input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x):
            # x: (batch, features)
            transformed = self.feature_transformer(x)  # (batch, hidden)
            attn_weights = self.attn(transformed)  # (batch, input_dim)
            attended = x * attn_weights  # izbor osobina
            combined = torch.cat([transformed, attended], dim=-1)
            return torch.sigmoid(self.classifier(combined)).squeeze(-1)


    class LSTMModel(nn.Module):
        """
        dvosmerniLSTM + attention pooling.
        ulaz: (batch, seq_len, n_numbers, features)
        izlaz: (batch, n_numbers) verovatnoća
        """
        def __init__(self, input_dim: int = NUM_FEATURES, hidden_dim: int = 64,
                     num_layers: int = 2, n_numbers: int = FRONT_N):
            super().__init__()
            self.n_numbers = n_numbers
            self.hidden_dim = hidden_dim
            self.input_proj = nn.Linear(input_dim, hidden_dim)

            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.2 if num_layers > 1 else 0,
            )
            # attention
            self.attn_fc = nn.Linear(hidden_dim * 2, 1)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            # x: (batch, seq_len, n_numbers, features)
            batch, seq, nums, feats = x.shape
            # spoji batch i nums dimenzija, nezavisno modeluj sekvencu za svaki broj
            x = x.reshape(batch * nums, seq, feats)  # (batch*nums, seq, feats)
            x = self.input_proj(x)  # (batch*nums, seq, hidden)
            lstm_out, _ = self.lstm(x)  # (batch*nums, seq, hidden*2)

            # attention pooling
            attn_scores = self.attn_fc(lstm_out).squeeze(-1)  # (batch*nums, seq)
            attn_weights = F.softmax(attn_scores, dim=-1)
            context = (lstm_out * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch*nums, hidden*2)

            out = torch.sigmoid(self.classifier(context)).squeeze(-1)  # (batch*nums,)
            return out.reshape(batch, nums)


    class TransformerModel(nn.Module):
        """
        Transformerenkoder + sekvencijalni pooling.
        ulaz: (batch, seq_len, n_numbers, features)
        izlaz: (batch, n_numbers) verovatnoća
        """
        def __init__(self, input_dim: int = NUM_FEATURES, d_model: int = 64,
                     nhead: int = 4, num_layers: int = 2, n_numbers: int = FRONT_N):
            super().__init__()
            self.n_numbers = n_numbers
            self.d_model = d_model
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.2, activation='relu',
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, x):
            # x: (batch, seq_len, n_numbers, features)
            batch, seq, nums, feats = x.shape
            x = x.reshape(batch * nums, seq, feats)
            x = self.input_proj(x) * math.sqrt(self.d_model)
            x = self.pos_encoder(x)
            x = self.transformer(x)  # (batch*nums, seq, d_model)

            # uzmi izlaz poslednjeg vremenskog koraka
            context = x[:, -1, :]  # (batch*nums, d_model)
            out = torch.sigmoid(self.classifier(context)).squeeze(-1)
            return out.reshape(batch, nums)


    # ============================================================
    # Trening
    # ============================================================

    class _DLTDataset(Dataset):
        """izIstorijagradi skup kliznim prozorom"""
        def __init__(self, draws: List[Tuple], builder: NeuralFeatureBuilder,
                     zone: str = 'front', seq_len: int = SEQ_LEN):
            self.draws = draws
            self.builder = builder
            self.zone = zone
            self.seq_len = seq_len
            self.n_numbers = FRONT_N if zone == 'front' else BACK_N
            # gradi indeks uzoraka: svaki trening uzorak je (seq_window, target_draw)
            self.sample_indices = []
            min_len = seq_len + 1
            for i in range(min_len, len(draws)):
                self.sample_indices.append(i)

        def __len__(self):
            return len(self.sample_indices)

        def __getitem__(self, idx):
            target_idx = self.sample_indices[idx]
            seq_start = target_idx - self.seq_len
            seq_data = self.draws[seq_start:target_idx]
            target = self.draws[target_idx]

            # gradi sekvencijalne osobine
            seq_feats = []
            for t in range(self.seq_len):
                window_data = self.draws[max(0, seq_start + t - self.builder.window):seq_start + t]
                if len(window_data) < 2:
                    window_data = self.draws[:max(2, self.builder.window)]
                feats_t = []
                n_max = self.n_numbers
                zone_name = 'front' if self.zone == 'front' else 'back'
                for n in range(1, n_max + 1):
                    feats_t.append(self.builder._compute_single_features(window_data, n, zone_name))
                seq_feats.append(feats_t)

            x = np.array(seq_feats, dtype=np.float32)  # (seq_len, n_numbers, features)
            mean = self.builder.front_mean if self.zone == 'front' else self.builder.back_mean
            std = self.builder.front_std if self.zone == 'front' else self.builder.back_std
            x = (x - mean) / std

            # labele
            target_nums = target[0] if self.zone == 'front' else target[1]
            y = np.zeros(self.n_numbers, dtype=np.float32)
            for n in target_nums:
                y[n - 1] = 1.0

            return torch.FloatTensor(x), torch.FloatTensor(y)


    class BaseTrainer:
        """bazna klasa trenera"""
        def __init__(self, model: nn.Module, model_name: str, lr: float = 1e-3,
                     device: str = None):
            self.model = model
            self.model_name = model_name
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3
            )
            self.best_loss = float('inf')
            self.patience_counter = 0
            self.max_patience = 5

        def train_epoch(self, loader: DataLoader) -> float:
            self.model.train()
            total_loss = 0.0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(x)
                # višestruka klasifikacija: BCE
                loss = F.binary_cross_entropy(pred, y, weight=torch.where(y > 0, 3.0, 1.0))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item() * x.size(0)
            return total_loss / len(loader.dataset)

        def validate(self, loader: DataLoader) -> float:
            self.model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = self.model(x)
                    loss = F.binary_cross_entropy(pred, y)
                    total_loss += loss.item() * x.size(0)
            return total_loss / len(loader.dataset)

        def fit(self, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 50, verbose: bool = True) -> Dict:
            history = {'train_loss': [], 'val_loss': []}
            for epoch in range(epochs):
                train_loss = self.train_epoch(train_loader)
                val_loss = self.validate(val_loader)
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                self.scheduler.step(val_loss)

                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"    [{self.model_name}] Epoha {epoch+1}/{epochs} "
                          f"trening={train_loss:.4f} valid={val_loss:.4f} "
                          f"(najbolje={self.best_loss:.4f})")

                if self.patience_counter >= self.max_patience:
                    if verbose:
                        print(f"    [{self.model_name}] Rana zaustavljanja na epohi {epoch+1}")
                    break

            return history

    class TabNetTrainer(BaseTrainer):
        """TabNetTrening"""
        def __init__(self, input_dim: int = NUM_FEATURES, hidden_dim: int = 64,
                     n_numbers: int = FRONT_N, lr: float = 1e-3):
            model = TabNetModel(input_dim, hidden_dim)
            super().__init__(model, 'TabNet', lr)
            self.input_dim = input_dim
            self.n_numbers = n_numbers

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            """
            ulaz: (n_numbers, features) ili (batch, n_numbers, features)
            izlaz: (n_numbers,) verovatnoća
            """
            self.model.eval()
            with torch.no_grad():
                if features.ndim == 2:
                    features = features[np.newaxis, :, :]  # (1, n, f) -> TabNetpotrebno(batch, f) nezavisna obrada, ali prvoreshape
                batch, n, f = features.shape
                # TabNetobrađuj svaki broj nezavisno, zato spljošti
                x = torch.FloatTensor(features.reshape(batch * n, f)).to(self.device)
                pred = self.model(x)  # (batch*n,)
                pred = pred.reshape(batch, n)
            return pred.cpu().numpy()[0] if batch == 1 else pred.cpu().numpy()


    class LSTMTrainer(BaseTrainer):
        """LSTMTrening"""
        def __init__(self, input_dim: int = NUM_FEATURES, hidden_dim: int = 64,
                     num_layers: int = 2, n_numbers: int = FRONT_N,
                     seq_len: int = SEQ_LEN, lr: float = 1e-3):
            model = LSTMModel(input_dim, hidden_dim, num_layers, n_numbers)
            super().__init__(model, 'LSTM', lr)
            self.input_dim = input_dim
            self.n_numbers = n_numbers
            self.seq_len = seq_len

        def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
            """
            ulaz: (seq_len, n_numbers, features)
            izlaz: (n_numbers,) verovatnoća
            """
            self.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # (1, seq, n, f)
                pred = self.model(x)  # (1, n)
            return pred.cpu().numpy()[0]


    class TransformerTrainer(BaseTrainer):
        """TransformerTrening"""
        def __init__(self, input_dim: int = NUM_FEATURES, d_model: int = 64,
                     nhead: int = 4, num_layers: int = 2, n_numbers: int = FRONT_N,
                     seq_len: int = SEQ_LEN, lr: float = 1e-3):
            model = TransformerModel(input_dim, d_model, nhead, num_layers, n_numbers)
            super().__init__(model, 'Transformer', lr)
            self.input_dim = input_dim
            self.n_numbers = n_numbers
            self.seq_len = seq_len

        def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
            """
            ulaz: (seq_len, n_numbers, features)
            izlaz: (n_numbers,) verovatnoća
            """
            self.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                pred = self.model(x)
            return pred.cpu().numpy()[0]

else:
    # kada PyTorch nije dostupanrezerva
    _rezerva_warned = False

    class _StubModel:
        def __init__(self, name: str):
            global _rezerva_warned
            if not _rezerva_warned:
                print(f"  ⚠️ [NeuralModels] PyTorch nije dostupan, {name} koristi rezervu")
                _rezerva_warned = True
            self.name = name

        def predict_proba(self, *args, **kwargs) -> np.ndarray:
            return np.ones(FRONT_N) * 0.5

    class TabNetTrainer:
        def __init__(self, **kwargs):
            self._model = _StubModel('TabNet')
        def predict_proba(self, features):
            return self._model.predict_proba(features)

    class LSTMTrainer:
        def __init__(self, **kwargs):
            self._model = _StubModel('LSTM')
        def predict_proba(self, sequence):
            return self._model.predict_proba(sequence)

    class TransformerTrainer:
        def __init__(self, **kwargs):
            self._model = _StubModel('Transformer')
        def predict_proba(self, sequence):
            return self._model.predict_proba(sequence)


# ============================================================
# ansambl neuronske mreže
# ============================================================

class NeuralEnsemble:
    """
    ansambl neuronske mreže sa tri modela.

    strategija korišćenja:
    - TabNet: klasifikacija na osnovu osobina pojedinačnog broja (brza inferencija)
    - LSTM: modelovanje vremenske zavisnosti kroz sekvence
    - Transformer: self-attention hvata globalne obrasce

    ansambl: ponderisana sredina izlaza tri modela
    """

    def __init__(self, draws: Optional[List[Tuple]] = None,
                 seq_len: int = SEQ_LEN, window: int = WINDOW,
                 train_epochs: int = 50, auto_train: bool = True,
                 verbose: bool = True):
        self.draws = draws
        self.seq_len = seq_len
        self.window = window
        self.train_epochs = train_epochs
        self.is_trained = False
        self.train_history = {}
        self._ensemble_weights = {'tabnet': 0.25, 'lstm': 0.35, 'transformer': 0.40}

        # graditelj osobina
        self.feature_builder = None
        # modeli
        self.tabnet_front = None
        self.tabnet_back = None
        self.lstm_front = None
        self.lstm_back = None
        self.transformer_front = None
        self.transformer_back = None

        if draws is not None and auto_train:
            self.train(draws, verbose=verbose)

    def _init_models(self, n_numbers: int, zone: str):
        """inicijalizacija tri modela za zadatu zonu"""
        if zone == 'front':
            self.tabnet_front = TabNetTrainer(n_numbers=n_numbers)
            self.lstm_front = LSTMTrainer(n_numbers=n_numbers, seq_len=self.seq_len)
            self.transformer_front = TransformerTrainer(n_numbers=n_numbers, seq_len=self.seq_len)
        else:
            self.tabnet_back = TabNetTrainer(n_numbers=n_numbers)
            self.lstm_back = LSTMTrainer(n_numbers=n_numbers, seq_len=self.seq_len)
            self.transformer_back = TransformerTrainer(n_numbers=n_numbers, seq_len=self.seq_len)

    def train(self, draws: List[Tuple], verbose: bool = True) -> Dict:
        """Trening svih modela na istoriji."""
        self.draws = draws
        if not TORCH_AVAILABLE:
            if verbose:
                print("  ⚠️ [NeuralEnsemble] PyTorch nije dostupan, preskačem trening")
            self.is_trained = True
            return {}

        if len(draws) < self.seq_len + 10:
            if verbose:
                print(f"  ⚠️ [NeuralEnsemble] Nedovoljno podataka ({len(draws)} kola), preskačem trening")
            self.is_trained = True
            return {}

        if verbose:
            print(f"  [NeuralEnsemble] Početak treninga ({len(draws)} kola)...")

        # gradi ekstraktor osobina
        self.feature_builder = NeuralFeatureBuilder(draws, self.window)

        # inicijalizacija modela
        self._init_models(FRONT_N, 'front')
        if BACK_N > 0:
            self._init_models(BACK_N, 'back')

        # pripremi podatke
        batch_size = min(64, max(8, len(draws) // 20))

        history = {}
        zone_models = [
            ('front', [
                (self.tabnet_front, 'TabNet-F'),
                (self.lstm_front, 'LSTM-F'),
                (self.transformer_front, 'TF-F'),
            ]),
        ]
        if BACK_N > 0:
            zone_models.append(('back', [
                (self.tabnet_back, 'TabNet-B'),
                (self.lstm_back, 'LSTM-B'),
                (self.transformer_back, 'TF-B'),
            ]))
        for zone, models in zone_models:
            zona_sr = "GLAVNA ZONA" if zone == "front" else "DODATNA ZONA"
            if verbose:
                print(f"  ── {zona_sr} ──")

            for trainer, name in models:
                if verbose:
                    print(f"    Trening {name}...")
                dataset = _DLTDataset(draws, self.feature_builder, zone, self.seq_len)
                if len(dataset) < 10:
                    continue
                split = int(len(dataset) * 0.8)
                train_ds, val_ds = torch.utils.data.random_split(
                    dataset, [split, len(dataset) - split],
                    generator=torch.Generator().manual_seed(39),
                )
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=batch_size)
                hist = trainer.fit(train_loader, val_loader, epochs=self.train_epochs, verbose=verbose)
                history[f'{name}'] = hist

        self.is_trained = True
        self.train_history = history

        if verbose:
            print(f"  ✅ [NeuralEnsemble] Trening završen")

        return history

    def train_all(self, epochs: int = 50, verbose: bool = True) -> Dict:
        """Alias za loto7_neural — trening sa zadatim brojem epoha."""
        self.train_epochs = epochs
        if not self.draws:
            return {}
        return self.train(self.draws, verbose=verbose)

    def predict_probabilities(self) -> Dict[str, Dict[int, float]]:
        """Verovatnoće po broju 1..39 za Loto 7/39."""
        draws = self.draws or []
        front_probs, back_probs = self._compute_number_probs(draws)
        out: Dict[str, Dict[int, float]] = {
            "front": {n: float(front_probs[n - 1]) for n in range(1, FRONT_N + 1)},
        }
        if BACK_N > 0:
            out["back"] = {n: float(back_probs[n - 1]) for n in range(1, BACK_N + 1)}
        return out

    def _compute_number_probs(self, draws: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        izračunaj verovatnoću svakog broja.

        Returns:
            front_probs: (39,) ndarray — verovatnoća pojavljivanja brojeva 1–39
            back_probs: (0,) ndarray — Loto 7/39 nema dodatnu zonu
        """
        if not self.is_trained:
            return np.ones(FRONT_N) * 0.5, np.ones(BACK_N) * 0.5

        if not TORCH_AVAILABLE:
            return np.ones(FRONT_N) * 0.5, np.ones(BACK_N) * 0.5

        fb = self.feature_builder

        # gradi sekvencijalne podatke
        front_seq, back_seq = fb.build_sequence(draws, self.seq_len)

        # TabNet: gradi osobine iz nedavnog prozora
        window_data = draws[-min(self.window, len(draws)):]
        front_tabnet_feats = np.array([
            fb._compute_single_features(window_data, n, 'front')
            for n in range(1, FRONT_N + 1)
        ], dtype=np.float32)
        front_tabnet_feats = (front_tabnet_feats - fb.front_mean) / fb.front_std

        back_tabnet_feats = None
        if BACK_N > 0:
            back_tabnet_feats = np.array([
                fb._compute_single_features(window_data, n, 'back')
                for n in range(1, BACK_N + 1)
            ], dtype=np.float32)
            back_tabnet_feats = (back_tabnet_feats - fb.back_mean) / fb.back_std

        # inferencija
        front_probs = np.zeros(FRONT_N)
        back_probs = np.zeros(max(BACK_N, 0))

        try:
            fp_t = self.tabnet_front.predict_proba(front_tabnet_feats[np.newaxis, :, :])
            fp_l = self.lstm_front.predict_proba(front_seq)
            fp_tf = self.transformer_front.predict_proba(front_seq)
            front_probs = (
                self._ensemble_weights['tabnet'] * fp_t +
                self._ensemble_weights['lstm'] * fp_l +
                self._ensemble_weights['transformer'] * fp_tf
            )
        except Exception as e:
            print(f"  ⚠️ [NeuralEnsemble] Greška inferencije glavne zone: {e}")
            front_probs = np.ones(FRONT_N) * 0.5

        if BACK_N > 0:
            try:
                bp_t = self.tabnet_back.predict_proba(back_tabnet_feats[np.newaxis, :, :])
                bp_l = self.lstm_back.predict_proba(back_seq)
                bp_tf = self.transformer_back.predict_proba(back_seq)
                back_probs = (
                    self._ensemble_weights['tabnet'] * bp_t +
                    self._ensemble_weights['lstm'] * bp_l +
                    self._ensemble_weights['transformer'] * bp_tf
                )
            except Exception as e:
                print(f"  ⚠️ [NeuralEnsemble] Greška inferencije dodatne zone: {e}")
                back_probs = np.ones(BACK_N) * 0.5

        return front_probs, back_probs

    def score_candidate(self, front: List[int], back: List[int],
                        draws: List[Tuple]) -> float:
        """
        za pojedinačnukandidataskor kombinacije brojeva (0~1).

        metoda ocenjivanja: kandidataprosečna log-verovatnoća brojeva
        - uzmi geometrijsku sredinu verovatnoća brojeva (prosek u log-prostoru)
        - normalizuj na0~1
        """
        front_probs, back_probs = self._compute_number_probs(draws)

        f_log_probs = [np.log(max(p, 1e-6)) for p in [front_probs[n - 1] for n in front]]
        f_score = np.exp(np.mean(f_log_probs))
        if BACK_N > 0 and back:
            b_log_probs = [np.log(max(p, 1e-6)) for p in [back_probs[n - 1] for n in back]]
            b_score = np.exp(np.mean(b_log_probs))
            combined = f_score * 0.7 + b_score * 0.3
        else:
            combined = f_score

        # normalizuj na 0~1
        normalized = 1.0 / (1.0 + np.exp(-5.0 * (combined - 0.15)))

        return float(normalized)

    def score_batch(self, candidates: List[Dict], draws: List[Tuple]) -> List[Dict]:
        """
        grupno ocenjivanjekandidatakombinacija.

        Args:
            candidates: [{'front': [5], 'back': [2]}, ...]
            draws: Istorija

        Returns:
            ista lista, svaki element dobija'neural_score'polje
        """
        if not candidates:
            return candidates

        front_probs, back_probs = self._compute_number_probs(draws)

        for c in candidates:
            front = c.get('front', [])
            back = c.get('back', [])
            if len(front) != FRONT_SELECT or (BACK_N > 0 and len(back) != BACK_SELECT):
                c['neural_score'] = 0.5
                continue

            f_log_probs = [np.log(max(front_probs[n - 1], 1e-6)) for n in front]
            f_score = np.exp(np.mean(f_log_probs))
            if BACK_N > 0 and back:
                b_log_probs = [np.log(max(back_probs[n - 1], 1e-6)) for n in back]
                b_score = np.exp(np.mean(b_log_probs))
                combined = f_score * 0.7 + b_score * 0.3
            else:
                combined = f_score
            normalized = float(1.0 / (1.0 + np.exp(-5.0 * (combined - 0.15))))
            c['neural_score'] = normalized

        return candidates

    def save(self, path: str):
        """sačuvaj model u fajl"""
        state = {
            'ensemble_weights': self._ensemble_weights,
            'seq_len': self.seq_len,
            'window': self.window,
            'is_trained': self.is_trained,
            'train_history': self.train_history,
        }
        if TORCH_AVAILABLE and self.is_trained:
            for name in ['tabnet_front', 'tabnet_back', 'lstm_front', 'lstm_back',
                         'transformer_front', 'transformer_back']:
                model = getattr(self, name, None)
                if model and hasattr(model, 'model'):
                    state[f'{name}_state'] = model.model.state_dict()
            if self.feature_builder:
                state['front_mean'] = self.feature_builder.front_mean.tolist()
                state['front_std'] = self.feature_builder.front_std.tolist()
                state['back_mean'] = self.feature_builder.back_mean.tolist()
                state['back_std'] = self.feature_builder.back_std.tolist()

        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str, draws: Optional[List[Tuple]] = None):
        """učitaj model"""
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            self._ensemble_weights = state.get('ensemble_weights', self._ensemble_weights)
            self.seq_len = state.get('seq_len', self.seq_len)
            self.window = state.get('window', self.window)
            self.is_trained = state.get('is_trained', False)
            self.train_history = state.get('train_history', {})

            if draws is not None:
                self.feature_builder = NeuralFeatureBuilder(draws, self.window)
                if 'front_mean' in state:
                    self.feature_builder.front_mean = np.array(state['front_mean'])
                    self.feature_builder.front_std = np.array(state['front_std'])
                    self.feature_builder.back_mean = np.array(state['back_mean'])
                    self.feature_builder.back_std = np.array(state['back_std'])

            if TORCH_AVAILABLE and self.is_trained:
                self._init_models(FRONT_N, 'front')
                if BACK_N > 0:
                    self._init_models(BACK_N, 'back')
                for name in ['tabnet_front', 'tabnet_back', 'lstm_front', 'lstm_back',
                             'transformer_front', 'transformer_back']:
                    model = getattr(self, name, None)
                    if model and hasattr(model, 'model') and f'{name}_state' in state:
                        model.model.load_state_dict(state[f'{name}_state'])

            return True
        except Exception as e:
            print(f"  ⚠️ [NeuralEnsemble] Greška učitavanja modela: {e}")
            return False


# Alias for framework compatibility
DLTFusionNet = NeuralEnsemble

__all__ = [
    'NeuralEnsemble',
    'NeuralFeatureBuilder',
    'TabNetTrainer',
    'LSTMTrainer',
    'TransformerTrainer',
    'FRONT_N',
    'BACK_N',
    'NUM_FEATURES',
    'SEQ_LEN',
]
