import os
import gc
import numerapi
import pyarrow.parquet as pq
import pandas as pd
import lightgbm as lgb

# --- 1. KREDENCJAŁY (Pobierane z GitHub Secrets) ---
print("Pobieranie kluczy z bezpiecznego środowiska serwera...")
PUBLIC_ID = os.environ.get("NUMERAI_PUBLIC_ID")
SECRET_KEY = os.environ.get("NUMERAI_SECRET_KEY")

if not PUBLIC_ID or not SECRET_KEY:
    raise ValueError("Brak NUMERAI_PUBLIC_ID lub NUMERAI_SECRET_KEY w zmiennych środowiskowych.")

# Twój MODEL_ID dla pierwszego slota
NOWY_MODEL_ID = "e171538a-7f83-47fe-bfb0-17eaab8076f7"

napi = numerapi.NumerAPI(PUBLIC_ID, SECRET_KEY)

# --- 2. POBIERANIE DANYCH ---
if not os.path.exists("train.parquet"):
    print("Pobieram brakujące dane train...")
    napi.download_dataset("v5.2/train.parquet", "train.parquet")

current_round = napi.get_current_round()
live_remote_path = "v5.2/live.parquet"
live_local_path = "live.parquet"

print(f"Pobieram dane live dla rundy {current_round}...")
napi.download_dataset(live_remote_path, live_local_path)

# --- 3. ODTWARZANIE STAREGO MODELU (20 000 WIERSZY) ---
print("\nOdtwarzanie oryginalnego modelu (20 000 wierszy)...")
pf_train = pq.ParquetFile("train.parquet")
first_batch = next(
    pf_train.iter_batches(batch_size=20000, use_pandas_metadata=True)
)
train_df = first_batch.to_pandas()

features = [f for f in train_df.columns if f.startswith("feature")]
target = "target"

if target not in train_df.columns:
    raise ValueError("Brak kolumny 'target' w train.parquet")

model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=3,
    n_jobs=-1,
    random_state=42
)

model.fit(train_df[features], train_df[target])
print("✅ Stary model wytrenowany ponownie!")

del train_df, first_batch, pf_train
gc.collect()

# --- 4. GENEROWANIE PROGNOZ ---
print("\nGenerowanie prognoz...")
pf_live = pq.ParquetFile(live_local_path)

all_preds = []
all_ids = []

for batch in pf_live.iter_batches(batch_size=10000, use_pandas_metadata=True):
    batch_df = batch.to_pandas()

    missing_features = [f for f in features if f not in batch_df.columns]
    if missing_features:
        raise ValueError(f"Brak feature'ów w live.parquet: {missing_features[:10]}")

    if "id" in batch_df.columns:
        batch_ids = batch_df["id"].tolist()
    else:
        batch_ids = batch_df.index.tolist()

    preds = model.predict(batch_df[features])
    preds = pd.Series(preds).clip(0, 1).tolist()

    all_ids.extend(batch_ids)
    all_preds.extend(preds)

# --- 5. ZAPIS SUBMISSION ---
print("\nZapis pliku submission...")
submission = pd.DataFrame({
    "id": all_ids,
    "prediction": all_preds
})

submission_file = "lucky_predictions.csv"
submission.to_csv(submission_file, index=False)
print(f"✅ Zapisano {submission_file}")

# --- 6. WYSYŁKA ---
print("\nWysyłka do Numerai...")
try:
    napi.upload_predictions(submission_file, model_id=NOWY_MODEL_ID)
    print("🎉 SUKCES! Twój model został wysłany z GitHub Actions!")
except Exception as e:
    print(f"❌ BŁĄD WYSYŁKI: {e}")
