import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL bağlantısı
engine = create_engine('postgresql://postgres:1234@localhost:5432/iyzico_db')

# Veritabanındaki iyzico_transactions tablosunu Pandas DataFrame'e oku
df = pd.read_sql("SELECT * FROM iyzico_transactions", engine)

# DataFrame'i CSV dosyası olarak dışa aktar
df.to_csv("iyzico_data_from_db.csv", index=False)

print("Veriler veritabanından başarıyla 'iyzico_data_from_db.csv' dosyasına aktarıldı!")