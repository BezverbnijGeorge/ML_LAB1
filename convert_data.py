import pandas as pd

# ==========================================
# 1. ЗАВАНТАЖЕННЯ ДАНИХ
# ==========================================
df = pd.read_csv('adac_raw_data.csv')

print("--- ТАБЛИЦЯ ДО ПЕРЕТВОРЕННЯ (Сирі дані) ---")
print(df[['vehicle_class', 'model', 'year', 'defect_rate']].head())

# ==========================================
# 2. ПЕРЕТВОРЕННЯ (Feature Engineering)
# ==========================================
# 1. Вік авто
df['age_years'] = 2025 - df['year']

# 2. Витягуємо марку (перше слово) та робимо точкові виправлення
df['brand'] = df['model'].apply(lambda x: x.split()[0])
df.loc[df['model'].str.contains('Mercedes'), 'brand'] = 'Mercedes'
df.loc[df['model'].str.contains('Land Rover'), 'brand'] = 'Land Rover'
df.loc[df['model'].str.contains('MINI'), 'brand'] = 'MINI'
df.loc[df['model'].str.contains('smart', case=False), 'brand'] = 'smart'

# 3. Нова логіка інженерних груп (Nissan об'єднано з Renault/Dacia)
group_mapping = {
    # VAG
    'VW': 'VAG', 'Audi': 'VAG', 'Skoda': 'VAG', 'SEAT': 'VAG', 'CUPRA': 'VAG', 'Porsche': 'VAG',
    # Stellantis
    'Peugeot': 'Stellantis', 'Citroen': 'Stellantis', 'Opel': 'Stellantis', 'Fiat': 'Stellantis',
    # Japanese (БЕЗ Nissan)
    'Toyota': 'Japanese', 'Mazda': 'Japanese', 'Suzuki': 'Japanese', 'Mitsubishi': 'Japanese', 'Honda': 'Japanese',
    # Mercedes-Benz Group
    'Mercedes': 'MB_Group', 'smart': 'MB_Group',
    # Ford
    'Ford': 'Ford_Europe',
    # BMW Group
    'BMW': 'BMW_Group', 'MINI': 'BMW_Group',
    # Альянс Renault-Nissan-Dacia
    'Renault': 'Renault_Dacia_Nissan', 'Dacia': 'Renault_Dacia_Nissan', 'Nissan': 'Renault_Dacia_Nissan',
    # Korean
    'Hyundai': 'Korean', 'Kia': 'Korean'
}

# Застосовуємо мапінг. Все, що не знайшлося в словнику (Tesla, Volvo, Land Rover, MG), отримує значення 'Other'
df['origin_group'] = df['brand'].map(group_mapping).fillna('Other')

# 4. Цільова змінна (Target): чи є ризик поломки високим (> 15 на 1000 авто)
df['is_high_risk'] = (df['defect_rate'] > 15.0).astype(int)

# Очищуємо від порожніх рядків (де не було статистики в оригіналі)
df_clean = df.dropna(subset=['defect_rate']).copy()

# ==========================================
# 3. ФОРМУВАННЯ ФІНАЛЬНОГО ДАТАСЕТУ
# ==========================================
# Залишаємо лише 3 ознаки та цільову змінну
final_columns = ['age_years', 'vehicle_class', 'origin_group', 'is_high_risk']
df_final = df_clean[final_columns]

print("\n--- ТАБЛИЦЯ ПІСЛЯ ПЕРЕТВОРЕННЯ (Готово для ML) ---")
print(df_final.head(10))

# Покажемо розподіл за новими групами, щоб перевірити, чи немає перекосів
print("\n--- РОЗПОДІЛ АВТОМОБІЛІВ ЗА ГРУПАМИ ---")
print(df_final['origin_group'].value_counts())

# Зберігаємо готовий датасет
df_final.to_csv('adac_ml_ready.csv', index=False)
print(f"\nГотово! Збережено рядків: {len(df_final)}. Датасет збережено як 'adac_ml_ready.csv'.")