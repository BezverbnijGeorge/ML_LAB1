import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager


def scrape_adac_authenticated():
    # Запускаємо браузер
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    url = "https://www.adac.de/rund-ums-fahrzeug/unfall-schaden-panne/adac-pannenstatistik/"
    driver.get(url)

    # =========================================================
    # МАГІЯ ТУТ: Зупиняємо скрипт і чекаємо на ваші дії
    # =========================================================
    print("\n" + "=" * 60)
    print("УВАГА: Браузер відкрито!")
    print("1. Увійдіть у свій акаунт ADAC у вікні браузера.")
    print("2. Закрийте всі банери з кукі-файлами, якщо вони є.")
    print("3. Переконайтеся, що таблиця зі статистикою стала видимою.")
    print("4. ПІСЛЯ ЦЬОГО поверніться сюди (в консоль) і натисніть ENTER.")
    print("=" * 60 + "\n")

    input("Натисніть ENTER, коли будете готові продовжувати парсинг...")
    # =========================================================

    print("Починаємо збір даних...")
    all_data = []

    # Знаходимо випадаючий список класів авто
    select_element = driver.find_element(By.ID, "breakdown-statistics-vehicle-class-select")
    select = Select(select_element)
    num_classes = len(select.options)

    # Далі йде той самий цикл, що і раніше
    for i in range(num_classes):
        class_name = select.options[i].text
        print(f"Парсимо: {class_name}...")

        select.select_by_index(i)
        time.sleep(2)  # Чекаємо, поки React оновить таблицю

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        table = soup.find('table', {'data-testid': 'breakdown-statistics-results'})
        if not table:
            print(f"Таблицю для {class_name} не знайдено!")
            continue

        thead = table.find('thead')
        headers = [th.text.replace('Перша реєстрація в', '').strip() for th in thead.find_all('th')]

        tbody = table.find('tbody')
        for tr in tbody.find_all('tr'):
            cols = tr.find_all('td')
            if not cols: continue

            model_name = cols[0].text.strip()
            row = {'vehicle_class': class_name, 'model': model_name}

            for idx, col in enumerate(cols[1:]):
                year = headers[idx + 1]
                val = col.text.strip()

                if val == '---':
                    row[year] = np.nan
                else:
                    row[year] = float(val.replace(',', '.'))

            all_data.append(row)

    driver.quit()

    # Перетворення у плоский формат
    df_wide = pd.DataFrame(all_data)
    years_cols = [c for c in df_wide.columns if c.isdigit()]
    df_final = df_wide.melt(
        id_vars=['vehicle_class', 'model'],
        value_vars=years_cols,
        var_name='year',
        value_name='defect_rate'
    )

    df_final = df_final.dropna(subset=['defect_rate']).reset_index(drop=True)
    df_final = df_final.sort_values(by=['vehicle_class', 'model', 'year'], ascending=[True, True, False])

    df_final.to_csv("adac_raw_data.csv", index=False)
    print("Готово! Дані успішно зібрано.")


if __name__ == "__main__":
    scrape_adac_authenticated()