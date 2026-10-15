import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="OLX KZ Laptop Deal Scorer", page_icon="💻", layout="wide")

@st.cache_data
def load_csv_data():
    df_full = pd.read_csv("data/olx_laptops_processed_features.csv")
    df_deals = pd.read_csv("data/olx_test_scored_deals.csv")
    return df_full, df_deals

@st.cache_resource
def load_artifacts():
    return joblib.load("models/laptop_scorer_artifacts.pkl")

df_full, df_deals = load_csv_data()
art = load_artifacts()
q_models = art['quantile_models']
ord_enc = art['ordinal_encoder']
cat_cols = art['cat_cols']
feature_cols = art['feature_cols']
group_medians = art['group_medians']
gpu_medians = art['gpu_medians']
global_med = art['global_med']

st.title("💻 Система оценки и поиска недооцененных ноутбуков на OLX.kz")
st.markdown("**Data Science Capstone Project:** *Квантильный градиентный бустинг (Q15 / Q50 / Q85) и Rule-Based парсинг объявлений*")

tab1, tab2, tab3 = st.tabs([
    " Сканер горячих сделок (Для реселлера)",
    " Калькулятор Trade-In и Справедливой цены",
    " Аналитика рынка РК (EDA)"
])

with tab1:
    st.subheader("Автоматический поиск выгодных объявлений (Тестовая витрина)")
    col1, col2, col3 = st.columns(3)
    with col1:
        verdict_filter = st.multiselect(
            "Статус сделки:",
            options=df_deals['deal_verdict'].unique(),
            default=['ГОРЯЧИЙ ЛОТ (Выкуп)', 'Выгодная сделка']
        )
    with col2:
        city_filter = st.multiselect(
            "Город:",
            options=sorted(df_deals['city_clean'].unique()),
            default=sorted(df_deals['city_clean'].unique())
        )
    with col3:
        min_profit = st.slider("Минимальная потенциальная выгода (₸):", 0, 250000, 30000, step=10000)

    filtered = df_deals[
        (df_deals['deal_verdict'].isin(verdict_filter)) &
        (df_deals['city_clean'].isin(city_filter)) &
        (df_deals['potential_profit_kzt'] >= min_profit)
    ].sort_values(by='potential_profit_kzt', ascending=False)

    m1, m2, m3 = st.columns(3)
    m1.metric("Найдено выгодных лотов", f"{len(filtered)} шт.")
    m2.metric("Средняя потенциальная выгода", f"{filtered['potential_profit_kzt'].mean():,.0f} ₸" if len(filtered) else "0 ₸")
    m3.metric("Средний дисконт к рынку", f"{filtered['discount_pct'].mean():.1f}%" if len(filtered) else "0%")

    show_cols = [
        'title', 'city', 'brand', 'gpu_exact', 'cpu_family', 'ram_gb', 'ssd_gb',
        'price_kzt', 'fast_buyout_q15', 'fair_price_q50', 'potential_profit_kzt', 'discount_pct', 'deal_verdict', 'url'
    ]
    st.dataframe(filtered[show_cols], use_container_width=True, height=460)

with tab2:
    st.subheader("Расчет ценового коридора ноутбука (Быстрый выкуп — Рынок — Оверпрайс)")
    c1, c2, c3 = st.columns(3)
    with c1:
        in_brand = st.selectbox("Бренд:", sorted(df_full['brand'].unique()))
        in_lineup = st.selectbox("Сегмент ноутбука:", sorted(df_full['lineup'].unique()), index=2)
        in_city = st.selectbox("Город:", sorted(df_full['city_clean'].unique()))
        in_price = st.number_input("Цена продавца в объявлении (₸):", min_value=25000, max_value=2200000, value=300000, step=10000)
    with c2:
        in_gpu = st.selectbox("Модель видеокарты (GPU):", sorted(df_full['gpu_exact'].unique()))
        gpu_score_map = df_full.groupby('gpu_exact')['gpu_score'].median().to_dict()
        in_cpu = st.selectbox("Семейство процессора (CPU):", sorted(df_full['cpu_family'].unique()))
        cpu_tier_map = df_full.groupby('cpu_family')['cpu_tier'].median().to_dict()
        in_year = st.slider("Ориентировочный год выпуска:", 2017, 2025, 2022)
    with c3:
        in_ram = st.select_slider("Оперативная память (RAM, ГБ):", options=[4, 8, 12, 16, 24, 32, 64], value=16)
        in_ssd = st.select_slider("Объем SSD (ГБ):", options=[128, 256, 512, 1024, 2048], value=512)
        in_screen = st.selectbox("Диагональ экрана:", [13.3, 14.0, 15.6, 16.0, 17.3, 18.0], index=2)
        in_hz = st.checkbox("Экран 144Hz+ / OLED / 2K", value=True)
        in_new = st.checkbox("Состояние нового / Запечатан", value=False)
        in_urgent = st.checkbox("Маркер «Срочно / Торг»", value=False)
        in_defect = st.checkbox("Есть дефекты / Старый офисный", value=False)

    hw_key = f"{in_brand}_{in_gpu}_{in_cpu}"
    hw_anchor = group_medians.get(hw_key, gpu_medians.get(in_gpu, global_med))

    row_df = pd.DataFrame([{
        'brand': in_brand, 'lineup': in_lineup, 'gpu_exact': in_gpu,
        'cpu_family': in_cpu, 'city_clean': in_city,
        'gpu_score': gpu_score_map.get(in_gpu, 3.0),
        'cpu_tier': cpu_tier_map.get(in_cpu, 4.0),
        'est_year': in_year, 'ram_gb': float(in_ram), 'ssd_gb': float(in_ssd),
        'screen_inch': float(in_screen), 'is_high_hz_or_oled': int(in_hz),
        'is_new_sealed': int(in_new), 'is_pawnshop_or_installment': 0,
        'is_urgent_or_bargain': int(in_urgent), 'has_defect_or_old': int(in_defect),
        'is_megapolis': int(in_city in ['Астана', 'Алматы']),
        'spec_detail_score': 5, 'hw_market_anchor': hw_anchor
    }])[feature_cols]

    row_enc = row_df.copy()
    row_enc[cat_cols] = ord_enc.transform(row_enc[cat_cols])

    p_q15 = round(np.expm1(q_models['q15'].predict(row_enc)[0]), -3)
    p_q50 = round(np.expm1(q_models['q50'].predict(row_enc)[0]), -3)
    p_q85 = round(np.expm1(q_models['q85'].predict(row_enc)[0]), -3)
    p_q15 = min(p_q15, round(p_q50 * 0.90, -3))
    p_q85 = max(p_q85, round(p_q50 * 1.10, -3))

    profit = p_q50 - in_price
    disc_pct = (profit / p_q50) * 100

    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(" Быстрый выкуп (Q15)", f"{p_q15:,.0f} ₸")
    k2.metric(" Справедливая цена (Q50)", f"{p_q50:,.0f} ₸")
    k3.metric(" Верх рынка (Q85)", f"{p_q85:,.0f} ₸")
    k4.metric(" Потенциал выгоды", f"{profit:+,.0f} ₸ ({disc_pct:+.1f}%)")

    if in_price <= p_q15 and not in_defect:
        st.success(" **ГОРЯЧИЙ ЛОТ!** Цена ниже 15-го перцентиля рынка. Рекомендуется к срочному выкупу под перепродажу.")
    elif in_price < p_q50:
        st.info(" **ВЫГОДНАЯ СДЕЛКА.** Цена ниже медианной рыночной стоимости.")
    elif in_price <= p_q85:
        st.warning(" **В РЫНКЕ.** Стандартное рыночное предложение.")
    else:
        st.error(" **ОВЕРПРАЙС.** Цена выше 85-го перцентиля рынка — покупка нецелесообразна.")

with tab3:
    st.subheader("Разведывательный анализ данных (EDA) рынка ноутбуков РК")
    st.image("eda_summary_charts.png", caption="Сводная аналитика цен по видеокартам, брендам и регионам Казахстана", use_container_width=True)
