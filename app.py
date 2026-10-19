import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="CRS Surveillance Dashboard", layout="wide")
st.title("CRS Surveillance Data Processing & Analytics Dashboard")

# ---------------------------------------------------------
# HELPER FUNCTIONS (MATCHING MS ACCESS LOGIC PRECISELY)
# ---------------------------------------------------------

def clean_str(val):
    if pd.isna(val):
        return ""
    return str(val).strip().lower()

def check_suspect(row):
    a_cols = ['A_CHD', 'A_CC', 'A_CG', 'A_PR', 'A_HI']
    has_gejala_a = any(clean_str(row.get(c)) in ['ya', 'y', '1', 'true'] for c in a_cols)
    
    age_y = row.get('AGE_Y')
    age_m = row.get('AGE_M')
    
    val_age_y = float(age_y) if pd.notna(age_y) and str(age_y).replace('.','',1).isdigit() else 0
    val_age_m = float(age_m) if pd.notna(age_m) and str(age_m).replace('.','',1).isdigit() else 0
    
    if (val_age_y < 1) and (val_age_m < 12) and has_gejala_a:
        return "Y"
    return "N"

def check_clinical(row):
    a_cols = ['A_CHD', 'A_CC', 'A_CG', 'A_PR', 'A_HI']
    b_cols = ['B_PU', 'B_MC', 'B_ME', 'B_IC', 'B_SP', 'B_DD', 'B_RBD']
    
    a_cnt = sum(1 for c in a_cols if clean_str(row.get(c)) in ['ya', 'y'])
    b_cnt = sum(1 for c in b_cols if clean_str(row.get(c)) in ['ya', 'y'])
    
    return "Y" if (a_cnt >= 2 or (a_cnt >= 1 and b_cnt >= 1)) else "N"

def run_all_updates(df):
    df = df.copy()
    if 'DOCHECK' in df.columns and 'DOB' in df.columns:
        docheck = pd.to_datetime(df['DOCHECK'], errors='coerce')
        dob = pd.to_datetime(df['DOB'], errors='coerce')
        days_diff = (docheck - dob).dt.days
        
        cond_empty_age = df['AGE_M'].isna() | df['AGE_Y'].isna() | ((df['AGE_M'].fillna(0) == 0) & (df['AGE_Y'].fillna(0) == 0) & df['AGE_D'].isna())
        df.loc[cond_empty_age, 'AGE_D'] = days_diff[cond_empty_age].fillna(0)
        df.loc[cond_empty_age, 'AGE_M'] = (days_diff[cond_empty_age] // 30).fillna(0)
        df.loc[cond_empty_age, 'AGE_Y'] = (days_diff[cond_empty_age] // 365).fillna(0)
        
    if 'H_SPEC_COLL' in df.columns:
        df['H_SPEC_COLL'] = df['H_SPEC_COLL'].fillna("Tidak Tahu")
        
        if 'H_DCOLLSER1' in df.columns:
            if 'H_DRES_IGM_S1' in df.columns:
                cond3 = df['H_DCOLLSER1'].isna() & df['H_DRES_IGM_S1'].notna() & (df['H_SPEC_COLL'].apply(clean_str) == 'ya')
                df.loc[cond3, 'H_DCOLLSER1'] = df.loc[cond3, 'H_DRES_IGM_S1']
            
            if 'DOCHECK' in df.columns:
                cond4 = df['H_DCOLLSER1'].isna() & (df['H_SPEC_COLL'].apply(clean_str) == 'ya')
                df.loc[cond4, 'H_DCOLLSER1'] = df.loc[cond4, 'DOCHECK']
                
    return df

def run_classification_final(df):
    df = df.copy()
    df['Suspect_CRS'] = df.apply(check_suspect, axis=1)
    df['Clinical_CRS'] = df.apply(check_clinical, axis=1)
    
    def classify_row(r):
        age_m = float(r['AGE_M']) if pd.notna(r.get('AGE_M')) and str(r.get('AGE_M')).replace('.','',1).isdigit() else 0
        igm1 = clean_str(r.get('H_IGM_S1'))
        igm2 = clean_str(r.get('H_IGM_S2'))
        igg1 = clean_str(r.get('H_IGG_S1'))
        igg2 = clean_str(r.get('H_IGG_S2'))
        clinical = r.get('Clinical_CRS', 'N')
        spec = clean_str(r.get('H_SPEC_COLL'))
        
        if (age_m < 6 and igm1 == "positif") or \
           (age_m < 1 and igm1 == "negatif" and igm2 == "positif") or \
           (age_m > 5 and igm1 == "positif" and igg1 == "positif") or \
           (age_m > 5 and igm1 == "negatif" and igg1 == "positif" and igg2 == "positif"):
            return "CRS pasti(Lab Positif)"
            
        if (1 <= age_m < 6 and igm1 == "negatif") or \
           (age_m < 1 and igm1 == "negatif" and igm2 == "negatif") or \
           (age_m > 5 and igg1 == "negatif") or \
           (age_m > 5 and igm1 == "negatif" and igg1 == "positif" and igg2 == "negatif"):
            return "Bukan CRS"
            
        if age_m < 1 and igm1 == "negatif" and pd.isna(r.get('H_IGM_S2')) and clinical == "Y":
            return "CRS Klinis"
        if age_m < 1 and igm1 == "negatif" and pd.isna(r.get('H_IGM_S2')) and clinical == "N":
            return "Bukan CRS"
        if age_m > 5 and igm1 == "negatif" and igg1 == "positif" and pd.isna(r.get('H_IGG_S2')) and clinical == "Y":
            return "CRS Klinis"
        if age_m > 5 and igm1 == "negatif" and igg1 == "positif" and pd.isna(r.get('H_IGG_S2')) and clinical == "N":
            return "Bukan CRS"
        if age_m < 12 and clinical == "Y" and spec in ["tidak", "tidak tahu"]:
            return "CRS Klinis"
        if age_m < 12 and clinical == "N" and spec in ["tidak", "tidak tahu"]:
            return "Bukan CRS"
        if age_m < 12 and clinical == "Y" and spec == "ya" and (pd.isna(r.get('H_IGM_S1')) and pd.isna(r.get('H_IGM_S2'))):
            return "CRS Klinis"
        if age_m < 12 and clinical == "N" and spec == "ya" and (pd.isna(r.get('H_IGM_S1')) and pd.isna(r.get('H_IGM_S2'))):
            return "Bukan CRS"
            
        return "Pending"

    df['FINAL_CLASS_UPDATED'] = df.apply(classify_row, axis=1)
    return df

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------

if 'df_current' not in st.session_state:
    st.session_state.df_current = None

step = st.sidebar.radio("Navigasi Langkah Kerja:", [
    "1. Import Data Excel",
    "2. Data Cleaning & Quality Checks",
    "3. Run Updates & Classification",
    "4. Visualisasi & Grafik (Save PNG)",
    "5. Export Reports (Bulletin & Non Suspect)"
])

# STEP 1: IMPORT
if step == "1. Import Data Excel":
    st.header("Langkah 1: Import Data")
    uploaded_file = st.file_uploader("Upload File Data Raw (.xlsx / .xls)", type=["xlsx", "xls"])
    if uploaded_file:
        df_uploaded = pd.read_excel(uploaded_file)
        st.session_state.df_current = df_uploaded.copy()
        st.success(f"Data Berhasil Diunggah! Total Record: {len(df_uploaded)}")
        st.dataframe(df_uploaded.head(5), use_container_width=True)

# STEP 2: QUALITY CHECKS
elif step == "2. Data Cleaning & Quality Checks":
    st.header("Langkah 2: Validation Checks")
    if st.session_state.df_current is None:
        st.warning("Unggah data pada Langkah 1 terlebih dahulu.")
    else:
        df = st.session_state.df_current
        check = st.selectbox("Pilih Jenis Pemeriksaan:", [
            "Cek01_Epid ID Kosong", "Cek02_Tgl Lahir Kosong", "Cek03_Umur Kosong",
            "Cek04_Umur > 11 Bulan", "Cek05_Kab/Prov Kosong", "Cek06_Epid ID Duplikat",
            "Cek07_Nama Duplikat", "Cek08_Pengambilan Spesimen Kosong"
        ])
        
        if "Cek01" in check: res = df[df['EPID_ID'].isna()]
        elif "Cek02" in check: res = df[df['DOB'].isna()]
        elif "Cek03" in check: res = df[df['AGE_M'].isna() | df['AGE_Y'].isna()]
        elif "Cek04" in check: res = df[df['AGE_M'] > 11]
        elif "Cek05" in check: res = df[df['DISTRICT'].isna() | df['PROVINCE'].isna()]
        elif "Cek06" in check: res = df[df.duplicated('EPID_ID', keep=False)].sort_values('EPID_ID')
        elif "Cek07" in check: res = df[df.duplicated('NAME', keep=False)].sort_values('NAME')
        elif "Cek08" in check: res = df[df['H_SPEC_COLL'].isna()]
            
        st.write(f"Ditemukan **{len(res)}** record bermasalah:")
        st.dataframe(res, use_container_width=True)

# STEP 3: UPDATES & CLASSIFICATION
elif step == "3. Run Updates & Classification":
    st.header("Langkah 3: Auto-Update & Klasifikasi CRS")
    if st.session_state.df_current is None:
        st.warning("Unggah data pada Langkah 1 terlebih dahulu.")
    else:
        if st.button("Jalankan Semua Update & Klasifikasi Otomatis"):
            df_updated = run_all_updates(st.session_state.df_current)
            df_classified = run_classification_final(df_updated)
            st.session_state.df_current = df_classified
            st.success("Proses Update Umur, Spesimen, dan Klasifikasi Selesai!")
            
        if 'Suspect_CRS' in st.session_state.df_current.columns:
            st.subheader("Ringkasan Hasil Klasifikasi:")
            st.write(st.session_state.df_current['Suspect_CRS'].value_counts().rename({'Y': 'Suspect CRS', 'N': 'Non Suspect CRS'}))
            st.dataframe(st.session_state.df_current[['EPID_ID', 'NAME', 'AGE_M', 'Suspect_CRS', 'FINAL_CLASS_UPDATED']].head(10), use_container_width=True)

# STEP 4: VISUALISASI & GRAFIK
elif step == "4. Visualisasi & Grafik (Save PNG)":
    st.header("Langkah 4: Visualisasi Data CRS Surveillance")
    st.info(" **Arahkan kursor ke pojok kanan atas grafik lalu klik ikon kamera ('Download plot as a png') untuk menyimpan grafik.**")
    
    if st.session_state.df_current is None:
        st.warning("Unggah data pada Langkah 1 terlebih dahulu.")
    else:
        df = st.session_state.df_current
        if 'FINAL_CLASS_UPDATED' not in df.columns:
            df = run_classification_final(run_all_updates(df))
            st.session_state.df_current = df

        col_a, col_b = st.columns(2)
        
        # Grafik 1: Proporsi Klasifikasi Final Kasus
        with col_a:
            st.subheader("1. Proporsi Klasifikasi Final Kasus")
            class_counts = df['FINAL_CLASS_UPDATED'].value_counts().reset_index()
            class_counts.columns = ['Klasifikasi', 'Jumlah']
            fig1 = px.bar(class_counts, x='Klasifikasi', y='Jumlah', color='Klasifikasi',
                          text='Jumlah', title="Distribusi Klasifikasi Akhir Kasus CRS",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig1.update_traces(textposition='outside')
            st.plotly_chart(fig1, use_container_width=True)

        # Grafik 2: Distribusi Kasus per Rumah Sakit Sentinel
        with col_b:
            st.subheader("2. Kasus per Rumah Sakit Sentinel")
            if 'HOS_NAME' in df.columns:
                hos_counts = df['HOS_NAME'].value_counts().reset_index().head(10)
                hos_counts.columns = ['Rumah Sakit', 'Jumlah Suspek']
                fig2 = px.bar(hos_counts, y='Rumah Sakit', x='Jumlah Suspek', orientation='h',
                              text='Jumlah Suspek', title="Top 10 RS Sentinel (Jumlah Suspek Terbanyak)",
                              color_discrete_sequence=['#2b5c8f'])
                fig2.update_layout(yaxis={'categoryorder':'total ascending'})
                fig2.update_traces(textposition='outside')
                st.plotly_chart(fig2, use_container_width=True)

        col_c, col_d = st.columns(2)
        
        # Grafik 3: Temuan Gejala Klinis Utama
        with col_c:
            st.subheader("3. Temuan Gejala Klinis Utama")
            gejala_cols = {'A_CHD': 'Kelainan Jantung (CHD)', 'A_CC': 'Katarak (CC)', 
                           'A_CG': 'Glaukoma (CG)', 'A_HI': 'Gangguan Pendengaran (HI)'}
            gejala_data = []
            for col_k, label in gejala_cols.items():
                if col_k in df.columns:
                    cnt = sum(1 for v in df[col_k] if clean_str(v) in ['ya', 'y'])
                    gejala_data.append({'Gejala': label, 'Jumlah Kasus': cnt})
            
            df_gejala = pd.DataFrame(gejala_data)
            fig3 = px.bar(df_gejala, x='Gejala', y='Jumlah Kasus', text='Jumlah Kasus',
                          title="Frekuensi Gejala Klinis Kategori A",
                          color_discrete_sequence=['#e66101'])
            fig3.update_traces(textposition='outside')
            st.plotly_chart(fig3, use_container_width=True)

        # Grafik 4: Distribusi Umur Kasus (Bulan)
        with col_d:
            st.subheader("4. Distribusi Umur Kasus (Bulan)")
            if 'AGE_M' in df.columns:
                fig4 = px.histogram(df[df['AGE_M'].notna()], x='AGE_M', nbins=12,
                                    title="Distribusi Umur Kasus (Bulan)",
                                    labels={'AGE_M': 'Umur (Bulan)'},
                                    color_discrete_sequence=['#5e3c99'])
                st.plotly_chart(fig4, use_container_width=True)

# STEP 5: EXPORT
elif step == "5. Export Reports (Bulletin & Non Suspect)":
    st.header("Langkah 5: Export Reports")
    if st.session_state.df_current is None:
        st.warning("Unggah data pada Langkah 1 terlebih dahulu.")
    else:
        df = st.session_state.df_current
        if 'Suspect_CRS' not in df.columns:
            df = run_classification_final(run_all_updates(df))
            st.session_state.df_current = df

        c1, c2 = st.columns(2)
        bln = c1.text_input("Bulan Pelaporan", "11")
        thn = c2.text_input("Tahun Pelaporan", "2026")
        
        # Bulletin Export
        df_bulletin = df[df['Suspect_CRS'] == 'Y']
        buf_b = BytesIO()
        with pd.ExcelWriter(buf_b, engine='openpyxl') as w: df_bulletin.to_excel(w, index=False)
        st.download_button(f"Download CRS_Bulletin_{bln} {thn}.xlsx ({len(df_bulletin)} records)", buf_b.getvalue(), f"CRS_Bulletin_{bln} {thn}.xlsx")

        # Non-Suspect Export
        df_non = df[df['Suspect_CRS'] == 'N']
        buf_n = BytesIO()
        with pd.ExcelWriter(buf_n, engine='openpyxl') as w: df_non.to_excel(w, index=False)
        st.download_button(f"Download CRS_DB Non Suspect {bln} {thn}.xlsx ({len(df_non)} records)", buf_n.getvalue(), f"CRS_DB Non Suspect {bln} {thn}.xlsx")
