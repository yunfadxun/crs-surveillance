import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="CRS Surveillance System", layout="wide")
st.title("CRS Surveillance Data Processing & Cleaning Dashboard")

# ---------------------------------------------------------
# HELPER FUNCTIONS (MATCHING MS ACCESS LOGIC PRECISELY)
# ---------------------------------------------------------

def clean_str(val):
    """Clean string values for precise matching."""
    if pd.isna(val):
        return ""
    return str(val).strip().lower()

def check_suspect(row):
    """
    Kriteria Qry1/Qry4 MS Access:
    IIf([AGE_Y]<1 And [AGE_M]<12 And ([A_CHD]="Ya" Or [A_CC]="Ya" Or ...),"Y","N")
    """
    # 1. Evaluasi Gejala A
    a_cols = ['A_CHD', 'A_CC', 'A_CG', 'A_PR', 'A_HI']
    has_gejala_a = any(clean_str(row.get(c)) in ['ya', 'y', '1', 'true'] for c in a_cols)
    
    # 2. Evaluasi Umur (Access menganggap NaN/Null pada Age sebagai < 1 / < 12)
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
    """Menggabungkan Qry1_Update Umur & Qry2-Qry4_Update Spesimen dalam 1 fungsi ringkas."""
    df = df.copy()
    
    # 1. Update Umur (Qry1_Update Umur)
    if 'DOCHECK' in df.columns and 'DOB' in df.columns:
        docheck = pd.to_datetime(df['DOCHECK'], errors='coerce')
        dob = pd.to_datetime(df['DOB'], errors='coerce')
        days_diff = (docheck - dob).dt.days
        
        cond_empty_age = df['AGE_M'].isna() | df['AGE_Y'].isna() | ((df['AGE_M'].fillna(0) == 0) & (df['AGE_Y'].fillna(0) == 0) & df['AGE_D'].isna())
        df.loc[cond_empty_age, 'AGE_D'] = days_diff[cond_empty_age].fillna(0)
        df.loc[cond_empty_age, 'AGE_M'] = (days_diff[cond_empty_age] // 30).fillna(0)
        df.loc[cond_empty_age, 'AGE_Y'] = (days_diff[cond_empty_age] // 365).fillna(0)
        
    # 2. Update Spesimen (Qry2, Qry3, Qry4)
    if 'H_SPEC_COLL' in df.columns:
        df['H_SPEC_COLL'] = df['H_SPEC_COLL'].fillna("Tidak Tahu")
        
        if 'H_DCOLLSER1' in df.columns:
            # Qry3: Update Tanggal Pengambilan Sampel 1
            if 'H_DRES_IGM_S1' in df.columns:
                cond3 = df['H_DCOLLSER1'].isna() & df['H_DRES_IGM_S1'].notna() & (df['H_SPEC_COLL'].apply(clean_str) == 'ya')
                df.loc[cond3, 'H_DCOLLSER1'] = df.loc[cond3, 'H_DRES_IGM_S1']
            
            # Qry4: Update Tanggal Pengambilan Sampel 2
            if 'DOCHECK' in df.columns:
                cond4 = df['H_DCOLLSER1'].isna() & (df['H_SPEC_COLL'].apply(clean_str) == 'ya')
                df.loc[cond4, 'H_DCOLLSER1'] = df.loc[cond4, 'DOCHECK']
                
    return df

def run_classification_final(df):
    """Proses Klasifikasi Final (Qry5 & Qry6)."""
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
        
        # Qry5_Update Final 1
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
            
        # Qry6_Update Final 2
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
    "4. Export Reports (Bulletin & Non Suspect)"
])

# STEP 1
if step == "1. Import Data Excel":
    st.header("Langkah 1: Import Data")
    uploaded_file = st.file_uploader("Upload File Data Raw (.xlsx / .xls)", type=["xlsx", "xls"])
    if uploaded_file:
        df_uploaded = pd.read_excel(uploaded_file)
        st.session_state.df_current = df_uploaded.copy()
        st.success(f"Data Berhasil Diunggah! Total Record: {len(df_uploaded)}")
        st.dataframe(df_uploaded.head(5), use_container_width=True)

# STEP 2
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

# STEP 3
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

# STEP 4
elif step == "4. Export Reports (Bulletin & Non Suspect)":
    st.header("Langkah 4: Export Reports")
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
