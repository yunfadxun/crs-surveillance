import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="CRS Surveillance System", layout="wide")

st.title("CRS Surveillance Data Processing & Cleaning Dashboard")
st.markdown("Aplikasi pengganti Microsoft Access untuk alur kerja pembersihan data, pengujian kriteria, dan otomatisasi klasifikasi CRS Surveillance.")

# ---------------------------------------------------------
# HELPER FUNCTIONS & LOGIC CONVERSIONS
# ---------------------------------------------------------

def check_suspect(row):
    """Cek01/Qry1: Kriteria Suspect CRS (Minimal 1 gejala Kategori A = 'Ya')"""
    cols = ['A_CHD', 'A_CC', 'A_CG', 'A_PR', 'A_HI']
    return "Y" if any(str(row.get(c, '')).strip().lower() == 'ya' for c in cols) else "N"

def check_clinical(row):
    """Cek Kriteria CRS Klinis (Gejala A & B)"""
    a_cols = ['A_CHD', 'A_CC', 'A_CG', 'A_PR', 'A_HI']
    b_cols = ['B_PU', 'B_MC', 'B_ME', 'B_IC', 'B_SP', 'B_DD', 'B_RBD']
    
    a_count = sum(1 for c in a_cols if str(row.get(c, '')).strip().lower() == 'ya')
    b_count = sum(1 for c in b_cols if str(row.get(c, '')).strip().lower() == 'ya')
    
    if a_count >= 2 or (a_count >= 1 and b_count >= 1):
        return "Y"
    return "N"

def run_update_umur(df):
    """Qry1_Update Umur: Hitung umur dalam hari/bulan jika kosong"""
    df = df.copy()
    if 'DOCHECK' in df.columns and 'DOB' in df.columns:
        docheck = pd.to_datetime(df['DOCHECK'], errors='coerce')
        dob = pd.to_datetime(df['DOB'], errors='coerce')
        
        # Hitung selisih hari & bulan
        days_diff = (docheck - dob).dt.days
        
        cond_empty = df['AGE_M'].isna() | df['AGE_Y'].isna() | ((df['AGE_M'].fillna(0) == 0) & (df['AGE_Y'].fillna(0) == 0) & df['AGE_D'].isna())
        
        df.loc[cond_empty, 'AGE_D'] = days_diff[cond_empty].fillna(0)
        df.loc[cond_empty, 'AGE_M'] = (days_diff[cond_empty] // 30).fillna(0)
        df.loc[cond_empty, 'AGE_Y'] = (days_diff[cond_empty] // 365).fillna(0)
    return df

def run_updates_specimen(df):
    """Qry2, Qry3, Qry4: Update Spesimen Null & Tanggal Pengambilan"""
    df = df.copy()
    
    # Qry2_Update Spec Coll if Null
    if 'H_SPEC_COLL' in df.columns:
        df['H_SPEC_COLL'] = df['H_SPEC_COLL'].fillna("Tidak Tahu")
        
    # Qry3_Update Tanggal Pengambilan Sampel 1
    if 'H_DCOLLSER1' in df.columns and 'H_DRES_IGM_S1' in df.columns and 'H_SPEC_COLL' in df.columns:
        cond3 = df['H_DCOLLSER1'].isna() & df['H_DRES_IGM_S1'].notna() & (df['H_SPEC_COLL'].str.lower() == 'ya')
        df.loc[cond3, 'H_DCOLLSER1'] = df.loc[cond3, 'H_DRES_IGM_S1']
        
    # Qry4_Update Tanggal Pengambilan Sampel 2
    if 'H_DCOLLSER1' in df.columns and 'DOCHECK' in df.columns and 'H_SPEC_COLL' in df.columns:
        cond4 = df['H_DCOLLSER1'].isna() & (df['H_SPEC_COLL'].str.lower() == 'ya')
        df.loc[cond4, 'H_DCOLLSER1'] = df.loc[cond4, 'DOCHECK']
        
    return df

def run_classification_final(df):
    """Qry5_Update Final 1 & Qry6_Update Final 2"""
    df = df.copy()
    df['Suspect_CRS'] = df.apply(check_suspect, axis=1)
    df['Clinical_CRS'] = df.apply(check_clinical, axis=1)
    
    def classify_row(r):
        age_m = r.get('AGE_M', np.nan)
        igm1 = str(r.get('H_IGM_S1', '')).strip().capitalize()
        igm2 = str(r.get('H_IGM_S2', '')).strip().capitalize()
        igg1 = str(r.get('H_IGG_S1', '')).strip().capitalize()
        igg2 = str(r.get('H_IGG_S2', '')).strip().capitalize()
        clinical = r.get('Clinical_CRS', 'N')
        spec = str(r.get('H_SPEC_COLL', '')).strip().capitalize()
        
        # Qry5_Update Final 1 Logic
        if (age_m < 6 and igm1 == "Positif") or \
           (age_m < 1 and igm1 == "Negatif" and igm2 == "Positif") or \
           (age_m > 5 and igm1 == "Positif" and igg1 == "Positif") or \
           (age_m > 5 and igm1 == "Negatif" and igg1 == "Positif" and igg2 == "Positif"):
            return "CRS pasti(Lab Positif)"
            
        if (1 <= age_m < 6 and igm1 == "Negatif") or \
           (age_m < 1 and igm1 == "Negatif" and igm2 == "Negatif") or \
           (age_m > 5 and igg1 == "Negatif") or \
           (age_m > 5 and igm1 == "Negatif" and igg1 == "Positif" and igg2 == "Negatif"):
            return "Bukan CRS"
            
        # Qry6_Update Final 2 Logic
        if age_m < 1 and igm1 == "Negatif" and pd.isna(r.get('H_IGM_S2')) and clinical == "Y":
            return "CRS Klinis"
        if age_m < 1 and igm1 == "Negatif" and pd.isna(r.get('H_IGM_S2')) and clinical == "N":
            return "Bukan CRS"
        if age_m > 5 and igm1 == "Negatif" and igg1 == "Positif" and pd.isna(r.get('H_IGG_S2')) and clinical == "Y":
            return "CRS Klinis"
        if age_m > 5 and igm1 == "Negatif" and igg1 == "Positif" and pd.isna(r.get('H_IGG_S2')) and clinical == "N":
            return "Bukan CRS"
        if age_m < 12 and clinical == "Y" and spec in ["Tidak", "Tidak Tahu"]:
            return "CRS Klinis"
        if age_m < 12 and clinical == "N" and spec in ["Tidak", "Tidak Tahu"]:
            return "Bukan CRS"
        if age_m < 12 and clinical == "Y" and spec == "Ya" and (pd.isna(r.get('H_IGM_S1')) and pd.isna(r.get('H_IGM_S2'))):
            return "CRS Klinis"
        if age_m < 12 and clinical == "N" and spec == "Ya" and (pd.isna(r.get('H_IGM_S1')) and pd.isna(r.get('H_IGM_S2'))):
            return "Bukan CRS"
            
        return "Pending"

    df['FINAL_CLASS_UPDATED'] = df.apply(classify_row, axis=1)
    return df

# ---------------------------------------------------------
# STREAMLIT INTERFACE & STEP-BY-STEP WORKFLOW
# ---------------------------------------------------------

st.sidebar.header("Langkah Workflow Operations")
step = st.sidebar.radio("Pilih Langkah Kerja:", [
    "1. Import Data Export Terbaru",
    "2. Data Cleaning & Validation Checks",
    "3. Run Updates (Umur & Spesimen)",
    "4. Processing Suspect & Final Classification",
    "5. Export Reports (Bulletin & Non Suspect)"
])

# INITIALIZE SESSION STATE DATA
if 'df_current' not in st.session_state:
    st.session_state.df_current = None
if 'df_backup' not in st.session_state:
    st.session_state.df_backup = None

# STEP 1: IMPORT & BACKUP DATA
if step == "1. Import Data Export Terbaru":
    st.header("Langkah 9-12: Import Data & Backup Table")
    
    uploaded_file = st.file_uploader("Upload file Excel hasil cleaning/export terbaru (.xlsx)", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        df_uploaded = pd.read_excel(uploaded_file)
        
        st.success(f"Berhasil mengunggah data! Total baris: {len(df_uploaded)}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Buat Backup Data saat ini"):
                st.session_state.df_backup = df_uploaded.copy()
                st.success("Backup data berhasil disimpan di session state.")
        
        with col2:
            if st.button("Jadikan Data Utama (Append / Load)"):
                st.session_state.df_current = df_uploaded.copy()
                st.success("Data berhasil dimuat ke dalam memori kerja aktif (1_CRS_DB_RAW).")

    if st.session_state.df_current is not None:
        st.subheader("Pratinjau Data Aktif (1_CRS_DB_RAW)")
        st.dataframe(st.session_state.df_current.head(10), use_container_width=True)

# STEP 2: DATA CLEANING & CHECKS
elif step == "2. Data Cleaning & Validation Checks":
    st.header("Langkah 13: Data Cleaning & Validation Queries")
    
    if st.session_state.df_current is None:
        st.warning("Silakan unggah data pada Langkah 1 terlebih dahulu.")
    else:
        df = st.session_state.df_current
        
        check_option = st.selectbox("Pilih Jenis Quality Check:", [
            "Cek01_Epid ID (Cek Nomor Epid Kosong)",
            "Cek02_Tgl Lahir Kosong",
            "Cek03_Umur Kosong",
            "Cek04_Umur > 11 Bulan",
            "Cek05_Kab Prov Kosong",
            "Cek06_Epid ID Duplikat",
            "Cek07_Nama Duplikat",
            "Cek08_Pengambilan Spesimen Kosong"
        ])
        
        res_df = pd.DataFrame()
        
        if "Cek01" in check_option:
            res_df = df[df['EPID_ID'].isna()]
        elif "Cek02" in check_option:
            res_df = df[df['DOB'].isna()]
        elif "Cek03" in check_option:
            cond = df['AGE_M'].isna() | df['AGE_Y'].isna() | ((df['AGE_M'] == 0) & (df['AGE_Y'] == 0) & df['AGE_D'].isna())
            res_df = df[cond]
        elif "Cek04" in check_option:
            res_df = df[df['AGE_M'] > 11]
        elif "Cek05" in check_option:
            res_df = df[df['DISTRICT'].isna() | df['PROVINCE'].isna()]
        elif "Cek06" in check_option:
            dup_epid = df[df.duplicated(subset=['EPID_ID'], keep=False)]
            res_df = dup_epid.sort_values('EPID_ID') if not dup_epid.empty else pd.DataFrame()
        elif "Cek07" in check_option:
            dup_name = df[df.duplicated(subset=['NAME'], keep=False)]
            res_df = dup_name.sort_values('NAME') if not dup_name.empty else pd.DataFrame()
        elif "Cek08" in check_option:
            res_df = df[df['H_SPEC_COLL'].isna()]
            
        st.subheader(f"Hasil Evaluasi: {check_option}")
        st.write(f"Ditemukan **{len(res_df)}** baris bermasalah.")
        st.dataframe(res_df, use_container_width=True)

# STEP 3: RUN UPDATES
elif step == "3. Run Updates (Umur & Spesimen)":
    st.header("Langkah 13 (Lanjutan): Execution Update Queries")
    
    if st.session_state.df_current is None:
        st.warning("Silakan unggah data pada Langkah 1 terlebih dahulu.")
    else:
        st.markdown("Jalankan otomatisasi update untuk memperbaiki field umur dan status pengambilan spesimen:")
        
        if st.button("Jalankan Qry1_Update Umur"):
            st.session_state.df_current = run_update_umur(st.session_state.df_current)
            st.success("Umur kasus berhasil diperbarui berdasarkan DOCHECK dan DOB!")
            
        if st.button("Jalankan Update Spesimen (Qry2, Qry3, Qry4)"):
            st.session_state.df_current = run_updates_specimen(st.session_state.df_current)
            st.success("Spesimen Null & Tanggal Pengambilan Spesimen berhasil di-update!")
            
        st.subheader("Pratinjau Data Setelah Update")
        st.dataframe(st.session_state.df_current[['EPID_ID', 'NAME', 'DOB', 'DOCHECK', 'AGE_Y', 'AGE_M', 'AGE_D', 'H_SPEC_COLL', 'H_DCOLLSER1']].head(10), use_container_width=True)

# STEP 4: CLASSIFICATION
elif step == "4. Processing Suspect & Final Classification":
    st.header("Langkah 14-15: Filtering Suspect & Processing Final Classification")
    
    if st.session_state.df_current is None:
        st.warning("Silakan unggah data pada Langkah 1 terlebih dahulu.")
    else:
        if st.button("Jalankan Klasifikasi Final CRS (Qry1, Qry2, Qry5, Qry6)"):
            st.session_state.df_current = run_classification_final(st.session_state.df_current)
            st.success("Klasifikasi Final CRS berhasil dihitung!")
            
        st.subheader("Hasil Klasifikasi Data Aktif")
        st.dataframe(st.session_state.df_current[['EPID_ID', 'NAME', 'AGE_M', 'Suspect_CRS', 'Clinical_CRS', 'FINAL_CLASS_UPDATED']].head(15), use_container_width=True)

# STEP 5: EXPORT REPORTS
elif step == "5. Export Reports (Bulletin & Non Suspect)":
    st.header("Langkah 16-17: Export Reports (Bulletin & Non Suspect)")
    
    if st.session_state.df_current is None:
        st.warning("Silakan unggah data pada Langkah 1 terlebih dahulu.")
    else:
        col_m, col_y = st.columns(2)
        with col_m:
            bulan_lapor = st.text_input("Bulan Pelaporan Berjalan (Contoh: 11)", "11")
        with col_y:
            tahun_lapor = st.text_input("Tahun Pelaporan Berjalan (Contoh: 2026)", "2026")
            
        df_active = st.session_state.df_current
        
        # Ensure classifications are run
        if 'Suspect_CRS' not in df_active.columns:
            df_active = run_classification_final(df_active)
            
        st.divider()
        
        # 1. Bulletin Export (Suspect Cases)
        st.subheader("1. Export CRS Bulletin (Qry3_CRS Bulletin / Table 4_CRS_Bulletin)")
        df_bulletin = df_active[df_active['Suspect_CRS'] == 'Y']
        st.write(f"Total Record Bulletin (Suspect CRS): **{len(df_bulletin)}**")
        
        file_name_bulletin = f"CRS_Bulletin_{bulan_lapor} {tahun_lapor}.xlsx"
        
        # Generate Excel download button
        from io import BytesIO
        output_b = BytesIO()
        with pd.ExcelWriter(output_b, engine='openpyxl') as writer:
            df_bulletin.to_excel(writer, index=False, sheet_name='4_CRS_Bulletin')
        
        st.download_button(
            label=f"Download {file_name_bulletin}",
            data=output_b.getvalue(),
            file_name=file_name_bulletin,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.divider()
        
        # 2. Non-Suspect Export
        st.subheader("2. Export CRS Non Suspect (Qry4_CRS db Non Suspect)")
        df_non_suspect = df_active[df_active['Suspect_CRS'] == 'N']
        st.write(f"Total Record Non Suspect: **{len(df_non_suspect)}**")
        
        file_name_non_suspect = f"CRS_DB Non Suspect {bulan_lapor} {tahun_lapor}.xlsx"
        
        output_ns = BytesIO()
        with pd.ExcelWriter(output_ns, engine='openpyxl') as writer:
            df_non_suspect.to_excel(writer, index=False, sheet_name='4_CRS db Non Suspect')
            
        st.download_button(
            label=f"Download {file_name_non_suspect}",
            data=output_ns.getvalue(),
            file_name=file_name_non_suspect,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
