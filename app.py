from io import BytesIO
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="CRS Surveillance System", layout="wide")
st.title("CRS Surveillance Data Processing & Analytics Dashboard")


# ---------------------------------------------------------
# HELPER FUNCTIONS (PRECISION LOGIC - FIXED CLINICAL COUNT)
# ---------------------------------------------------------


def clean_str(val):
  if pd.isna(val):
    return ""
  return str(val).strip().lower()


def check_suspect(row):
  """Kriteria Suspek CRS (Qry1 / Kemenkes):

  Usia Y < 1 AND M < 12 DAN Memiliki minimal 1 gejala Kategori A = 'Ya'
  """
  a_cols = ["A_CHD", "A_CC", "A_CG", "A_PR", "A_HI"]
  has_gejala_a = any(
      clean_str(row.get(c)) in ["ya", "y", "1", "true"] for c in a_cols
  )

  age_y = (
      float(row.get("AGE_Y"))
      if pd.notna(row.get("AGE_Y"))
      and str(row.get("AGE_Y")).replace(".", "", 1).isdigit()
      else 0
  )
  age_m = (
      float(row.get("AGE_M"))
      if pd.notna(row.get("AGE_M"))
      and str(row.get("AGE_M")).replace(".", "", 1).isdigit()
      else 0
  )

  return "Y" if (age_y < 1 and age_m < 12 and has_gejala_a) else "N"


def check_clinical(row):
  """Kriteria CRS Klinis (Gejala Kategori A & B)"""
  a_cols = ["A_CHD", "A_CC", "A_CG", "A_PR", "A_HI"]
  b_cols = ["B_PU", "B_MC", "B_ME", "B_IC", "B_SP", "B_DD", "B_RBD"]

  a_cnt = sum(1 for c in a_cols if clean_str(row.get(c)) in ["ya", "y"])
  b_cnt = sum(1 for c in b_cols if clean_str(row.get(c)) in ["ya", "y"])

  return "Y" if (a_cnt >= 2 or (a_cnt >= 1 and b_cnt >= 1)) else "N"


def run_all_updates(df):
  """Update Umur & Tanggal Spesimen"""
  df = df.copy()
  if "DOCHECK" in df.columns and "DOB" in df.columns:
    docheck = pd.to_datetime(df["DOCHECK"], errors="coerce")
    dob = pd.to_datetime(df["DOB"], errors="coerce")
    days_diff = (docheck - dob).dt.days

    cond_empty = (
        df["AGE_M"].isna()
        | df["AGE_Y"].isna()
        | (
            (df["AGE_M"].fillna(0) == 0)
            & (df["AGE_Y"].fillna(0) == 0)
            & df["AGE_D"].isna()
        )
    )
    df.loc[cond_empty, "AGE_D"] = days_diff[cond_empty].fillna(0)
    df.loc[cond_empty, "AGE_M"] = (days_diff[cond_empty] // 30).fillna(0)
    df.loc[cond_empty, "AGE_Y"] = (days_diff[cond_empty] // 365).fillna(0)

  if "H_SPEC_COLL" in df.columns:
    df["H_SPEC_COLL"] = df["H_SPEC_COLL"].fillna("Tidak Tahu")
    if "H_DCOLLSER1" in df.columns:
      if "H_DRES_IGM_S1" in df.columns:
        cond3 = (
            df["H_DCOLLSER1"].isna()
            & df["H_DRES_IGM_S1"].notna()
            & (df["H_SPEC_COLL"].apply(clean_str) == "ya")
        )
        df.loc[cond3, "H_DCOLLSER1"] = df.loc[cond3, "H_DRES_IGM_S1"]
      if "DOCHECK" in df.columns:
        cond4 = df["H_DCOLLSER1"].isna() & (
            df["H_SPEC_COLL"].apply(clean_str) == "ya"
        )
        df.loc[cond4, "H_DCOLLSER1"] = df.loc[cond4, "DOCHECK"]

  return df


def run_classification_final(df):
  """Klasifikasi Final CRS HANYA untuk Kasus Suspek CRS (Bulletin).

  Diperbaiki untuk menangani toleransi spesimen kosong & batas umur < 12 bulan.
  """
  df = df.copy()
  df["Suspect_CRS"] = df.apply(check_suspect, axis=1)
  df["Clinical_CRS"] = df.apply(check_clinical, axis=1)

  def classify_row(r):
    if r["Suspect_CRS"] != "Y":
      return "Discarded / Non Suspect"

    age_m = (
        float(r["AGE_M"])
        if pd.notna(r.get("AGE_M"))
        and str(r.get("AGE_M")).replace(".", "", 1).isdigit()
        else 0
    )
    igm1 = clean_str(r.get("H_IGM_S1"))
    igm2 = clean_str(r.get("H_IGM_S2"))
    igg1 = clean_str(r.get("H_IGG_S1"))
    igg2 = clean_str(r.get("H_IGG_S2"))
    clinical = r.get("Clinical_CRS", "N")
    spec = clean_str(r.get("H_SPEC_COLL"))

    # Helper cek ketersediaan hasil tes
    no_igm1 = (igm1 == "") or (igm1 == "none") or pd.isna(r.get("H_IGM_S1"))
    no_igm2 = (igm2 == "") or (igm2 == "none") or pd.isna(r.get("H_IGM_S2"))
    no_igg1 = (igg1 == "") or (igg1 == "none") or pd.isna(r.get("H_IGG_S1"))
    no_igg2 = (igg2 == "") or (igg2 == "none") or pd.isna(r.get("H_IGG_S2"))

    # 1. CRS PASTI (LAB POSITIF) - Qry5
    if (
        (age_m < 6 and igm1 == "positif")
        or (age_m < 1 and igm1 == "negatif" and igm2 == "positif")
        or (age_m > 5 and igm1 == "positif" and igg1 == "positif")
        or (
            age_m > 5
            and igm1 == "negatif"
            and igg1 == "positif"
            and igg2 == "positif"
        )
    ):
      return "CRS Pasti (Lab Positif)"

    # 2. BUKAN CRS (DISCARDED FROM LAB) - Qry5
    if (
        (1 <= age_m < 6 and igm1 == "negatif")
        or (age_m < 1 and igm1 == "negatif" and igm2 == "negatif")
        or (age_m > 5 and igg1 == "negatif")
        or (
            age_m > 5
            and igm1 == "negatif"
            and igg1 == "positif"
            and igg2 == "negatif"
        )
    ):
      return "Bukan CRS"

    # 3. EVALUASI CRS KLINIS vs BUKAN CRS - Qry6
    # Kasus Usia < 1 Bulan
    if age_m < 1 and igm1 == "negatif" and no_igm2:
      return "CRS Klinis" if clinical == "Y" else "Bukan CRS"

    # Kasus Usia > 5 Bulan dengan IgG
    if age_m > 5 and igm1 == "negatif" and igg1 == "positif" and no_igg2:
      return "CRS Klinis" if clinical == "Y" else "Bukan CRS"

    # Kasus Usia < 12 Bulan tanpa spesimen / spesimen tidak tahu / hasil lab belum ada
    if age_m <= 12:
      if spec in ["tidak", "tidak tahu", "", "null"]:
        return "CRS Klinis" if clinical == "Y" else "Bukan CRS"
      if spec == "ya" and no_igm1 and no_igm2 and no_igg1 and no_igg2:
        return "CRS Klinis" if clinical == "Y" else "Bukan CRS"

    # Jika memenuhi kriteria klinis dan belum terkelompokkan di atas -> CRS Klinis
    if clinical == "Y":
      return "CRS Klinis"

    return "Pending"

  df["FINAL_CLASS_UPDATED"] = df.apply(classify_row, axis=1)
  return df


# ---------------------------------------------------------
# UI & NAVIGATION
# ---------------------------------------------------------

if "df_current" not in st.session_state:
  st.session_state.df_current = None

step = st.sidebar.radio(
    "Navigasi Langkah Kerja:",
    [
        "1. Import Data Excel",
        "2. Data Cleaning & Checks",
        "3. Run Updates & Classification",
        "4. Visualisasi Grafik (Save PNG)",
        "5. Export Reports",
    ],
)

# 1. IMPORT
if step == "1. Import Data Excel":
  st.header("Langkah 1: Import Data")
  uploaded_file = st.file_uploader(
      "Upload File Raw Excel (.xlsx / .xls)", type=["xlsx", "xls"]
  )
  if uploaded_file:
    df_uploaded = pd.read_excel(uploaded_file)
    st.session_state.df_current = df_uploaded.copy()
    st.success(f"Data Berhasil Diunggah! Total Record: {len(df_uploaded)}")
    st.dataframe(df_uploaded.head(5), use_container_width=True)

# 2. CHECKS
elif step == "2. Data Cleaning & Checks":
  st.header("Langkah 2: Validation Checks")
  if st.session_state.df_current is None:
    st.warning("Unggah data pada Langkah 1 terlebih dahulu.")
  else:
    df = st.session_state.df_current
    check = st.selectbox(
        "Pilih Jenis Pemeriksaan:",
        [
            "Cek01_Epid ID Kosong",
            "Cek02_Tgl Lahir Kosong",
            "Cek03_Umur Kosong",
            "Cek04_Umur > 11 Bulan",
            "Cek05_Kab/Prov Kosong",
            "Cek06_Epid ID Duplikat",
            "Cek07_Nama Duplikat",
            "Cek08_Pengambilan Spesimen Kosong",
        ],
    )

    if "Cek01" in check:
      res = df[df["EPID_ID"].isna()]
    elif "Cek02" in check:
      res = df[df["DOB"].isna()]
    elif "Cek03" in check:
      res = df[df["AGE_M"].isna() | df["AGE_Y"].isna()]
    elif "Cek04" in check:
      res = df[df["AGE_M"] > 11]
    elif "Cek05" in check:
      res = df[df["DISTRICT"].isna() | df["PROVINCE"].isna()]
    elif "Cek06" in check:
      res = df[df.duplicated("EPID_ID", keep=False)].sort_values("EPID_ID")
    elif "Cek07" in check:
      res = df[df.duplicated("NAME", keep=False)].sort_values("NAME")
    elif "Cek08" in check:
      res = df[df["H_SPEC_COLL"].isna()]

    st.write(f"Ditemukan **{len(res)}** record bermasalah:")
    st.dataframe(res, use_container_width=True)

# 3. UPDATES & CLASSIFICATION
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

    if "Suspect_CRS" in st.session_state.df_current.columns:
      df_suspect_only = st.session_state.df_current[
          st.session_state.df_current["Suspect_CRS"] == "Y"
      ]
      summary_counts = df_suspect_only["FINAL_CLASS_UPDATED"].value_counts()

      st.subheader("Ringkasan Hasil Klasifikasi Suspek CRS (Khusus Bulletin):")
      c1, c2, c3, c4 = st.columns(4)
      c1.metric(
          "CRS Pasti (Lab Positif)",
          summary_counts.get("CRS Pasti (Lab Positif)", 0),
      )
      c2.metric("CRS Klinis", summary_counts.get("CRS Klinis", 0))
      c3.metric("Bukan CRS", summary_counts.get("Bukan CRS", 0))
      c4.metric("Pending", summary_counts.get("Pending", 0))

      st.dataframe(
          df_suspect_only[
              ["EPID_ID", "NAME", "AGE_M", "FINAL_CLASS_UPDATED"]
          ].head(10),
          use_container_width=True,
      )

# 4. VISUALISASI
elif step == "4. Visualisasi Grafik (Save PNG)":
  st.header("Langkah 4: Visualisasi Grafik CRS Surveillance")
  st.info(
      "💡 **Arahkan kursor ke pojok kanan atas grafik lalu klik ikon kamera 📷"
      " ('Download plot as a png') untuk menyimpan grafik.**"
  )

  if st.session_state.df_current is None:
    st.warning("Unggah data pada Langkah 1 terlebih dahulu.")
  else:
    df = st.session_state.df_current
    if "FINAL_CLASS_UPDATED" not in df.columns:
      df = run_classification_final(run_all_updates(df))
      st.session_state.df_current = df

    df_suspect = df[df["Suspect_CRS"] == "Y"]

    col_a, col_b = st.columns(2)
    with col_a:
      st.subheader("1. Klasifikasi Kasus Suspek CRS")
      counts = df_suspect["FINAL_CLASS_UPDATED"].value_counts().reset_index()
      counts.columns = ["Klasifikasi", "Jumlah"]
      fig1 = px.bar(
          counts,
          x="Klasifikasi",
          y="Jumlah",
          color="Klasifikasi",
          text="Jumlah",
          color_discrete_map={
              "CRS Pasti (Lab Positif)": "#e41a1c",
              "CRS Klinis": "#ff7f00",
              "Bukan CRS": "#4daf4a",
              "Pending": "#984ea3",
          },
      )
      fig1.update_traces(textposition="outside")
      st.plotly_chart(fig1, use_container_width=True)

    with col_b:
      st.subheader("2. Top 10 RS Sentinel (Suspek Terbanyak)")
      if "HOS_NAME" in df_suspect.columns:
        hos_counts = (
            df_suspect["HOS_NAME"].value_counts().reset_index().head(10)
        )
        hos_counts.columns = ["Rumah Sakit", "Jumlah Suspek"]
        fig2 = px.bar(
            hos_counts,
            y="Rumah Sakit",
            x="Jumlah Suspek",
            orientation="h",
            text="Jumlah Suspek",
            color_discrete_sequence=["#2b5c8f"],
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"})
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
      st.subheader("3. Temuan Gejala Klinis Kategori A")
      gejala_cols = {
          "A_CHD": "Kelainan Jantung (CHD)",
          "A_CC": "Katarak (CC)",
          "A_CG": "Glaukoma (CG)",
          "A_HI": "Gangguan Pendengaran (HI)",
      }
      gejala_data = [
          {
              "Gejala": label,
              "Jumlah Kasus": sum(
                  1 for v in df_suspect[k] if clean_str(v) in ["ya", "y"]
              ),
          }
          for k, label in gejala_cols.items()
          if k in df_suspect.columns
      ]
      fig3 = px.bar(
          pd.DataFrame(gejala_data),
          x="Gejala",
          y="Jumlah Kasus",
          text="Jumlah Kasus",
          color_discrete_sequence=["#e66101"],
      )
      fig3.update_traces(textposition="outside")
      st.plotly_chart(fig3, use_container_width=True)

    with col_d:
      st.subheader("4. Distribusi Umur Ibu Kasus")
      if "MOTHER_AGE" in df_suspect.columns:
        df_mother = df_suspect[
            df_suspect["MOTHER_AGE"].notna() & (df_suspect["MOTHER_AGE"] > 0)
        ]
        fig4 = px.histogram(
            df_mother,
            x="MOTHER_AGE",
            nbins=10,
            labels={"MOTHER_AGE": "Umur Ibu (Tahun)"},
            color_discrete_sequence=["#377eb8"],
        )
        st.plotly_chart(fig4, use_container_width=True)

# 5. EXPORT
elif step == "5. Export Reports":
  st.header("Langkah 5: Export Reports")
  if st.session_state.df_current is None:
    st.warning("Unggah data pada Langkah 1 terlebih dahulu.")
  else:
    df = st.session_state.df_current
    if "Suspect_CRS" not in df.columns:
      df = run_classification_final(run_all_updates(df))
      st.session_state.df_current = df

    c1, c2 = st.columns(2)
    bln, thn = c1.text_input("Bulan Pelaporan", "09"), c2.text_input(
        "Tahun Pelaporan", "2026"
    )

    # Bulletin Export
    df_bulletin = df[df["Suspect_CRS"] == "Y"]
    buf_b = BytesIO()
    with pd.ExcelWriter(buf_b, engine="openpyxl") as w:
      df_bulletin.to_excel(w, index=False)
    st.download_button(
        f"Download CRS_Bulletin_{bln} {thn}.xlsx ({len(df_bulletin)} records)",
        buf_b.getvalue(),
        f"CRS_Bulletin_{bln} {thn}.xlsx",
    )

    # Non-Suspect Export
    df_non = df[df["Suspect_CRS"] == "N"]
    buf_n = BytesIO()
    with pd.ExcelWriter(buf_n, engine="openpyxl") as w:
      df_non.to_excel(w, index=False)
    st.download_button(
        f"Download CRS_DB Non Suspect {bln} {thn}.xlsx ({len(df_non)} records)",
        buf_n.getvalue(),
        f"CRS_DB Non Suspect {bln} {thn}.xlsx",
    )
