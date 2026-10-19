from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CRS Surveillance System",
    layout="wide",
)

st.title("CRS Surveillance Data Processing & Analytics Dashboard")


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_str(val):
    """Normalize string values and handle missing values."""

    if pd.isna(val):
        return ""

    return str(val).strip().lower()


def normalize_result(value):
    """Normalize laboratory test results."""

    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    if value in ["", "none", "nan", "null", "nat"]:
        return ""

    if value in ["positif", "positive", "pos"]:
        return "positif"

    if value in ["negatif", "negative", "neg"]:
        return "negatif"

    return value


def normalize_specimen(value):
    """Normalize specimen collection status."""

    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    if value in ["", "none", "nan", "null", "nat"]:
        return ""

    return value


def is_yes(value):
    """Identify affirmative responses."""

    return clean_str(value) in ["ya", "y", "1", "true"]


def get_numeric_age(value):
    """Convert age to a numeric value, returning None if invalid."""

    if pd.isna(value):
        return None

    try:
        age = float(value)

        if not np.isfinite(age) or age < 0:
            return None

        return age

    except (TypeError, ValueError):
        return None


# =========================================================
# SUSPECT CRS CLASSIFICATION
# =========================================================

def check_suspect(row):
    """
    Suspect CRS criteria:

    Age < 12 months AND at least one Category A symptom.

    Category A:
    A_CHD, A_CC, A_CG, A_PR, A_HI

    AGE_Y and AGE_M are both checked to preserve the
    original age criteria in the source application.
    """

    a_cols = [
        "A_CHD",
        "A_CC",
        "A_CG",
        "A_PR",
        "A_HI",
    ]

    has_gejala_a = any(
        is_yes(row.get(col))
        for col in a_cols
    )

    age_y = get_numeric_age(row.get("AGE_Y"))
    age_m = get_numeric_age(row.get("AGE_M"))

    # Missing age is not automatically considered zero.
    if age_y is None or age_m is None:
        return "N"

    if age_y < 1 and age_m < 12 and has_gejala_a:
        return "Y"

    return "N"


# =========================================================
# CLINICAL CRS CLASSIFICATION
# =========================================================

def check_clinical(row):
    """
    Clinical CRS criteria based on the Access formula.

    Clinical:
    - At least 2 Category A symptoms; OR
    - At least 1 Category A symptom AND at least
      1 Category B symptom.

    Otherwise:
    - Non Clinical.
    """

    a_cols = [
        "A_CHD",
        "A_CC",
        "A_CG",
        "A_PR",
        "A_HI",
    ]

    b_cols = [
        "B_PU",
        "B_MC",
        "B_ME",
        "B_IC",
        "B_DD",
        "B_SP",
        "B_RBD",
    ]

    a_count = sum(
        1
        for col in a_cols
        if is_yes(row.get(col))
    )

    b_count = sum(
        1
        for col in b_cols
        if is_yes(row.get(col))
    )

    if a_count >= 2 or (a_count >= 1 and b_count >= 1):
        return "Clinical"

    return "Non Clinical"


# =========================================================
# AUTOMATIC AGE AND SPECIMEN UPDATES
# =========================================================

def run_all_updates(df):
    """
    Update age and specimen collection dates.

    Age is calculated from DOCHECK and DOB only when
    the existing age information is considered incomplete.

    Specimen collection date is filled using the available
    IgM result date or DOCHECK, following the original logic.
    """

    df = df.copy()

    # -----------------------------------------------------
    # 1. UPDATE AGE
    # -----------------------------------------------------

    if "DOCHECK" in df.columns and "DOB" in df.columns:

        docheck = pd.to_datetime(
            df["DOCHECK"],
            errors="coerce",
        )

        dob = pd.to_datetime(
            df["DOB"],
            errors="coerce",
        )

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

        # Do not assign negative ages.
        valid_days = days_diff.notna() & (days_diff >= 0)

        cond_update = cond_empty & valid_days

        df.loc[cond_update, "AGE_D"] = days_diff[cond_update]

        df.loc[cond_update, "AGE_M"] = (
            days_diff[cond_update] // 30
        )

        df.loc[cond_update, "AGE_Y"] = (
            days_diff[cond_update] // 365
        )

    # -----------------------------------------------------
    # 2. UPDATE SPECIMEN COLLECTION DATE
    # -----------------------------------------------------

    if "H_SPEC_COLL" in df.columns:

        df["H_SPEC_COLL"] = df["H_SPEC_COLL"].fillna(
            "Tidak Tahu"
        )

        if "H_DCOLLSER1" in df.columns:

            # Fill using IgM result date if available.
            if "H_DRES_IGM_S1" in df.columns:

                cond3 = (
                    df["H_DCOLLSER1"].isna()
                    & df["H_DRES_IGM_S1"].notna()
                    & (
                        df["H_SPEC_COLL"]
                        .apply(clean_str)
                        == "ya"
                    )
                )

                df.loc[cond3, "H_DCOLLSER1"] = (
                    df.loc[cond3, "H_DRES_IGM_S1"]
                )

            # Otherwise, use DOCHECK.
            if "DOCHECK" in df.columns:

                cond4 = (
                    df["H_DCOLLSER1"].isna()
                    & (
                        df["H_SPEC_COLL"]
                        .apply(clean_str)
                        == "ya"
                    )
                )

                df.loc[cond4, "H_DCOLLSER1"] = (
                    df.loc[cond4, "DOCHECK"]
                )

    return df


# =========================================================
# FINAL CRS CLASSIFICATION
# =========================================================

def run_classification_final(df):
    """
    Final CRS classification based on the latest FIN_CLASS_C.

    Classification codes:
        1 = CRS Pasti (Lab Positif)
        2 = CRS Klinis
        3 = Bukan CRS

    Records that do not meet any classification condition
    are assigned Pending.

    Non-suspect records are assigned:
        Discarded / Non Suspect

    Laboratory columns:
        H_IGM_S1
        H_IGM_S2
        H_IGG_S1
        H_IGG_S2
    """

    df = df.copy()

    # -----------------------------------------------------
    # 1. DETERMINE SUSPECT AND CLINICAL STATUS
    # -----------------------------------------------------

    df["Suspect_CRS"] = df.apply(
        check_suspect,
        axis=1,
    )

    df["Clinical"] = df.apply(
        check_clinical,
        axis=1,
    )

    # -----------------------------------------------------
    # 2. CLASSIFICATION FUNCTION
    # -----------------------------------------------------

    def classify_row(row):

        # Non-suspect records are not classified further.
        if row["Suspect_CRS"] != "Y":

            return pd.Series({
                "FINAL_CLASS_CODE": np.nan,
                "FINAL_CLASS_UPDATED": "Discarded / Non Suspect",
            })

        # Age in months.
        age_m = get_numeric_age(row.get("AGE_M"))

        # Missing or invalid age.
        if age_m is None:

            return pd.Series({
                "FINAL_CLASS_CODE": np.nan,
                "FINAL_CLASS_UPDATED": "Pending",
            })

        # Normalize laboratory results.
        igm1 = normalize_result(
            row.get("H_IGM_S1")
        )

        igm2 = normalize_result(
            row.get("H_IGM_S2")
        )

        igg1 = normalize_result(
            row.get("H_IGG_S1")
        )

        igg2 = normalize_result(
            row.get("H_IGG_S2")
        )

        clinical = row.get(
            "Clinical",
            "Non Clinical",
        )

        specimen = normalize_specimen(
            row.get("H_SPEC_COLL")
        )

        # -------------------------------------------------
        # RULE 1: CRS PASTI (LAB POSITIF) = 1
        # -------------------------------------------------

        if (
            (
                age_m < 6
                and igm1 == "positif"
            )
            or (
                age_m < 1
                and igm1 == "negatif"
                and igm2 == "positif"
            )
            or (
                age_m > 5
                and igm1 == "positif"
                and igg1 == "positif"
            )
            or (
                age_m > 5
                and igm1 == "negatif"
                and igg1 == "positif"
                and igg2 == "positif"
            )
        ):

            code = 1

        # -------------------------------------------------
        # RULE 2: BUKAN CRS BASED ON LAB RESULTS = 3
        # -------------------------------------------------

        elif (
            (
                1 <= age_m < 6
                and igm1 == "negatif"
            )
            or (
                age_m < 1
                and igm1 == "negatif"
                and igm2 == "negatif"
            )
            or (
                age_m > 5
                and igg1 == "negatif"
            )
            or (
                age_m > 5
                and igm1 == "negatif"
                and igg1 == "positif"
                and igg2 == "negatif"
            )
        ):

            code = 3

        # -------------------------------------------------
        # RULE 3: CLINICAL CLASSIFICATION WHEN RESULTS
        # ARE INCOMPLETE
        # -------------------------------------------------

        elif (
            age_m < 1
            and igm1 == "negatif"
            and igm2 == ""
        ):

            if clinical == "Clinical":
                code = 2
            else:
                code = 3

        elif (
            age_m > 5
            and igm1 == "negatif"
            and igg1 == "positif"
            and igg2 == ""
        ):

            if clinical == "Clinical":
                code = 2
            else:
                code = 3

        # -------------------------------------------------
        # RULE 4: NO SPECIMEN COLLECTED OR UNKNOWN
        # -------------------------------------------------

        elif (
            age_m < 12
            and clinical == "Clinical"
            and specimen in [
                "tidak",
                "tidak tahu",
            ]
        ):

            code = 2

        elif (
            age_m < 12
            and clinical == "Non Clinical"
            and specimen in [
                "tidak",
                "tidak tahu",
            ]
        ):

            code = 3

        # -------------------------------------------------
        # RULE 5: NOT CLASSIFIABLE
        # -------------------------------------------------

        else:

            code = np.nan

        # -------------------------------------------------
        # 3. CONVERT CODE TO CLASSIFICATION LABEL
        # -------------------------------------------------

        labels = {
            1: "CRS Pasti (Lab Positif)",
            2: "CRS Klinis",
            3: "Bukan CRS",
        }

        label = labels.get(
            code,
            "Pending",
        )

        return pd.Series({
            "FINAL_CLASS_CODE": code,
            "FINAL_CLASS_UPDATED": label,
        })

    # -----------------------------------------------------
    # 4. APPLY CLASSIFICATION
    # -----------------------------------------------------

    results = df.apply(
        classify_row,
        axis=1,
    )

    df["FINAL_CLASS_CODE"] = results[
        "FINAL_CLASS_CODE"
    ]

    df["FINAL_CLASS_UPDATED"] = results[
        "FINAL_CLASS_UPDATED"
    ]

    return df


# =========================================================
# SESSION STATE
# =========================================================

if "df_current" not in st.session_state:
    st.session_state.df_current = None


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

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


# =========================================================
# STEP 1: IMPORT DATA
# =========================================================

if step == "1. Import Data Excel":

    st.header("Langkah 1: Import Data")

    uploaded_file = st.file_uploader(
        "Upload File Raw Excel (.xlsx / .xls)",
        type=["xlsx", "xls"],
    )

    if uploaded_file is not None:

        try:

            df_uploaded = pd.read_excel(
                uploaded_file
            )

            st.session_state.df_current = (
                df_uploaded.copy()
            )

            st.success(
                "Data berhasil diunggah! "
                f"Total record: {len(df_uploaded):,}"
            )

            st.write(
                f"Total kolom: {len(df_uploaded.columns):,}"
            )

            st.dataframe(
                df_uploaded.head(10),
                use_container_width=True,
            )

        except Exception as e:

            st.error(
                f"Gagal membaca file Excel: {e}"
            )


# =========================================================
# STEP 2: DATA VALIDATION
# =========================================================

elif step == "2. Data Cleaning & Checks":

    st.header("Langkah 2: Validation Checks")

    if st.session_state.df_current is None:

        st.warning(
            "Unggah data pada Langkah 1 terlebih dahulu."
        )

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

        # -------------------------------------------------
        # CHECK 01: MISSING EPID ID
        # -------------------------------------------------

        if "Cek01" in check:

            if "EPID_ID" in df.columns:

                res = df[
                    df["EPID_ID"].isna()
                    | (
                        df["EPID_ID"]
                        .astype(str)
                        .str.strip()
                        .eq("")
                    )
                ]

            else:

                res = pd.DataFrame()

        # -------------------------------------------------
        # CHECK 02: MISSING DATE OF BIRTH
        # -------------------------------------------------

        elif "Cek02" in check:

            if "DOB" in df.columns:

                res = df[
                    df["DOB"].isna()
                ]

            else:

                res = pd.DataFrame()

        # -------------------------------------------------
        # CHECK 03: MISSING AGE
        # -------------------------------------------------

        elif "Cek03" in check:

            required = [
                "AGE_M",
                "AGE_Y",
            ]

            if all(
                col in df.columns
                for col in required
            ):

                res = df[
                    df["AGE_M"].isna()
                    | df["AGE_Y"].isna()
                ]

            else:

                res = pd.DataFrame()

        # -------------------------------------------------
        # CHECK 04: AGE ABOVE 11 MONTHS
        # -------------------------------------------------

        elif "Cek04" in check:

            if "AGE_M" in df.columns:

                res = df[
                    pd.to_numeric(
                        df["AGE_M"],
                        errors="coerce",
                    ) > 11
                ]

            else:

                res = pd.DataFrame()

        # -------------------------------------------------
        # CHECK 05: MISSING DISTRICT OR PROVINCE
        # -------------------------------------------------

        elif "Cek05" in check:

            required = [
                "DISTRICT",
                "PROVINCE",
            ]

            if all(
                col in df.columns
                for col in required
            ):

                res = df[
                    df["DISTRICT"].isna()
                    | df["PROVINCE"].isna()
                ]

            else:

                res = pd.DataFrame()

        # -------------------------------------------------
        # CHECK 06: DUPLICATE EPID ID
        # -------------------------------------------------

        elif "Cek06" in check:

            if "EPID_ID" in df.columns:

                valid_id = (
                    df["EPID_ID"].notna()
                    & df["EPID_ID"]
                    .astype(str)
                    .str.strip()
                    .ne("")
                )

                res = df[
                    valid_id
                    & df.duplicated(
                        "EPID_ID",
                        keep=False,
                    )
                ].sort_values(
                    "EPID_ID"
                )

            else:

                res = pd.DataFrame()

        # -------------------------------------------------
        # CHECK 07: DUPLICATE NAME
        # -------------------------------------------------

        elif "Cek07" in check:

            if "NAME" in df.columns:

                valid_name = (
                    df["NAME"].notna()
                    & df["NAME"]
                    .astype(str)
                    .str.strip()
                    .ne("")
                )

                res = df[
                    valid_name
                    & df.duplicated(
                        "NAME",
                        keep=False,
                    )
                ].sort_values(
                    "NAME"
                )

            else:

                res = pd.DataFrame()

        # -------------------------------------------------
        # CHECK 08: MISSING SPECIMEN COLLECTION STATUS
        # -------------------------------------------------

        elif "Cek08" in check:

            if "H_SPEC_COLL" in df.columns:

                res = df[
                    df["H_SPEC_COLL"].isna()
                    | (
                        df["H_SPEC_COLL"]
                        .astype(str)
                        .str.strip()
                        .eq("")
                    )
                ]

            else:

                res = pd.DataFrame()

        # -------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------

        st.write(
            f"Ditemukan **{len(res):,}** record bermasalah."
        )

        st.dataframe(
            res,
            use_container_width=True,
        )

        if not res.empty:

            buffer = BytesIO()

            with pd.ExcelWriter(
                buffer,
                engine="openpyxl",
            ) as writer:

                res.to_excel(
                    writer,
                    index=False,
                    sheet_name="Validation Results",
                )

            st.download_button(
                label="Download Validation Results",
                data=buffer.getvalue(),
                file_name="CRS_Validation_Results.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )


# =========================================================
# STEP 3: UPDATES & CLASSIFICATION
# =========================================================

elif step == "3. Run Updates & Classification":

    st.header(
        "Langkah 3: Auto-Update & Klasifikasi CRS"
    )

    if st.session_state.df_current is None:

        st.warning(
            "Unggah data pada Langkah 1 terlebih dahulu."
        )

    else:

        st.info(
            "Proses ini akan memperbarui informasi umur, "
            "tanggal pengambilan spesimen, status suspek, "
            "status klinis, dan klasifikasi final CRS."
        )

        if st.button(
            "Jalankan Semua Update & Klasifikasi Otomatis",
            type="primary",
        ):

            try:

                df_updated = run_all_updates(
                    st.session_state.df_current
                )

                df_classified = run_classification_final(
                    df_updated
                )

                st.session_state.df_current = (
                    df_classified
                )

                st.success(
                    "Proses update dan klasifikasi selesai!"
                )

            except Exception as e:

                st.error(
                    f"Terjadi kesalahan saat klasifikasi: {e}"
                )

        # -------------------------------------------------
        # DISPLAY CLASSIFICATION SUMMARY
        # -------------------------------------------------

        df = st.session_state.df_current

        if "FINAL_CLASS_UPDATED" in df.columns:

            df_suspect_only = df[
                df["Suspect_CRS"] == "Y"
            ]

            summary_counts = (
                df_suspect_only[
                    "FINAL_CLASS_UPDATED"
                ]
                .value_counts()
            )

            st.subheader(
                "Ringkasan Hasil Klasifikasi Suspek CRS"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "CRS Pasti (Lab Positif)",
                summary_counts.get(
                    "CRS Pasti (Lab Positif)",
                    0,
                ),
            )

            c2.metric(
                "CRS Klinis",
                summary_counts.get(
                    "CRS Klinis",
                    0,
                ),
            )

            c3.metric(
                "Bukan CRS",
                summary_counts.get(
                    "Bukan CRS",
                    0,
                ),
            )

            c4.metric(
                "Pending",
                summary_counts.get(
                    "Pending",
                    0,
                ),
            )

            st.write(
                "Total kasus suspek: "
                f"**{len(df_suspect_only):,}**"
            )

            display_cols = [
                col
                for col in [
                    "EPID_ID",
                    "NAME",
                    "AGE_Y",
                    "AGE_M",
                    "Clinical",
                    "H_SPEC_COLL",
                    "H_IGM_S1",
                    "H_IGM_S2",
                    "H_IGG_S1",
                    "H_IGG_S2",
                    "FINAL_CLASS_CODE",
                    "FINAL_CLASS_UPDATED",
                ]
                if col in df_suspect_only.columns
            ]

            st.dataframe(
                df_suspect_only[
                    display_cols
                ].head(20),
                use_container_width=True,
            )

            # Download classified data.
            buffer = BytesIO()

            with pd.ExcelWriter(
                buffer,
                engine="openpyxl",
            ) as writer:

                df.to_excel(
                    writer,
                    index=False,
                    sheet_name="CRS Classification",
                )

            st.download_button(
                label="Download Full Classified Dataset",
                data=buffer.getvalue(),
                file_name="CRS_Classified_Data.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )


# =========================================================
# STEP 4: VISUALIZATION
# =========================================================

elif step == "4. Visualisasi Grafik (Save PNG)":

    st.header(
        "Langkah 4: Visualisasi Grafik CRS Surveillance"
    )

    st.info(
        "Arahkan kursor ke pojok kanan atas grafik, "
        "kemudian klik ikon kamera untuk mengunduh grafik PNG."
    )

    if st.session_state.df_current is None:

        st.warning(
            "Unggah data pada Langkah 1 terlebih dahulu."
        )

    else:

        df = st.session_state.df_current.copy()

        # Run classification if it has not been performed.
        if "FINAL_CLASS_UPDATED" not in df.columns:

            df = run_all_updates(df)

            df = run_classification_final(df)

            st.session_state.df_current = df

        df_suspect = df[
            df["Suspect_CRS"] == "Y"
        ].copy()

        # -------------------------------------------------
        # CHART 1: FINAL CLASSIFICATION
        # -------------------------------------------------

        col_a, col_b = st.columns(2)

        with col_a:

            st.subheader(
                "1. Klasifikasi Kasus Suspek CRS"
            )

            counts = (
                df_suspect[
                    "FINAL_CLASS_UPDATED"
                ]
                .value_counts()
                .reindex(
                    [
                        "CRS Pasti (Lab Positif)",
                        "CRS Klinis",
                        "Bukan CRS",
                        "Pending",
                    ],
                    fill_value=0,
                )
                .rename_axis("Klasifikasi")
                .reset_index(name="Jumlah")
            )

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

            fig1.update_traces(
                textposition="outside"
            )

            fig1.update_layout(
                showlegend=False,
                yaxis_title="Jumlah Kasus",
                xaxis_title="Klasifikasi",
            )

            st.plotly_chart(
                fig1,
                use_container_width=True,
            )

        # -------------------------------------------------
        # CHART 2: TOP 10 SENTINEL HOSPITALS
        # -------------------------------------------------

        with col_b:

            st.subheader(
                "2. Top 10 RS Sentinel "
                "(Suspek Terbanyak)"
            )

            if "HOS_NAME" in df_suspect.columns:

                hos_counts = (
                    df_suspect["HOS_NAME"]
                    .dropna()
                    .value_counts()
                    .head(10)
                    .rename_axis("Rumah Sakit")
                    .reset_index(name="Jumlah Suspek")
                )

                fig2 = px.bar(
                    hos_counts,
                    y="Rumah Sakit",
                    x="Jumlah Suspek",
                    orientation="h",
                    text="Jumlah Suspek",
                    color_discrete_sequence=[
                        "#2b5c8f"
                    ],
                )

                fig2.update_layout(
                    yaxis={
                        "categoryorder": "total ascending"
                    },
                    xaxis_title="Jumlah Suspek",
                    yaxis_title="Rumah Sakit",
                )

                fig2.update_traces(
                    textposition="outside"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True,
                )

            else:

                st.warning(
                    "Kolom HOS_NAME tidak ditemukan."
                )

        # -------------------------------------------------
        # CHART 3: CATEGORY A SYMPTOMS
        # -------------------------------------------------

        col_c, col_d = st.columns(2)

        with col_c:

            st.subheader(
                "3. Temuan Gejala Klinis Kategori A"
            )

            gejala_cols = {
                "A_CHD": "Kelainan Jantung (CHD)",
                "A_CC": "Katarak (CC)",
                "A_CG": "Glaukoma (CG)",
                "A_PR": "Retinopati Pigmen (PR)",
                "A_HI": "Gangguan Pendengaran (HI)",
            }

            gejala_data = []

            for col, label in gejala_cols.items():

                if col in df_suspect.columns:

                    count = sum(
                        1
                        for value in df_suspect[col]
                        if is_yes(value)
                    )

                    gejala_data.append({
                        "Gejala": label,
                        "Jumlah Kasus": count,
                    })

            if gejala_data:

                fig3 = px.bar(
                    pd.DataFrame(gejala_data),
                    x="Gejala",
                    y="Jumlah Kasus",
                    text="Jumlah Kasus",
                    color_discrete_sequence=[
                        "#e66101"
                    ],
                )

                fig3.update_traces(
                    textposition="outside"
                )

                fig3.update_layout(
                    xaxis_title="Gejala",
                    yaxis_title="Jumlah Kasus",
                )

                st.plotly_chart(
                    fig3,
                    use_container_width=True,
                )

            else:

                st.warning(
                    "Kolom gejala Kategori A tidak ditemukan."
                )

        # -------------------------------------------------
        # CHART 4: MOTHER AGE DISTRIBUTION
        # -------------------------------------------------

        with col_d:

            st.subheader(
                "4. Distribusi Umur Ibu Kasus"
            )

            if "MOTHER_AGE" in df_suspect.columns:

                df_mother = df_suspect.copy()

                df_mother["MOTHER_AGE"] = pd.to_numeric(
                    df_mother["MOTHER_AGE"],
                    errors="coerce",
                )

                df_mother = df_mother[
                    df_mother["MOTHER_AGE"].notna()
                    & (df_mother["MOTHER_AGE"] > 0)
                ]

                if not df_mother.empty:

                    fig4 = px.histogram(
                        df_mother,
                        x="MOTHER_AGE",
                        nbins=10,
                        labels={
                            "MOTHER_AGE": "Umur Ibu (Tahun)"
                        },
                        color_discrete_sequence=[
                            "#377eb8"
                        ],
                    )

                    fig4.update_layout(
                        xaxis_title="Umur Ibu (Tahun)",
                        yaxis_title="Jumlah Kasus",
                    )

                    st.plotly_chart(
                        fig4,
                        use_container_width=True,
                    )

                else:

                    st.info(
                        "Tidak ada data umur ibu yang valid."
                    )

            else:

                st.warning(
                    "Kolom MOTHER_AGE tidak ditemukan."
                )


# =========================================================
# STEP 5: EXPORT REPORTS
# =========================================================

elif step == "5. Export Reports":

    st.header(
        "Langkah 5: Export Reports"
    )

    if st.session_state.df_current is None:

        st.warning(
            "Unggah data pada Langkah 1 terlebih dahulu."
        )

    else:

        df = st.session_state.df_current.copy()

        # Ensure classification has been performed.
        if "FINAL_CLASS_UPDATED" not in df.columns:

            df = run_all_updates(df)

            df = run_classification_final(df)

            st.session_state.df_current = df

        # -------------------------------------------------
        # REPORTING PERIOD
        # -------------------------------------------------

        c1, c2 = st.columns(2)

        bln = c1.text_input(
            "Bulan Pelaporan",
            "09",
        )

        thn = c2.text_input(
            "Tahun Pelaporan",
            "2026",
        )

        # -------------------------------------------------
        # BULLETIN EXPORT
        # Suspect CRS cases only
        # -------------------------------------------------

        st.subheader(
            "1. CRS Bulletin"
        )

        df_bulletin = df[
            df["Suspect_CRS"] == "Y"
        ].copy()

        buf_b = BytesIO()

        with pd.ExcelWriter(
            buf_b,
            engine="openpyxl",
        ) as writer:

            df_bulletin.to_excel(
                writer,
                index=False,
                sheet_name="CRS Bulletin",
            )

        st.download_button(
            label=(
                f"Download CRS_Bulletin_{bln}_{thn}.xlsx "
                f"({len(df_bulletin):,} records)"
            ),
            data=buf_b.getvalue(),
            file_name=(
                f"CRS_Bulletin_{bln}_{thn}.xlsx"
            ),
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            key="download_bulletin",
        )

        # -------------------------------------------------
        # NON-SUSPECT EXPORT
        # -------------------------------------------------

        st.subheader(
            "2. CRS Database - Non Suspect"
        )

        df_non = df[
            df["Suspect_CRS"] == "N"
        ].copy()

        buf_n = BytesIO()

        with pd.ExcelWriter(
            buf_n,
            engine="openpyxl",
        ) as writer:

            df_non.to_excel(
                writer,
                index=False,
                sheet_name="Non Suspect",
            )

        st.download_button(
            label=(
                f"Download CRS_DB_Non_Suspect_{bln}_{thn}.xlsx "
                f"({len(df_non):,} records)"
            ),
            data=buf_n.getvalue(),
            file_name=(
                f"CRS_DB_Non_Suspect_{bln}_{thn}.xlsx"
            ),
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            key="download_non_suspect",
        )

        # -------------------------------------------------
        # COMPLETE DATABASE EXPORT
        # -------------------------------------------------

        st.subheader(
            "3. Full CRS Database"
        )

        buf_full = BytesIO()

        with pd.ExcelWriter(
            buf_full,
            engine="openpyxl",
        ) as writer:

            df.to_excel(
                writer,
                index=False,
                sheet_name="Full CRS Database",
            )

        st.download_button(
            label=(
                f"Download CRS_Full_Database_{bln}_{thn}.xlsx "
                f"({len(df):,} records)"
            ),
            data=buf_full.getvalue(),
            file_name=(
                f"CRS_Full_Database_{bln}_{thn}.xlsx"
            ),
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            key="download_full_database",
        )
