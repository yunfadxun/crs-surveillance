import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CRS Surveillance Data Processing & Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

APP_TITLE = "CRS Surveillance Data Processing & Analytics Dashboard"

A_COLUMNS = [
    "A_CHD",
    "A_CC",
    "A_CG",
    "A_PR",
    "A_HI",
]

B_COLUMNS = [
    "B_PU",
    "B_MC",
    "B_ME",
    "B_IC",
    "B_SP",
    "B_DD",
    "B_RBD",
]

REQUIRED_COLUMNS = [
    "AGE_Y",
    "AGE_M",
    "AGE_D",
    "H_IGM_S1",
    "H_IGM_S2",
    "H_IGG_S1",
    "H_IGG_S2",
    "H_SPEC_COLL",
    "Suspect_CRS",
    "Clinical",
]

CLASSIFICATION_ORDER = [
    "CRS Pasti (Lab Positif)",
    "CRS Klinis",
    "Bukan CRS",
    "Pending",
    "Discarded / Non Suspect",
]

CLASSIFICATION_COLORS = {
    "CRS Pasti (Lab Positif)": "#d62728",
    "CRS Klinis": "#ff7f0e",
    "Bukan CRS": "#2ca02c",
    "Pending": "#9467bd",
    "Discarded / Non Suspect": "#7f7f7f",
}


# ============================================================
# GENERAL HELPER FUNCTIONS
# ============================================================

def clean_str(value):
    """
    Normalize values into lowercase strings.
    Missing values become empty strings.
    """
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def is_missing(value):
    """
    Identify missing values, including empty strings
    and strings containing only whitespace.
    """
    if pd.isna(value):
        return True

    return str(value).strip() == ""


def is_yes(value):
    """
    Recognize affirmative values.
    """
    return clean_str(value) in [
        "ya",
        "y",
        "1",
        "true",
    ]


def is_no(value):
    """
    Recognize negative values.
    """
    return clean_str(value) in [
        "tidak",
        "n",
        "no",
        "0",
        "false",
    ]


def normalize_lab_result(value):
    """
    Standardize laboratory result values.

    Examples:
        POSITIF -> positif
        Negatif -> negatif
        Empty -> ""
    """
    return clean_str(value)


def to_numeric_or_nan(value):
    """
    Convert a value to numeric.
    Invalid or missing values become NaN.
    """
    if is_missing(value):
        return np.nan

    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_columns(df):
    """
    Check whether the uploaded dataset contains
    the columns required by the processing functions.
    """
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    missing_symptom_columns = [
        col for col in A_COLUMNS + B_COLUMNS
        if col not in df.columns
    ]

    return missing_columns, missing_symptom_columns


def run_validation_checks(df):
    """
    Generate basic data-quality checks.

    These checks do not automatically remove records.
    They are intended to help identify issues in the data.
    """
    results = []

    results.append({
        "Validation Check": "Total records",
        "Records Flagged": len(df),
        "Description": "Total number of records in the dataset.",
    })

    if "AGE_Y" in df.columns:
        age_y = pd.to_numeric(
            df["AGE_Y"],
            errors="coerce",
        )

        invalid_age = (
            age_y.notna()
            & ((age_y < 0) | (age_y > 120))
        )

        results.append({
            "Validation Check": "Potentially invalid AGE_Y",
            "Records Flagged": int(invalid_age.sum()),
            "Description": "AGE_Y is outside the range of 0–120 years.",
        })

    if "AGE_M" in df.columns:
        age_m = pd.to_numeric(
            df["AGE_M"],
            errors="coerce",
        )

        invalid_age = age_m.notna() & (age_m < 0)

        results.append({
            "Validation Check": "Negative AGE_M",
            "Records Flagged": int(invalid_age.sum()),
            "Description": "AGE_M contains a negative value.",
        })

    if "H_IGM_S1" in df.columns:
        invalid_igm = (
            df["H_IGM_S1"].notna()
            & ~df["H_IGM_S1"].astype(str).str.strip().str.lower().isin(
                ["positif", "negatif", "equivocal", ""]
            )
        )

        results.append({
            "Validation Check": "Unrecognized H_IGM_S1 values",
            "Records Flagged": int(invalid_igm.sum()),
            "Description": (
                "Values other than Positif, Negatif, or Equivocal. "
                "Please verify the source data."
            ),
        })

    if "H_IGG_S1" in df.columns:
        invalid_igg = (
            df["H_IGG_S1"].notna()
            & ~df["H_IGG_S1"].astype(str).str.strip().str.lower().isin(
                ["positif", "negatif", "equivocal", ""]
            )
        )

        results.append({
            "Validation Check": "Unrecognized H_IGG_S1 values",
            "Records Flagged": int(invalid_igg.sum()),
            "Description": (
                "Values other than Positif, Negatif, or Equivocal. "
                "Please verify the source data."
            ),
        })

    return pd.DataFrame(results)


# ============================================================
# SUSPECT CRS CLASSIFICATION
# ============================================================

def check_suspect(row):
    """
    Determine whether a record meets the suspect CRS criteria
    used in the existing application.

    Criteria:
        Age < 1 year AND at least one A symptom is present.

    Note:
        Missing or invalid age is not automatically treated as zero.
    """
    has_gejala_a = any(
        is_yes(row.get(col))
        for col in A_COLUMNS
    )

    age_y = to_numeric_or_nan(row.get("AGE_Y"))
    age_m = to_numeric_or_nan(row.get("AGE_M"))

    # Prefer AGE_Y when available.
    if pd.notna(age_y):
        age_under_one = age_y < 1

    elif pd.notna(age_m):
        age_under_one = age_m < 12

    else:
        age_under_one = False

    return "Y" if age_under_one and has_gejala_a else "N"


# ============================================================
# CLINICAL CRS CLASSIFICATION
# ============================================================

def check_clinical(row):
    """
    Determine clinical CRS status using the existing criteria.

    Clinical CRS:
        At least 2 A symptoms
        OR
        At least 1 A symptom and at least 1 B symptom.
    """
    a_count = sum(
        1
        for col in A_COLUMNS
        if is_yes(row.get(col))
    )

    b_count = sum(
        1
        for col in B_COLUMNS
        if is_yes(row.get(col))
    )

    return "Y" if (
        a_count >= 2
        or (a_count >= 1 and b_count >= 1)
    ) else "N"


# ============================================================
# DATA UPDATES
# ============================================================

def run_all_updates(df):
    """
    Update derived age variables and specimen information.

    The original AGE_D, AGE_M, and AGE_Y values are retained
    when already populated, except where AGE_M or AGE_Y is
    missing and can be derived from AGE_D.

    Missing specimen collection status is standardized
    to 'Tidak Tahu'.
    """
    df = df.copy()

    # --------------------------------------------------------
    # Ensure age columns exist
    # --------------------------------------------------------

    for col in ["AGE_D", "AGE_M", "AGE_Y"]:
        if col not in df.columns:
            df[col] = np.nan

    # --------------------------------------------------------
    # Normalize age fields
    # --------------------------------------------------------

    for col in ["AGE_D", "AGE_M", "AGE_Y"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Derive age in months and years from age in days
    # when the derived value is missing.
    # --------------------------------------------------------

    age_d_available = df["AGE_D"].notna()

    age_m_missing = df["AGE_M"].isna()
    age_y_missing = df["AGE_Y"].isna()

    df.loc[
        age_m_missing & age_d_available,
        "AGE_M",
    ] = (
        df.loc[
            age_m_missing & age_d_available,
            "AGE_D",
        ] // 30
    )

    df.loc[
        age_y_missing & age_d_available,
        "AGE_Y",
    ] = (
        df.loc[
            age_y_missing & age_d_available,
            "AGE_D",
        ] // 365
    )

    # --------------------------------------------------------
    # Standardize specimen collection status
    # --------------------------------------------------------

    if "H_SPEC_COLL" not in df.columns:
        df["H_SPEC_COLL"] = np.nan

    df["H_SPEC_COLL"] = df["H_SPEC_COLL"].apply(
        lambda x: "Tidak Tahu" if is_missing(x) else str(x).strip()
    )

    # --------------------------------------------------------
    # Derive H_DCOLLSER1 when specimen collection is marked
    # as "Ya" but the collection date is missing.
    # --------------------------------------------------------

    if "H_DCOLLSER1" not in df.columns:
        df["H_DCOLLSER1"] = np.nan

    if "H_DRES_IGM_S1" not in df.columns:
        df["H_DRES_IGM_S1"] = np.nan

    if "DOCHECK" in df.columns:
        fallback_date = df["H_DRES_IGM_S1"].combine_first(
            df["DOCHECK"]
        )
    else:
        fallback_date = df["H_DRES_IGM_S1"]

    missing_collection_date = df["H_DCOLLSER1"].apply(is_missing)

    specimen_collected = df["H_SPEC_COLL"].apply(
        lambda x: clean_str(x) == "ya"
    )

    fill_date = missing_collection_date & specimen_collected

    df.loc[fill_date, "H_DCOLLSER1"] = fallback_date.loc[
        fill_date
    ]

    # --------------------------------------------------------
    # Calculate Suspect_CRS and Clinical
    # --------------------------------------------------------

    df["Suspect_CRS"] = df.apply(
        check_suspect,
        axis=1,
    )

    df["Clinical"] = df.apply(
        check_clinical,
        axis=1,
    )

    return df


# ============================================================
# FINAL CRS CLASSIFICATION
# Based on the latest Access FIN_CLASS_C formula
# ============================================================

def run_classification_final(df):
    """
    Classify suspected CRS cases using the latest FIN_CLASS_C
    Access logic.

    Classification labels:
        1 = CRS Pasti (Lab Positif)
        2 = CRS Klinis
        3 = Bukan CRS
        No matching rule = Pending

    Only records classified as Suspect_CRS = Y are evaluated.
    Non-suspected records are labelled Discarded / Non Suspect.
    """

    df = df.copy()

    def classify_row(row):

        # ----------------------------------------------------
        # Step 1: Only classify suspected CRS cases
        # ----------------------------------------------------

        suspect = clean_str(row.get("Suspect_CRS"))

        if suspect not in ["y", "ya", "1", "true"]:
            return "Discarded / Non Suspect"

        # ----------------------------------------------------
        # Step 2: Read age in months
        # ----------------------------------------------------

        age_m = to_numeric_or_nan(row.get("AGE_M"))

        if pd.isna(age_m):
            return "Pending"

        # ----------------------------------------------------
        # Step 3: Normalize laboratory results
        # ----------------------------------------------------

        igm1 = normalize_lab_result(
            row.get("H_IGM_S1")
        )

        igm2 = normalize_lab_result(
            row.get("H_IGM_S2")
        )

        igg1 = normalize_lab_result(
            row.get("H_IGG_S1")
        )

        igg2 = normalize_lab_result(
            row.get("H_IGG_S2")
        )

        igm2_missing = is_missing(
            row.get("H_IGM_S2")
        )

        igg2_missing = is_missing(
            row.get("H_IGG_S2")
        )

        # ----------------------------------------------------
        # Step 4: Read clinical classification
        # ----------------------------------------------------

        clinical = clean_str(
            row.get("Clinical")
        )

        clinical_y = clinical in [
            "y",
            "ya",
            "1",
            "true",
        ]

        clinical_n = clinical in [
            "n",
            "tidak",
            "0",
            "false",
        ]

        # ----------------------------------------------------
        # Step 5: Read specimen collection status
        # ----------------------------------------------------

        specimen = clean_str(
            row.get("H_SPEC_COLL")
        )

        specimen_not_collected_or_unknown = specimen in [
            "tidak",
            "tidak tahu",
        ]

        # ----------------------------------------------------
        # Step 6: CRS Pasti (Lab Positif)
        #
        # Following FIN_CLASS_C:
        #
        # AGE_M < 6 AND IGM1 positive
        # OR
        # AGE_M < 1 AND IGM1 negative AND IGM2 positive
        # OR
        # AGE_M > 5 AND IGM1 positive AND IGG1 positive
        # OR
        # AGE_M > 5 AND IGM1 negative AND IGG1 positive
        # AND IGG2 positive
        # ----------------------------------------------------

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
            return "CRS Pasti (Lab Positif)"

        # ----------------------------------------------------
        # Step 7: Bukan CRS based on laboratory results
        # ----------------------------------------------------

        if (
            (
                age_m >= 1
                and age_m < 6
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
            return "Bukan CRS"

        # ----------------------------------------------------
        # Step 8: Clinical classification for age < 1 month
        #
        # IGM1 negative and IGM2 missing
        # ----------------------------------------------------

        if (
            age_m < 1
            and igm1 == "negatif"
            and igm2_missing
            and clinical_y
        ):
            return "CRS Klinis"

        if (
            age_m < 1
            and igm1 == "negatif"
            and igm2_missing
            and clinical_n
        ):
            return "Bukan CRS"

        # ----------------------------------------------------
        # Step 9: Clinical classification for age > 5 months
        #
        # IGM1 negative, IGG1 positive, and IGG2 missing
        # ----------------------------------------------------

        if (
            age_m > 5
            and igm1 == "negatif"
            and igg1 == "positif"
            and igg2_missing
            and clinical_y
        ):
            return "CRS Klinis"

        if (
            age_m > 5
            and igm1 == "negatif"
            and igg1 == "positif"
            and igg2_missing
            and clinical_n
        ):
            return "Bukan CRS"

        # ----------------------------------------------------
        # Step 10: No specimen collected or collection status
        # is unknown, for age < 12 months
        # ----------------------------------------------------

        if (
            age_m < 12
            and clinical_y
            and specimen_not_collected_or_unknown
        ):
            return "CRS Klinis"

        if (
            age_m < 12
            and clinical_n
            and specimen_not_collected_or_unknown
        ):
            return "Bukan CRS"

        # ----------------------------------------------------
        # Step 11: No FIN_CLASS_C condition matched
        # ----------------------------------------------------

        return "Pending"

    df["Final_Classification"] = df.apply(
        classify_row,
        axis=1,
    )

    return df


# ============================================================
# SUMMARY TABLES
# ============================================================

def generate_classification_summary(df):
    """
    Generate classification counts and percentages.
    """

    if "Final_Classification" not in df.columns:
        return pd.DataFrame(
            columns=[
                "Classification",
                "Count",
                "Percentage",
            ]
        )

    counts = (
        df["Final_Classification"]
        .value_counts()
        .reindex(CLASSIFICATION_ORDER, fill_value=0)
    )

    total_records = len(df)

    summary = pd.DataFrame({
        "Classification": counts.index,
        "Count": counts.values,
    })

    if total_records > 0:
        summary["Percentage"] = (
            summary["Count"] / total_records * 100
        ).round(2)
    else:
        summary["Percentage"] = 0.0

    return summary


def generate_suspect_summary(df):
    """
    Generate a summary of suspected and non-suspected records.
    """

    counts = (
        df["Suspect_CRS"]
        .map({
            "Y": "Suspect CRS",
            "N": "Non-Suspect",
        })
        .fillna("Unknown")
        .value_counts()
    )

    summary = counts.rename_axis(
        "Suspect Status"
    ).reset_index(
        name="Count"
    )

    return summary


# ============================================================
# EXCEL EXPORT
# ============================================================

def create_excel_export(df, validation_df, summary_df):
    """
    Export processed records, summaries, and validation checks
    into a multi-sheet Excel workbook.
    """

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Processed Data",
            index=False,
        )

        summary_df.to_excel(
            writer,
            sheet_name="Classification Summary",
            index=False,
        )

        validation_df.to_excel(
            writer,
            sheet_name="Validation Checks",
            index=False,
        )

        suspect_summary = generate_suspect_summary(df)

        suspect_summary.to_excel(
            writer,
            sheet_name="Suspect Summary",
            index=False,
        )

    output.seek(0)

    return output.getvalue()


# ============================================================
# STREAMLIT DASHBOARD
# ============================================================

st.title(APP_TITLE)

st.markdown(
    """
    This dashboard processes CRS surveillance data, checks basic
    data quality, determines suspect and clinical status, and
    classifies suspected cases using the latest `FIN_CLASS_C`
    logic provided from Microsoft Access.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Data Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload CRS surveillance Excel file",
    type=["xlsx", "xlsm"],
    help="Upload the Excel file containing the CRS surveillance dataset.",
)

st.sidebar.caption(
    "The uploaded data will be processed in the current session."
)


# ============================================================
# MAIN PROCESSING
# ============================================================

if uploaded_file is None:

    st.info(
        "Please upload a CRS surveillance Excel file to begin."
    )

    st.markdown(
        """
        ### Processing workflow

        1. Upload the source Excel file.
        2. Validate required columns.
        3. Update derived age and specimen information.
        4. Determine `Suspect_CRS` and `Clinical`.
        5. Apply the latest `FIN_CLASS_C` classification logic.
        6. Review the dashboard and download the processed data.
        """
    )

    st.stop()


# ============================================================
# READ EXCEL
# ============================================================

try:

    excel_file = pd.ExcelFile(uploaded_file)

    available_sheets = excel_file.sheet_names

    selected_sheet = st.selectbox(
        "Select worksheet",
        options=available_sheets,
        index=0,
    )

    df_original = pd.read_excel(
        uploaded_file,
        sheet_name=selected_sheet,
    )

except Exception as e:

    st.error(
        f"Unable to read the uploaded Excel file: {e}"
    )

    st.stop()


if df_original.empty:

    st.warning(
        "The selected worksheet contains no records."
    )

    st.stop()


# ============================================================
# COLUMN VALIDATION
# ============================================================

missing_columns, missing_symptom_columns = validate_columns(
    df_original
)

if missing_columns:

    st.warning(
        "Some expected columns are missing. "
        "The application will create missing columns where "
        "necessary, but please verify that the correct worksheet "
        "has been selected."
    )

    with st.expander("View missing columns"):

        st.write(
            "Missing required columns:",
            missing_columns,
        )

        st.write(
            "Missing symptom columns:",
            missing_symptom_columns,
        )


# ============================================================
# CREATE MISSING COLUMNS
# ============================================================

df = df_original.copy()

for col in REQUIRED_COLUMNS:
    if col not in df.columns:
        df[col] = np.nan

for col in A_COLUMNS + B_COLUMNS:
    if col not in df.columns:
        df[col] = np.nan


# ============================================================
# RUN PROCESSING
# ============================================================

with st.spinner("Processing CRS surveillance data..."):

    df = run_all_updates(df)

    df = run_classification_final(df)

    validation_df = run_validation_checks(df)

    summary_df = generate_classification_summary(df)


st.success("Data processing completed.")


# ============================================================
# OVERVIEW METRICS
# ============================================================

st.header("1. Dataset Overview")

total_records = len(df)

suspect_count = int(
    df["Suspect_CRS"].apply(
        lambda x: clean_str(x) in ["y", "ya", "1", "true"]
    ).sum()
)

non_suspect_count = total_records - suspect_count

confirmed_count = int(
    (
        df["Final_Classification"]
        == "CRS Pasti (Lab Positif)"
    ).sum()
)

clinical_count = int(
    (
        df["Final_Classification"]
        == "CRS Klinis"
    ).sum()
)

not_crs_count = int(
    (
        df["Final_Classification"]
        == "Bukan CRS"
    ).sum()
)

pending_count = int(
    (
        df["Final_Classification"]
        == "Pending"
    ).sum()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Records",
    f"{total_records:,}",
)

col2.metric(
    "Suspect CRS",
    f"{suspect_count:,}",
)

col3.metric(
    "Non-Suspect",
    f"{non_suspect_count:,}",
)

col4.metric(
    "Pending",
    f"{pending_count:,}",
)


# ============================================================
# FINAL CLASSIFICATION METRICS
# ============================================================

st.header("2. Final Classification")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "CRS Pasti (Lab Positif)",
    f"{confirmed_count:,}",
)

m2.metric(
    "CRS Klinis",
    f"{clinical_count:,}",
)

m3.metric(
    "Bukan CRS",
    f"{not_crs_count:,}",
)

m4.metric(
    "Pending",
    f"{pending_count:,}",
)


# ============================================================
# CLASSIFICATION SUMMARY TABLE
# ============================================================

st.subheader("Classification Summary")

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# CLASSIFICATION CHART
# ============================================================

st.subheader("Classification Distribution")

chart_df = summary_df[
    summary_df["Classification"] != "Discarded / Non Suspect"
].copy()

if not chart_df.empty:

    fig = px.bar(
        chart_df,
        x="Classification",
        y="Count",
        text="Count",
        color="Classification",
        category_orders={
            "Classification": CLASSIFICATION_ORDER
        },
        color_discrete_map=CLASSIFICATION_COLORS,
        title="Final Classification of CRS Surveillance Records",
    )

    fig.update_layout(
        xaxis_title="Classification",
        yaxis_title="Number of Records",
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# SUSPECT STATUS SUMMARY
# ============================================================

st.header("3. Suspect Status")

suspect_summary = generate_suspect_summary(df)

st.dataframe(
    suspect_summary,
    use_container_width=True,
    hide_index=True,
)

suspect_chart = px.pie(
    suspect_summary,
    names="Suspect Status",
    values="Count",
    title="Suspect vs Non-Suspect Records",
    hole=0.4,
)

st.plotly_chart(
    suspect_chart,
    use_container_width=True,
)


# ============================================================
# VALIDATION CHECKS
# ============================================================

st.header("4. Data Validation Checks")

st.dataframe(
    validation_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# PROCESSED DATA
# ============================================================

st.header("5. Processed Surveillance Data")

st.caption(
    "Review the processed records below. "
    "The original columns are retained, with derived fields "
    "and classification results added or updated."
)

filter_options = [
    "All Records",
    "Suspect CRS",
    "CRS Pasti (Lab Positif)",
    "CRS Klinis",
    "Bukan CRS",
    "Pending",
    "Discarded / Non Suspect",
]

selected_filter = st.selectbox(
    "Filter records",
    options=filter_options,
)

if selected_filter == "All Records":

    display_df = df.copy()

elif selected_filter == "Suspect CRS":

    display_df = df[
        df["Suspect_CRS"].apply(
            lambda x: clean_str(x) in ["y", "ya", "1", "true"]
        )
    ].copy()

else:

    display_df = df[
        df["Final_Classification"] == selected_filter
    ].copy()


st.write(
    f"Showing **{len(display_df):,}** of "
    f"**{total_records:,}** records."
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DOWNLOAD PROCESSED DATA
# ============================================================

st.header("6. Download Results")

excel_bytes = create_excel_export(
    df=df,
    validation_df=validation_df,
    summary_df=summary_df,
)

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

download_filename = (
    f"CRS_Surveillance_Processed_{timestamp}.xlsx"
)

st.download_button(
    label="Download Processed Excel File",
    data=excel_bytes,
    file_name=download_filename,
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
)

st.caption(
    "The downloaded workbook contains Processed Data, "
    "Classification Summary, Validation Checks, and Suspect Summary."
)
