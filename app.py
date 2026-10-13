
from pathlib import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# COEUS | KEPLER EXOPLANET CANDIDATE CLASSIFICATION
# ============================================================

st.set_page_config(
    page_title="Coeus — Kepler Research",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "1.0.0"
DATA_PATH = Path(__file__).parent / "data" / "kepler_koi.csv"
RANDOM_STATE = 42
THRESHOLD = 0.50

FEATURES = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
]

FEATURE_INFO = {
    "koi_period": {
        "label": "Orbital period",
        "unit": "days",
        "description": "Estimated orbital period of the candidate.",
    },
    "koi_duration": {
        "label": "Transit duration",
        "unit": "hours",
        "description": "Duration of the observed transit event.",
    },
    "koi_depth": {
        "label": "Transit depth",
        "unit": "ppm",
        "description": "Measured transit depth in parts per million.",
    },
    "koi_prad": {
        "label": "Planet radius estimate",
        "unit": "Earth radii",
        "description": "Estimated radius relative to Earth.",
    },
    "koi_teq": {
        "label": "Equilibrium temperature",
        "unit": "K",
        "description": "Estimated equilibrium temperature in kelvin.",
    },
    "koi_insol": {
        "label": "Incident stellar flux",
        "unit": "Earth flux",
        "description": "Estimated incident flux relative to Earth.",
    },
    "koi_model_snr": {
        "label": "Transit signal-to-noise",
        "unit": "SNR",
        "description": "Signal-to-noise statistic for the transit model.",
    },
    "koi_steff": {
        "label": "Stellar effective temperature",
        "unit": "K",
        "description": "Estimated effective temperature of the host star.",
    },
    "koi_slogg": {
        "label": "Stellar surface gravity",
        "unit": "log(g)",
        "description": "Estimated logarithmic stellar surface gravity.",
    },
    "koi_srad": {
        "label": "Stellar radius",
        "unit": "Solar radii",
        "description": "Estimated stellar radius relative to the Sun.",
    },
}

NAVIGATION = [
    "Overview",
    "Data Explorer",
    "Classifier",
    "Research & Methods",
]

NAVY = "#244E7A"
BLUE = "#6D91B8"
TEAL = "#36796F"
GREY = "#8793A2"
GRID = "#E1E6ED"
INK = "#18283F"
MUTED = "#617084"


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --paper: #F5F6F8;
        --white: #FFFFFF;
        --ink: #18283F;
        --muted: #617084;
        --line: #D9DEE6;
        --blue: #244E7A;
        --blue-light: #EAF0F7;
        --font-system: -apple-system, BlinkMacSystemFont,
            "SF Pro Text", "SF Pro Display",
            "Helvetica Neue", Arial, sans-serif;
    }

    html, body, .stApp, [class*="css"],
    button, input, textarea, select {
        font-family: var(--font-system) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-system) !important;
        color: var(--ink) !important;
        letter-spacing: -0.025em;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
    }

    .block-container {
        max-width: 1320px;
        padding-top: 2rem;
        padding-bottom: 3.5rem;
    }

    p, li {
        line-height: 1.7;
        color: #39495D;
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 5px;
        padding: 17px 19px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.82rem !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-weight: 650;
    }

    .coeus-wordmark {
        color: var(--ink);
        font-family: var(--font-system);
        font-size: 2.25rem;
        font-weight: 750;
        letter-spacing: -0.065em;
        line-height: 1.1;
    }

    .coeus-subtitle {
        color: var(--muted);
        font-size: 0.70rem;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        margin-top: 7px;
    }

    .eyebrow, .small-label {
        color: var(--blue);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
    }

    .hero {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-top: 3px solid var(--blue);
        padding: clamp(22px, 4vw, 38px);
        margin-bottom: 26px;
        border-radius: 5px;
    }

    .hero h1 {
        font-family: var(--font-system) !important;
        font-size: clamp(2rem, 4vw, 3.1rem);
        font-weight: 720;
        line-height: 1.12;
        letter-spacing: -0.045em;
        margin: 10px 0 15px;
    }

    .hero p {
        max-width: 820px;
        color: var(--muted);
        font-size: 1rem;
        margin-bottom: 0;
    }

    .research-note {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-left: 3px solid var(--blue);
        padding: 17px 20px;
        margin: 14px 0 22px;
        border-radius: 3px;
    }

    .research-note p {
        margin-bottom: 0;
    }

    .footer {
        border-top: 1px solid var(--line);
        margin-top: 42px;
        padding-top: 18px;
        color: var(--muted);
        font-size: 0.80rem;
        line-height: 1.8;
    }

    .stButton > button,
.stButton > button *,
button[data-testid="baseButton-primary"],
button[data-testid="baseButton-primary"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
}

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .stLinkButton > a:hover {
        background: #173754;
        color: #FFFFFF;
        border-color: #173754;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 4px;
    }

    hr {
        border-color: var(--line);
        margin: 1.6rem 0;
    }

    a {
        color: var(--blue);
    }

    @media (max-width: 640px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 22px;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            transition-duration: 0.01ms !important;
            scroll-behavior: auto !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(path_string, modified_time):
    del modified_time  # Used as a cache invalidation key.

    data = pd.read_csv(path_string)

    required = ["koi_disposition"] + FEATURES
    missing_columns = [
        column for column in required if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    data = data[required].copy()
    data["koi_disposition"] = (
        data["koi_disposition"].astype("string").str.strip().str.upper()
    )

    data = data[
        data["koi_disposition"].isin(["CANDIDATE", "FALSE POSITIVE"])
    ].copy()

    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")

    data[FEATURES] = data[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.dropna(subset=FEATURES, how="all").copy()

    if data.empty:
        raise ValueError(
            "No usable observations remain after data validation."
        )

    data["target"] = (
        data["koi_disposition"] == "CANDIDATE"
    ).astype(int)

    return data


@st.cache_resource(show_spinner="Training the baseline classifier...")
def train_model(path_string, modified_time):
    data = load_data(path_string, modified_time)

    X = data[FEATURES].copy()
    y = data["target"].copy()

    if y.nunique() < 2:
        raise ValueError(
            "Training requires both CANDIDATE and FALSE POSITIVE "
            "observations. Check the dataset labels."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "There are too few observations in one class for a "
            "stratified train-test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Remove features that are entirely missing in the training set.
    usable_features = [
        feature for feature in FEATURES
        if X_train[feature].notna().any()
    ]

    if not usable_features:
        raise ValueError(
            "Every selected feature is missing in the training data."
        )

    X_train = X_train[usable_features]
    X_test = X_test[usable_features]

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Confusion matrix": confusion_matrix(
            y_test, predictions, labels=[1, 0]
        ),
        "Test size": len(y_test),
        "Test labels": y_test,
        "Test predictions": predictions,
        "Test scores": probabilities,
        "Features": usable_features,
        "Train size": len(y_train),
    }

    return pipeline, metrics


# ============================================================
# STARTUP CHECKS
# ============================================================

if not DATA_PATH.exists():
    st.error(
        "Coeus could not find its dataset. Place kepler_koi.csv "
        "inside the project's data folder."
    )
    st.code("Kepler-Planet-Classifier/data/kepler_koi.csv")
    st.stop()

try:
    modified_time = DATA_PATH.stat().st_mtime
    df = load_data(str(DATA_PATH), modified_time)
    model, metrics = train_model(str(DATA_PATH), modified_time)
except Exception as exc:
    st.error(
        "Coeus could not load the dataset or train its classifier."
    )
    st.info(
        "Check the CSV file, required column names, and whether "
        "both archive classes are present."
    )
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()


# ============================================================
# PLOT HELPERS
# ============================================================

def style_axes(fig, ax):
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.tick_params(colors="#536276", labelsize=9)
    ax.xaxis.label.set_color("#39495D")
    ax.yaxis.label.set_color("#39495D")
    ax.title.set_color(INK)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def add_footer():
    st.markdown(
        """
        <div class="footer">
            <strong>COEUS</strong><br>
            Independent educational research project.<br>
            Data source: NASA Exoplanet Archive, operated by
            Caltech/IPAC under contract with NASA.<br>
            Coeus is not affiliated with or endorsed by NASA or Caltech.
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_label(feature):
    info = FEATURE_INFO[feature]
    return f"{info['label']} ({info['unit']})"


def render_distribution_chart(data):
    counts = (
        data["koi_disposition"]
        .value_counts()
        .reindex(["CANDIDATE", "FALSE POSITIVE"])
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(9, 3.8))
    style_axes(fig, ax)

    bars = ax.bar(
        counts.index,
        counts.values,
        color=[NAVY, BLUE],
        width=0.52,
    )

    ax.set_ylabel("Number of observations")
    ax.set_title(
        "Recorded archive classifications",
        loc="left",
        fontsize=12,
        pad=14,
        fontweight="semibold",
    )

    for bar in bars:
        height = int(bar.get_height())
        ax.annotate(
            f"{height:,}",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color=INK,
        )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="coeus-wordmark">Coeus</div>
        <div class="coeus-subtitle">Kepler research workspace</div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "RESEARCH WORKSPACE",
        NAVIGATION,
        label_visibility="visible",
    )

    st.divider()

    st.markdown(
        """
        <div class="small-label">DATA PROVIDER</div>
        <p>NASA Exoplanet Archive</p>
        <div class="small-label">BASELINE MODEL</div>
        <p>Class-weighted logistic regression</p>
        <div class="small-label">PROJECT VERSION</div>
        <p>1.0.0</p>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Exploratory analysis of historical Kepler Objects of Interest."
    )


# ============================================================
# PAGE HEADER
# ============================================================

if page == "Overview":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Astronomical data analysis</div>
            <h1>Understanding transit signals through data.</h1>
            <p>
                Coeus is an educational machine-learning project exploring
                whether stellar and transit measurements from the Kepler
                Objects of Interest catalogue can help distinguish
                recorded candidate labels from recorded false-positive
                labels.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Coeus research workspace</div>
            <h1>Kepler exoplanet candidate classification</h1>
            <p>
                Explore archived observations, examine a baseline
                classifier, and review the methodology and limitations.
                Model outputs are exploratory and are not planet
                confirmations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":
    st.caption("01 / OVERVIEW")
    st.title("Research overview")

    st.write(
        "The Kepler space telescope detected many exoplanet candidates "
        "by measuring small, repeated decreases in a star's brightness. "
        "These events are called transits. Other phenomena can also "
        "produce transit-like signals, so candidates require further "
        "validation."
    )

    total = len(df)
    candidate_count = int((df["target"] == 1).sum())
    false_positive_count = int((df["target"] == 0).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Usable observations", f"{total:,}")
    c2.metric("Recorded candidates", f"{candidate_count:,}")
    c3.metric("Recorded false positives", f"{false_positive_count:,}")

    st.markdown("### Classification distribution")
    render_distribution_chart(df)

    st.markdown("### Analytical framework")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Input measurements")
        st.write(
            "Ten numerical variables describe orbital period, transit "
            "duration and depth, estimated planetary properties, transit "
            "signal strength, and host-star properties."
        )

    with right:
        st.markdown("#### Computational method")
        st.write(
            "A preprocessing pipeline imputes missing values, "
            "standardises numerical features, and fits class-weighted "
            "logistic regression on a stratified training split."
        )

    st.markdown(
        """
        <div class="research-note">
            <strong>Scope and interpretation</strong>
            <p>
                Coeus learns to reproduce patterns in historical archive
                labels. These labels are not independent ground truth for
                this model, and a model prediction cannot confirm the
                existence or absence of an exoplanet.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Continue your exploration")

    explore_col, classify_col = st.columns(2)

    with explore_col:
        st.markdown("#### Data Explorer")
        st.write(
            "Compare selected measurements and inspect the records "
            "used in the analysis."
        )
        if st.button("Explore the data", use_container_width=True):
            st.session_state["navigation_target"] = "Data Explorer"
            st.rerun()

    with classify_col:
        st.markdown("#### Classifier")
        st.write(
            "Enter a set of measurements and inspect the baseline "
            "model's output."
        )
        if st.button("Open the classifier", use_container_width=True):
            st.session_state["navigation_target"] = "Classifier"
            st.rerun()


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":
    st.caption("02 / EXPLORATORY DATA ANALYSIS")
    st.title("Data Explorer")

    st.write(
        "Explore relationships between selected measurements and "
        "the historical archive classifications."
    )

    first, second = st.columns(2)

    with first:
        x_feature = st.selectbox(
            "Horizontal variable",
            FEATURES,
            index=FEATURES.index("koi_period"),
            format_func=feature_label,
        )

    with second:
        y_feature = st.selectbox(
            "Vertical variable",
            FEATURES,
            index=FEATURES.index("koi_prad"),
            format_func=feature_label,
        )

    max_points = st.select_slider(
        "Maximum observations plotted",
        options=[500, 1000, 2000, 4000, 6000],
        value=2000,
    )

    category_filter = st.multiselect(
        "Archive classifications",
        ["CANDIDATE", "FALSE POSITIVE"],
        default=["CANDIDATE", "FALSE POSITIVE"],
    )

    chart_data = df[
        [x_feature, y_feature, "koi_disposition"]
    ].dropna()

    chart_data = chart_data[
        chart_data["koi_disposition"].isin(category_filter)
    ]

    if len(chart_data) > max_points:
        chart_data = chart_data.sample(
            max_points,
            random_state=RANDOM_STATE,
        )

    if chart_data.empty:
        st.warning(
            "No observations match the current filters. "
            "Select at least one archive classification."
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        style_axes(fig, ax)

        for category, colour in [
            ("CANDIDATE", NAVY),
            ("FALSE POSITIVE", BLUE),
        ]:
            subset = chart_data[
                chart_data["koi_disposition"] == category
            ]

            if subset.empty:
                continue

            ax.scatter(
                subset[x_feature],
                subset[y_feature],
                s=19,
                alpha=0.58,
                color=colour,
                label=category,
                edgecolors="none",
                rasterized=True,
            )

        ax.set_xlabel(feature_label(x_feature))
        ax.set_ylabel(feature_label(y_feature))
        ax.set_title(
            f"{FEATURE_INFO[x_feature]['label']} vs. "
            f"{FEATURE_INFO[y_feature]['label']}",
            loc="left",
            fontsize=12,
            pad=14,
            fontweight="semibold",
        )
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            f"{len(chart_data):,} observations displayed. "
            "Points may overlap. Visual association does not establish "
            "causation or independently validate a planet."
        )

    st.divider()
    st.markdown("### Data quality")

    missing_counts = df[FEATURES].isna().sum()
    missing_table = pd.DataFrame({
        "Feature": [feature_label(f) for f in FEATURES],
        "Missing observations": [
            int(missing_counts[f]) for f in FEATURES
        ],
        "Missing (%)": [
            float(missing_counts[f] / len(df) * 100)
            for f in FEATURES
        ],
    })

    st.dataframe(
        missing_table,
        hide_index=True,
        use_container_width=True,
    )

    st.divider()
    st.markdown("### Observation records")

    filter_choice = st.selectbox(
        "Filter archive disposition",
        ["All", "CANDIDATE", "FALSE POSITIVE"],
    )

    selected_columns = ["koi_disposition"] + FEATURES
    table = df[selected_columns]

    if filter_choice != "All":
        table = table[
            table["koi_disposition"] == filter_choice
        ]

    st.caption(f"{len(table):,} matching records")

    st.dataframe(
        table.head(500),
        hide_index=True,
        use_container_width=True,
        height=400,
    )

    st.caption(
        "The table displays at most 500 rows. The plotted sample "
        "and table filter are separate controls."
    )


# ============================================================
# CLASSIFIER
# ============================================================

elif page == "Classifier":
    st.caption("03 / MODEL INTERFACE")
    st.title("Classifier")

    st.write(
        "Enter measurements to inspect how the baseline model "
        "classifies a hypothetical observation."
    )

    st.markdown(
        """
        <div class="research-note">
            <strong>How to interpret this tool</strong>
            <p>
                Default inputs use the dataset medians. Input limits use
                the 1st and 99th percentiles of available measurements.
                These bounds support exploration but do not guarantee
                that a combination of values is physically plausible.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    usable_features = metrics["Features"]

    with st.form("coeus_classifier_form"):
        inputs = {}
        left, right = st.columns(2)

        for index, feature in enumerate(usable_features):
            column = left if index < (len(usable_features) + 1) // 2 else right
            values = df[feature].dropna()

            if values.empty:
                continue

            median = float(values.median())
            lower = float(values.quantile(0.01))
            upper = float(values.quantile(0.99))

            if not np.isfinite(lower) or not np.isfinite(upper):
                lower, upper = median - 1.0, median + 1.0

            if lower >= upper:
                lower, upper = median - 1.0, median + 1.0

            with column:
                inputs[feature] = st.number_input(
                    feature_label(feature),
                    min_value=float(lower),
                    max_value=float(upper),
                    value=float(np.clip(median, lower, upper)),
                    format="%.5f",
                    help=FEATURE_INFO[feature]["description"]
                    + f" Dataset median: {median:.5g}.",
                    key=f"input_{feature}",
                )

        submitted = st.form_submit_button(
            "Run classification",
            use_container_width=True,
        )

    if submitted:
        observation = pd.DataFrame(
            [[inputs[feature] for feature in usable_features]],
            columns=usable_features,
        )

        candidate_score = float(
            model.predict_proba(observation)[0][1]
        )

        predicted_label = (
            "CANDIDATE"
            if candidate_score >= THRESHOLD
            else "FALSE POSITIVE"
        )

        st.divider()
        st.markdown("### Classification result")

        result_col, score_col = st.columns(2)

        with result_col:
            st.markdown("**Predicted archive-label class**")
            st.subheader(predicted_label)
            st.caption(
                f"Decision threshold: {THRESHOLD:.2f}"
            )

        with score_col:
            st.metric(
                "Candidate-class model score",
                f"{candidate_score:.1%}",
            )
            st.progress(candidate_score)

        st.warning(
            "This score is not necessarily a calibrated probability "
            "that a planet exists. It is the model's output for the "
            "recorded CANDIDATE class, not a scientific confirmation."
        )

        with st.expander("Inspect input measurements"):
            display_inputs = observation.T.reset_index()
            display_inputs.columns = ["Feature", "Input value"]
            display_inputs["Description"] = display_inputs[
                "Feature"
            ].map(lambda f: FEATURE_INFO[f]["description"])
            st.dataframe(
                display_inputs,
                hide_index=True,
                use_container_width=True,
            )


# ============================================================
# RESEARCH & METHODS
# ============================================================

elif page == "Research & Methods":
    st.caption("04 / METHODS AND REFERENCES")
    st.title("Research & Methods")

    st.markdown("## Model evaluation")

    st.write(
        "The following results are calculated on a stratified test "
        "split held out from model fitting. The decision threshold is "
        f"{THRESHOLD:.2f}. These results describe this split and should "
        "not be assumed to generalise to new releases or other surveys."
    )

    a, b, c = st.columns(3)
    a.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
    b.metric(
        "Balanced accuracy",
        f"{metrics['Balanced accuracy']:.1%}",
    )
    c.metric("F1 score", f"{metrics['F1 score']:.3f}")

    d, e = st.columns(2)
    d.metric("Precision", f"{metrics['Precision']:.1%}")
    e.metric("Recall", f"{metrics['Recall']:.1%}")

    st.caption(
        f"Training observations: {metrics['Train size']:,} · "
        f"Test observations: {metrics['Test size']:,} · "
        f"Random seed: {RANDOM_STATE}"
    )

    st.markdown("### Confusion matrix")

    matrix = metrics["Confusion matrix"]

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual candidate", "Actual false positive"],
        columns=[
            "Predicted candidate",
            "Predicted false positive",
        ],
    )

    st.dataframe(
        matrix_df,
        use_container_width=True,
    )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor("#FFFFFF")

    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], [
        "Predicted candidate",
        "Predicted false positive",
    ])
    ax.set_yticks([0, 1], [
        "Actual candidate",
        "Actual false positive",
    ])
    ax.set_title("Confusion matrix", loc="left", pad=14)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Recorded label")

    midpoint = matrix.max() / 2 if matrix.size and matrix.max() else 0

    for row in range(2):
        for col in range(2):
            value = int(matrix[row, col])
            ax.text(
                col,
                row,
                f"{value:,}",
                ha="center",
                va="center",
                color="white" if value > midpoint else INK,
                fontsize=12,
                fontweight="semibold",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("## Computational methodology")

    st.markdown(
        """
        1. **Data preparation:** retain observations labelled
           `CANDIDATE` or `FALSE POSITIVE`; convert selected features
           to numeric values and treat infinite values as missing.
        2. **Partitioning:** use a stratified 80/20 train-test split
           with a fixed random seed.
        3. **Missing values:** impute missing values using medians
           learned from the training partition.
        4. **Feature scaling:** standardise the numerical variables.
        5. **Classification:** fit class-weighted logistic regression.
        6. **Evaluation:** calculate accuracy, balanced accuracy,
           precision, recall, F1 score and the confusion matrix.
        """
    )

    st.markdown("## Research paper")

    st.write(
        "**Kepler Data Validation I: Architecture, Diagnostic Tests, "
        "and Data Products for Vetting Transiting Planet Candidates**"
    )
    st.caption("Twicken et al. (2018) · NASA Technical Reports Server")

    st.write(
        "This paper describes Kepler's data-validation architecture "
        "and diagnostic products for assessing transit-like signals. "
        "It is background literature, not a description of the "
        "Coeus machine-learning model."
    )

    paper_url = (
        "https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/"
        "20180001990.pdf"
    )

    google_viewer_url = (
        "https://docs.google.com/gview?embedded=1&url="
        + quote(paper_url, safe="")
    )

    try:
        components.iframe(
            google_viewer_url,
            height=680,
            scrolling=True,
        )
    except Exception:
        st.info(
            "The embedded PDF viewer could not be displayed. "
            "Use the original PDF link below."
        )

    st.link_button(
        "Open original NASA-hosted PDF",
        paper_url,
    )

    st.markdown("## Data provenance and acknowledgment")

    st.write(
        "**Data source:** NASA Exoplanet Archive, Kepler Objects of "
        "Interest (KOI) cumulative table."
    )

    st.markdown(
        "[Dataset DOI: 10.26133/NEA4](https://doi.org/10.26133/NEA4)"
    )

    st.markdown(
        "[KOI table documentation]"
        "(https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html)"
    )

    st.markdown(
        "[NASA Exoplanet Archive acknowledgment guidance]"
        "(https://exoplanetarchive.nexsci.services/docs/acknowledge.html)"
    )

    st.markdown(
        """
        **Recommended acknowledgment**

        This research has made use of the NASA Exoplanet Archive,
        which is operated by the California Institute of Technology,
        under contract with the National Aeronautics and Space
        Administration under the Exoplanet Exploration Program.
        """
    )

    st.markdown("### Related literature")

    st.markdown(
        """
        - Batalha et al. (2013), Kepler planet-candidate catalogue.
        - Burke et al. (2014), Kepler planet-candidate catalogue.
        - Rowe et al. (2015), Kepler planet-candidate catalogue.
        - Mullally et al. (2015), Kepler planet-candidate catalogue.
        - Coughlin et al. (2016), Kepler data validation and candidate work.
        - Thompson et al. (2018), final Kepler DR25 candidate catalogue.
        - Christiansen et al. (2025), NASA Exoplanet Archive overview.
        """
    )

    st.markdown("## Limitations")

    st.markdown(
        """
        - The target is a recorded archive disposition, not independently
          verified planetary status.
        - The cumulative KOI table combines information from different
          releases and may not form a uniform sample for every analysis.
        - Random splitting can place observations associated with the
          same host star in both training and test sets.
        - A stronger study could use a defined catalogue release and
          validation grouped by host star.
        - Class weighting changes the training objective and can affect
          the interpretation of model scores.
        - No independent calibration procedure is performed here.
        - This baseline is an educational exploration, not a validated
          exoplanet discovery or confirmation system.
        """
    )


# ============================================================
# FOOTER
# ============================================================

add_footer()
