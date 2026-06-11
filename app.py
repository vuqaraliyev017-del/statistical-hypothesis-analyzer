
import io
import itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from docx import Document


ALPHA_DEFAULT = 0.05


@dataclass
class AnalysisResult:
    analysis_type: str
    variables: str
    h0: str
    h1: str
    statistic_name: str
    statistic_value: Optional[float]
    p_value: Optional[float]
    decision: str
    summary: str
    details: str = ""


def clean_column_name(col: str) -> str:
    return (
        str(col)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
    )


def load_data(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Yalnız CSV və Excel faylları dəstəklənir.")

    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # Numeric kimi görünən object sütunları çevirməyə çalışırıq
    for col in df.columns:
        if col not in numeric_cols:
            converted = pd.to_numeric(df[col], errors="coerce")
            non_null_ratio = converted.notna().mean()
            if non_null_ratio >= 0.75:
                df[col] = converted
                numeric_cols.append(col)
                if col in categorical_cols:
                    categorical_cols.remove(col)

    # Az unikal dəyəri olan numeric sütunlar categorical kimi də istifadə oluna bilər
    for col in numeric_cols.copy():
        if df[col].nunique(dropna=True) <= 10:
            if col not in categorical_cols:
                categorical_cols.append(col)

    return numeric_cols, categorical_cols


def pvalue_decision(p_value: Optional[float], alpha: float) -> str:
    if p_value is None or np.isnan(p_value):
        return "Decision verilə bilmədi"
    return "H0 rejected / H1 accepted" if p_value < alpha else "H0 not rejected / H1 declined"


def format_p(p: Optional[float]) -> str:
    if p is None or np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


def normality_test(df: pd.DataFrame, numeric: str, group: Optional[str] = None) -> str:
    rows = []
    if group:
        for g, sub in df[[numeric, group]].dropna().groupby(group):
            values = sub[numeric]
            if len(values) >= 3:
                stat, p = stats.shapiro(values)
                rows.append(f"{group}={g}: Shapiro-Wilk p={format_p(p)}")
            else:
                rows.append(f"{group}={g}: normality üçün müşahidə sayı azdır")
    else:
        values = df[numeric].dropna()
        if len(values) >= 3:
            stat, p = stats.shapiro(values)
            rows.append(f"Shapiro-Wilk p={format_p(p)}")
        else:
            rows.append("Normality üçün müşahidə sayı azdır")
    return "; ".join(rows)


def descriptive_statistics(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    desc = df[numeric_cols].describe().T
    desc["missing"] = df[numeric_cols].isna().sum()
    desc["median"] = df[numeric_cols].median()
    desc["skewness"] = df[numeric_cols].skew()
    desc["kurtosis"] = df[numeric_cols].kurtosis()
    return desc.reset_index().rename(columns={"index": "variable"})


def independent_ttest(df: pd.DataFrame, numeric: str, group: str, alpha: float) -> AnalysisResult:
    data = df[[numeric, group]].dropna()
    groups = list(data[group].dropna().unique())

    if len(groups) != 2:
        return AnalysisResult(
            "Independent Samples t-test",
            f"{numeric} by {group}",
            "",
            "",
            "t",
            None,
            None,
            "Uyğun deyil",
            f"{group} sütununda tam 2 qrup olmalıdır. Hazırda {len(groups)} qrup var."
        )

    g1, g2 = groups[0], groups[1]
    x1 = data.loc[data[group] == g1, numeric]
    x2 = data.loc[data[group] == g2, numeric]

    if len(x1) < 2 or len(x2) < 2:
        return AnalysisResult(
            "Independent Samples t-test",
            f"{numeric} by {group}",
            "",
            "",
            "t",
            None,
            None,
            "Uyğun deyil",
            "Hər qrupda ən azı 2 müşahidə olmalıdır."
        )

    stat, p = stats.ttest_ind(x1, x2, equal_var=False, nan_policy="omit")
    decision = pvalue_decision(p, alpha)

    h0 = f"{g1} və {g2} qrupları arasında {numeric} üzrə orta göstəricidə statistik fərq yoxdur."
    h1 = f"{g1} və {g2} qrupları arasında {numeric} üzrə orta göstəricidə statistik fərq vardır."
    summary = (
        f"Independent t-test nəticəsinə görə p-value={format_p(p)}. "
        f"{'p < alpha olduğu üçün H0 rədd edilir və qruplar arasında statistik əhəmiyyətli fərq var.' if p < alpha else 'p ≥ alpha olduğu üçün H0 rədd edilmir və statistik əhəmiyyətli fərq tapılmadı.'}"
    )

    details = (
        f"{g1}: n={len(x1)}, mean={x1.mean():.4f}, std={x1.std():.4f}; "
        f"{g2}: n={len(x2)}, mean={x2.mean():.4f}, std={x2.std():.4f}. "
        f"Assumption check: {normality_test(data, numeric, group)}"
    )

    return AnalysisResult("Independent Samples t-test", f"{numeric} by {group}", h0, h1, "t", float(stat), float(p), decision, summary, details)


def paired_ttest(df: pd.DataFrame, before: str, after: str, alpha: float) -> AnalysisResult:
    data = df[[before, after]].dropna()
    if len(data) < 2:
        return AnalysisResult("Paired Samples t-test", f"{before} vs {after}", "", "", "t", None, None, "Uyğun deyil", "Ən azı 2 paired müşahidə lazımdır.")

    stat, p = stats.ttest_rel(data[before], data[after])
    decision = pvalue_decision(p, alpha)

    h0 = f"{before} və {after} ölçmələri arasında orta göstəricidə statistik fərq yoxdur."
    h1 = f"{before} və {after} ölçmələri arasında orta göstəricidə statistik fərq vardır."
    summary = (
        f"Paired t-test nəticəsinə görə p-value={format_p(p)}. "
        f"{'p < alpha olduğu üçün H0 rədd edilir və iki ölçmə arasında statistik əhəmiyyətli fərq var.' if p < alpha else 'p ≥ alpha olduğu üçün H0 rədd edilmir və statistik əhəmiyyətli fərq tapılmadı.'}"
    )
    diff = data[after] - data[before]
    details = f"n={len(data)}, mean difference={diff.mean():.4f}, std difference={diff.std():.4f}. Difference normality: {normality_test(pd.DataFrame({'diff': diff}), 'diff')}"

    return AnalysisResult("Paired Samples t-test", f"{before} vs {after}", h0, h1, "t", float(stat), float(p), decision, summary, details)


def one_way_anova(df: pd.DataFrame, numeric: str, group: str, alpha: float) -> Tuple[AnalysisResult, Optional[pd.DataFrame]]:
    data = df[[numeric, group]].dropna()
    grouped = [sub[numeric].values for _, sub in data.groupby(group)]

    if len(grouped) < 3:
        result = AnalysisResult(
            "One-way ANOVA",
            f"{numeric} by {group}",
            "",
            "",
            "F",
            None,
            None,
            "Uyğun deyil",
            "ANOVA üçün ən azı 3 qrup lazımdır. 2 qrup varsa t-test istifadə edin."
        )
        return result, None

    if any(len(x) < 2 for x in grouped):
        result = AnalysisResult(
            "One-way ANOVA",
            f"{numeric} by {group}",
            "",
            "",
            "F",
            None,
            None,
            "Uyğun deyil",
            "Hər qrupda ən azı 2 müşahidə olmalıdır."
        )
        return result, None

    stat, p = stats.f_oneway(*grouped)
    decision = pvalue_decision(p, alpha)

    h0 = f"{group} qrupları arasında {numeric} üzrə orta göstəricidə statistik fərq yoxdur."
    h1 = f"{group} qruplarından ən azı biri üzrə {numeric} orta göstəricisi fərqlidir."
    summary = (
        f"One-way ANOVA nəticəsinə görə p-value={format_p(p)}. "
        f"{'p < alpha olduğu üçün H0 rədd edilir və qruplar arasında statistik əhəmiyyətli fərq var.' if p < alpha else 'p ≥ alpha olduğu üçün H0 rədd edilmir və qruplar arasında statistik əhəmiyyətli fərq tapılmadı.'}"
    )

    details = f"Assumption check: {normality_test(data, numeric, group)}"

    tukey_df = None
    if p < alpha:
        try:
            tukey = pairwise_tukeyhsd(endog=data[numeric], groups=data[group], alpha=alpha)
            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        except Exception:
            tukey_df = None

    return AnalysisResult("One-way ANOVA", f"{numeric} by {group}", h0, h1, "F", float(stat), float(p), decision, summary, details), tukey_df


def correlation_analysis(df: pd.DataFrame, x: str, y: str, alpha: float, method: str = "pearson") -> AnalysisResult:
    data = df[[x, y]].dropna()
    if len(data) < 3:
        return AnalysisResult("Correlation", f"{x} and {y}", "", "", "r", None, None, "Uyğun deyil", "Correlation üçün ən azı 3 müşahidə lazımdır.")

    if method == "spearman":
        stat, p = stats.spearmanr(data[x], data[y])
        test_name = "Spearman correlation"
        stat_name = "rho"
    else:
        stat, p = stats.pearsonr(data[x], data[y])
        test_name = "Pearson correlation"
        stat_name = "r"

    h0 = f"{x} və {y} arasında statistik əhəmiyyətli əlaqə yoxdur."
    h1 = f"{x} və {y} arasında statistik əhəmiyyətli əlaqə vardır."
    decision = pvalue_decision(p, alpha)

    direction = "müsbət" if stat > 0 else "mənfi"
    strength_abs = abs(stat)
    if strength_abs < 0.3:
        strength = "zəif"
    elif strength_abs < 0.7:
        strength = "orta"
    else:
        strength = "güclü"

    summary = (
        f"{test_name} nəticəsinə görə correlation coefficient={stat:.4f}, p-value={format_p(p)}. "
        f"{'p < alpha olduğu üçün statistik əhəmiyyətli əlaqə var.' if p < alpha else 'p ≥ alpha olduğu üçün statistik əhəmiyyətli əlaqə tapılmadı.'} "
        f"Əlaqənin istiqaməti {direction}, gücü isə {strength} səviyyədədir."
    )

    return AnalysisResult(test_name, f"{x} and {y}", h0, h1, stat_name, float(stat), float(p), decision, summary, f"n={len(data)}")


def chi_square_test(df: pd.DataFrame, cat1: str, cat2: str, alpha: float) -> Tuple[AnalysisResult, Optional[pd.DataFrame]]:
    data = df[[cat1, cat2]].dropna()
    if data.empty:
        return AnalysisResult("Chi-square test", f"{cat1} and {cat2}", "", "", "chi2", None, None, "Uyğun deyil", "Məlumat yoxdur."), None

    table = pd.crosstab(data[cat1], data[cat2])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return AnalysisResult("Chi-square test", f"{cat1} and {cat2}", "", "", "chi2", None, None, "Uyğun deyil", "Hər iki categorical dəyişəndə ən azı 2 kateqoriya olmalıdır."), table

    stat, p, dof, expected = stats.chi2_contingency(table)

    h0 = f"{cat1} və {cat2} dəyişənləri arasında asılılıq yoxdur."
    h1 = f"{cat1} və {cat2} dəyişənləri arasında statistik əhəmiyyətli asılılıq vardır."
    decision = pvalue_decision(p, alpha)
    summary = (
        f"Chi-square test nəticəsinə görə p-value={format_p(p)}. "
        f"{'p < alpha olduğu üçün H0 rədd edilir və dəyişənlər arasında statistik əhəmiyyətli asılılıq var.' if p < alpha else 'p ≥ alpha olduğu üçün H0 rədd edilmir və statistik əhəmiyyətli asılılıq tapılmadı.'}"
    )
    details = f"chi-square={stat:.4f}, dof={dof}, n={len(data)}"

    return AnalysisResult("Chi-square test", f"{cat1} and {cat2}", h0, h1, "chi2", float(stat), float(p), decision, summary, details), table


def linear_regression(df: pd.DataFrame, y: str, x_cols: List[str], alpha: float) -> Tuple[AnalysisResult, pd.DataFrame, str]:
    cols = [y] + x_cols
    data = df[cols].dropna()

    if len(data) <= len(x_cols) + 2:
        return AnalysisResult("Linear Regression", f"{y} ~ {', '.join(x_cols)}", "", "", "F", None, None, "Uyğun deyil", "Regression üçün müşahidə sayı çox azdır."), pd.DataFrame(), ""

    X = sm.add_constant(data[x_cols])
    Y = data[y]
    model = sm.OLS(Y, X).fit()

    f_p = float(model.f_pvalue) if model.f_pvalue is not None else np.nan
    h0 = f"{', '.join(x_cols)} dəyişənləri birlikdə {y} dəyişənini statistik əhəmiyyətli izah etmir."
    h1 = f"{', '.join(x_cols)} dəyişənlərindən ən azı biri {y} dəyişənini statistik əhəmiyyətli izah edir."
    decision = pvalue_decision(f_p, alpha)
    summary = (
        f"Linear regression modelinin F-test p-value={format_p(f_p)}, R²={model.rsquared:.4f}, adjusted R²={model.rsquared_adj:.4f}. "
        f"{'Model ümumi olaraq statistik əhəmiyyətlidir.' if f_p < alpha else 'Model ümumi olaraq statistik əhəmiyyətli deyil.'}"
    )

    coef_df = pd.DataFrame({
        "variable": model.params.index,
        "coefficient": model.params.values,
        "p_value": model.pvalues.values,
        "significant": ["Yes" if p < alpha else "No" for p in model.pvalues.values]
    })

    detail_lines = []
    for _, row in coef_df.iterrows():
        if row["variable"] != "const":
            sign = "müsbət" if row["coefficient"] > 0 else "mənfi"
            sig = "statistik əhəmiyyətlidir" if row["p_value"] < alpha else "statistik əhəmiyyətli deyil"
            detail_lines.append(
                f"{row['variable']}: coefficient={row['coefficient']:.4f}, p={format_p(row['p_value'])}, təsir {sign} və {sig}."
            )

    try:
        vif_df = pd.DataFrame()
        if len(x_cols) >= 2:
            vif_df["variable"] = x_cols
            vif_df["VIF"] = [variance_inflation_factor(data[x_cols].values, i) for i in range(len(x_cols))]
            vif_text = " VIF: " + "; ".join([f"{r.variable}={r.VIF:.2f}" for r in vif_df.itertuples()])
        else:
            vif_text = ""
    except Exception:
        vif_text = ""

    return AnalysisResult("Linear Regression", f"{y} ~ {', '.join(x_cols)}", h0, h1, "F", float(model.fvalue), f_p, decision, summary, " ".join(detail_lines) + vif_text), coef_df, model.summary().as_text()


def auto_suggest_tests(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
    suggestions = []

    for group in categorical_cols:
        n_groups = df[group].nunique(dropna=True)
        for num in numeric_cols:
            if group == num:
                continue
            if n_groups == 2:
                suggestions.append(f"Independent t-test: {num} by {group}")
            elif 3 <= n_groups <= 10:
                suggestions.append(f"One-way ANOVA: {num} by {group}")

    for x, y in itertools.combinations(numeric_cols, 2):
        suggestions.append(f"Correlation: {x} and {y}")

    for c1, c2 in itertools.combinations(categorical_cols, 2):
        if df[c1].nunique(dropna=True) <= 20 and df[c2].nunique(dropna=True) <= 20:
            suggestions.append(f"Chi-square: {c1} and {c2}")

    return suggestions[:100]


def result_to_dict(r: AnalysisResult) -> Dict:
    return {
        "Analysis Type": r.analysis_type,
        "Variables": r.variables,
        "H0": r.h0,
        "H1": r.h1,
        "Statistic": r.statistic_name,
        "Statistic Value": r.statistic_value,
        "p-value": r.p_value,
        "Decision": r.decision,
        "Summary": r.summary,
        "Details": r.details,
    }


def build_excel_report(results: List[AnalysisResult], desc_df: Optional[pd.DataFrame] = None, extra_tables: Optional[Dict[str, pd.DataFrame]] = None) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if desc_df is not None and not desc_df.empty:
            desc_df.to_excel(writer, sheet_name="Descriptive", index=False)

        if results:
            pd.DataFrame([result_to_dict(r) for r in results]).to_excel(writer, sheet_name="Hypothesis Tests", index=False)

        if extra_tables:
            for name, table in extra_tables.items():
                safe_name = name[:31].replace("/", "_").replace("\\", "_")
                table.to_excel(writer, sheet_name=safe_name, index=True)

    output.seek(0)
    return output.getvalue()


def build_word_report(results: List[AnalysisResult], desc_df: Optional[pd.DataFrame] = None) -> bytes:
    doc = Document()
    doc.add_heading("Statistical Analysis Report", 0)

    doc.add_paragraph("Bu report avtomatik olaraq yaradılıb. Qərarlar alpha səviyyəsinə əsasən verilib.")

    if desc_df is not None and not desc_df.empty:
        doc.add_heading("1. Descriptive Statistics", level=1)
        table = doc.add_table(rows=1, cols=len(desc_df.columns))
        hdr_cells = table.rows[0].cells
        for i, col in enumerate(desc_df.columns):
            hdr_cells[i].text = str(col)

        for _, row in desc_df.head(30).iterrows():
            cells = table.add_row().cells
            for i, col in enumerate(desc_df.columns):
                val = row[col]
                cells[i].text = f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)

    if results:
        doc.add_heading("2. Hypothesis Test Results", level=1)
        for idx, r in enumerate(results, start=1):
            doc.add_heading(f"{idx}. {r.analysis_type}: {r.variables}", level=2)
            doc.add_paragraph(f"H0: {r.h0}")
            doc.add_paragraph(f"H1: {r.h1}")
            doc.add_paragraph(f"{r.statistic_name}: {r.statistic_value}")
            doc.add_paragraph(f"p-value: {format_p(r.p_value)}")
            doc.add_paragraph(f"Decision: {r.decision}")
            doc.add_paragraph(f"Summary: {r.summary}")
            if r.details:
                doc.add_paragraph(f"Details: {r.details}")

    output = io.BytesIO()
    doc.save(output)
    output.seek(0)
    return output.getvalue()


st.set_page_config(page_title="Statistical Hypothesis Analyzer", layout="wide")

st.title("📊 Statistical Hypothesis Analyzer")
st.caption("Excel/CSV yüklə → hipotez qur → t-test, ANOVA, correlation, chi-square və regression nəticələrini avtomatik al.")

with st.sidebar:
    st.header("Settings")
    alpha = st.number_input("Significance level / alpha", min_value=0.001, max_value=0.20, value=ALPHA_DEFAULT, step=0.01)
    st.info("Qayda: p-value < alpha olduqda H0 rejected / H1 accepted.")

uploaded_file = st.file_uploader("Excel və ya CSV faylını yüklə", type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Fayl oxunmadı: {e}")
        st.stop()

    numeric_cols, categorical_cols = detect_columns(df)

    st.subheader("Data Preview")
    st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")
    st.dataframe(df.head(100), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Numeric columns**")
        st.write(numeric_cols)
    with col_b:
        st.markdown("**Categorical columns**")
        st.write(categorical_cols)

    results: List[AnalysisResult] = []
    extra_tables = {}

    tabs = st.tabs([
        "Descriptive",
        "Auto Suggestions",
        "t-test",
        "ANOVA",
        "Correlation",
        "Chi-square",
        "Regression",
        "Report Export"
    ])

    with tabs[0]:
        st.subheader("Descriptive Statistics")
        selected_nums = st.multiselect("Numeric variables", numeric_cols, default=numeric_cols[:5])
        desc_df = descriptive_statistics(df, selected_nums)
        if not desc_df.empty:
            st.dataframe(desc_df, use_container_width=True)
        else:
            st.warning("Numeric dəyişən seçilməyib.")

    with tabs[1]:
        st.subheader("Automatic Hypothesis Suggestions")
        suggestions = auto_suggest_tests(df, numeric_cols, categorical_cols)
        if suggestions:
            for s in suggestions:
                st.write("• " + s)
        else:
            st.warning("Avtomatik test təklifi üçün uyğun dəyişən kombinasiyası tapılmadı.")

    with tabs[2]:
        st.subheader("t-test")

        ttest_type = st.radio("t-test type", ["Independent samples", "Paired samples"], horizontal=True)

        if ttest_type == "Independent samples":
            c1, c2 = st.columns(2)
            with c1:
                num = st.selectbox("Numeric dependent variable", numeric_cols, key="tt_num")
            with c2:
                group = st.selectbox("Group variable with 2 groups", categorical_cols, key="tt_group")

            if st.button("Run independent t-test"):
                r = independent_ttest(df, num, group, alpha)
                st.session_state.setdefault("results", []).append(r)
                st.success(r.decision)
                st.write(result_to_dict(r))

        else:
            c1, c2 = st.columns(2)
            with c1:
                before = st.selectbox("Before / variable 1", numeric_cols, key="paired_before")
            with c2:
                after = st.selectbox("After / variable 2", numeric_cols, key="paired_after")

            if st.button("Run paired t-test"):
                r = paired_ttest(df, before, after, alpha)
                st.session_state.setdefault("results", []).append(r)
                st.success(r.decision)
                st.write(result_to_dict(r))

    with tabs[3]:
        st.subheader("One-way ANOVA")
        c1, c2 = st.columns(2)
        with c1:
            num = st.selectbox("Numeric dependent variable", numeric_cols, key="anova_num")
        with c2:
            group = st.selectbox("Group variable with 3+ groups", categorical_cols, key="anova_group")

        if st.button("Run ANOVA"):
            r, tukey_df = one_way_anova(df, num, group, alpha)
            st.session_state.setdefault("results", []).append(r)
            st.success(r.decision)
            st.write(result_to_dict(r))

            if tukey_df is not None:
                st.markdown("**Post-hoc Tukey HSD**")
                st.dataframe(tukey_df, use_container_width=True)
                st.session_state.setdefault("extra_tables", {})[f"Tukey_{num}_{group}"] = tukey_df

    with tabs[4]:
        st.subheader("Correlation")
        c1, c2, c3 = st.columns(3)
        with c1:
            x = st.selectbox("Variable X", numeric_cols, key="corr_x")
        with c2:
            y = st.selectbox("Variable Y", numeric_cols, key="corr_y")
        with c3:
            method = st.selectbox("Method", ["pearson", "spearman"])

        if st.button("Run correlation"):
            r = correlation_analysis(df, x, y, alpha, method)
            st.session_state.setdefault("results", []).append(r)
            st.success(r.decision)
            st.write(result_to_dict(r))

    with tabs[5]:
        st.subheader("Chi-square Test")
        c1, c2 = st.columns(2)
        with c1:
            cat1 = st.selectbox("Categorical variable 1", categorical_cols, key="chi1")
        with c2:
            cat2 = st.selectbox("Categorical variable 2", categorical_cols, key="chi2")

        if st.button("Run chi-square"):
            r, table = chi_square_test(df, cat1, cat2, alpha)
            st.session_state.setdefault("results", []).append(r)
            st.success(r.decision)
            st.write(result_to_dict(r))

            if table is not None:
                st.markdown("**Crosstab**")
                st.dataframe(table, use_container_width=True)
                st.session_state.setdefault("extra_tables", {})[f"Crosstab_{cat1}_{cat2}"] = table

    with tabs[6]:
        st.subheader("Linear Regression")
        c1, c2 = st.columns(2)
        with c1:
            y = st.selectbox("Dependent variable / Y", numeric_cols, key="reg_y")
        with c2:
            x_cols = st.multiselect("Independent variables / X", [c for c in numeric_cols if c != y], key="reg_x")

        if st.button("Run regression"):
            if not x_cols:
                st.warning("Ən azı bir X dəyişəni seç.")
            else:
                r, coef_df, model_text = linear_regression(df, y, x_cols, alpha)
                st.session_state.setdefault("results", []).append(r)
                st.success(r.decision)
                st.write(result_to_dict(r))
                if not coef_df.empty:
                    st.markdown("**Coefficients**")
                    st.dataframe(coef_df, use_container_width=True)
                    st.session_state.setdefault("extra_tables", {})[f"Regression_{y}"] = coef_df
                with st.expander("Full statsmodels output"):
                    st.text(model_text)

    with tabs[7]:
        st.subheader("Report Export")
        saved_results = st.session_state.get("results", [])
        saved_extra = st.session_state.get("extra_tables", {})
        selected_nums_export = numeric_cols
        desc_export = descriptive_statistics(df, selected_nums_export)

        if saved_results:
            st.markdown("**Saved analysis results**")
            st.dataframe(pd.DataFrame([result_to_dict(r) for r in saved_results]), use_container_width=True)
        else:
            st.info("Hələ nəticə yoxdur. Əvvəl testlərdən birini run et.")

        excel_bytes = build_excel_report(saved_results, desc_export, saved_extra)
        word_bytes = build_word_report(saved_results, desc_export)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download Excel Report",
                data=excel_bytes,
                file_name="statistical_analysis_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with c2:
            st.download_button(
                "Download Word Report",
                data=word_bytes,
                file_name="statistical_analysis_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

        if st.button("Clear saved results"):
            st.session_state["results"] = []
            st.session_state["extra_tables"] = {}
            st.success("Saved results cleared.")

else:
    st.info("Başlamaq üçün Excel və ya CSV faylı yüklə.")
