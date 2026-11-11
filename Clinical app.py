"""저·무칼로리 감미 음료 임상시험 결과 탐색 + 로지스틱 회귀 (스트림릿 앱)

데이터: McGlynn 등(2022) JAMA Network Open 메타분석 보충자료를 엑셀로 옮긴 것
        (연구별 평균차이 MD와 표준오차 SE. 참가자 개인 데이터가 아님)
실행:   streamlit run clinical_app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="제로음료 임상시험 탐색", page_icon="🧪", layout="wide")

XLSX_SOURCE = "clinical_trials.xlsx"  # 저장소 안의 엑셀 파일 이름


def find_xlsx():
    """엑셀 파일을 찾습니다. 이 파일이 있는 폴더, 실행 폴더, data 폴더 순으로 보고,
    이름이 조금 달라도 '임상시험'이나 'clinical'이 들어간 .xlsx가 있으면 그것을 씁니다."""
    roots = [Path(__file__).resolve().parent, Path.cwd()]
    for root in roots:
        for name in (XLSX_SOURCE, f"data/{XLSX_SOURCE}"):
            if (root / name).exists():
                return str(root / name)
    for root in roots:
        found = sorted(root.glob("**/*임상시험*.xlsx")) + sorted(root.glob("**/*clinical*.xlsx"))
        if found:
            return str(found[0])
    return None

IND_KO = {
    "Body Weight": "체중",
    "BMI": "BMI",
    "Body Fat": "체지방률",
    "Waist Circumference": "허리둘레",
    "HbA1c": "당화혈색소",
    "Fasting Blood Glucose": "공복혈당",
    "2-Hour Post-Prandial Glucose": "식후 2시간 혈당",
}
TREAT_ALIAS = {"LNCSBs": "LNCSB", "SSBs": "SSB", "Water": "Water"}


# ---------------------------------------------------------------- 데이터
def parse_pooled_comparison(text):
    """논문 종합분석의 'LNCSBs for SSBs'(SSB를 LNCSB로 대체)를 'SSB→LNCSB' 형태로 바꿉니다."""
    a, b = [w.strip() for w in text.split(" for ")]
    return f"{TREAT_ALIAS.get(b, b)}→{TREAT_ALIAS.get(a, a)}"


@st.cache_data
def load_clinical(source):
    d = pd.read_excel(source, sheet_name="개별연구_원자료")
    d = d.rename(columns={"연구(저자·연도)": "연구_전체", "Treatment 1": "T1", "Treatment 2": "T2",
                          "Mean Difference": "MD", "Standard Error": "SE"})
    d["지표_ko"] = d["지표"].map(IND_KO).fillna(d["지표"])
    d["연구"] = d["연구_전체"].str.replace(r"\s*\(.*\)", "", regex=True)  # 같은 논문의 여러 행을 묶는 이름
    d["비교"] = d["T1"] + "→" + d["T2"]

    p = pd.read_excel(source, sheet_name="종합분석_결과")
    p = p.rename(columns={"종합 평균차이(MD)": "MD", "95% CI 하한": "하한", "95% CI 상한": "상한",
                          "근거 확실성(논문 GRADE 표기)": "GRADE"})
    p["지표_ko"] = p["지표"].map(IND_KO).fillna(p["지표"])
    p["대체"] = p["비교"].map(parse_pooled_comparison)
    return d, p


def prepare(d, convention):
    """MD의 부호 해석을 적용하고 분류용 열을 만듭니다.
    convention == 'T2-T1': MD = Treatment 2 - Treatment 1 (Treatment 1을 Treatment 2로 바꿨을 때의 변화)
    """
    x = d.copy()
    x["효과"] = x["MD"] if convention == "T2-T1" else -x["MD"]
    x["감소"] = (x["효과"] < 0).astype(int)          # 정답: 바꿨더니 지표가 낮아졌는가 (MD=0은 '감소 아님')
    x["z"] = x["효과"] / x["SE"]
    log_se = np.log(x["SE"])
    # 같은 지표 안에서 SE가 큰 편(=불확실한 연구)일수록 큰 값. 지표마다 단위가 달라서 지표 안에서 표준화합니다.
    x["불확실성z"] = log_se.groupby(x["지표"]).transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
    return x


# ---------------------------------------------------------------- 로지스틱 회귀
FEATURE_OPTIONS = {
    "비교(어떤 음료를 어떤 음료로 바꿨나)": ("cat", "비교"),
    "지표(체중, 혈당 등)": ("cat", "지표_ko"),
    "불확실성(연구의 SE)": ("num", "불확실성z"),
}


def build_model(cat_cols, num_cols, C):
    parts = []
    if cat_cols:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    if num_cols:
        parts.append(("num", StandardScaler(), num_cols))
    return make_pipeline(ColumnTransformer(parts), LogisticRegression(C=C, max_iter=1000))


def run_logreg(x, feature_labels, C):
    cat = [FEATURE_OPTIONS[f][1] for f in feature_labels if FEATURE_OPTIONS[f][0] == "cat"]
    num = [FEATURE_OPTIONS[f][1] for f in feature_labels if FEATURE_OPTIONS[f][0] == "num"]
    X, y, groups = x[cat + num], x["감소"], x["연구"]
    # 같은 연구의 행은 서로 닮아서, 연구 하나를 통째로 빼고 학습한 뒤 그 연구의 행을 맞히게 합니다.
    proba = cross_val_predict(build_model(cat, num, C), X, y, cv=LeaveOneGroupOut(),
                              groups=groups, method="predict_proba")[:, 1]
    model = build_model(cat, num, C).fit(X, y)
    return model, X, y, proba


# ---------------------------------------------------------------- 화면
st.title("🧪 저·무칼로리 감미 음료 임상시험 탐색")
st.caption("McGlynn 등(2022) JAMA Network Open 체계적 문헌고찰·메타분석의 보충자료 표를 엑셀로 옮긴 데이터입니다. "
           "연구별 요약값이며 참가자 개인 데이터가 아니고, 성인 대상 임상시험이라 청소년에게 그대로 적용할 수 없습니다.")

uploaded = st.sidebar.file_uploader("다른 엑셀로 바꿔 보기 (선택)", type="xlsx")
source = uploaded if uploaded is not None else find_xlsx()
if source is None:
    here = Path(__file__).resolve().parent
    st.error(f"'{XLSX_SOURCE}' 파일을 찾을 수 없습니다. GitHub 저장소에서 clinical_app.py와 같은 위치에 엑셀 파일을 올렸는지, "
             f"이름이 정확히 {XLSX_SOURCE}인지 확인하세요. 급하면 왼쪽 사이드바에서 엑셀을 직접 올려도 됩니다.")
    st.caption("앱이 지금 보고 있는 폴더의 파일: " + ", ".join(sorted(f.name for f in here.iterdir())))
    st.stop()
raw, pooled = load_clinical(source)

st.sidebar.header("설정")
conv_label = st.sidebar.radio(
    "MD 부호 해석",
    ["Treatment 2 − Treatment 1 (기본)", "Treatment 1 − Treatment 2"],
    help="원본 엑셀은 MD의 부호를 논문에서 확인해야 한다고 안내합니다. 기본값은 '논문의 종합 결과와 방향이 맞는' 해석입니다. 소개 탭의 점검 표를 보세요.",
)
convention = "T2-T1" if conv_label.startswith("Treatment 2") else "T1-T2"
x = prepare(raw, convention)

tab0, tab1, tab2, tab3, tab4 = st.tabs(["데이터 소개", "논문 종합 결과", "개별 연구 보기", "로지스틱 회귀", "데이터 표"])

# ===== 탭 0: 소개 =====
with tab0:
    st.markdown("""
**비교하는 세 가지 음료**
- **SSB**: 설탕이 든 음료
- **LNCSB**: 저·무칼로리 감미 음료(우리가 보는 '제로음료'에 해당)
- **Water**: 물

**데이터의 한 행** = 한 연구에서 한 지표(체중, BMI, 체지방률, 허리둘레, 당화혈색소, 공복혈당, 식후 2시간 혈당)를
두 음료(Treatment 1 → Treatment 2)로 바꿔 비교한 평균차이(MD)와 표준오차(SE)입니다.
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("비교(행) 수", len(raw))
    c2.metric("연구(논문) 수", raw["연구"].nunique())
    c3.metric("지표 수", raw["지표"].nunique())

    st.subheader("MD 부호 해석 점검")
    st.markdown("""
엑셀은 'MD의 부호는 논문의 비교 방향을 확인해 해석하라'고 안내합니다. 이 앱은 **MD = Treatment 2 − Treatment 1**
(Treatment 1을 Treatment 2로 바꿨을 때의 변화)로 해석했습니다. 논문의 종합 결과에서 'SSB를 LNCSB로 바꾸면 체중이 줄어든다(MD −1.06kg)'인데,
개별 연구에서도 SSB→LNCSB 비교 대부분이 이 해석에서 음수(감소)로 나오기 때문입니다. 아래 표로 직접 확인하세요.
""")
    chk = x.groupby("비교").agg(행수=("감소", "size"), 감소한_행=("감소", "sum")).reset_index()
    chk["감소 비율"] = (chk["감소한_행"] / chk["행수"]).round(2)
    st.dataframe(chk)
    st.caption("사이드바에서 부호 해석을 바꾸면 이 표와 아래 모든 분석이 함께 바뀝니다. 부호 해석이 틀리면 결론의 방향이 뒤집히니, 인용 전 논문 원문으로 꼭 확인하세요.")

# ===== 탭 1: 논문 종합 결과 =====
with tab1:
    st.markdown("논문이 여러 연구를 합쳐 계산한 **종합 평균차이**와 95% 신뢰구간입니다. 점이 효과, 가로선이 신뢰구간이고, **선이 0(점선)을 지나면 뚜렷한 차이가 있다고 보기 어렵습니다.**")
    ind = st.selectbox("지표", list(dict.fromkeys(pooled["지표_ko"])), key="pooled_ind")
    sub = pooled[pooled["지표_ko"] == ind]
    unit = sub["단위"].iloc[0]
    fig = go.Figure(go.Scatter(
        x=sub["MD"], y=sub["대체"], mode="markers", marker=dict(size=11),
        error_x=dict(type="data", symmetric=False, array=sub["상한"] - sub["MD"], arrayminus=sub["MD"] - sub["하한"]),
        hovertemplate="%{y}<br>MD %{x}<extra></extra>"))
    fig.add_vline(x=0, line_dash="dash")
    fig.update_layout(xaxis_title=f"종합 평균차이 ({unit})", yaxis_title="바꾼 방향 (앞 → 뒤)", margin=dict(t=10, b=10))
    st.plotly_chart(fig)
    st.dataframe(sub[["대체", "MD", "하한", "상한", "p값", "비교 수", "GRADE"]].round(3))
    st.caption("'SSB→LNCSB'는 SSB를 LNCSB로 바꿨을 때의 변화입니다. GRADE는 논문이 표기한 근거의 확실성이고, 대부분 'Low'라서 해석에 조심해야 합니다.")

# ===== 탭 2: 개별 연구 =====
with tab2:
    st.markdown("연구마다의 효과와 95% 신뢰구간(효과 ± 1.96×SE)입니다. 연구마다 결과가 들쭉날쭉하고 신뢰구간이 넓은 것을 볼 수 있습니다.")
    ind2 = st.selectbox("지표", list(dict.fromkeys(x["지표_ko"])), key="indiv_ind")
    s2 = x[x["지표_ko"] == ind2].copy()
    s2["연구(비교)"] = s2["연구_전체"] + " (" + s2["비교"] + ")"
    s2["신뢰구간 반폭"] = 1.96 * s2["SE"]
    s2 = s2.sort_values(["비교", "효과"])
    fig = px.scatter(s2, x="효과", y="연구(비교)", color="비교", error_x="신뢰구간 반폭",
                     category_orders={"연구(비교)": list(s2["연구(비교)"])})
    fig.add_vline(x=0, line_dash="dash")
    fig.update_traces(marker=dict(size=9))
    fig.update_layout(xaxis_title=f"효과 ({s2['단위'].iloc[0]}), 0보다 작으면 감소", yaxis_title="",
                      height=max(350, 28 * len(s2) + 120), margin=dict(t=10, b=10))
    st.plotly_chart(fig)
    st.caption(f"행 {len(s2)}개. 신뢰구간이 0을 지나는 연구가 많아 개별 연구만으로는 방향을 단정하기 어렵습니다.")

# ===== 탭 3: 로지스틱 회귀 =====
with tab3:
    st.markdown("**질문**: 음료를 바꿨을 때 지표가 **낮아졌는지(효과 < 0)** 를, 어떤 비교인지·어떤 지표인지로 설명할 수 있을까?")
    st.caption("정답은 '효과 < 0이면 1, 아니면 0'입니다(MD가 0인 행은 0). 체중, BMI, 체지방률, 허리둘레, 당화혈색소, 혈당은 모두 낮을수록 좋은 쪽의 지표입니다. "
               "신뢰구간을 보지 않고 방향만 보는 거친 정답이라, 효과가 거의 0인 연구도 감소/증가로 갈립니다.")

    c1, c2 = st.columns([2, 1])
    feats = c1.multiselect("입력(특징)", list(FEATURE_OPTIONS), default=list(FEATURE_OPTIONS)[:2])
    C = c2.slider("규제 세기 C (작을수록 강한 규제)", 0.1, 5.0, 1.0, 0.1)
    if not feats:
        st.info("입력을 하나 이상 고르세요.")
        st.stop()

    model, X, y, proba = run_logreg(x, feats, C)
    pred = (proba >= 0.5).astype(int)

    st.subheader("모델 성능")
    st.caption("연구(논문) 하나를 통째로 빼고 학습한 뒤 그 연구의 행을 맞히는 방식(Leave-One-Study-Out)으로 잰 값입니다. 같은 연구의 행이 학습과 평가에 섞여 점수가 부풀려지는 것을 막기 위해서입니다.")
    base = max(y.mean(), 1 - y.mean())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("기준선 정확도", f"{base:.2f}", help="전부 다수 클래스로 찍었을 때의 정확도")
    m2.metric("정확도", f"{accuracy_score(y, pred):.2f}")
    m3.metric("균형 정확도", f"{balanced_accuracy_score(y, pred):.2f}")
    m4.metric("ROC AUC", f"{roc_auc_score(y, proba):.2f}")
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    st.dataframe(pd.DataFrame([[tn, fp], [fn, tp]], index=["실제: 감소 아님", "실제: 감소"], columns=["예측: 감소 아님", "예측: 감소"]))
    st.caption(f"행 {len(x)}개, 연구 {x['연구'].nunique()}개 (감소 {int(y.sum())}개, 감소 아님 {int((1 - y).sum())}개). "
               "행이 적고 연구 사이 차이가 커서 점수는 크게 흔들릴 수 있습니다.")

    st.subheader("어떤 조건이 영향을 줄까 (오즈비)")
    names = [n.replace("cat__", "").replace("num__", "").replace("지표_ko_", "지표_")
             for n in model.named_steps["columntransformer"].get_feature_names_out()]
    coef = model.named_steps["logisticregression"].coef_[0]
    odds = pd.DataFrame({"조건": names, "오즈비": np.exp(coef)}).sort_values("오즈비")
    fig = px.bar(odds, x="오즈비", y="조건", orientation="h", text_auto=".2f")
    fig.add_vline(x=1, line_dash="dash")
    fig.update_layout(margin=dict(t=10, b=10), height=max(300, 30 * len(odds) + 100))
    st.plotly_chart(fig)
    st.caption("오즈비가 1보다 크면 그 조건에서 '감소'일 가능성이 높아지고(오른쪽), 1보다 작으면 낮아집니다(왼쪽). "
               "범주 변수는 모든 범주를 넣고 규제로 안정시킨 값이라, 같은 변수 안의 범주끼리 서로 비교해서 보세요. 인과관계가 아니라 이 데이터 안의 경향입니다.")

    st.subheader("예측해 보기")
    cc = st.columns(3)
    row = {}
    if "비교(어떤 음료를 어떤 음료로 바꿨나)" in feats:
        row["비교"] = cc[0].selectbox("비교", sorted(x["비교"].unique()))
    if "지표(체중, 혈당 등)" in feats:
        row["지표_ko"] = cc[1].selectbox("지표", sorted(x["지표_ko"].unique()))
    if "불확실성(연구의 SE)" in feats:
        row["불확실성z"] = cc[2].slider("불확실성 (지표 안에서 표준화, 클수록 불확실)", -2.0, 2.0, 0.0, 0.1)
    p_new = float(model.predict_proba(pd.DataFrame([row])[list(X.columns)])[0, 1])
    st.metric("지표가 낮아질 확률(모델)", f"{p_new * 100:.1f}%")
    st.progress(p_new)
    st.caption("전체 데이터로 학습한 모델의 값입니다. 한 번의 임상시험 결과를 예측하거나 효과를 증명하는 값이 아닙니다.")

    wrong = x.assign(예측확률=proba.round(3))[pred != y.to_numpy()]
    with st.expander(f"잘못 맞힌 행 {len(wrong)}개 보기"):
        st.dataframe(wrong[["연구_전체", "지표_ko", "비교", "효과", "SE", "감소", "예측확률"]])

    st.warning("이 분석은 '데이터에서 보이는 경향'을 정리한 것입니다. 논문이 여러 연구를 합쳐 계산한 종합 결과(논문 종합 결과 탭)가 더 믿을 만한 근거이고, "
               "이 모델이 그 결론을 바꾸지는 못합니다. 건강 판단이나 처방의 근거로 쓰지 마세요.")

# ===== 탭 4: 데이터 표 =====
with tab4:
    show = x[["표", "지표_ko", "단위", "연구_전체", "T1", "T2", "MD", "SE", "효과", "감소", "원문 PDF 페이지"]]
    st.dataframe(show)
    st.download_button("표를 CSV로 받기", show.to_csv(index=False).encode("utf-8-sig"),
                       file_name="clinical_rows.csv", mime="text/csv")
    st.caption("원본: McGlynn ND 등. JAMA Netw Open. 2022;5(3):e222092. doi:10.1001/jamanetworkopen.2022.2092 (엑셀의 '출처와_주의사항' 시트 참고)")
