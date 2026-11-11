from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

st.set_page_config(page_title="제로음료 영양성분 탐색", page_icon="🥤", layout="wide")

# 데이터 위치: 저장소 안의 파일 경로입니다.
# GitHub raw 링크를 쓰려면 아래 줄을 링크로 바꾸세요.
# 예) "https://raw.githubusercontent.com/내계정/저장소이름/main/data/zero_drinks.csv"
DATA_SOURCE = "zero_drinks.csv"


def find_data():
    """CSV 파일을 찾습니다.
    1) 이 파일이 있는 폴더와 실행 폴더, 그 아래 data 폴더의 zero_drinks.csv
    2) 그래도 없으면 어느 폴더든 '식품명, 음료종류, 에너지_kcal' 열이 있는 .csv (요약 파일 등은 열이 달라 제외)"""
    roots = [Path(__file__).resolve().parent, Path.cwd()]
    for root in roots:
        for name in (DATA_SOURCE, f"data/{DATA_SOURCE}"):
            if (root / name).exists():
                return str(root / name)
    for root in roots:
        for f in sorted(root.glob("**/*.csv")):
            try:
                header = f.open(encoding="utf-8-sig").readline()
            except (OSError, UnicodeDecodeError):
                continue
            if all(k in header for k in ("식품명", "음료종류", "에너지_kcal")):
                return str(f)
    return None

# 지방_g, 탄수화물_g는 결측치가 많아(45개 중 35개) 그래프에서 뺐습니다. 데이터 표에는 그대로 남아 있습니다.
NUTRIENTS = ["에너지_kcal", "단백질_g", "당류_g", "나트륨_mg", "포화지방_g"]
NUMERIC = NUTRIENTS + ["식품중량"]


@st.cache_data
def load_data(source):
    df = pd.read_csv(source, encoding="utf-8-sig")
    df["데이터생성일"] = pd.to_datetime(df["데이터생성일"])
    df["생성월"] = df["데이터생성일"].dt.strftime("%Y-%m")
    return df


# ---------------------------------------------------------------- 데이터 불러오기
st.title("🥤 제로음료 영양성분 탐색")
st.caption("출처: 식품의약품안전처 식품영양성분 DB (화면을 보고 직접 옮겨 적은 사본, 자세한 내용은 data/출처.md)")

uploaded = st.sidebar.file_uploader("다른 CSV로 바꿔 보기 (선택)", type="csv")
source = uploaded if uploaded is not None else find_data()
if source is None:
    here = Path(__file__).resolve().parent
    st.error(f"'{DATA_SOURCE}' 파일을 찾을 수 없습니다. GitHub 저장소에서 app.py와 같은 위치(또는 data 폴더)에 CSV를 올렸는지, "
             f"이름이 정확히 {DATA_SOURCE}인지 확인하세요. 급하면 왼쪽 사이드바에서 CSV를 직접 올려도 됩니다.")
    st.caption("앱이 지금 보고 있는 폴더의 파일: " + ", ".join(sorted(f.name for f in here.iterdir())))
    st.stop()
df_all = load_data(source)

type_order = list(dict.fromkeys(df_all["음료종류"]))  # 파일에 나온 순서대로

# ---------------------------------------------------------------- 사이드바 필터
st.sidebar.header("필터")
unit_choice = st.sidebar.radio(
    "기준단위",
    ["100ml 기준만", "100g 기준만", "전체"],
    help="영양성분 값이 100ml 기준인 제품과 100g 기준인 제품이 섞여 있어서, 기본은 100ml 기준만 봅니다.",
)
types = st.sidebar.multiselect("음료종류", type_order, default=type_order)
bins = st.sidebar.slider("히스토그램 구간 수", 5, 20, 10)

df = df_all[df_all["음료종류"].isin(types)]
if unit_choice == "100ml 기준만":
    df = df[df["기준단위"] == "100ml"]
elif unit_choice == "100g 기준만":
    df = df[df["기준단위"] == "100g"]

st.sidebar.caption(f"전체 {len(df_all)}개 중 {len(df)}개를 보여주고 있습니다.")
if df.empty:
    st.warning("조건에 맞는 제품이 없습니다. 왼쪽 필터를 바꿔 보세요.")
    st.stop()



# 하루 섭취 계산기와 음료 바꾸기에 쓰는 영양성분과 하루 기준치(편집 가능한 기본값)
# 식약처 '1일 영양성분 기준치'(열량 2,000kcal 기준): 나트륨 2,000mg, 포화지방 15g, 당류 100g
INTAKE_COLS = {
    "에너지_kcal": ("에너지", "kcal", 2000),
    "나트륨_mg": ("나트륨", "mg", 2000),
    "포화지방_g": ("포화지방", "g", 15),
    "당류_g": ("당류", "g", 100),
}


def calc_intake(data, cups):
    """마신 음료의 영양성분 합계 표를 만듭니다.
    cups: {식품명: 잔 수}
    한 잔 값 = 100ml(또는 100g)당 값 x 식품중량 / 100. 값이 비어 있으면 결측(NaN)으로 둡니다."""
    rows = []
    for name, n in cups.items():
        r = data[data["식품명"] == name].iloc[0]
        row = {"음료": name, "잔 수": n, "1잔 용량": f"{r['식품중량']:g}{r['중량단위']}"}
        for col in INTAKE_COLS:
            row[col] = r[col] * r["식품중량"] / 100 * n
        rows.append(row)
    return pd.DataFrame(rows)



# 분류 모델: 영양성분으로 '아이스티인지 아닌지'를 맞히는 모델 3개 (train_models.py와 같은 설정)
MODEL_FEATURES = ["에너지_kcal", "단백질_g", "나트륨_mg", "포화지방_g"]
MODEL_TARGET = "아이스티"
MODEL_NAMES = ["로지스틱 회귀", "결정나무", "랜덤포레스트"]


def make_model(name):
    if name == "로지스틱 회귀":
        # 값이 한쪽으로 치우쳐 있어 log1p로 누르고, 표준화한 뒤 규제 로지스틱 회귀를 씁니다.
        return make_pipeline(
            FunctionTransformer(np.log1p),
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000),
        )
    if name == "결정나무":
        # 깊이와 잎의 최소 제품 수를 제한해서 적은 데이터를 외우지 못하게 합니다.
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, class_weight="balanced", random_state=0)
    return RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=2,
                                  class_weight="balanced", random_state=0)


@st.cache_resource
def train_model(data, name):
    """100ml 기준 제품만 써서 학습하고, 제품 하나씩 빼고 맞힌 예측(Leave-One-Out)도 함께 돌려줍니다."""
    d = data[data["기준단위"] == "100ml"].dropna(subset=MODEL_FEATURES).reset_index(drop=True)
    X = d[MODEL_FEATURES]
    y = (d["음료종류"] == MODEL_TARGET).astype(int)
    loo = cross_val_predict(make_model(name), X, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    model = make_model(name).fit(X, y)
    return model, d, y, loo



# ------------------------------------------------ 음료 바꾸기: 교체 쌍 데이터와 로지스틱 회귀
SWAP_METRICS = {"에너지": "에너지_kcal", "나트륨": "나트륨_mg", "포화지방": "포화지방_g"}  # '다른 성분의 변화' 입력에 쓰는 성분
SWAP_TARGETS = {**SWAP_METRICS, "당류": "당류_g"}  # 낮아지는지 볼 수 있는 성분 (당류는 빈 값이 많음)
FEAT_TYPE, FEAT_SIZE, FEAT_OTHER = "음료종류(지금 → 바꿀)", "용량 변화", "다른 성분의 변화"


def serving_table(data, target):
    """한 잔 값 = 100ml(또는 100g)당 값 x 식품중량 / 100.
    '다른 성분' 세 개와 정답을 만들 성분(target)의 값이 모두 있는 음료만 남깁니다."""
    s = data.copy()
    for k, col in SWAP_TARGETS.items():
        s["한잔_" + k] = s[col] * s["식품중량"] / 100
    need = ["한잔_" + k for k in SWAP_METRICS] + ["한잔_" + target]
    return s.dropna(subset=need).reset_index(drop=True)


def make_pairs(data, target):
    """모든 (지금 음료 A -> 바꿀 음료 B) 쌍을 만들고, 정답 y = 'B로 바꾸면 한 잔 target이 낮아지는가'."""
    s = serving_table(data, target)
    n = len(s)
    ia, ib = np.where(~np.eye(n, dtype=bool))
    A = s.iloc[ia].reset_index(drop=True)
    B = s.iloc[ib].reset_index(drop=True)
    P = pd.DataFrame({
        "a": ia, "b": ib,
        "A종류": A["음료종류"], "B종류": B["음료종류"],
        "로그용량비": np.log(B["식품중량"] / A["식품중량"]),
    })
    tcol = "한잔_" + target
    P["y"] = (B[tcol] < A[tcol]).astype(int)
    for k in SWAP_METRICS:
        if k != target:
            P["Δ" + k] = np.log1p(B["한잔_" + k]) - np.log1p(A["한잔_" + k])
    same = (B[tcol] == A[tcol]).to_numpy()
    return P[~same].reset_index(drop=True), s  # 값이 같은 쌍은 '낮아졌는가'를 물을 수 없어 뺍니다.


def swap_columns(feats, target):
    cat = ["A종류", "B종류"] if FEAT_TYPE in feats else []
    num = (["로그용량비"] if FEAT_SIZE in feats else []) + (["Δ" + k for k in SWAP_METRICS if k != target] if FEAT_OTHER in feats else [])
    return cat, num


def build_swap_model(cat, num, C):
    parts = []
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    if num:
        parts.append(("num", StandardScaler(), num))
    return make_pipeline(ColumnTransformer(parts), LogisticRegression(C=C, max_iter=2000))


@st.cache_resource(show_spinner="모델을 평가하는 중...")
def run_swap_model(data, target, feats, C):
    """제품 하나(A)를 통째로 빼고 학습한 뒤, 그 제품이 'A'인 쌍을 맞히게 합니다(Leave-One-Product-Out).
    빠진 제품이 들어간 쌍은 학습에서 모두 제외합니다."""
    P, s = make_pairs(data, target)
    cat, num = swap_columns(feats, target)
    X = P[cat + num]
    proba = pd.Series(np.nan, index=P.index)
    for p in sorted(P["a"].unique()):
        te = P["a"] == p
        tr = (P["a"] != p) & (P["b"] != p)
        m = build_swap_model(cat, num, C).fit(X[tr], P.loc[tr, "y"])
        proba[te] = m.predict_proba(X[te])[:, 1]
    model = build_swap_model(cat, num, C).fit(X, P["y"])
    return model, P, s, proba, cat, num


# ---------------------------------------------------------------- 그래프 도우미
def show_hist(col, why):
    """숫자 열: 값이 어디에 몰려 있는지 보는 히스토그램"""
    d = df.dropna(subset=[col])
    st.subheader(col)
    st.caption(f"히스토그램 · {why}")
    if d.empty:
        st.info("표시할 값이 없습니다.")
        return
    fig = px.histogram(d, x=col, nbins=bins)
    fig.update_layout(yaxis_title="제품 수", bargap=0.05, margin=dict(t=10, b=10))
    st.plotly_chart(fig)
    st.caption(f"값이 있는 제품 {len(d)}개 · 결측 {len(df) - len(d)}개")


# ---------------------------------------------------------------- 하루 기준치 (계산기와 음료 바꾸기 탭이 함께 사용)
with st.sidebar.expander("하루 기준치 (계산기용)"):
    st.caption("식약처 영양표시의 1일 영양성분 기준치(2,000kcal 기준)입니다. 개인마다 필요한 양은 다르니 필요하면 바꾸세요.")
    refs = {}
    for col, (label, unit, default) in INTAKE_COLS.items():
        refs[col] = st.number_input(f"{label} ({unit})", min_value=1.0, value=float(default), step=1.0, key=f"ref_{col}")


# ---------------------------------------------------------------- 탭
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["열별 분포", "음료종류별 비교", "관계 보기(산점도)", "데이터 표", "하루 섭취 계산기", "분류 모델", "음료 바꾸기"])

# ===== 탭 1: 열마다 알맞은 그래프 =====
with tab1:
    st.markdown("열의 성격(범주, 숫자, 날짜)에 맞는 그래프를 하나씩 보여줍니다. 번호, 식품코드, 식품명은 제품마다 달라서 그래프 없이 '데이터 표' 탭에서 확인합니다.")

    st.subheader("결측치 현황")
    st.caption("막대그래프 · 열마다 비어 있는 칸이 몇 개인지. 비어 있는 것은 0이 아니라 원본에 값이 없다는 뜻입니다.")
    miss = df[NUMERIC].isna().sum().reset_index()
    miss.columns = ["열", "결측 개수"]
    fig = px.bar(miss, x="열", y="결측 개수", text="결측 개수")
    fig.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig)

    left, right = st.columns(2)

    with left:
        st.subheader("음료종류")
        st.caption("가로 막대그래프 · 종류가 많은 범주형이라 원그래프보다 막대가 비교하기 쉽습니다.")
        cnt = df["음료종류"].value_counts().rename_axis("음료종류").reset_index(name="제품 수")
        fig = px.bar(cnt, x="제품 수", y="음료종류", orientation="h", text="제품 수")
        fig.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(t=10, b=10))
        st.plotly_chart(fig)

    with right:
        st.subheader("기준단위")
        st.caption("도넛(원) 그래프 · 범주가 2개뿐이라 전체에서 차지하는 비율을 보기 좋습니다. (필터와 상관없이 전체 데이터)")
        cnt = df_all["기준단위"].value_counts().rename_axis("기준단위").reset_index(name="제품 수")
        fig = px.pie(cnt, names="기준단위", values="제품 수", hole=0.5)
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig)
        st.caption("중량단위(ml, g)는 기준단위와 똑같이 나뉘어서 따로 그리지 않았습니다.")

    left, right = st.columns(2)
    with left:
        show_hist("식품중량", "제품 1개 용량이 어느 크기에 몰려 있는지 (기준단위가 섞이면 ml와 g이 함께 나옵니다)")
    with right:
        st.subheader("데이터생성일")
        st.caption("막대그래프 · 원본 데이터가 언제 만들어졌는지 월별 제품 수")
        cnt = df["생성월"].value_counts().rename_axis("생성월").reset_index(name="제품 수").sort_values("생성월")
        fig = px.bar(cnt, x="생성월", y="제품 수", text="제품 수")
        fig.update_xaxes(type="category")
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig)

    st.divider()
    st.markdown("#### 영양성분 (숫자 열)")
    pairs = [
        ("에너지_kcal", "대부분 낮은 값에 몰리는지, 튀는 제품이 있는지"),
        ("단백질_g", "제품별 단백질 분포"),
        ("당류_g", "제로 음료라도 당류가 0이 아닌 제품이 있는지"),
        ("나트륨_mg", "제품별 나트륨 분포"),
        ("포화지방_g", "제품별 포화지방 분포"),
    ]
    for i in range(0, len(pairs), 2):
        cols = st.columns(2)
        for c, (col, why) in zip(cols, pairs[i:i + 2]):
            with c:
                show_hist(col, why)

# ===== 탭 2: 음료종류별 비교 =====
with tab2:
    st.markdown("음료종류에 따라 영양성분이 어떻게 다른지 비교합니다.")
    if unit_choice == "전체":
        st.warning("100ml 기준과 100g 기준이 섞여 있어 커피 값이 왜곡될 수 있습니다. 왼쪽에서 기준단위를 하나로 고르는 것을 권합니다.")

    metric = st.selectbox("비교할 영양성분", NUTRIENTS)
    d = df.dropna(subset=[metric]).copy()
    if d.empty:
        st.info("이 영양성분은 지금 조건에서 값이 있는 제품이 없습니다.")
    else:
        counts = d["음료종류"].value_counts()
        d["종류(n)"] = d["음료종류"].map(lambda t: f"{t} (n={counts[t]})")
        order = [f"{t} (n={counts[t]})" for t in type_order if t in counts.index]

        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"음료종류별 {metric} 평균")
            st.caption("막대그래프 · 종류별 평균 비교 (n은 값이 있는 제품 수)")
            avg = d.groupby("종류(n)")[metric].mean().reindex(order).reset_index()
            fig = px.bar(avg, x="종류(n)", y=metric, text_auto=".2f")
            fig.update_layout(xaxis_title="", yaxis_title=f"평균 {metric}", margin=dict(t=10, b=10))
            st.plotly_chart(fig)
        with c2:
            st.subheader(f"음료종류별 {metric} 분포")
            st.caption("상자그림 · 제품 하나하나의 점과 퍼짐 (점에 마우스를 올리면 제품명)")
            fig = px.box(d, x="종류(n)", y=metric, points="all", hover_name="식품명",
                         category_orders={"종류(n)": order})
            fig.update_layout(xaxis_title="", margin=dict(t=10, b=10))
            st.plotly_chart(fig)

        no_value = [t for t in types if t not in counts.index]
        if no_value:
            st.caption("값이 하나도 없어 그래프에서 빠진 종류: " + ", ".join(no_value))
        st.caption("제품이 1~2개뿐인 종류는 평균이 그 제품의 값과 거의 같습니다. 종류 사이의 차이를 말할 때는 n을 함께 보세요.")

# ===== 탭 3: 산점도 =====
with tab3:
    st.markdown("두 숫자 열의 관계를 점으로 봅니다. 점 하나가 제품 하나입니다.")
    c1, c2 = st.columns(2)
    x = c1.selectbox("가로축", NUMERIC, index=NUMERIC.index("당류_g"))
    y = c2.selectbox("세로축", NUMERIC, index=0)
    if x == y:
        st.info("가로축과 세로축을 서로 다른 열로 골라 주세요.")
    else:
        d = df.dropna(subset=[x, y])
        if d.empty:
            st.info("두 열 모두 값이 있는 제품이 없습니다.")
        else:
            fig = px.scatter(d, x=x, y=y, color="음료종류", hover_name="식품명",
                             category_orders={"음료종류": type_order})
            fig.update_traces(marker=dict(size=10))
            st.plotly_chart(fig)
            st.caption(f"두 열 모두 값이 있는 제품 {len(d)}개 (전체 {len(df)}개 중). 점이 몰려 보이는 것은 같은 값의 제품이 겹쳐 있기 때문일 수 있습니다.")

# ===== 탭 4: 데이터 표 =====
with tab4:
    shown = df.drop(columns=["생성월"])
    st.dataframe(shown)
    st.download_button(
        "지금 보이는 표를 CSV로 받기",
        shown.to_csv(index=False).encode("utf-8-sig"),
        file_name="zero_drinks_filtered.csv",
        mime="text/csv",
    )


# ===== 탭 5: 하루 섭취 계산기 =====
with tab5:
    st.markdown("오늘 마신 음료와 잔 수를 고르면, 에너지·나트륨·포화지방·당류가 **하루 기준치의 몇 %**인지 계산합니다. "
                "한 잔 값은 '100ml(또는 100g)당 값 × 제품 용량 ÷ 100'으로 구하며, 왼쪽 필터와 상관없이 전체 제품에서 고릅니다.")

    st.caption("하루 기준치는 왼쪽 사이드바의 '하루 기준치'에서 바꿀 수 있습니다.")

    picked = st.multiselect("오늘 마신 음료", list(df_all["식품명"]))
    if not picked:
        st.info("음료를 하나 이상 고르면 계산 결과가 나옵니다.")
    else:
        cups = {}
        for name in picked:
            cups[name] = st.number_input(f"{name} · 잔 수", min_value=1, max_value=10, value=1, key=f"cups_{name}")

        res = calc_intake(df_all, cups)
        totals = {col: res[col].sum() for col in INTAKE_COLS}  # 결측은 합계에서 빠짐

        st.subheader("하루 기준치 대비")
        for c, (col, (label, unit, _)) in zip(st.columns(4), INTAKE_COLS.items()):
            pct = totals[col] / refs[col] * 100
            c.metric(f"{label} 합계", f"{totals[col]:,.1f} {unit}", f"기준치의 {pct:.1f}%", delta_color="off")

        for col, (label, unit, _) in INTAKE_COLS.items():
            miss = res.loc[res[col].isna(), "음료"].tolist()
            if miss:
                st.warning(f"{label} 값이 비어 있는 음료({', '.join(miss)})는 합계에서 빠졌습니다. 실제 {label} 합계는 이보다 클 수 있어요.")

        long = []
        for col, (label, unit, _) in INTAKE_COLS.items():
            for _, r in res.dropna(subset=[col]).iterrows():
                long.append({"영양성분": label, "음료": r["음료"], "기준치 대비(%)": r[col] / refs[col] * 100})
        if long:
            fig = px.bar(pd.DataFrame(long), x="영양성분", y="기준치 대비(%)", color="음료")
            fig.add_hline(y=100, line_dash="dash", annotation_text="하루 기준치 100%")
            fig.update_layout(barmode="stack", margin=dict(t=10, b=10))
            st.plotly_chart(fig)

        st.subheader("음료별 계산 내용")
        st.dataframe(res.rename(columns={c: f"{v[0]} ({v[1]})" for c, v in INTAKE_COLS.items()}))

    st.caption("참고용 계산입니다. 개인의 건강 상태에 따른 조언이 아니며, 값은 식약처 DB 화면을 보고 직접 옮겨 적은 것이라 원본과 다를 수 있습니다. "
               "실제로 마신 용량이나 제조 방법에 따라 차이가 날 수 있습니다.")


# ===== 탭 6: 분류 모델 =====
with tab6:
    st.markdown(f"영양성분(에너지, 단백질, 나트륨, 포화지방)만 보고 **'{MODEL_TARGET}'인지 아닌지**를 맞히는 모델 3개를 비교합니다. "
                "100ml 기준 제품만 쓰고, 결측이 많은 지방·탄수화물·당류는 입력에서 뺐습니다.")

    # 세 모델의 성능을 한 표로 비교
    rows = []
    for nm in MODEL_NAMES:
        _, md0, my0, loo0 = train_model(df_all, nm)
        p0 = (loo0 >= 0.5).astype(int)
        rows.append({"모델": nm, "정확도": accuracy_score(my0, p0), "균형 정확도": balanced_accuracy_score(my0, p0),
                     "F1": f1_score(my0, p0), "ROC AUC": roc_auc_score(my0, loo0)})
    base = max(my0.mean(), 1 - my0.mean())
    st.subheader("모델 비교")
    st.caption("제품 하나씩 빼고 학습한 뒤 그 제품을 맞히게 하는 방식(Leave-One-Out)으로 잰 값입니다. 학습에 쓴 제품으로 잰 점수가 아닙니다.")
    st.dataframe(pd.DataFrame(rows).round(3))
    st.caption(f"기준선 정확도(전부 '아이스티 아님'으로 찍기) {base:.2f}. 사용한 제품 {len(md0)}개 (정답 1: {int(my0.sum())}개, 정답 0: {int((1 - my0).sum())}개). "
               "표본이 작아서 모델 사이의 차이는 우연일 수 있고, 여러 모델 중 잘 나온 것을 고르면 실제보다 낙관적으로 보일 수 있습니다.")

    name_sel = st.selectbox("자세히 볼 모델", MODEL_NAMES)
    model, md, my, loo = train_model(df_all, name_sel)
    pred = (loo >= 0.5).astype(int)

    st.subheader("예측해 보기")
    mode = st.radio("예측할 제품", ["데이터에 있는 제품 고르기", "직접 입력"], horizontal=True)
    if mode == "데이터에 있는 제품 고르기":
        name = st.selectbox("제품", list(md["식품명"]))
        i = int(md.index[md["식품명"] == name][0])
        prob = float(loo[i])
        actual = "아이스티" if my[i] == 1 else "아이스티 아님"
        st.dataframe(md.loc[[i], ["식품명", "음료종류"] + MODEL_FEATURES])
        st.caption("이 예측은 그 제품을 뺀 채로 학습한 모델의 결과입니다.")
    else:
        vals = {}
        steps = {"에너지_kcal": 1.0, "단백질_g": 0.01, "나트륨_mg": 1.0, "포화지방_g": 0.01}
        for c, col in zip(st.columns(4), MODEL_FEATURES):
            vals[col] = c.number_input(col, min_value=0.0, value=float(md[col].median()), step=steps[col])
        st.caption("100ml당 값을 넣으세요. 기본값은 이 데이터의 중앙값입니다.")
        prob = float(model.predict_proba(pd.DataFrame([vals]))[0, 1])
        actual = None

    st.metric(f"'{MODEL_TARGET}'일 확률 ({name_sel})", f"{prob * 100:.1f}%")
    st.progress(prob)
    verdict = f"모델은 {MODEL_TARGET}로 판단합니다." if prob >= 0.5 else f"모델은 {MODEL_TARGET}가 아니라고 판단합니다."
    st.write(verdict + (f" (실제: {actual})" if actual else ""))

    st.subheader(f"{name_sel}이(가) 본 것")
    if name_sel == "로지스틱 회귀":
        st.caption(f"막대가 오른쪽이면 '{MODEL_TARGET}일 가능성'을 높이고, 왼쪽이면 낮춥니다. 로그 변환 후 표준화한 입력 기준 계수입니다.")
        coef = model.named_steps["logisticregression"].coef_[0]
        cdf = pd.DataFrame({"성분": MODEL_FEATURES, "계수": coef}).sort_values("계수")
        fig = px.bar(cdf, x="계수", y="성분", orientation="h", text_auto=".2f")
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig)
    elif name_sel == "결정나무":
        st.caption("위에서부터 조건을 따라 내려가면 판정이 나옵니다. (class 1 = 아이스티)")
        st.code(export_text(model, feature_names=MODEL_FEATURES))
    else:
        st.caption("값이 클수록 여러 나무에서 분류에 많이 쓰인 성분입니다. 영향의 방향(높을수록 좋은지 나쁜지)은 알려주지 않습니다.")
        idf = pd.DataFrame({"성분": MODEL_FEATURES, "중요도": model.feature_importances_}).sort_values("중요도")
        fig = px.bar(idf, x="중요도", y="성분", orientation="h", text_auto=".2f")
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig)

    wrong = md.assign(예측확률=loo.round(3), 정답=my)[pred != my.to_numpy()]
    with st.expander(f"{name_sel}이(가) 잘못 맞힌 제품 {len(wrong)}개 보기"):
        st.dataframe(wrong[["식품명", "음료종류", "정답", "예측확률"]])

    st.caption("이 모델들은 데이터 안의 패턴(아이스티는 나트륨과 에너지가 낮은 경향)을 보여줄 뿐, 제로음료의 건강 효과를 증명하지 않습니다.")


# ===== 탭 7: 음료 바꾸기 =====
with tab7:
    st.markdown("지금 마시는 제로음료를 **다른 제로음료로 바꾸면** 한 잔(그리고 하루) 영양성분이 어떻게 달라지는지 보고, "
                "'바꾸면 낮아질까?'를 로지스틱 회귀로 살펴봅니다. 왼쪽 필터와 상관없이 전체 음료에서 고릅니다.")
    names_all = list(df_all["식품명"])
    c1, c2, c3 = st.columns([3, 3, 1])
    drink_a = c1.selectbox("지금 마시는 음료 (A)", names_all, index=0, key="swap_a")
    drink_b = c2.selectbox("바꿀 음료 (B)", names_all, index=min(len(names_all) - 1, 7), key="swap_b")
    cups_n = c3.number_input("하루 잔 수", 1, 10, 1, key="swap_cups")

    st.subheader("바꾸면 달라지는 양")
    if drink_a == drink_b:
        st.info("A와 B를 서로 다른 음료로 골라 주세요.")
    else:
        res = calc_intake(df_all, {drink_a: cups_n, drink_b: cups_n})
        rows = []
        for col, (label, unit, _d) in INTAKE_COLS.items():
            ref = refs[col]
            a_v, b_v = res.loc[0, col], res.loc[1, col]
            if pd.isna(a_v) or pd.isna(b_v):
                rows.append({"성분": f"{label} ({unit})", "A": a_v, "B": b_v, "변화(B−A)": np.nan,
                             "기준치 대비 변화(%p)": np.nan, "방향": "값이 없어 비교 불가"})
                continue
            diff = b_v - a_v
            pp = diff / ref * 100
            direction = "낮아짐 ▼" if diff < 0 else ("높아짐 ▲" if diff > 0 else "같음")
            if diff != 0 and abs(pp) < 1:
                direction += " (차이 작음)"
            rows.append({"성분": f"{label} ({unit})", "A": a_v, "B": b_v, "변화(B−A)": diff,
                         "기준치 대비 변화(%p)": pp, "방향": direction})
        cmp_df = pd.DataFrame(rows)
        st.dataframe(cmp_df.round(2))
        st.caption(f"하루 {cups_n}잔 기준입니다. '차이 작음'은 하루 기준치의 1%p 미만 차이라는 뜻이고, 원본 값이 반올림돼 있어 이 정도 차이는 의미를 두기 어렵습니다. "
                   "값이 낮아졌다고 무조건 좋은 것은 아니며, 이 계산은 개인 건강 조언이 아닙니다.")

        chg = cmp_df.dropna(subset=["기준치 대비 변화(%p)"])
        if not chg.empty:
            parts = ", ".join(f"{r['성분'].split(' (')[0]} {r['기준치 대비 변화(%p)']:+.1f}%p" for _, r in chg.iterrows())
            st.write(f"하루 {cups_n}잔을 A에서 B로 바꾸면 **하루 기준치 대비** {parts} 달라집니다.")
            fig = px.bar(chg, x="기준치 대비 변화(%p)", y="성분", orientation="h", text_auto=".1f")
            fig.add_vline(x=0, line_dash="dash")
            fig.update_layout(margin=dict(t=10, b=10), height=max(220, 60 * len(chg) + 80))
            st.plotly_chart(fig)
            st.caption("0보다 왼쪽이면 줄고 오른쪽이면 늘어납니다. 기준치는 식약처 1일 영양성분 기준치(2,000kcal 기준)이며 사이드바에서 바꿀 수 있습니다. "
                       "기준치는 영양 정보를 비교하기 위한 값이지 어떤 음료가 건강에 좋은지 판정하는 기준이 아닙니다.")

        long = []
        for col, (label, unit, _d) in INTAKE_COLS.items():
            ref = refs[col]
            for who, row_i in (("A: 지금", 0), ("B: 바꾼 뒤", 1)):
                v = res.loc[row_i, col]
                if pd.notna(v):
                    long.append({"성분": label, "음료": who, "하루 기준치 대비(%)": v / ref * 100})
        if long:
            fig = px.bar(pd.DataFrame(long), x="성분", y="하루 기준치 대비(%)", color="음료", barmode="group")
            fig.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig)

    st.divider()
    st.subheader("바꾸면 낮아질까? (로지스틱 회귀)")
    st.markdown("모든 (A → B) 쌍을 만들어, **B로 바꾸면 한 잔의 선택한 성분이 낮아지는가**(1) 아닌가(0)를 맞힙니다. "
                "값이 똑같은 쌍은 뺍니다. 정답이 음료의 영양성분 값에서 바로 나오므로, 입력에는 정답을 만든 성분 자체를 넣지 않습니다.")
    cc1, cc2 = st.columns(2)
    target = cc1.selectbox("낮아지는지 볼 성분", list(SWAP_TARGETS), index=1)
    sw_feats = cc2.multiselect("입력(특징)", [FEAT_TYPE, FEAT_SIZE, FEAT_OTHER], default=[FEAT_TYPE, FEAT_SIZE, FEAT_OTHER],
                               help=f"'다른 성분의 변화'는 {target}이(가) 아닌 나머지 성분이 바뀐 정도입니다.")
    if not sw_feats:
        st.info("입력을 하나 이상 고르세요.")
    else:
        sw_model, P, s, sw_proba, sw_cat, sw_num = run_swap_model(df_all, target, tuple(sw_feats), 1.0)
        sw_y = P["y"]
        if target == "당류":
            n_zero = int((s["한잔_당류"] == 0).sum())
            st.warning(f"당류는 값이 비어 있는 음료가 많아, 당류 값이 있는 음료 {len(s)}개만 사용합니다. 그중 {n_zero}개는 당류가 0이라서 "
                       "'당류가 0인 음료로 바꾸면 낮아진다'는 거의 단순한 규칙이 점수를 크게 끌어올립니다. 값이 같은 쌍(0과 0)은 빠지고, "
                       "값이 비어 있는 음료가 빠지면서 데이터가 한쪽으로 치우쳐 있으니 점수를 그대로 믿지 마세요.")
        sw_pred = (sw_proba >= 0.5).astype(int)

        st.caption("음료 하나를 통째로 빼고 학습한 뒤 그 음료가 'A'인 쌍을 맞히게 하는 방식(Leave-One-Product-Out)으로 잰 값입니다. "
                   "빠진 음료가 들어간 쌍은 학습에 쓰지 않습니다. 정답이 대칭(A→B가 1이면 B→A는 0)이라 기준선 정확도는 0.5입니다.")
        m1, m2, m3 = st.columns(3)
        m1.metric("정확도", f"{accuracy_score(sw_y, sw_pred):.2f}")
        m2.metric("ROC AUC", f"{roc_auc_score(sw_y, sw_proba):.2f}")
        m3.metric("쌍 개수", f"{len(P):,}", help=f"음료 {len(s)}개로 만든 (A→B) 쌍 중 값이 다른 쌍")

        nm = [n.replace("cat__A종류_", "지금: ").replace("cat__B종류_", "바꿀: ")
               .replace("num__로그용량비", "용량 변화(클수록 더 큰 컵으로)").replace("num__Δ", "다른 성분 변화: ")
              for n in sw_model.named_steps["columntransformer"].get_feature_names_out()]
        odds = pd.DataFrame({"조건": nm, "오즈비": np.exp(sw_model.named_steps["logisticregression"].coef_[0])}).sort_values("오즈비")
        fig = px.bar(odds, x="오즈비", y="조건", orientation="h", text_auto=".2f")
        fig.add_vline(x=1, line_dash="dash")
        fig.update_layout(margin=dict(t=10, b=10), height=max(300, 26 * len(odds) + 100))
        st.plotly_chart(fig)
        st.caption(f"오즈비가 1보다 크면 그 조건에서 '바꾸면 {target}이(가) 낮아질' 가능성이 높아지고, 1보다 작으면 낮아집니다. "
                   "인과관계가 아니라 이 데이터 안의 경향이며, 같은 변수 안의 범주끼리 서로 비교해서 보세요.")

        st.markdown(f"**위에서 고른 A → B에 적용**")
        if drink_a == drink_b:
            st.info("위에서 A와 B를 서로 다른 음료로 고르면 이 쌍의 예측을 볼 수 있습니다.")
        else:
            ia = s.index[s["식품명"] == drink_a]
            ib = s.index[s["식품명"] == drink_b]
            if len(ia) == 0 or len(ib) == 0:
                st.info(f"고른 음료 중 하나는 필요한 성분 값(에너지, 나트륨, 포화지방, {target})이 비어 있어 모델에서 빠졌습니다.")
            else:
                hit = P.index[(P["a"] == ia[0]) & (P["b"] == ib[0])]
                if len(hit) == 0:
                    st.info(f"두 음료의 한 잔 {target} 값이 같아 비교 대상이 아닙니다.")
                else:
                    pr = float(sw_proba[hit[0]])
                    real_low = bool(sw_y[hit[0]])
                    st.metric(f"바꾸면 한 잔 {target}이(가) 낮아질 확률(모델)", f"{pr * 100:.1f}%")
                    st.progress(pr)
                    st.write(f"실제: 한 잔 {target}이(가) " + ("낮아집니다." if real_low else "높아집니다.")
                             + " (이 예측은 A 음료를 뺀 채로 학습한 모델의 결과입니다.)")

    st.warning("이 모델이 맞히는 것은 '데이터에서 한 잔 값이 어떻게 달라지는가'이지 건강에 좋은지가 아닙니다. 용량이 큰 음료로 바꾸면 한 잔 값이 커지는 효과가 크게 섞여 있고, "
               "표본이 음료 40여 개뿐이며 쌍들은 서로 독립이 아닙니다. 점수는 참고로만 보세요.")
