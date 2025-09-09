import streamlit as st
import random
import pandas as pd
from dataclasses import dataclass, field

# =========================
# 定数・タスクリスト
# =========================
PC = 0.2   # 苦痛が生じているライン
PD = 0.5   # 重大事態ライン

# 1..11 → 表示は A-1..A-11（通常）
NORMAL_TASKS = list(range(1, 12))
# 12..19 → 表示は B-1..B-8（重大事態）
MAJOR_TASKS = list(range(12, 20))

# =========================
# 研修向けシナリオ文（課題の状況提示）
# =========================
TASK_LABELS = {
    1: "学校内である教員が、いじめがありそうな状況を把握しました。この教員は学年主任／生徒指導主事に報告するでしょうか？",
    2: "教務・生徒指導の関係者が、初動段階で状況共有と役割分担のための会議をすぐに開催するべき場面です。会議は開かれるでしょうか？",
    3: "校内での意思決定ラインを正式に立ち上げるべく、校長へ報告する局面です。報告は行われるでしょうか？",
    4: "被害児童生徒に安心できる場を確保し、早期に事実と気持ちを丁寧に聴き取る場面です。適切な聞き取りは行われるでしょうか？",
    5: "被害者保護者に学校の見立てと支援方針を丁寧に説明し、連携を図るべき局面です。早期の報告・相談は行われるでしょうか？",
    6: "座席・動線・見守りなどの即時の安全配慮と心理的支援が必要です。安全確保と心のケアは実施されるでしょうか？",
    7: "加害とされる側や周囲の児童生徒から客観的に事実確認を行う局面です。十分な確認は行われるでしょうか？",
    8: "被害者の意向と安全に配慮しながら、段階的に必要な指導を行う場面です。適切な指導は行われるでしょうか？",
    9: "校長が学校設置者へ正式に報告し、外部の支援を仰ぐ局面です。報告は行われるでしょうか？",
    10:"心理・福祉・医療の専門職（SC等）と早期に連携する場面です。連携は実施されるでしょうか？",
    11:"状況が落ち着いても継続的なフォローが必要です。被害者への支援は継続されるでしょうか？",
    12:"欠席日数30日などの要件を満たしたかを精査し、重大事態として正式認定する局面です。認定は行われるでしょうか？",
    13:"重大事態の報告を学校設置者へ速やかに行う場面です。早期報告は行われるでしょうか？",
    14:"設置者とともに第三者性・専門性を担保した調査組織を早期に設置する局面です。設置は行われるでしょうか？",
    15:"重大事態調査の目的・手続等を関係者に適切に説明する局面です。説明は行われるでしょうか？",
    16:"重大事態対応中であっても、被害者の安全確保と心理的支援を切らさない場面です。支援は継続されるでしょうか？",
    17:"重大事態対応中、加害者・保護者へ必要な説明と再発防止策を講じる局面です。適切な説明等は行われるでしょうか？",
    18:"学習保障（出席認定等）や進路相談を具体化する局面です。適切な配慮は行われるでしょうか？",
    19:"新たな被害防止のため、学級・学年・学校全体に必要な説明を行う局面です。周知は適切に行われるでしょうか？"
}

# A/B選択肢の説明（研修用）
TASK_CHOICES = {
    1: ("A: 学年主任／生徒指導主事にすぐ共有する", "B: 自分だけで様子見する"),
    2: ("A: 校内いじめ対策会議を即日〜数日内に開催", "B: 会議は後回し／開かない"),
    3: ("A: 校長に正式に報告する", "B: 校長へはまだ報告しない"),
    4: ("A: 安心できる場で被害者から早期に聴き取る", "B: 聴き取りを遅らせる／形式的に済ませる"),
    5: ("A: 早期に保護者へ説明・相談し連携する", "B: 説明不足で連絡も遅い"),
    6: ("A: 即時の安全配慮と心のケアを行う", "B: 安全配慮・心理支援を後回し"),
    7: ("A: 関係生徒から客観的に事実確認を行う", "B: 伝聞・噂で判断し確認を怠る"),
    8: ("A: 被害者の意向と安全に配慮した段階的指導", "B: 指導を曖昧に／被害者意向を無視"),
    9: ("A: 設置者へ速やかに報告する", "B: 報告をしない／遅らせる"),
    10:("A: SC等専門職と早期連携する", "B: 専門職に相談しない／先延ばし"),
    11:("A: 支援を継続しフォローを切らさない", "B: 一度の対応で打ち切る／疎かにする"),
    12:("A: 重大事態として正式認定を判断する", "B: 認定を避ける／遅らせる"),
    13:("A: 重大事態を設置者へ早期報告する", "B: 先延ばし／不十分な報告"),
    14:("A: 設置者と調査組織を早期に設置する", "B: 組織設置を遅らせる／不十分な体制"),
    15:("A: 調査の目的・手続等を適切に説明する", "B: 説明不足で誤解を招く"),
    16:("A: 調査中も安全確保と心理支援を継続する", "B: 調査を理由に支援を止める"),
    17:("A: 加害者・保護者へ必要な説明と再発防止策を講じる", "B: 説明不足／場当たり対応"),
    18:("A: 学習保障・進路相談を具体化する", "B: 学習面の配慮が不十分"),
    19:("A: 周囲への必要な説明で再発防止を図る", "B: 周知を怠る／場当たり対応")
}

# =========================
# 状態とロジック
# =========================
@dataclass
class GameState:
    t: int = 1
    p: float = 0.3
    s: float = 0.0
    m: float = 0.6
    initial_m: float = 0.6

    s_halved_m_applied: bool = False
    last_choice: str | None = None
    consecutive_A: int = 0

    normal_pending: list = field(default_factory=lambda: NORMAL_TASKS.copy())
    major_pending: list = field(default_factory=lambda: MAJOR_TASKS.copy())
    mode: str = "normal"
    log: list = field(default_factory=list)

    just_entered_major: bool = False
    pending_task: int | None = None
    pending_choice: str | None = None

def can_choose_A_by_dependency(k: int, history: dict):
    """依存関係：Aが選べない場合は具体的理由を返す"""
    def was_A(t): return history.get(t) == 'A'
    def was_B(t): return history.get(t) == 'B'
    rules = {
        2: (lambda: was_B(1),  "課題A-1（学年主任／生徒指導主事への共有）をBにしたため"),
        3: (lambda: was_B(1),  "課題A-1（学年主任／生徒指導主事への共有）をBにしたため"),
        5: (lambda: was_B(1),  "課題A-1（学年主任／生徒指導主事への共有）をBにしたため"),
        9: (lambda: was_B(3),  "課題A-3（校長への正式報告）をBにしたため"),
        10:(lambda: was_B(3),  "課題A-3（校長への正式報告）をBにしたため"),
        11:(lambda: was_B(2) or was_B(3), "課題A-2またはA-3をBにしたため"),
        12:(lambda: not was_A(3), "課題A-3（校長への正式報告）でAを選んでいないため"),
        13:(lambda: was_B(9),  "課題A-9（設置者への報告）をBにしたため"),
        14:(lambda: was_B(13), "課題B-2（重大事態の早期報告）をBにしたため"),
        15:(lambda: was_B(13), "課題B-2（重大事態の早期報告）をBにしたため"),
        16:(lambda: was_B(11), "課題A-11（支援の継続）をBにしたため"),
        17:(lambda: was_B(15), "課題B-4（調査の説明）をBにしたため"),
    }
    if k in rules and rules[k][0]():
        return False, rules[k][1]
    return True, None

def preview_effects(state: GameState, choice: str):
    """適用前の差分（プレビュー）"""
    dp = -0.05 if choice == 'A' else +0.2
    ds = +0.05 if choice == 'A' else -0.2
    dm = 0.0
    if choice == 'A' and state.last_choice == 'A':
        dm += 0.1  # 連続Aで+0.1
    new_s = state.s + ds
    half = (not state.s_halved_m_applied) and (new_s <= -0.5)
    return dp, ds, dm, half

def apply_choice(state: GameState, task_id: int, choice: str, history: dict):
    """選択の適用"""
    dp, ds, dm, half = preview_effects(state, choice)

    # p, s 更新
    state.p = max(0.0, state.p + dp)
    state.s = state.s + ds

    # m 更新（連続A加点／s<=-0.5で半減）
    if choice == 'A':
        if state.last_choice == 'A':
            state.consecutive_A += 1
            if state.consecutive_A >= 2:
                state.m = min(1.0, state.m + 0.1)
        else:
            state.consecutive_A = 1
    else:
        state.consecutive_A = 0

    if half:
        state.m *= 0.5
        state.s_halved_m_applied = True

    # タスク消化
    if task_id in state.normal_pending:
        state.normal_pending.remove(task_id)
    if task_id in state.major_pending:
        state.major_pending.remove(task_id)

    # モード判定（p>=0.5で重大事態モードへ）
    if state.p >= PD and state.mode != 'major':
        state.mode = 'major'
        state.just_entered_major = True
    else:
        state.just_entered_major = False

    # 終了判定
    ended = False
    if state.mode == 'normal':
        if 11 not in state.normal_pending and len(state.normal_pending) == 0:
            ended = True
    else:
        if 19 not in state.major_pending and len(state.major_pending) == 0:
            ended = True

    # 次ターン
    state.t += 1
    state.last_choice = choice
    return ended

def next_available_task(state: GameState):
    """強制昇順：次の課題（A-1.. or B-1..）"""
    pool = state.normal_pending if state.mode == 'normal' else state.major_pending
    return min(pool) if pool else None

def check_A_gate(state: GameState, task_id: int, history: dict, a_gate: dict):
    """A可否の抽選（依存関係OKの場合のみ）"""
    dep_ok, _ = can_choose_A_by_dependency(task_id, history)
    if not dep_ok:
        return False
    gate = a_gate.get(task_id)
    if gate is None:
        # この抽選は「抽選を開始」ボタン側で行う（r表示のため）
        return None
    return gate

# 1) モンテカルロの自動選択を修正（mの二重掛けを排除）
def auto_pick(state, task_id, history, a_gate):
    dep_ok, _ = can_choose_A_by_dependency(task_id, history)
    if not dep_ok:
        return 'B'
    if task_id not in a_gate:
        a_gate[task_id] = (random.random() < state.m)  # ← ここだけ m
    return 'A' if a_gate[task_id] else 'B'            # ← 2回目の m を削除


def run_one_game(seed=None, init_m=None):
    """モンテカルロ1ゲーム"""
    if seed is not None:
        random.seed(seed)
    m0 = init_m if init_m is not None else random.choice([0.3, 0.6, 0.9])
    stt = GameState(m=m0, initial_m=m0)
    history, a_gate = {}, {}
    while True:
        task_id = next_available_task(stt)
        if task_id is None:
            break
        # 自動でA/B
        choice = auto_pick(stt, task_id, history, a_gate)
        history[task_id] = choice
        ended = apply_choice(stt, task_id, choice, history)
        if ended or stt.t >= 20:
            stt.t = 20
            break
    goal = stt.p < PC
    return {
        'goal': goal,
        'p_final': stt.p,
        's_final': stt.s,
        'm_final': stt.m,
        'major_mode_ratio': 1 if stt.mode == 'major' else 0
    }

# =========================
# UI（タイトルと説明）
# =========================
st.set_page_config(page_title="いじめ対応のゲーム教材（論文準拠）", layout="wide")

st.title("学校のいじめ対応のゲーム教材（論文モデル準拠・研修用デモ）")
st.markdown(
    """
**使い方**  
学校は、時間が進むごとに、**いじめ防止対策推進法に従った選択肢A**か、**従わない選択肢B**か、いずれかの選択をします。  
ただし、**過去に必要な対応をしていない場合**には、**Aが選べない**ことがあります。  
また、**Aが選べるのは、組織風土（=A選択の確率）の抽選で選択可となった場合だけ**です。  
左で「新しいゲーム」を初期化し、右で課題を順に選びます。下部の「モンテカルロ」で初期mごとの達成率を確認できます。
    """
)

# =========================
# ユーティリティ：新規ゲームの完全初期化
# =========================
def _hard_reset(m0: float):
    new_state = GameState(m=m0, initial_m=m0)
    new_state.t = 1; new_state.p = 0.3; new_state.s = 0.0
    new_state.mode = 'normal'; new_state.last_choice = None; new_state.consecutive_A = 0
    new_state.normal_pending = list(range(1, 12))
    new_state.major_pending = list(range(12, 20))
    new_state.log = []; new_state.just_entered_major = False
    new_state.pending_task = None; new_state.pending_choice = None

    st.session_state["state"] = new_state
    st.session_state["history"] = {}
    st.session_state["ended"] = False

    # 抽選結果・直近乱数・実測値カウンタを確実に初期化
    st.session_state["a_gate"] = {}
    st.session_state["gate_last_r"] = {}
    st.session_state["gate_stats"] = {"draws": 0, "success": 0}

    # ボタンkeyの一意化用 run_id を増やす
    st.session_state["run_id"] = st.session_state.get("run_id", 0) + 1

    # ランダム抽選モードの一時値もリセット
    st.session_state["_m_draw"] = None

    st.rerun()

# =========================
# Sidebar（初期mのランダム抽選フローつき）
# =========================
with st.sidebar:
    st.header("新しいゲームを初期化")

    init_mode = st.radio(
        "初期mの設定",
        ["ランダム(0.9/0.6/0.3)", "手動指定"],
        index=0,
        key="sidebar_init_mode_radio"
    )

    # セッション変数の安全な初期確保
    if "state" not in st.session_state:
        st.session_state["state"] = GameState(m=random.choice([0.3, 0.6, 0.9]), initial_m=0.6)
    if "history" not in st.session_state:
        st.session_state["history"] = {}
    if "ended" not in st.session_state:
        st.session_state["ended"] = False
    if "a_gate" not in st.session_state:
        st.session_state["a_gate"] = {}
    if "gate_last_r" not in st.session_state:
        st.session_state["gate_last_r"] = {}
    if "gate_stats" not in st.session_state:
        st.session_state["gate_stats"] = {"draws": 0, "success": 0}
    if "run_id" not in st.session_state:
        st.session_state["run_id"] = 0

    if init_mode == "ランダム(0.9/0.6/0.3)":
        st.info(
            "学校の組織風土がどの程度いじめにきちんと対応できるものかをランダムに設定します。\n"
            "組織風土 m は、各選択で学校が『いじめ防止対策推進法』に沿った適切な選択（A）をできる確率です。"
        )
        if "_m_draw" not in st.session_state:
            st.session_state["_m_draw"] = None

        if st.button("組織風土を抽選する", key="btn_draw_m"):
            st.session_state["_m_draw"] = random.choice([0.3, 0.6, 0.9])

        if st.session_state["_m_draw"] is not None:
            m0 = st.session_state["_m_draw"]
            st.success(f"mは {m0:.1f} となりました。現在、学校が法に沿った適切な選択(A)をできる確率は **{int(m0*100)}%** です。")
            if st.button("この組織風土でゲームを開始", key="btn_start_random"):
                _hard_reset(m0)
    else:
        init_m = st.select_slider(
            "初期m",
            options=[0.3, 0.6, 0.9],
            value=0.6,
            key="sidebar_init_m_slider"
        )
        if st.button("新しいゲームを開始", key="btn_start_manual"):
            _hard_reset(float(init_m))

# 参照（以降はこのエイリアスを使用）
state: GameState = st.session_state["state"]
history: dict = st.session_state["history"]
a_gate: dict = st.session_state["a_gate"]
run_id: int = st.session_state.get("run_id", 0)

# =========================
# モンテカルロ（常時表示）
# =========================
st.markdown("---")
with st.expander("モンテカルロ試行", expanded=False):
    trials = st.slider("試行回数", 10, 2000, 500, step=10, key=f"mc_trials_slider_{run_id}")
    init_m_opt = st.selectbox(
        "初期mの固定",
        ["固定しない(ランダム)", 0.3, 0.6, 0.9],
        index=0,
        key=f"mc_init_m_selectbox_{run_id}"
    )
    seed = st.number_input("乱数シード（任意）", value=0, step=1, key=f"mc_seed_input_{run_id}")
    if st.button("試行を実行", key=f"mc_run_button_{run_id}"):
        results = []
        for i in range(trials):
            res = run_one_game(
                seed=int(seed) + i,
                init_m=None if init_m_opt == "固定しない(ランダム)" else float(init_m_opt)
            )
            results.append(res)
        df = pd.DataFrame(results)
        st.write(f"**ゴール達成率**: {df['goal'].mean()*100:.1f}%")
        st.write(f"**重大事態移行率**: {df['major_mode_ratio'].mean()*100:.1f}%")
        st.write("**最終値の平均**:")
        st.table(df[['p_final', 's_final', 'm_final']].mean().to_frame(name='mean'))
        st.dataframe(df, use_container_width=True)

# =========================
# メインUI（2カラム）
# =========================
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("現在の状態")

    # 残り課題数のバー：A-n → 残り=11-n、B-n → 残り=8-n
    next_id = next_available_task(state)
    if next_id is None:
        rem = 0
        total = 11 if state.mode == 'normal' else 8
    else:
        if next_id <= 11:
            n = next_id
            rem = 11 - n
            total = 11
        else:
            n = next_id - 11
            rem = 8 - n
            total = 8
    st.write("**時間（t）**：ゴールまでの残り（課題数）")
    progress = 0.0 if total == 0 else (total - rem) / total
    st.progress(progress)
    st.caption(f"残り課題数: {rem} / {total}")

    # 苦痛p
    st.write("**苦痛 p**：0.2以上=『苦痛あり』、0.5以上=『重大事態移行』")
    st.progress(min(1.0, max(0.0, state.p)))
    st.caption(f"現在のp: {state.p:.3f} ｜ しきい値 pc=0.2, pd=0.5")

    # 信頼s
    st.write("**信頼 s**：-0.5以下になると m が一度だけ半減")
    s_norm = (state.s + 1.0) / 2.0
    st.progress(min(1.0, max(0.0, s_norm)))
    s_note = "（半減条件未発動）" if not state.s_halved_m_applied else "（半減は既に一度発動済み）"
    st.caption(f"現在のs: {state.s:.3f} ｜ しきい値 -0.5 {s_note}")

    # 組織風土m
    st.write("**組織風土 m**（= A選択の確率）")
    st.metric("A選択確率", f"{state.m*100:.0f}%")
    st.caption("mが高いほどAを実行しやすく、連続Aによりmが上がります（最大1.0）。")

with col2:
    st.subheader("このターンの選択")

    # 重大事態メッセージ
    if state.just_entered_major:
        st.error("**被害者の苦痛が一定の値（0.5）以上になったので、苦痛が深刻となり、学校の対応がうまくいかないまま時間が進んでしまいました。欠席日数が30日以上となったので、ここからは重大事態モードとなり、重大事態としての対応が適切かどうかが問われることとなります。**")
        st.markdown("### 重大事態モードに移行しました")
        state.just_entered_major = False

    if st.session_state["ended"] or state.t >= 20:
        st.success("被害児童生徒が卒業を迎え、ゲームはいったん終了となりました。『結果の評価』を読み、振り返りを行ってください。")
    else:
        task_id = next_available_task(state)
        if task_id is None:
            st.warning("進めるタスクがありません。終了します。")
            state.t = 20
            st.session_state["ended"] = True
        else:
            # 課題番号（A-* or B-*）
            code = f"A-{task_id}" if task_id <= 11 else f"B-{task_id-11}"
            st.markdown(f"**課題{code}**：{TASK_LABELS[task_id]}")
            a_text, b_text = TASK_CHOICES.get(task_id, ("A", "B"))
            st.write(a_text)
            st.write(b_text)

            # 依存関係 → A不可なら抽選スキップ
            dep_ok, dep_reason = can_choose_A_by_dependency(task_id, history)
            gate = a_gate.get(task_id)

            if not dep_ok:
                st.warning(f"過去の対応の欠落により、この課題ではAは選択できません（理由：{dep_reason}）。Bを選んでください。")
            else:
                # A選択可否の抽選（未抽選なら実行）
                if gate is None:
                    st.caption("それでは組織風土（A選択確率）に基づき、Aが選べるか抽選します。『抽選を開始』を押してください。")
                    if st.button("抽選を開始", key=f"draw_{task_id}_{run_id}"):
                        r = random.random()
                        gate = (r < state.m)
                        a_gate[task_id] = gate

                        # 乱数と実測率の記録
                        st.session_state["gate_last_r"][task_id] = r
                        st.session_state["gate_stats"]["draws"] += 1
                        if gate:
                            st.session_state["gate_stats"]["success"] += 1
                        st.rerun()

                # 抽選結果の説明
                if gate is not None:
                    prob = int(state.m * 100)
                    if gate:
                        st.info(f"選択確率{prob}%で抽選した結果、**Aの選択が可能**となりました。どちらを選びますか？")
                    else:
                        st.warning(f"選択確率{prob}%で抽選した結果、**Aの選択はできない**こととなりました。**Bを選んでください。**")

                    # 今回の乱数と実測値
                    r_shown = st.session_state.get("gate_last_r", {}).get(task_id)
                    if r_shown is not None:
                        st.caption(f"【今回の乱数 r】{r_shown:.3f} ／ 【判定基準】r < m (= {state.m:.3f}) → {'A可' if gate else 'A不可'}")
                    g = st.session_state.get("gate_stats", {"draws": 0, "success": 0})
                    emp = (g["success"] / g["draws"] * 100) if g["draws"] else 0.0
                    st.caption(f"【A抽選 実測値】 {g['success']} / {g['draws']}（{emp:.1f}%）｜【理論値】{state.m*100:.0f}%")

                    # 検証用：同課題で再抽選したいとき（未確定の間だけ）
                    if state.pending_task is None and state.pending_choice is None:
                        if st.button("再抽選（検証用）", key=f"redraw_{task_id}_{run_id}"):
                            a_gate.pop(task_id, None)
                            st.session_state["gate_last_r"].pop(task_id, None)
                            st.rerun()

            # プレビュー＆確定
            allow_A = dep_ok and bool(gate)
            if allow_A:
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Aを選ぶ（プレビュー）", key=f"prev_A_{task_id}_{run_id}"):
                        state.pending_task = task_id
                        state.pending_choice = 'A'
                        st.rerun()
                with c2:
                    if st.button("Bを選ぶ（プレビュー）", key=f"prev_B_{task_id}_{run_id}"):
                        state.pending_task = task_id
                        state.pending_choice = 'B'
                        st.rerun()
            else:
                if st.button("Bを選ぶ（プレビュー）", key=f"prev_B_only_{task_id}_{run_id}"):
                    state.pending_task = task_id
                    state.pending_choice = 'B'
                    st.rerun()

            if state.pending_task == task_id and state.pending_choice is not None:
                dp, ds, dm, half = preview_effects(state, state.pending_choice)
                new_p = max(0.0, state.p + dp)
                new_s = state.s + ds
                new_m = state.m
                m_explain = []
                if state.pending_choice == 'A' and state.last_choice == 'A':
                    m_explain.append("連続Aにより m が +0.100")
                    new_m = min(1.0, new_m + 0.1)
                if half:
                    m_explain.append("sが -0.500 以下となるため m が半減")
                    new_m = new_m * 0.5

                st.markdown("#### 変化のプレビュー（教員研修用説明）")
                st.write(f"- **苦痛(p)** は {'0.050減少' if state.pending_choice=='A' else '0.200増加'} するので、 {state.p:.3f} → **{new_p:.3f}** となります。")
                st.write(f"- **信頼(s)** は {'0.050増加' if state.pending_choice=='A' else '0.200減少'} するので、 {state.s:.3f} → **{new_s:.3f}** となります。")
                if m_explain:
                    st.write("- **組織風土(m)** は " + "、".join(m_explain) + f" ため、 → **{new_m:.3f}** となります。")
                else:
                    st.write(f"- **組織風土(m)** は変化しないため、 → **{new_m:.3f}** のままです。")
                st.write(f"- **モード** は、苦痛が 0.500 以上で重大事態に移行します（今回の見込み: {'重大事態' if new_p >= PD else '通常'}）。")

                if allow_A:
                    c_ok, c_cancel = st.columns(2)
                    with c_ok:
                        if st.button("この内容で確定", key=f"commit_{task_id}_{run_id}"):
                            choice = state.pending_choice
                            state.pending_choice = None
                            state.pending_task = None
                            history[task_id] = choice
                            ended = apply_choice(state, task_id, choice, history)
                            state.log.append({
                                't': state.t-1, 'task': task_id, 'label': TASK_LABELS[task_id],
                                'choice': choice, 'p': round(state.p, 3), 's': round(state.s, 3),
                                'm': round(state.m, 3), 'mode': state.mode
                            })
                            a_gate.pop(task_id, None)
                            if ended:
                                st.session_state["ended"] = True
                            st.rerun()
                    with c_cancel:
                        if st.button("やっぱり選び直す", key=f"cancel_{task_id}_{run_id}"):
                            state.pending_choice = None
                            state.pending_task = None
                            st.rerun()
                else:
                    if st.button("この内容で確定", key=f"commit_onlyB_{task_id}_{run_id}"):
                        choice = state.pending_choice
                        state.pending_choice = None
                        state.pending_task = None
                        history[task_id] = choice
                        ended = apply_choice(state, task_id, choice, history)
                        state.log.append({
                            't': state.t-1, 'task': task_id, 'label': TASK_LABELS[task_id],
                            'choice': choice, 'p': round(state.p, 3), 's': round(state.s, 3),
                            'm': round(state.m, 3), 'mode': state.mode
                        })
                        a_gate.pop(task_id, None)
                        if ended:
                            st.session_state["ended"] = True
                        st.rerun()

# =========================
# ログと評価
# =========================
st.markdown("---")
st.subheader("ログ（直近から順）")
if state.log:
    df = pd.DataFrame(state.log)
    st.dataframe(df.iloc[::-1], use_container_width=True)
else:
    st.info("まだログはありません。上で選択を進めてください。")

if st.session_state.get("ended") or state.t >= 20:
    st.markdown("### 結果の評価")
    goal = state.p < PC
    trust = state.s
    m_final = state.m
    m_initial = state.initial_m

    # pの解釈
    if goal:
        p_msg = "被害児童生徒の苦痛は十分に抑えられました（p<0.200）。"
    elif state.p >= PD:
        p_msg = "重大事態となってからも、被害児童生徒の苦痛が深刻に増大してしまいました（p≥0.500）。"
    else:
        p_msg = "苦痛は一定程度残りました（0.200≤p<0.500）。"

    # sの解釈
    if trust >= 0.2:
        s_msg = "学校への信頼は概ね回復しています（s≥0.200）。"
    elif trust >= 0:
        s_msg = "学校への信頼はなんとか維持されています（0≤s<0.200）。"
    elif trust <= -0.5:
        s_msg = "学校が被害者側から深刻な不信を抱かれる状況です（s≤-0.500）。"
    else:
        s_msg = "学校への信頼が低下しています（-0.500<s<0）。"

    # mの解釈
    if m_final - m_initial > 0.1:
        m_msg = "学校の組織風土は、いじめ対応を通じて**明確に改善**しました。"
    elif m_final >= m_initial:
        m_msg = "学校の組織風土は、いじめ対応を通じて**少し改善**または維持されました。"
    else:
        m_msg = "学校の組織風土は、いじめ対応を通じて**低下**しました。"

    # 総合評価（初期mも踏まえて）
    if goal and trust >= 0 and m_final >= m_initial:
        overall = "総合的に見て、当該校は適切にいじめ対応ができたと評価できます。"
    elif (not goal) and state.p >= PD and trust < 0:
        overall = "総合的に見て、当該校はいじめ対応が不十分で、重大事態後も苦痛・不信が悪化したと評価されます。"
    else:
        overall = "総合的に見て、対応には改善の余地が残ります。初動と継続支援、説明責任の徹底が重要です。"

    st.write(f"- **苦痛(p)**: {p_msg}（最終p={state.p:.3f}）")
    st.write(f"- **信頼(s)**: {s_msg}（最終s={state.s:.3f}）")
    st.write(f"- **組織風土(m)**: {m_msg}（初期m={m_initial:.3f} → 最終m={m_final:.3f}）")
    st.write(f"- **総合評価**: {overall}")

# =========================
# フッター（著作権表記）
# =========================
st.markdown("---")
st.caption("© 2025 Daisuke Fujikawa — 本教材は研究用試作品。改変・再配布は自由ですが、クレジットを必ず明記してください。")
