import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
import requests

# ── API URLs ──────────────────────────────────────────────────────
CLUSTERING_API = "https://mazenmaher26-aroundu-location-clustering.hf.space"
ANOMALY_API    = "https://mazenmaher26-aroundu-anomaly-detection.hf.space"

# ═══════════════════════════════════════════════════════════
# PAGE SETUP
# ═══════════════════════════════════════════════════════════
st.set_page_config(page_title="AroundU | Admin Dashboard", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=DM+Sans:wght@400;500;600&display=swap');

    /* ── Root palette ── */
    :root {
        --brand-dark:    #1D3143;
        --brand-mid:     #2F5C85;
        --brand-blue:    #315687;
        --brand-teal:    #619FB8;
        --brand-light:   #61A3BB;
        --brand-muted:   #65797E;
        --brand-white:   #FFFFFF;
        --bg-page:       #EDF2F6;
        --bg-card:       #FFFFFF;
        --border-subtle: #C8D9E6;
    }

    /* ── Page background ── */
    .main, .block-container { background-color: var(--bg-page) !important; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--brand-dark) 0%, #243C52 100%) !important;
        min-width: 270px !important;
        border-right: 1px solid rgba(97,163,187,0.15);
    }
    section[data-testid="stSidebar"] * { color: var(--brand-white) !important; }
    section[data-testid="stSidebar"] .stDateInput input {
        background: rgba(97,159,184,0.12) !important;
        border: 1px solid rgba(97,159,184,0.3) !important;
        color: #1D3143 !important;
        border-radius: 8px !important;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: var(--bg-card);
        border-left: 4px solid var(--brand-teal);
        padding: 18px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(29,49,67,0.08);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(47,92,133,0.14);
    }
    div[data-testid="stMetric"] label {
        color: var(--brand-muted) !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--brand-dark) !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 13px !important;
        font-weight: 600 !important;
    }

    /* ── Plot / chart wrapper ── */
    .plot-container {
        background: var(--bg-card);
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 2px 12px rgba(29,49,67,0.07);
        margin-bottom: 24px;
    }

    /* ── Headings ── */
    h1 {
        font-family: 'Sora', sans-serif !important;
        color: var(--brand-dark) !important;
        font-weight: 700 !important;
        font-size: 1.7rem !important;
        letter-spacing: -0.3px;
        padding-bottom: 4px;
        border-bottom: 3px solid var(--brand-teal);
        display: inline-block;
        margin-bottom: 1.2rem !important;
    }
    h2, h3 {
        font-family: 'Sora', sans-serif !important;
        color: var(--brand-mid) !important;
        font-weight: 600 !important;
    }

    /* ── Subheader ── */
    [data-testid="stHeadingWithActionElements"] h2,
    [data-testid="stHeadingWithActionElements"] h3 {
        color: var(--brand-mid) !important;
        font-family: 'Sora', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border-left: 3px solid var(--brand-light);
        padding-left: 10px;
    }

    /* ── Dataframe table ── */
    [data-testid="stDataFrame"] {
        border-radius: 10px !important;
        overflow: hidden;
        border: 1px solid var(--border-subtle) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: var(--brand-mid) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 6px 14px !important;
        transition: background 0.2s ease;
    }
    .stButton > button:hover {
        background: var(--brand-dark) !important;
    }

    /* ── Selectbox / text input ── */
    .stSelectbox > div > div,
    .stTextInput > div > div > input {
        border-radius: 8px !important;
        border: 1px solid var(--border-subtle) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── Divider ── */
    hr { border-color: var(--border-subtle) !important; }

    /* ── Section divider label ── */
    .section-label {
        font-size: 11px;
        font-weight: 700;
        color: rgba(255,255,255,0.55);
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin: 10px 0 4px 0;
    }

    /* ── Alert badges ── */
    .badge-red {
        background: rgba(239,68,68,0.15); color: #EF4444;
        padding: 2px 10px; border-radius: 20px;
        font-size: 12px; font-weight: 700;
        border: 1px solid rgba(239,68,68,0.3);
    }
    .badge-green {
        background: rgba(97,163,187,0.15); color: var(--brand-light);
        padding: 2px 10px; border-radius: 20px;
        font-size: 12px; font-weight: 700;
        border: 1px solid rgba(97,163,187,0.3);
    }
    .badge-yellow {
        background: rgba(245,158,11,0.15); color: #F59E0B;
        padding: 2px 10px; border-radius: 20px;
        font-size: 12px; font-weight: 700;
        border: 1px solid rgba(245,158,11,0.3);
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: var(--brand-teal) !important; }

    /* ── Info / warning / error banners ── */
    .stInfo    { background: rgba(97,163,187,0.1) !important; border-left-color: var(--brand-teal) !important; }
    .stWarning { background: rgba(245,158,11,0.08) !important; }
    .stError   { background: rgba(239,68,68,0.08) !important; }
    </style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# MOCK DATA ENGINE  (fixed + expanded)
# ═══════════════════════════════════════════════════════════
@st.cache_data
def load_all_platform_data():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2024-03-01", freq="D")

    # ── Time series ──────────────────────────────────────────
    df_ts = pd.DataFrame({
        "Date":           dates,
        "Visits":         np.random.randint(2000, 8000,  len(dates)),
        "New_Users":      np.random.randint(30,   150,   len(dates)),
        "New_Owners":     np.random.randint(1,    8,     len(dates)),
        "Saves":          np.random.randint(100,  500,   len(dates)),
        "Directions":     np.random.randint(200,  900,   len(dates)),
        "Calls":          np.random.randint(50,   250,   len(dates)),
        "Reviews":        np.random.randint(20,   120,   len(dates)),
        "Chats":          np.random.randint(100,  500,   len(dates)),   # ✅ BUG FIX: was 'Chat_Sessions'
        "Resolved_Chats": np.random.randint(80,   450,   len(dates)),
    })

    # ── Places ───────────────────────────────────────────────
    cats      = ["Restaurant", "Cafe", "Pharmacy", "House", "Super Market"]
    districts = ["City Center", "University Area", "Nile Corniche",
                 "New Beni Suef", "El Wasta", "Biba"]

    places = []
    for i in range(250):
        places.append({
            "Place_ID": f"P-{1000+i}",
            "Name":     f"Place {1000+i}",
            "Category": np.random.choice(cats),
            "District": np.random.choice(districts),
            "Visits":   np.random.randint(100,  10000),
            "Saves":    np.random.randint(10,   2000),
            "Rating":   round(np.random.uniform(1.5, 5.0), 1),
            "Reviews":  np.random.randint(0,    300),
            "Status":   np.random.choice(
                ["Active", "Suspended", "Pending Approval"], p=[0.80, 0.10, 0.10]
            ),
            "Added":    np.random.choice(
                pd.date_range("2023-01-01", "2024-03-01", freq="D")
            ),
        })
    df_places = pd.DataFrame(places)

    # ── Users ────────────────────────────────────────────────
    users = []
    for i in range(500):
        users.append({
            "User_ID":    f"U-{2000+i}",
            "Name":       f"User {2000+i}",
            "District":   np.random.choice(districts),
            "Reviews":    np.random.randint(0, 50),
            "Saves":      np.random.randint(0, 100),
            "Status":     np.random.choice(["Active", "Suspended"], p=[0.90, 0.10]),
            "Joined":     str(np.random.choice(pd.date_range("2023-01-01", "2024-03-01", freq="D")))[:10],
            "Last_Login": str(np.random.choice(pd.date_range("2024-01-01", "2024-03-01", freq="D")))[:10],
        })
    df_users = pd.DataFrame(users)

    # ── Flagged reviews ──────────────────────────────────────
    review_texts = [
        "خدمة سيئة جداً ولا أنصح بالذهاب",
        "الأكل كان بارداً والمكان غير نظيف",
        "تأخير كبير في التوصيل",
        "الأسعار مبالغ فيها جداً",
        "موظفين غير محترمين مع العملاء",
        "جودة رديئة مقارنة بالسعر",
        "المكان مغلق ولم يتم التحديث",
        "طلبت ولم يصلني الطلب نهائياً",
        "تجربة سيئة للغاية لن أعود",
        "الإدارة لا تتجاوب مع الشكاوى",
        "وجدت شعر في الطعام",
        "الحمامات غير نظيفة على الإطلاق",
        "الموظفون يتجاهلون العملاء تماماً",
        "أسوأ تجربة في حياتي",
        "لا أنصح أي شخص بالذهاب هنا",
    ]
    flagged_reviews = pd.DataFrame({
        "Review_ID": [f"R-{400+i}" for i in range(15)],
        "User":      [f"User {2000+i}" for i in range(15)],
        "Place":     np.random.choice([f"Place {1000+i}" for i in range(20)], 15),
        "Review":    review_texts,
        "Rating":    np.random.randint(1, 3, 15),
        "Date":      [str(d)[:10] for d in pd.date_range("2024-02-01", periods=15, freq="D")],
    })

    # ── Pending owners ───────────────────────────────────────
    pending_owners = pd.DataFrame({
        "Owner_ID":  [f"OWN-{i}" for i in range(1, 8)],
        "Name":      [f"Owner {i}" for i in range(1, 8)],
        "Business":  [f"Business {i}" for i in range(1, 8)],
        "Category":  np.random.choice(cats, 7),
        "Submitted": [str(d)[:10] for d in pd.date_range("2024-02-20", periods=7, freq="D")],
    })

    # ── Chatbot query types ──────────────────────────────────
    chat_types = pd.DataFrame({
        "Type": ["Menu", "Hours", "Location", "Pricing", "Other"],
        "Val":  [45, 25, 15, 10, 5],
    })

    # ── Top chatbot places ───────────────────────────────────
    top_chat_places = pd.DataFrame({
        "Place":    [f"Place {1000+i}" for i in range(8)],
        "Chats":    np.random.randint(200, 2000, 8),
        "Resolved": np.random.randint(150, 1800, 8),
    }).sort_values("Chats", ascending=False).reset_index(drop=True)

    # ── Admin action log ─────────────────────────────────────
    admin_log = pd.DataFrame({
        "Admin":     ["Admin_01", "Admin_01", "SuperAdmin", "Admin_02", "Admin_01",
                      "SuperAdmin", "Admin_02"],
        "Action":    ["Approved Place P-1002", "Suspended User U-882",
                      "Deleted Review R-442",  "Approved Owner OWN-3",
                      "Suspended Place P-1045","Added Category 'Gym'",
                      "Rejected Owner OWN-6"],
        "Target_ID": ["P-1002","U-882","R-442","OWN-3","P-1045","—","OWN-6"],
        "Timestamp": ["10 mins ago","1 hr ago","Yesterday",
                      "2 days ago","2 days ago","3 days ago","3 days ago"],
    })

    return df_ts, df_places, df_users, flagged_reviews, pending_owners, chat_types, top_chat_places, admin_log


with st.spinner("Loading platform data..."):
    (df_ts, df_places, df_users,
     flagged_reviews, pending_owners,
     chat_types, top_chat_places, admin_log) = load_all_platform_data()


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;padding:4px 0 12px 0;">
            <div style="background:linear-gradient(135deg,#2F5C85,#61A3BB);
                        border-radius:10px;width:38px;height:38px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:20px;">🏙️</div>
            <div>
                <div style="font-family:'Sora',sans-serif;font-weight:700;
                            font-size:20px;color:white;letter-spacing:-0.3px;">AroundU</div>
                <div style="font-size:11px;color:rgba(97,163,187,0.9);font-weight:500;">
                    Admin Intelligence
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=[
            "Overview", "Places Analytics", "User Analytics",
            "Reviews", "Chatbot", "Category Analytics",
            "Moderation", "Anomaly Detection", "Location Logic",
        ],
        icons=[
            "grid-1x2", "shop", "people",
            "star", "robot", "tags",
            "shield-lock", "exclamation-triangle", "geo-alt",
        ],
        default_index=0,
        styles={
            "container": {
                 "background-color": "#1D3143",
                 "border-radius": "14px",
                 "padding": "8px",
                 "border": "1px solid rgba(97,163,187,0.15)",
                 },
            "nav-link":{"font-size": "15px","font-weight": "500", "font-family": "'DM Sans', sans-serif", "text-align": "left", "color": "rgba(255,255,255,0.75)","border-radius": "10px","padding": "12px 16px","margin": "2px 0",},
            "nav-link-selected": {"background-color": "#2F5C85", "color": "white", "font-weight": "700",  "border-radius": "10px",},
            "icon": {"color": "#61A3BB" , "font-size": "18px",},
        },
    )

    st.divider()
    st.markdown("### 📅 Date Range")
    date_range = st.date_input(
        "Choose period:",
        value=(datetime(2024, 1, 1), datetime(2024, 1, 30)),
    )

    # Alert counts in sidebar
    st.divider()
    st.markdown('<p class="section-label">⚠️ Alerts</p>', unsafe_allow_html=True)
    st.markdown(
        f'🚩 Flagged reviews &nbsp; <span class="badge-red">{len(flagged_reviews)}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'👤 Pending owners &nbsp;&nbsp; <span class="badge-yellow">{len(pending_owners)}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'🚫 Suspended users &nbsp; <span class="badge-red">{len(df_users[df_users["Status"]=="Suspended"])}</span>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════
# FILTER + HELPERS
# ═══════════════════════════════════════════════════════════
if isinstance(date_range, tuple) and len(date_range) == 2:
    start = pd.to_datetime(date_range[0])
    end   = pd.to_datetime(date_range[1])
    df_filtered = df_ts[(df_ts["Date"] >= start) & (df_ts["Date"] <= end)]
    days_diff   = (end - start).days + 1
    df_prev     = df_ts[
        (df_ts["Date"] >= start - timedelta(days=days_diff)) &
        (df_ts["Date"] <  start)
    ]
else:
    st.warning("Please select a valid start AND end date.")
    st.stop()

# ✅ BUG FIX: safe delta — handles empty prev period
def get_delta(col):
    if df_prev.empty or df_filtered.empty:
        return None
    curr = df_filtered[col].sum()
    prev = df_prev[col].sum()
    if prev == 0:
        return None
    pct = int(((curr - prev) / prev) * 100)
    return f"{pct:+}%"

def empty_state(msg="No data available for the selected period."):
    st.info(f"ℹ️ {msg}")

TEMPLATE = "plotly_white"


# ═══════════════════════════════════════════════════════════
# 1. OVERVIEW
# ═══════════════════════════════════════════════════════════
if selected == "Overview":
    st.title("📊 Platform Performance Overview")

    if df_filtered.empty:
        empty_state()
        st.stop()

    # KPI Row 1
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Visits",     f"{df_filtered['Visits'].sum():,}",    get_delta("Visits"))
    k2.metric("New Users",        f"{df_filtered['New_Users'].sum():,}",  get_delta("New_Users"))
    k3.metric("Saved Places",     f"{df_filtered['Saves'].sum():,}",      get_delta("Saves"))
    k4.metric("Direction Clicks", f"{df_filtered['Directions'].sum():,}", get_delta("Directions"))

    # KPI Row 2
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Call Clicks",   f"{df_filtered['Calls'].sum():,}",   get_delta("Calls"))
    k6.metric("Total Reviews", f"{df_filtered['Reviews'].sum():,}", get_delta("Reviews"))
    k7.metric("Active Places", len(df_places[df_places["Status"] == "Active"]))
    chats = df_filtered["Chats"].sum()
    res   = df_filtered["Resolved_Chats"].sum()
    k8.metric("Bot Resolution", f"{(res/chats*100):.1f}%" if chats > 0 else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row A: heatmap + signup velocity
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("⏰ Platform Visiting Hours")
        hours     = [f"{i}:00" for i in range(24)]
        days      = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        heat_data = np.random.randint(10, 100, size=(7, 24))
        heat_data[:, 18:22] += 60
        fig_heat = px.imshow(heat_data, x=hours, y=days,
                             color_continuous_scale=[[0,"#E8EFF5"],[0.5,"#619FB8"],[1,"#1D3143"]],
                             aspect="auto", template=TEMPLATE)
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_b:
        st.subheader("🚀 Signup Velocity")
        fig_v = px.bar(df_filtered, x="Date", y=["New_Users","New_Owners"],
                       barmode="group",
                       color_discrete_sequence=["#2F5C85","#61A3BB"],
                       template=TEMPLATE)
        st.plotly_chart(fig_v, use_container_width=True)

    # ✅ BUG FIX: col_c paired with col_d — no longer alone in full row
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("🛡️ Place Status Distribution")
        fig_st = px.pie(df_places, names="Status", hole=0.6,
                        color_discrete_sequence=["#2F5C85","#61A3BB","#65797E"],
                        template=TEMPLATE)
        st.plotly_chart(fig_st, use_container_width=True)

    with col_d:
        st.subheader("📈 Current vs Previous Period")
        if not df_prev.empty:
            metrics   = ["Visits","Reviews","Saves","Chats"]
            curr_vals = [df_filtered[m].sum() for m in metrics]
            prev_vals = [df_prev[m].sum()     for m in metrics]
            gdf = pd.DataFrame({
                "Metric": metrics * 2,
                "Value":  curr_vals + prev_vals,
                "Period": ["Current"] * 4 + ["Previous"] * 4,
            })
            fig_g = px.bar(gdf, x="Metric", y="Value", color="Period",
                           barmode="group", text_auto=".2s",
                           color_discrete_map={
                               "Current":  "#2F5C85",
                               "Previous": "#C8D9E6",
                           },
                           template=TEMPLATE)
            st.plotly_chart(fig_g, use_container_width=True)
        else:
            empty_state("No previous period available for comparison.")


# ═══════════════════════════════════════════════════════════
# 2. PLACES ANALYTICS
# ═══════════════════════════════════════════════════════════
elif selected == "Places Analytics":
    st.title("🏘️ Places Analytics")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Total Places",     len(df_places))
    p2.metric("Active",           len(df_places[df_places["Status"] == "Active"]))
    p3.metric("Pending Approval", len(df_places[df_places["Status"] == "Pending Approval"]))
    p4.metric("Suspended",        len(df_places[df_places["Status"] == "Suspended"]))

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 Places per Category")
        cat_counts = df_places["Category"].value_counts().reset_index()
        cat_counts.columns = ["Category","Count"]
        fig_cc = px.bar(cat_counts, x="Category", y="Count",
                        color="Count", color_continuous_scale=[[0,"#65797E"],[0.5,"#619FB8"],[1,"#1D3143"]],
                        text_auto=True, template=TEMPLATE)
        st.plotly_chart(fig_cc, use_container_width=True)

    with c2:
        st.subheader("🏆 Most Visited Places (Top 8)")
        top_v = df_places.nlargest(8, "Visits")
        fig_tv = px.bar(top_v, x="Visits", y="Name", orientation="h",
                        color="Visits",color_continuous_scale=[[0,"#65797E"],[0.5,"#619FB8"],[1,"#1D3143"]],
                        template=TEMPLATE)
        fig_tv.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig_tv, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("❤️ Most Saved Places (Top 8)")
        top_s = df_places.nlargest(8, "Saves")
        fig_ts = px.bar(top_s, x="Saves", y="Name", orientation="h",
                        color="Saves", color_continuous_scale=[[0,"#D4E8EE"],[1,"#315687"]],
                        template=TEMPLATE)
        fig_ts.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig_ts, use_container_width=True)

    with c4:
        st.subheader("⭐ Average Rating per Category")
        avg_c = df_places.groupby("Category")["Rating"].mean().reset_index()
        avg_c["Rating"] = avg_c["Rating"].round(2)
        fig_rc = px.bar(avg_c, x="Category", y="Rating",
                        color="Rating",color_continuous_scale=[[0,"#65797E"],[0.5,"#619FB8"],[1,"#1D3143"]],
                        text_auto=".2f", template=TEMPLATE)
        fig_rc.update_layout(yaxis_range=[0, 5])
        st.plotly_chart(fig_rc, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("🌟 Highest Rated Places")
        st.dataframe(
            df_places.nlargest(8, "Rating")[["Name","Category","District","Rating","Visits"]]
            .reset_index(drop=True),
            use_container_width=True,
        )
    with c6:
        st.subheader("⚠️ Lowest Rated (Needs Attention)")
        st.dataframe(
            df_places.nsmallest(8, "Rating")[["Name","Category","District","Rating","Visits"]]
            .reset_index(drop=True),
            use_container_width=True,
        )

    st.subheader("🆕 Recently Added Places")
    recent = df_places.nlargest(10, "Added").copy()
    recent["Added"] = recent["Added"].astype(str).str[:10]
    st.dataframe(
        recent[["Name","Category","District","Status","Added","Rating"]].reset_index(drop=True),
        use_container_width=True,
    )

    # ✅ NEW: Search & filter
    st.subheader("🔍 Search & Filter All Places")
    cs1, cs2, cs3 = st.columns(3)
    with cs1:
        srch = st.text_input("Search by name...")
    with cs2:
        cat_f = st.selectbox("Category", ["All"] + sorted(df_places["Category"].unique().tolist()))
    with cs3:
        sta_f = st.selectbox("Status", ["All","Active","Suspended","Pending Approval"])

    fp = df_places.copy()
    if srch:
        fp = fp[fp["Name"].str.contains(srch, case=False)]
    if cat_f != "All":
        fp = fp[fp["Category"] == cat_f]
    if sta_f != "All":
        fp = fp[fp["Status"] == sta_f]

    st.caption(f"Showing {len(fp)} places")
    st.dataframe(
        fp[["Place_ID","Name","Category","District","Rating","Visits","Saves","Status"]]
        .reset_index(drop=True),
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════
# 3. USER ANALYTICS  (expanded)
# ═══════════════════════════════════════════════════════════
elif selected == "User Analytics":
    st.title("👥 User Analytics")

    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Total Users",         len(df_users))
    u2.metric("New Users (Period)",  df_filtered["New_Users"].sum(), get_delta("New_Users"))
    u3.metric("Suspended Users",     len(df_users[df_users["Status"] == "Suspended"]))
    u4.metric("New Owners (Period)", df_filtered["New_Owners"].sum(), get_delta("New_Owners"))

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 User Growth Trend")
        fig_ug = px.area(df_filtered, x="Date", y="New_Users",
                         color_discrete_sequence=["#2F5C85"], template=TEMPLATE)
        st.plotly_chart(fig_ug, use_container_width=True)

    with col2:
        # ✅ NEW: users by district
        st.subheader("📍 Users by District")
        dist_u = df_users["District"].value_counts().reset_index()
        dist_u.columns = ["District","Users"]
        fig_du = px.bar(dist_u, x="Users", y="District", orientation="h",
                        color="Users", color_continuous_scale=[[0,"#E8EFF5"],[1,"#2F5C85"]],
                        template=TEMPLATE)
        fig_du.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig_du, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        # ✅ NEW: most active users
        st.subheader("🏅 Most Active Users (by Reviews)")
        top_u = df_users.nlargest(10, "Reviews")[
            ["User_ID","Name","District","Reviews","Saves","Status"]
        ]
        st.dataframe(top_u.reset_index(drop=True), use_container_width=True)

    with col4:
        # ✅ NEW: retention pie
        st.subheader("🔄 User Retention")
        ret_df = pd.DataFrame({
            "Type":  ["Returning","New"],
            "Count": [int(df_filtered["New_Users"].sum() * 0.60),
                      int(df_filtered["New_Users"].sum() * 0.40)],
        })
        fig_ret = px.pie(ret_df, values="Count", names="Type", hole=0.55,
                         color_discrete_sequence=["#2F5C85","#61A3BB"],                         template=TEMPLATE)
        st.plotly_chart(fig_ret, use_container_width=True)

    # ✅ NEW: search users
    st.subheader("🔍 Search Users")
    su1, su2 = st.columns(2)
    with su1:
        u_srch = st.text_input("Search by name or ID...")
    with su2:
        u_sta  = st.selectbox("Filter by status", ["All","Active","Suspended"])

    fu = df_users.copy()
    if u_srch:
        fu = fu[
            fu["Name"].str.contains(u_srch, case=False) |
            fu["User_ID"].str.contains(u_srch, case=False)
        ]
    if u_sta != "All":
        fu = fu[fu["Status"] == u_sta]

    st.caption(f"Showing {len(fu)} users")
    st.dataframe(
        fu[["User_ID","Name","District","Reviews","Saves","Status","Joined","Last_Login"]]
        .reset_index(drop=True),
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════
# 4. REVIEWS  (expanded)
# ═══════════════════════════════════════════════════════════
elif selected == "Reviews":
    st.title("⭐ Reviews & Sentiment Analysis")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total Reviews (Period)", f"{df_filtered['Reviews'].sum():,}", get_delta("Reviews"))
    r2.metric("Positive Sentiment",     "75%")
    r3.metric("Negative Sentiment",     "25%")
    r4.metric("Flagged Reviews",        len(flagged_reviews))

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("😊 Positive vs Negative Ratio")
        fig_pie = px.pie(values=[75, 25], names=["Positive","Negative"],
                         hole=0.55,
                         color_discrete_sequence=["#2F5C85","#65797E"],
                         template=TEMPLATE)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # ✅ NEW: reviews over time
        st.subheader("📅 Reviews Over Time")
        fig_rt = px.area(df_filtered, x="Date", y="Reviews",
                         color_discrete_sequence=["#619FB8"], template=TEMPLATE)
        st.plotly_chart(fig_rt, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("⭐ Average Rating per Category")
        avg_c = df_places.groupby("Category")["Rating"].mean().reset_index()
        avg_c["Rating"] = avg_c["Rating"].round(2)
        fig_rc = px.bar(avg_c, x="Category", y="Rating",
                        color="Rating", color_continuous_scale=[[0,"#65797E"],[0.5,"#619FB8"],[1,"#1D3143"]],
                        text_auto=".2f", template=TEMPLATE)
        fig_rc.update_layout(yaxis_range=[0, 5])
        st.plotly_chart(fig_rc, use_container_width=True)

    with col4:
        # ✅ NEW: most reviewed places
        st.subheader("🏆 Most Reviewed Places (Top 8)")
        top_rev = df_places.nlargest(8, "Reviews")
        fig_mr  = px.bar(top_rev, x="Reviews", y="Name", orientation="h",
                         color="Reviews", color_continuous_scale=[[0,"#D4E8EE"],[1,"#315687"]],
                         template=TEMPLATE)
        fig_mr.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig_mr, use_container_width=True)

    # ✅ NEW: places with no reviews
    st.subheader("😶 Places With No Reviews Yet")
    no_rev = df_places[df_places["Reviews"] == 0][
        ["Place_ID","Name","Category","District","Status"]
    ]
    if no_rev.empty:
        st.success("All places have at least one review.")
    else:
        st.caption(f"{len(no_rev)} places have no reviews")
        st.dataframe(no_rev.reset_index(drop=True), use_container_width=True)

    # ✅ NEW: flagged reviews table
    st.subheader("🚩 Flagged Reviews")
    st.caption("Reviews reported by users — admin action required")
    st.dataframe(
        flagged_reviews[["Review_ID","User","Place","Review","Rating","Date"]]
        .reset_index(drop=True),
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════
# 5. CHATBOT  (fixed + expanded)
# ═══════════════════════════════════════════════════════════
elif selected == "Chatbot":
    st.title("🤖 Chatbot Intelligence")

    # ✅ BUG FIX: was 'Chat_Sessions' — column is named 'Chats'
    total_chats = df_filtered["Chats"].sum()
    resolved    = df_filtered["Resolved_Chats"].sum()
    res_pct     = (resolved / total_chats * 100) if total_chats > 0 else 0

    ch1, ch2, ch3, ch4 = st.columns(4)
    ch1.metric("Total Chat Sessions", f"{total_chats:,}",  get_delta("Chats"))
    ch2.metric("Resolved Chats",      f"{resolved:,}",     get_delta("Resolved_Chats"))
    ch3.metric("Avg Resolution Rate", f"{res_pct:.1f}%")
    ch4.metric("Unresolved Queries",  f"{total_chats - resolved:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        # ✅ NEW: chat volume over time
        st.subheader("📅 Chat Volume Over Time")
        fig_cv = px.area(df_filtered, x="Date", y="Chats",
                         color_discrete_sequence=["#315687"], template=TEMPLATE)
        st.plotly_chart(fig_cv, use_container_width=True)

    with col2:
        st.subheader("🔍 Query Types Distribution")
        fig_qt = px.pie(chat_types, values="Val", names="Type", hole=0.55,
                        color_discrete_sequence=["#2F5C85","#619FB8","#315687","#65797E","#1D3143"],
                        template=TEMPLATE)
        st.plotly_chart(fig_qt, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        # ✅ NEW: most active chatbot places
        st.subheader("🏆 Most Active Chatbot Places")
        fig_tcp = px.bar(top_chat_places.head(8), x="Chats", y="Place",
                         orientation="h", color="Chats",
                         color_continuous_scale=[[0,"#D8E8F0"],[1,"#1D3143"]], template=TEMPLATE)
        fig_tcp.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig_tcp, use_container_width=True)

    with col4:
        # ✅ NEW: resolved vs unresolved per place
        st.subheader("✅ Resolved vs Unresolved per Place")
        tcp = top_chat_places.head(6).copy()
        tcp["Unresolved"] = tcp["Chats"] - tcp["Resolved"]
        tcp_m = tcp.melt(id_vars="Place", value_vars=["Resolved","Unresolved"],
                         var_name="Status", value_name="Count")
        fig_ru = px.bar(tcp_m, x="Place", y="Count", color="Status",
                        barmode="stack",
                        color_discrete_map={"Resolved":"#2F5C85","Unresolved":"#65797E"},
                        template=TEMPLATE)
        fig_ru.update_xaxes(tickangle=30)
        st.plotly_chart(fig_ru, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# 6. CATEGORY ANALYTICS  ✅ ENTIRELY NEW SECTION
# ═══════════════════════════════════════════════════════════
elif selected == "Category Analytics":
    st.title("🏷️ Category Analytics")

    ca1, ca2, ca3, ca4 = st.columns(4)
    ca1.metric("Total Categories",      len(df_places["Category"].unique()))
    ca2.metric("Most Places In",        df_places["Category"].value_counts().idxmax())
    ca3.metric("Most Visited Category", df_places.groupby("Category")["Visits"].sum().idxmax())
    ca4.metric("Most Saved Category",   df_places.groupby("Category")["Saves"].sum().idxmax())

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Places per Category")
        cc = df_places["Category"].value_counts().reset_index()
        cc.columns = ["Category","Count"]
        fig_cc = px.bar(cc, x="Category", y="Count",
                        color="Count", color_continuous_scale=[[0,"#E8EFF5"],[1,"#2F5C85"]],
                        text_auto=True, template=TEMPLATE)
        st.plotly_chart(fig_cc, use_container_width=True)

    with col2:
        st.subheader("👁️ Total Visits per Category")
        cv = df_places.groupby("Category")["Visits"].sum().reset_index()
        fig_cv = px.bar(cv, x="Category", y="Visits",
                        color="Visits", color_continuous_scale=[[0,"#D4E8EE"],[1,"#315687"]],
                        text_auto=".2s", template=TEMPLATE)
        st.plotly_chart(fig_cv, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("❤️ Total Saves per Category")
        cs = df_places.groupby("Category")["Saves"].sum().reset_index()
        fig_cs = px.pie(cs, values="Saves", names="Category", hole=0.5,
                       color_discrete_sequence=["#1D3143","#2F5C85","#619FB8","#61A3BB","#65797E"],
                        template=TEMPLATE)
        st.plotly_chart(fig_cs, use_container_width=True)

    with col4:
        st.subheader("⭐ Average Rating per Category")
        ar = df_places.groupby("Category")["Rating"].mean().reset_index()
        ar["Rating"] = ar["Rating"].round(2)
        fig_ar = px.bar(ar, x="Category", y="Rating",
                        color="Rating", color_continuous_scale=[[0,"#65797E"],[0.5,"#619FB8"],[1,"#1D3143"]],
                        text_auto=".2f", template=TEMPLATE)
        fig_ar.update_layout(yaxis_range=[0, 5])
        st.plotly_chart(fig_ar, use_container_width=True)

    st.subheader("📋 Category Summary Table")
    cat_summary = df_places.groupby("Category").agg(
        Places       =("Place_ID", "count"),
        Total_Visits =("Visits",   "sum"),
        Total_Saves  =("Saves",    "sum"),
        Avg_Rating   =("Rating",   "mean"),
        Total_Reviews=("Reviews",  "sum"),
    ).reset_index()
    cat_summary["Avg_Rating"] = cat_summary["Avg_Rating"].round(2)
    st.dataframe(cat_summary, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# 7. MODERATION  (fully rebuilt)
# ═══════════════════════════════════════════════════════════
elif selected == "Moderation":
    st.title("🛡️ Moderation Center")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flagged Reviews",    len(flagged_reviews))
    m2.metric("Pending Owners",     len(pending_owners))
    m3.metric("Suspended Users",    len(df_users[df_users["Status"] == "Suspended"]))
    m4.metric("Suspended Places",   len(df_places[df_places["Status"] == "Suspended"]))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Flagged reviews — hidden until "View All" pressed ────
    rev_header_col, rev_btn_col = st.columns([4, 1])
    with rev_header_col:
        st.subheader("🚩 Flagged Reviews — Pending Action")
        st.caption("Reported by users · Admin decision required")
    with rev_btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            f"👁️ View All ({len(flagged_reviews)})",
            key="view_all_reviews",
            use_container_width=True,
        ):
            st.session_state.show_all_reviews = not st.session_state.get("show_all_reviews", False)

    if not st.session_state.get("show_all_reviews", False):
        st.markdown(
            f"""
            <div style="background:rgba(245,158,11,0.10);border-left:4px solid #F59E0B;
                        border-radius:8px;padding:14px 18px;color:#92400E;">
                <b>⚠️ {len(flagged_reviews)} reviews</b> are flagged and waiting for your review.
                Press <b>View All</b> to inspect and take action.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info(f"Showing all {len(flagged_reviews)} flagged reviews")
        for _, row in flagged_reviews.iterrows():
            col_info, col_act = st.columns([4, 1])
            with col_info:
                st.markdown(
                    f"**{row['Review_ID']}** &nbsp;·&nbsp; 📍 {row['Place']} "
                    f"&nbsp;·&nbsp; ⭐ {row['Rating']} &nbsp;·&nbsp; 👤 {row['User']}"
                )
                st.caption(f"_{row['Review']}_")
            with col_act:
                st.button("🗑️ Delete", key=f"del_{row['Review_ID']}")
                st.button("✅ Keep",   key=f"keep_{row['Review_ID']}")
            st.divider()

    # ── Pending owners — hidden until "View All" pressed ─────
    own_header_col, own_btn_col = st.columns([4, 1])
    with own_header_col:
        st.subheader("👤 New Owners Pending Verification")
        st.caption("Submitted accounts awaiting admin approval")
    with own_btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            f"👁️ View All ({len(pending_owners)})",
            key="view_all_owners",
            use_container_width=True,
        ):
            st.session_state.show_all_owners = not st.session_state.get("show_all_owners", False)

    if not st.session_state.get("show_all_owners", False):
        st.markdown(
            f"""
            <div style="background:rgba(47,92,133,0.08);border-left:4px solid #2F5C85;
                        border-radius:8px;padding:14px 18px;color:#1D3143;">
                <b>📋 {len(pending_owners)} owner requests</b> are pending verification.
                Press <b>View All</b> to review and approve or reject them.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info(f"Showing all {len(pending_owners)} pending owner requests")
        for _, row in pending_owners.iterrows():
            col_info, col_act = st.columns([4, 1])
            with col_info:
                st.markdown(
                    f"**{row['Owner_ID']}** &nbsp;·&nbsp; {row['Name']} "
                    f"&nbsp;·&nbsp; {row['Business']} &nbsp;·&nbsp; 🏷️ {row['Category']}"
                )
                st.caption(f"Submitted: {row['Submitted']}")
            with col_act:
                st.button("✅ Approve", key=f"apr_{row['Owner_ID']}")
                st.button("❌ Reject",  key=f"rej_{row['Owner_ID']}")
            st.divider()

    # ── Suspended users ──────────────────────────────────────
    st.subheader("🚫 Suspended Users")
    sus_u = df_users[df_users["Status"] == "Suspended"][
        ["User_ID","Name","District","Reviews","Joined"]
    ].head(10).reset_index(drop=True)
    st.dataframe(sus_u, use_container_width=True)

    # ── Suspended places ─────────────────────────────────────
    st.subheader("🏚️ Suspended Places")
    sus_p = df_places[df_places["Status"] == "Suspended"][
        ["Place_ID","Name","Category","District","Rating"]
    ].head(10).reset_index(drop=True)
    st.dataframe(sus_p, use_container_width=True)

    # ── Admin action history ─────────────────────────────────
    st.subheader("📋 Admin Action History")
    st.table(admin_log)



# ═══════════════════════════════════════════════════════════
# 8. ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════
elif selected == "Anomaly Detection":
    st.title("🚨 Anomaly Detection")

    # ── Fetch anomalies from API ──────────────────────────────────
    @st.cache_data(ttl=120)
    def fetch_anomalies():
        """Calls POST /detect on the anomaly detection API."""
        try:
            np.random.seed(99)
            districts = [
                (29.0661, 31.0994, 0, "City Center"),
                (29.0300, 31.0600, 1, "Biba"),
                (29.0800, 31.1200, 2, "New Beni Suef"),
                (29.0670, 31.0920, 3, "South District"),
                (29.0750, 31.1050, 4, "University Area"),
                (29.0580, 31.0850, 5, "Nile Corniche"),
            ]
            visits = []
            base = datetime(2026, 1, 1, 8, 0, 0)
            for _ in range(300):
                lat, lon, cluster, district = districts[np.random.randint(0, 6)]
                visits.append({
                    "user_id":    int(np.random.randint(1, 200)),
                    "place_id":   int(np.random.randint(1, 50)),
                    "user_lat":   round(lat + np.random.normal(0, 0.005), 6),
                    "user_lon":   round(lon + np.random.normal(0, 0.005), 6),
                    "visited_at": str(base + timedelta(
                        days=int(np.random.randint(0, 60)),
                        hours=int(np.random.randint(8, 23)),
                        minutes=int(np.random.randint(0, 60)))),
                    "cluster":    cluster,
                })
            # Inject bot
            bot_base = datetime(2026, 2, 20, 14, 0, 0)
            for i in range(15):
                visits.append({
                    "user_id": 999, "place_id": int(np.random.randint(1, 50)),
                    "user_lat": 29.0661, "user_lon": 31.0994,
                    "visited_at": str(bot_base + timedelta(seconds=i*20)),
                    "cluster": 0,
                })
            # Inject traffic spike
            spike_base = datetime(2026, 2, 15, 19, 0, 0)
            for i in range(30):
                visits.append({
                    "user_id": int(np.random.randint(200, 400)), "place_id": 5,
                    "user_lat": 29.0661, "user_lon": 31.0994,
                    "visited_at": str(spike_base + timedelta(minutes=int(np.random.randint(0, 60)))),
                    "cluster": 0,
                })

            resp = requests.post(
                f"{ANOMALY_API}/detect",
                json={"visits": visits},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["anomalies"], data["total_anomalies"]
        except Exception:
            pass
        return [], 0

    @st.cache_data(ttl=120)
    def fetch_anomaly_summary(anomalies):
        """Calls POST /summary on the anomaly detection API."""
        try:
            if not anomalies:
                return []
            resp = requests.post(
                f"{ANOMALY_API}/summary",
                json={"anomalies": anomalies},
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()["summary"]
        except Exception:
            pass
        return []

    with st.spinner("Fetching anomaly data..."):
        anomalies, total_anomalies = fetch_anomalies()

    summary = fetch_anomaly_summary(anomalies) if anomalies else []

    # ── KPI cards ─────────────────────────────────────────────────
    high_count   = len([a for a in anomalies if a["severity"] == "High"])
    medium_count = len([a for a in anomalies if a["severity"] == "Medium"])
    top_district = summary[0]["cluster"] if summary else "N/A"

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Anomalies",     total_anomalies)
    a2.metric("🔴 High Severity",    high_count)
    a3.metric("🟡 Medium Severity",  medium_count)
    a4.metric("Most Affected Cluster", f"Cluster {top_district}" if summary else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)

    if not anomalies:
        st.info("ℹ️ No anomalies detected or API unavailable.")
    else:
        col1, col2 = st.columns(2)

        # ── Anomaly by Type chart ─────────────────────────────────
        with col1:
            st.subheader("📊 Anomalies by Type")
            type_counts = pd.DataFrame(anomalies)["anomaly_type"].value_counts().reset_index()
            type_counts.columns = ["Type", "Count"]
            fig_at = px.bar(type_counts, x="Count", y="Type", orientation="h",
                            color="Count", color_continuous_scale=[[0,"#C8D9E6"],[1,"#1D3143"]],
                            text_auto=True, template=TEMPLATE)
            fig_at.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_at, use_container_width=True)

        # ── Severity Distribution ─────────────────────────────────
        with col2:
            st.subheader("🔴 Severity Distribution")
            sev_counts = pd.DataFrame(anomalies)["severity"].value_counts().reset_index()
            sev_counts.columns = ["Severity", "Count"]
            fig_sev = px.pie(sev_counts, values="Count", names="Severity", hole=0.55,
                color_discrete_sequence=["#1D3143", "#619FB8"],
                template=TEMPLATE)
            st.plotly_chart(fig_sev, use_container_width=True)

        # ── Anomaly Heatmap per Cluster ───────────────────────────
        if summary:
            st.subheader("🗺️ Anomaly Activity per Cluster")
            sum_df = pd.DataFrame(summary)
            fig_cl = px.bar(sum_df, x="cluster", y="total_anomalies",
                            color="total_anomalies", color_continuous_scale=[[0,"#C8D9E6"],[1,"#1D3143"]],
                            text_auto=True, template=TEMPLATE,
                            labels={"cluster": "Cluster", "total_anomalies": "Total Anomalies"})
            fig_cl.update_xaxes(type="category")
            st.plotly_chart(fig_cl, use_container_width=True)

        # ── Live Anomaly Feed ─────────────────────────────────────
        st.subheader("📋 Live Anomaly Feed")
        st.caption(f"Showing {len(anomalies)} flagged anomalies — sorted by score")

        for a in anomalies:
            col_info, col_act = st.columns([5, 1])
            severity_badge = "🔴" if a["severity"] == "High" else "🟡"
            with col_info:
                st.markdown(
                    f"{severity_badge} **{a['anomaly_type'].replace('_', ' ').title()}** "
                    f"&nbsp;·&nbsp; Score: `{a['score']}` "
                    f"&nbsp;·&nbsp; Place: `{a['place_id']}` "
                    f"&nbsp;·&nbsp; User: `{a['user_id']}` "
                    f"&nbsp;·&nbsp; Cluster: `{a['cluster']}`"
                )
                st.caption(f"_{a['details']}_")
            with col_act:
                if a["severity"] == "High":
                    st.button("🚫 Suspend", key=f"sus_{a['place_id']}_{a['user_id']}_{a['anomaly_type']}")
                st.button("✅ Dismiss", key=f"dis_{a['place_id']}_{a['user_id']}_{a['anomaly_type']}")
            st.divider()


# ═══════════════════════════════════════════════════════════
# 9. LOCATION LOGIC
# ═══════════════════════════════════════════════════════════
elif selected == "Location Logic":
    st.title("📍 Location Intelligence: Beni Suef")

    BS_LAT, BS_LON = 29.0661, 31.0994

    # ── Fetch real hotspots from clustering API ───────────────────
    @st.cache_data(ttl=300)
    def fetch_heatmap():
        """Calls POST /heatmap on the clustering API with sample visits."""
        try:
            sample_visits = []
            districts = [
                (29.0661, 31.0994, 0), (29.0300, 31.0600, 1),
                (29.0800, 31.1200, 2), (29.0670, 31.0920, 3),
                (29.0750, 31.1050, 4), (29.0580, 31.0850, 5),
            ]
            np.random.seed(42)
            for lat, lon, cluster in districts:
                n = np.random.randint(30, 80)
                for _ in range(n):
                    sample_visits.append({
                        "lat": round(lat + np.random.normal(0, 0.005), 6),
                        "lon": round(lon + np.random.normal(0, 0.005), 6),
                        "cluster": cluster,
                        "district": ["City Center","Biba","New Beni Suef",
                                     "South District","University Area","Nile Corniche"][cluster],
                    })
            resp = requests.post(
                f"{CLUSTERING_API}/heatmap",
                json={"visits": sample_visits},
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()["hotspots"]
        except Exception:
            pass
        return None

    @st.cache_data(ttl=300)
    def fetch_opportunities():
        """Calls POST /opportunities on the clustering API."""
        try:
            np.random.seed(42)
            districts_list = ["City Center","Biba","New Beni Suef",
                               "South District","University Area","Nile Corniche"]
            cats = ["Restaurant","Cafe","Pharmacy","House","Super Market"]
            visits = []
            for i, d in enumerate(districts_list):
                n = np.random.randint(60, 400)
                for _ in range(n):
                    visits.append({"cluster": i, "district": d})
            places = []
            for d in districts_list[:3]:
                for c in cats[:3]:
                    places.append({"category": c, "district": d})
            resp = requests.post(
                f"{CLUSTERING_API}/opportunities",
                json={"visits": visits, "places": places},
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()["opportunities"]
        except Exception:
            pass
        return None

    hotspots      = fetch_heatmap()
    opportunities = fetch_opportunities()

    # ── Heatmap ───────────────────────────────────────────────────
    if hotspots:
        map_data = pd.DataFrame(hotspots)
        map_data.rename(columns={"lon": "lon", "lat": "lat", "intensity": "intensity"}, inplace=True)
        st.caption("🟢 Live data from Location Clustering API")
    else:
        st.caption("⚠️ Using fallback data — API unavailable")
        map_data = pd.DataFrame({
            "lat":       np.random.uniform(BS_LAT - 0.015, BS_LAT + 0.015, 300),
            "lon":       np.random.uniform(BS_LON - 0.015, BS_LON + 0.015, 300),
            "intensity": np.random.randint(1, 100, 300),
        })

    fig_map = px.density_mapbox(
        map_data, lat="lat", lon="lon", z="intensity", radius=15,
        center=dict(lat=BS_LAT, lon=BS_LON), zoom=12.5,
        mapbox_style="open-street-map", height=550,
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🗺️ Most Active Districts")
        dc = df_places["District"].value_counts().reset_index()
        dc.columns = ["District","Places"]
        fig_dc = px.bar(dc, x="Places", y="District", orientation="h",
                        color="Places", color_continuous_scale=[[0,"#E8EFF5"],[1,"#2F5C85"]],
                        template=TEMPLATE)
        fig_dc.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig_dc, use_container_width=True)

    with col2:
        st.subheader("🏷️ Places per District per Category")
        dcat = df_places.groupby(["District","Category"]).size().reset_index(name="Count")
        fig_dcat = px.bar(dcat, x="District", y="Count", color="Category",
                  barmode="stack",
                  color_discrete_sequence=["#1D3143","#2F5C85","#619FB8","#61A3BB","#65797E"],
                  template=TEMPLATE)
        fig_dcat.update_xaxes(tickangle=30)
        st.plotly_chart(fig_dcat, use_container_width=True)

    # ── Opportunity Map — real API data ───────────────────────────
    st.subheader("💡 Opportunity Map")
    if opportunities:
        st.caption("🟢 Live data from Location Clustering API")
        for opp in opportunities[:6]:
            urgency = opp["urgency"]
            msg     = opp["message"]
            if urgency == "High":
                st.error(f"🔴 {msg}")
            elif urgency == "Medium":
                st.warning(f"🟡 {msg}")
            else:
                st.info(f"🔵 {msg}")
    else:
        st.caption("⚠️ Using fallback data — API unavailable")
        st.success("📍 'New Beni Suef' has high search volume for Pharmacies but 0 registered.")
        st.info   ("📍 'Nile Corniche' has the highest concentration of Direction Clicks.")
        st.warning("📍 'El Wasta' has only 1 Cafe — potential market gap.")
