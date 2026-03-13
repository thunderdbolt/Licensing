import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta
from flexlog_parser import parse_flex_debug_log

st.set_page_config(page_title="License Log Review", layout="wide", page_icon="📊")
st.title("License Log Review (FlexNet debug log)")

# -----------------------------
# 1. Configuration & Definitions
# -----------------------------
PIPESIM_FEATURES = {
    "BP", "PIPESIM_LINUX", "ledaflowpm_2phase", "ledaflowpm_3phase",
    "psim compositional", "PVTToolbox_Advanced_Gas", "PVTToolbox_Base",
    "NETENGINE", "PIPESIM-NET", "PIPESIM Adv Well", "PIPESYS", "sym fe fa",
    "Shell", "tpa corr", "PIPESIM", "PIPESIM Python Toolkit", "PSIMENGINE",
}
OLGA_FEATURES = {
    "compmodule", "etohtracking", "megtracking", "meohtracking", "mfCTdll",
    "single_co2", "single_h2o", "single_other", "steam", "tracertracking",
    "complexfluid", "pump", "plugin", "tuning", "waxdeposition", "batch",
    "geometry", "gui", "hidef", "MEPO_RMO", "olga", "OLGA_fluids",
    "OLGA_XLS", "profgen", "server", "spt_mt_hdpm", "spt_mt_olgas_2p",
    "spt_mt_olgas_3p", "spt_mt_OLGAS-3PHASE", "water", "olga_co2_reach",
    "olga_co2_tide", "mfctdll", "OLGA_comptrack", "olga_neo_solver",
    "olga_new_enegry", "well", "corrosion", "ifecorrmod", "olga_eqn_tca",
    "bundle", "femtherm", "femthermtool", "femthermviewer", "soil",
    "hydratekinetics", "parallel_MR", "OVIP-OLGA", "OVIP-Tools", "rocx",
    "slugtracking", "olga_ovip", "well-extended", "wellsgui", "olga_wax",
}
PIPESIM_FEATURES_L = {f.strip().lower() for f in PIPESIM_FEATURES}
OLGA_FEATURES_L = {f.strip().lower() for f in OLGA_FEATURES}

def feature_in_software(feature: str, software: str) -> bool:
    f = (feature or "").strip().lower()
    if software == "PIPESIM":
        return f in PIPESIM_FEATURES_L
    if software == "OLGA":
        return f in OLGA_FEATURES_L
    if software == "SYMMETRY":
        return f.startswith("sym") or f.startswith("vmg")
    return False

@st.cache_data(show_spinner=True)
def parse_log_bytes(file_bytes: bytes):
    text = file_bytes.decode("utf-8", errors="replace")
    parsed = parse_flex_debug_log(text.splitlines())
    events = pd.DataFrame(parsed["events"])
    reservations = pd.DataFrame(parsed["reservations"])
    max_rules = pd.DataFrame(parsed["max_rules"])
    group_rules = pd.DataFrame(parsed["group_rules"])
    
    for df in (events, reservations, max_rules, group_rules):
        if "timestamp" in df.columns and not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
    if not events.empty and "timestamp" in events.columns:
        events["day"] = events["timestamp"].dt.date
        # NEW: Add hour and weekday for Heatmaps
        events["hour"] = events["timestamp"].dt.hour
        events["weekday"] = events["timestamp"].dt.day_name()
        
    if "reason" in events.columns:
        events["reason"] = events["reason"].replace({None: pd.NA})
        
    return events, reservations, max_rules, group_rules, parsed.get("meta", {})

# NEW: Helper for Concurrency
def calculate_concurrency(df):
    """Calculates concurrent usage based on OUT (+1) and IN (-1) events."""
    usage = df[df["event"].isin(["OUT", "IN"])].copy()
    if usage.empty:
        return pd.DataFrame()
    usage["change"] = usage["event"].map({"OUT": 1, "IN": -1})
    usage = usage.sort_values("timestamp")
    # Cumulative sum per feature
    usage["concurrency"] = usage.groupby("feature")["change"].cumsum()
    # Handle negative concurrency (artifacts from missing start of log)
    usage["concurrency"] = usage.groupby("feature")["concurrency"].transform(lambda x: x - x.min())
    return usage

# -----------------------------
# 2. Upload & Parsing
# -----------------------------
uploaded = st.file_uploader(
    "Upload a debug log (.log / .txt). Year-long files are OK.",
    type=["log", "txt"],
    key="uploader",
)
if not uploaded:
    st.info("Upload a log file to begin.")
    st.stop()

if "log_bytes" not in st.session_state or st.session_state.get("log_name") != uploaded.name:
    st.session_state["log_bytes"] = uploaded.getvalue()
    st.session_state["log_name"] = uploaded.name

events, reservations, max_rules, group_rules, meta = parse_log_bytes(st.session_state["log_bytes"])

st.sidebar.header("Filters")
if events.empty or "timestamp" not in events.columns:
    st.warning("No IN/OUT/DENIED/UNSUPPORTED events detected (with current patterns).")
    st.stop()

# -----------------------------
# 3. Sidebar Filters (Original)
# -----------------------------
def _reset_filters():
    for k in [
        "view_mode", "day_sel", "date_range", "software_sel",
        "include_unsupported", "event_sel",
        "feature_sel", "slb_feature_sel", "client_sel", "y_axis",
        "denied_reason_sel"
    ]:
        if k in st.session_state:
            del st.session_state[k]

st.sidebar.button("Reset filters", on_click=_reset_filters)

# Time window
day_min = events["day"].min()
day_max = events["day"].max()
view_mode = st.sidebar.radio(
    "Time window",
    ["Single day", "Date range"],
    index=0,
    key="view_mode",
)
if view_mode == "Single day":
    day_sel = st.sidebar.date_input(
        "Day",
        value=day_max,
        min_value=day_min,
        max_value=day_max,
        key="day_sel",
    )
    start_day, end_day = day_sel, day_sel
else:
    default_start = max(day_min, day_max - timedelta(days=6))
    start_day, end_day = st.sidebar.date_input(
        "Date range",
        value=(default_start, day_max),
        min_value=day_min,
        max_value=day_max,
        key="date_range",
    )
range_events = events[(events["day"] >= start_day) & (events["day"] <= end_day)].copy()

# Software filter
software_sel = st.sidebar.multiselect(
    "Software (optional)",
    options=["PIPESIM", "OLGA", "SYMMETRY"],
    default=[],
    key="software_sel",
)
if software_sel:
    feats = range_events["feature"].astype(str)
    mask = pd.Series(False, index=range_events.index)
    for s in software_sel:
        mask = mask | feats.map(lambda x: feature_in_software(x, s))
    range_events = range_events[mask].copy()

# Event types
include_unsupported = st.sidebar.checkbox("Include UNSUPPORTED", value=False, key="include_unsupported")
all_event_types = sorted(range_events["event"].dropna().unique())
event_options = all_event_types if include_unsupported else [e for e in all_event_types if e != "UNSUPPORTED"]
prev_sel = st.session_state.get("event_sel", None)
if prev_sel is not None:
    pruned = [e for e in prev_sel if e in event_options]
    if not pruned and event_options:
        st.session_state["event_sel"] = event_options
event_sel = st.sidebar.multiselect(
    "Event type(s)",
    options=event_options,
    default=event_options,
    key="event_sel",
)

# Feature filters
all_features = pd.Series(range_events["feature"].dropna().unique())
slb_mask = all_features.astype(str).str.contains("slb", case=False, na=False)
slb_features = sorted(all_features[slb_mask].tolist())
normal_features = sorted(all_features[~slb_mask].tolist())
feature_sel = st.sidebar.multiselect(
    "Feature(s) (optional)",
    options=normal_features,
    default=[],
    key="feature_sel",
)
slb_feature_sel = st.sidebar.multiselect(
    "Slb Internal Feature(s) (optional)",
    options=slb_features,
    default=[],
    key="slb_feature_sel",
)

# DENIED reason filter
denied_reasons = sorted(
    range_events.loc[range_events["event"] == "DENIED", "reason"]
    .dropna()
    .astype(str)
    .unique()
)
denied_reason_sel = st.sidebar.multiselect(
    "DENIED reason (optional)",
    options=denied_reasons,
    default=[],
    key="denied_reason_sel",
)

# Client filter
clients = sorted(range_events["client"].dropna().unique())
client_sel = st.sidebar.multiselect(
    "Client/User token(s) (optional)",
    options=clients,
    default=[],
    key="client_sel",
)
y_axis = st.sidebar.radio("Timeline Y-axis", ["client", "feature"], index=0, key="y_axis")

# -----------------------------
# 4. Apply filters
# -----------------------------
filtered = range_events
if event_sel:
    filtered = filtered[filtered["event"].isin(event_sel)]
combined_features = []
if feature_sel:
    combined_features.extend(feature_sel)
if slb_feature_sel:
    combined_features.extend(slb_feature_sel)
if combined_features:
    filtered = filtered[filtered["feature"].isin(combined_features)]
if client_sel:
    filtered = filtered[filtered["client"].isin(client_sel)]
# Apply denied reason filter ONLY to DENIED events
if denied_reason_sel:
    denied_mask = (filtered["event"] == "DENIED")
    keep_denied = filtered.loc[denied_mask, "reason"].fillna("").astype(str).isin(denied_reason_sel)
    filtered = pd.concat([filtered.loc[~denied_mask], filtered.loc[denied_mask].loc[keep_denied]], axis=0)

# Downsample for plotting if huge
MAX_POINTS = 60000
plot_df = filtered.sort_values("timestamp")
if len(plot_df) > MAX_POINTS:
    plot_df = plot_df.sample(MAX_POINTS, random_state=1).sort_values("timestamp")
    st.info(f"Showing a random sample of {MAX_POINTS:,} events for plotting (filtered set has {len(filtered):,}).")

# -----------------------------
# 5. Executive Summary (NEW)
# -----------------------------
st.markdown("### Executive Summary")
total_events = len(filtered)
denial_events = filtered[filtered["event"] == "DENIED"]
total_denials = len(denial_events)
unique_users = filtered["client"].nunique()
denial_rate = (total_denials / total_events * 100) if total_events > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{total_events:,}")
col2.metric("Total Denials", f"{total_denials:,}", delta_color="inverse")
col3.metric("Denial Rate", f"{denial_rate:.2f}%")
col4.metric("Active Users", f"{unique_users}")
st.divider()

# -----------------------------
# 6. Tabs (Mixed Old & New)
# -----------------------------
tab1, tab_conc, tab_heat, tab2, tab3, tab4, tab5 = st.tabs([
    "Timeline", 
    "Concurrency (New)", 
    "Heatmap & Users (New)", 
    "Denied reasons", 
    "Reservations", 
    "MAX rules", 
    "Tables & meta"
])

# --- TAB 1: Timeline (Original) ---
with tab1:
    title = f"Timestamp plot — {start_day.isoformat()} → {end_day.isoformat()}" if view_mode == "Date range" else f"Timestamp plot — {start_day.isoformat()}"
    st.subheader(title)
    if plot_df.empty:
        st.info("No events match the current filters.")
    else:
        fig = px.scatter(
            plot_df,
            x="timestamp",
            y=y_axis,
            color="event",
            hover_data=["feature", "client", "reason", "version", "daemon"],
            render_mode="webgl",
        )
        st.plotly_chart(fig, use_container_width=True)
    if view_mode == "Date range" and not filtered.empty:
        st.subheader("Daily totals (filtered)")
        daily = (
            filtered.assign(day=pd.to_datetime(filtered["timestamp"]).dt.date)
            .groupby(["day", "event"])
            .size()
            .reset_index(name="count")
        )
        fig2 = px.bar(daily, x="day", y="count", color="event")
        st.plotly_chart(fig2, use_container_width=True)

# --- TAB: Concurrency (New) ---
with tab_conc:
    st.subheader("Concurrent License Usage (Estimated)")
    st.caption("Calculated by tracking cumulative OUT (+1) and IN (-1) events over time.")
    
    concurrency_df = calculate_concurrency(filtered)
    
    if not concurrency_df.empty:
        fig_conc = px.line(
            concurrency_df, 
            x="timestamp", 
            y="concurrency", 
            color="feature",
            title="Concurrent Usage by Feature",
            render_mode="webgl"
        )
        st.plotly_chart(fig_conc, use_container_width=True)
        
        # Max Concurrency Table
        st.subheader("Peak Concurrency per Feature")
        max_conc = concurrency_df.groupby("feature")["concurrency"].max().reset_index()
        max_conc.columns = ["Feature", "Max Concurrent Licenses Used"]
        st.dataframe(max_conc, use_container_width=True)
    else:
        st.info("Not enough IN/OUT data in the filtered selection to calculate concurrency.")

# --- TAB: Heatmap & Users (Revamped) ---
with tab_heat:
    st.markdown("### User Behavior & Patterns")
    st.caption("Deep dive into user activity, work patterns, and individual user audits.")

    # 1. MACRO PATTERNS (Collapsible)
    with st.expander("Global Work Patterns (When are people active?)", expanded=True):
        if not filtered.empty:
            heatmap_data = filtered.groupby(["weekday", "hour"]).size().reset_index(name="count")
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            fig_heat = px.density_heatmap(
                heatmap_data, 
                x="hour", 
                y="weekday", 
                z="count", 
                nbinsx=24,
                category_orders={"weekday": days_order},
                color_continuous_scale="Viridis",
                title="System Load: Day of Week vs. Hour of Day"
            )
            fig_heat.update_layout(xaxis_title="Hour (0-23)", yaxis_title="Day of Week")
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No data available for heatmaps.")

    st.divider()

    # 2. COMBINED ACTIVITY VS DENIALS (The "Friction" Chart)
    st.markdown("### Activity vs. Friction")
    st.caption("Compare how active a user is (Total Events) vs. how often they get blocked (Denials).")
    
    if not filtered.empty:
        # --- NEW: Radio Button Control ---
        view_mode = st.radio(
            "Select View Mode:", 
            ["Both (Activity + Denials)", "Activity Only (Approved)", "Denials Only"],
            index=0,
            horizontal=True,
            key="friction_radio"
        )

        # 1. Prepare Base Data
        # Ensure we handle missing client names so they don't disappear
        base_df = filtered.copy()
        base_df["client"] = base_df["client"].fillna("Unknown")
        
        # Create Status Column
        base_df["status"] = base_df["event"].apply(lambda x: "Denied" if x == "DENIED" else "Approved")

        # 2. Filter Data based on Radio Button
        if view_mode == "Activity Only (Approved)":
            chart_data = base_df[base_df["status"] == "Approved"].copy()
            chart_title = "Top Users by Successful Activity"
        elif view_mode == "Denials Only":
            chart_data = base_df[base_df["status"] == "Denied"].copy()
            chart_title = "Top Users by Denial Count"
        else:
            # "Both"
            chart_data = base_df.copy()
            chart_title = "User Activity: Success vs. Failure (Top 50 by Volume)"

        # 3. Calculate Top 100 Users *based on the filtered view*
        # This ensures if we look at "Denials Only", we see the top denied users, 
        # not just the top active users who happen to have 0 denials.
        if not chart_data.empty:
            top_users = chart_data["client"].value_counts().head(50).index.tolist()
            chart_data = chart_data[chart_data["client"].isin(top_users)]
            
            grouped = chart_data.groupby(["client", "status"]).size().reset_index(name="count")
            
            # 4. Plot
            fig_combined = px.bar(
                grouped, 
                x="count", 
                y="client", 
                color="status", 
                orientation='h',
                title=chart_title,
                # Explicit colors: Red for bad, Green for good
                color_discrete_map={"Denied": "#FF4B4B", "Approved": "#2E8B57"}, 
                text_auto=True
            )
            
            fig_combined.update_layout(
                barmode='stack', 
                yaxis={'categoryorder':'total ascending'}, # Sort by total length of the bar
                xaxis_title="Event Count",
                yaxis_title="User",
                legend_title="Status"
            )
            st.plotly_chart(fig_combined, use_container_width=True)
        else:
            st.info(f"No events found for the selected mode: {view_mode}")


    st.divider()

    # 3. WHO USES WHAT? (User vs Feature Matrix)
    st.markdown("### The 'Who Uses What' Matrix")
    st.caption("Identify which users depend on specific modules (X-axis = Feature, Y-axis = User).")
    
    if not filtered.empty:
        # We limit to top 30 users to keep the chart readable
        user_counts = filtered["client"].value_counts()
        top_n_users = user_counts.head(50).index.tolist() 
        matrix_df = filtered[filtered["client"].isin(top_n_users)].copy()
        
        matrix_data = matrix_df.groupby(["client", "feature"]).size().reset_index(name="count")
        
        fig_matrix = px.density_heatmap(
            matrix_data,
            x="feature",
            y="client",
            z="count",
            color_continuous_scale="Blues", # Changed to blue for professional look
            title="User vs. Feature Intensity (Limit to Top 50 Users)"
        )
        fig_matrix.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_matrix, use_container_width=True)

    st.divider()

    # 4. INDIVIDUAL USER INSPECTOR (Drill Down)
    st.markdown("### Individual User Inspector")
    st.caption("Select a specific user to see their personal timeline and favorite features.")
    
    unique_users = sorted(filtered["client"].unique())
    
    if unique_users:
        selected_user = st.selectbox("Select User to Inspect:", unique_users)
        
        # Filter data for this user
        user_df = filtered[filtered["client"] == selected_user].sort_values("timestamp")
        
        # Metrics for this user
        u_events = len(user_df)
        u_denials = len(user_df[user_df["event"] == "DENIED"])
        u_features = user_df["feature"].nunique()
        u_top_feature = user_df["feature"].mode()[0] if not user_df.empty else "N/A"
        
        # Display 4-column metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Interactions", u_events)
        m2.metric("Total Denials", u_denials, delta_color="inverse")
        m3.metric("Distinct Features", u_features)
        m4.metric("Favorite Feature", u_top_feature)
        
        # User Timeline Plot
        st.markdown(f"**Activity Timeline: {selected_user}**")
        fig_user_time = px.scatter(
            user_df,
            x="timestamp",
            y="feature",
            color="event",
            symbol="event",
            color_discrete_map={"DENIED": "red", "OUT": "green", "IN": "blue", "UNSUPPORTED": "gray"},
            height=350,
            hover_data=["reason"]
        )
        st.plotly_chart(fig_user_time, use_container_width=True)
        
        # User Feature Distribution Pie
        col_pie1, col_pie2 = st.columns(2)
        with col_pie1:
            st.markdown(f"**Usage by Feature**")
            # Count only successful checkouts for accurate usage stats
            user_feat_counts = user_df[user_df["event"] == "OUT"]["feature"].value_counts().reset_index()
            if user_feat_counts.empty:
                user_feat_counts = user_df["feature"].value_counts().reset_index() # Fallback
            
            user_feat_counts.columns = ["Feature", "Count"]
            
            fig_user_pie = px.pie(
                user_feat_counts, 
                names="Feature", 
                values="Count", 
                hole=0.4
            )
            st.plotly_chart(fig_user_pie, use_container_width=True)
            
    else:
        st.info("No users found in current filter.")

with tab2:
    st.subheader("DENIED reasons (counts)")
    denied = filtered[filtered["event"] == "DENIED"].copy()
    if denied.empty:
        st.caption("No DENIED events for current filters/time window.")
    else:
        denied["reason"] = denied["reason"].fillna("(no reason parsed)")
        counts = (
            denied.groupby(["feature", "reason"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        st.dataframe(counts, use_container_width=True)

        # 1. PIE CHART
        st.markdown("#### Distribution of Reasons")
        # Global count of reasons (ignoring feature breakdown for the pie)
        global_counts = denied["reason"].value_counts().reset_index()
        global_counts.columns = ["Reason", "Count"]
        
        fig_pie = px.pie(
            global_counts, 
            values="Count", 
            names="Reason", 
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# --- TAB 3: Reservations (Original) ---
with tab3:
    st.subheader("Reservations detected in log")
    if reservations.empty:
        st.caption("No RESERVING entries detected.")
    else:
        res = reservations.copy()
        if "timestamp" in res.columns and not res["timestamp"].isna().all():
            res["day"] = pd.to_datetime(res["timestamp"]).dt.date
            res = res[(res["day"] >= start_day) & (res["day"] <= end_day)]
        st.dataframe(res.sort_values("timestamp"), use_container_width=True)

# --- TAB 4: MAX rules (Original) ---
with tab4:
    st.subheader("MAX rules detected in log")
    if max_rules.empty:
        st.caption("No MAX rules detected.")
    else:
        mr = max_rules.copy()
        if "timestamp" in mr.columns and not mr["timestamp"].isna().all():
            mr["day"] = pd.to_datetime(mr["timestamp"]).dt.date
            mr = mr[(mr["day"] >= start_day) & (mr["day"] <= end_day)]
        st.dataframe(mr.sort_values(["feature", "user"]), use_container_width=True)

# --- TAB 5: Tables & meta (Original + Download) ---
with tab5:
    st.subheader("Filtered events table")
    st.dataframe(filtered.sort_values("timestamp"), use_container_width=True)
    
    # NEW: Download Button
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="license_log_export.csv",
        mime="text/csv",
    )
    
    st.subheader("Parser meta (anchors / timestamp markers)")
    st.json({
        "date_anchors_detected": meta.get("date_anchors", [])[:10],
        "timestamp_markers_detected": meta.get("timestamp_markers", [])[:10],
        "warnings": meta.get("warnings", []),
    })
