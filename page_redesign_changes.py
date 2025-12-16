# ============================================================================
# LOOKER-STYLE DASHBOARD REDESIGN
# Instructions: Integrate these changes into your existing page.py
# ============================================================================

# STEP 1: Add this CSS injection function right after your imports (around line 31)
# ============================================================================

def inject_looker_style_css():
    """Inject custom CSS for Looker-style dashboard appearance"""
    st.markdown("""
    <style>
    /* Compact metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 300 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        color: #666 !important;
    }
    
    /* Tighter spacing */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }
    
    /* KPI indicator dots */
    .kpi-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }
    .kpi-green { background-color: #22c55e; }
    .kpi-yellow { background-color: #eab308; }
    .kpi-red { background-color: #ef4444; }
    
    /* Industry benchmark text */
    .benchmark-text {
        font-size: 0.7rem;
        color: #999;
        margin-top: -10px;
    }
    
    /* Summary stats styling */
    .summary-stat-row {
        display: flex;
        justify-content: space-between;
        padding-top: 0.75rem;
        border-top: 1px solid #eee;
        margin-top: 0.75rem;
    }
    
    /* Section dividers */
    hr {
        margin: 1rem 0 !important;
        border-color: #eee !important;
    }
    
    /* Compact selectbox */
    .stSelectbox > div > div {
        padding: 0.25rem 0.5rem !important;
    }
    
    /* Chart title styling */
    .chart-title {
        font-size: 0.9rem;
        font-weight: 500;
        color: #333;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


# STEP 2: Replace your create_overview_cards function (around line 506) with this:
# ============================================================================

def create_overview_cards(metrics):
    """Create compact overview cards in Looker style"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Impressions", f"{metrics['impressions']:,}")
    with col2:
        st.metric("Total Clicks", f"{metrics['clicks']:,}")
    with col3:
        st.metric("Video Starts", f"{metrics['starts']:,}")
    with col4:
        st.metric("Video Completes", f"{metrics['completes']:,}")


# STEP 3: Replace your create_kpi_cards function (around line 521) with this:
# ============================================================================

def create_kpi_cards(metrics, selected_kpis=None):
    """Create KPI cards with colored indicators like Looker"""
    if selected_kpis is None:
        selected_kpis = ['CTR', 'VCR']
    
    available_kpis = {
        'CTR': {'value': metrics['ctr'], 'format': ':.4f', 'unit': '%', 'industry': '0.05-0.10%', 'good_threshold': 0.10},
        'VCR': {'value': metrics['vcr'], 'format': ':.2f', 'unit': '%', 'industry': '70-85%', 'good_threshold': 70},
        'CPC': {'value': metrics['cpc'], 'format': ':.2f', 'unit': '', 'industry': 'Varies by industry', 'good_threshold': 5},
        'CPA': {'value': metrics['cpa'], 'format': ':.2f', 'unit': '', 'industry': 'Varies by industry', 'good_threshold': 50}
    }
    
    valid_kpis = [kpi for kpi in selected_kpis if kpi in available_kpis and available_kpis[kpi]['value'] > 0]
    
    if not valid_kpis:
        st.info("No KPI data available for selected metrics")
        return
    
    num_cols = min(len(valid_kpis), 4)
    cols = st.columns(num_cols)
    
    for i, kpi in enumerate(valid_kpis[:4]):
        kpi_data = available_kpis[kpi]
        value = kpi_data['value']
        
        # Determine indicator color
        if kpi in ['CTR']:
            color_class = "kpi-green" if value >= kpi_data['good_threshold'] else "kpi-red"
        elif kpi == 'VCR':
            color_class = "kpi-green" if value >= kpi_data['good_threshold'] else "kpi-yellow"
        else:
            color_class = "kpi-green" if value < kpi_data['good_threshold'] else "kpi-yellow"
        
        # Format value
        if kpi_data['format'] == ':.4f':
            formatted_value = f"{value:.4f}{kpi_data['unit']}"
        else:
            formatted_value = f"{value:.2f}{kpi_data['unit']}"
        
        with cols[i % num_cols]:
            # Custom HTML for indicator dot
            st.markdown(f'<span class="kpi-indicator {color_class}"></span> **{kpi}**', unsafe_allow_html=True)
            st.markdown(f"<h2 style='margin:0; font-weight:300;'>{formatted_value}</h2>", unsafe_allow_html=True)
            st.markdown(f'<p class="benchmark-text">Industry average: {kpi_data["industry"]}</p>', unsafe_allow_html=True)


# STEP 4: Replace your create_device_chart function (around line 1216) with this:
# ============================================================================

def create_device_chart(df):
    """Create compact device type pie chart like Looker"""
    if 'Device Type' not in df.columns:
        return
    
    device_data = df.groupby('Device Type').agg({
        'Impressions': 'sum',
        'Clicks': 'sum'
    }).reset_index()
    
    # Looker-style colors
    colors = ['#4285f4', '#a8c7fa', '#ea4335', '#fbcfe8', '#34a853']
    
    fig = go.Figure(data=[go.Pie(
        labels=device_data['Device Type'],
        values=device_data['Impressions'],
        hole=0,
        textinfo='percent',
        textposition='inside',
        insidetextorientation='horizontal',
        hovertemplate='<b>%{label}</b><br>Impressions=%{value:,.0f}<extra></extra>',
        marker=dict(colors=colors, line=dict(color='#ffffff', width=2))
    )])
    
    fig.update_layout(
        title=None,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        ),
        margin=dict(l=10, r=80, t=10, b=10),
        height=280
    )
    
    st.markdown("**Impressions by Device Type**")
    st.plotly_chart(fig, use_container_width=True)


# STEP 5: Replace your create_enhanced_daily_trend function (around line 1229) with this:
# ============================================================================

def create_enhanced_daily_trend(df, selected_kpi='Impressions', days_back=14):
    """Create enhanced daily trend with compact summary stats like Looker"""
    if 'Date' not in df.columns:
        st.warning("No date column found in data")
        return
    
    df_clean = df[~df['Date'].astype(str).str.contains('Filter|:', case=False, na=False)].copy()
    
    try:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['Date'])
    except:
        st.error("Unable to parse date column")
        return
    
    if days_back > 0:
        end_date = df_clean['Date'].max()
        start_date = end_date - pd.Timedelta(days=days_back)
        df_clean = df_clean[df_clean['Date'] >= start_date]
    
    if len(df_clean) == 0:
        st.warning(f"No data available for the last {days_back} days")
        return
    
    # Prepare aggregation
    agg_dict = {}
    if 'Impressions' in df_clean.columns:
        agg_dict['Impressions'] = 'sum'
    if 'Clicks' in df_clean.columns:
        agg_dict['Clicks'] = 'sum'
    
    cost_columns = ['Total Media Cost (Advertiser Currency)', 'Media Cost', 'Cost', 'Spend', 'Revenue (Partner Currency)']
    for col in cost_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            agg_dict['Cost'] = (col, 'sum')
            break
    
    conversion_columns = ['Total Conversions', 'Conversions', 'Post-Click Conversions', 'Post-View Conversions']
    for col in conversion_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            agg_dict['Conversions'] = (col, 'sum')
            break
    
    if 'Starts (Video)' in df_clean.columns:
        agg_dict['Video Starts'] = ('Starts (Video)', 'sum')
    if 'Complete Views (Video)' in df_clean.columns:
        agg_dict['Video Completes'] = ('Complete Views (Video)', 'sum')
    
    final_agg_dict = {}
    for new_name, agg_info in agg_dict.items():
        if isinstance(agg_info, tuple):
            col_name, func = agg_info
            final_agg_dict[col_name] = func
        else:
            final_agg_dict[new_name] = agg_info
    
    if not final_agg_dict:
        st.warning("No aggregatable columns found in the data")
        return
    
    daily_data = df_clean.groupby('Date').agg(final_agg_dict).reset_index()
    
    column_mapping = {}
    for new_name, agg_info in agg_dict.items():
        if isinstance(agg_info, tuple):
            col_name, func = agg_info
            if col_name in daily_data.columns:
                column_mapping[col_name] = new_name
    daily_data.rename(columns=column_mapping, inplace=True)
    
    # Calculate derived KPIs
    if 'Clicks' in daily_data.columns and 'Impressions' in daily_data.columns:
        daily_data['CTR'] = (daily_data['Clicks'] / daily_data['Impressions'] * 100).round(4)
        daily_data['CTR'] = daily_data['CTR'].replace([float('inf'), -float('inf')], 0).fillna(0)
    
    if 'Video Completes' in daily_data.columns and 'Video Starts' in daily_data.columns:
        daily_data['VCR'] = (daily_data['Video Completes'] / daily_data['Video Starts'] * 100).round(2)
        daily_data['VCR'] = daily_data['VCR'].replace([float('inf'), -float('inf')], 0).fillna(0)
    
    if 'Cost' in daily_data.columns and 'Clicks' in daily_data.columns:
        daily_data['CPC'] = (daily_data['Cost'] / daily_data['Clicks']).round(2)
        daily_data['CPC'] = daily_data['CPC'].replace([float('inf'), -float('inf')], 0).fillna(0)
    
    if 'Cost' in daily_data.columns and 'Conversions' in daily_data.columns:
        daily_data['CPA'] = (daily_data['Cost'] / daily_data['Conversions']).round(2)
        daily_data['CPA'] = daily_data['CPA'].replace([float('inf'), -float('inf')], 0).fillna(0)
    
    if selected_kpi not in daily_data.columns:
        available_kpis = [col for col in daily_data.columns if col != 'Date']
        st.warning(f"'{selected_kpi}' not available. Available: {', '.join(available_kpis)}")
        selected_kpi = available_kpis[0] if available_kpis else 'Impressions'
    
    # Create Looker-style line chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_data['Date'],
        y=daily_data[selected_kpi],
        mode='lines+markers',
        name=selected_kpi,
        line=dict(color='#4285f4', width=2),
        marker=dict(size=6, color='#4285f4'),
        hovertemplate=f'<b>%{{x|%b %d, %Y}}</b><br>{selected_kpi}: %{{y:,.2f}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{selected_kpi} Trend - Last {days_back} Days',
        title_font_size=14,
        xaxis_title='Date',
        yaxis_title=selected_kpi,
        hovermode='x unified',
        margin=dict(l=40, r=20, t=40, b=40),
        height=280,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#f5f5f5',
        tickformat='%b %d\n%Y'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#f5f5f5'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Compact summary stats row (like Looker)
    unit = "%" if selected_kpi in ['CTR', 'VCR'] else ""
    avg_val = daily_data[selected_kpi].mean()
    peak_val = daily_data[selected_kpi].max()
    trend_val = daily_data[selected_kpi].iloc[-1] - daily_data[selected_kpi].iloc[0] if len(daily_data) > 1 else 0
    trend_icon = "📈" if trend_val >= 0 else "📉"
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Days of Data", len(daily_data))
    with col2:
        st.metric("Average Impressions", f"{avg_val:,.2f}{unit}")
    with col3:
        st.metric("Peak Impressions", f"{peak_val:,.2f}{unit}")
    with col4:
        st.metric("Trend", f"{trend_icon} {trend_val:+,.0f}")


# STEP 6: Update your main dashboard section (around line 1929-1976) to use tighter layout:
# ============================================================================

# Replace this section in your main() function where you display the charts:
"""
# In main(), around lines 1929-1976, update the layout to be more compact:

        # Inject the custom CSS at the start of the dashboard
        inject_looker_style_css()
        
        st.subheader("Overview")
        create_overview_cards(metrics)
        
        st.subheader("Key Performance Indicators")
        
        # KPI selection - more compact
        available_kpis = ['CTR', 'VCR']
        if metrics['cost'] > 0:
            available_kpis.extend(['CPC'])
        if metrics['conversions'] > 0:
            available_kpis.extend(['CPA'])
        
        default_kpis = [kpi for kpi in st.session_state.kpi_settings.get('selected_kpis', ['CTR', 'VCR']) if kpi in available_kpis]
        if not default_kpis:
            default_kpis = ['CTR', 'VCR'] if 'CTR' in available_kpis else available_kpis[:2]
        
        selected_kpis = st.multiselect(
            "Select KPIs to display:",
            options=available_kpis,
            default=default_kpis,
            help="Choose which Key Performance Indicators to show",
            key="kpi_multiselect"
        )
        
        if selected_kpis != st.session_state.kpi_settings.get('selected_kpis', []):
            st.session_state.kpi_settings['selected_kpis'] = selected_kpis
        
        if selected_kpis:
            create_kpi_cards(metrics, selected_kpis)
        else:
            st.info("Please select at least one KPI to display")
        
        # Side-by-side charts (Device Pie + Daily Trend)
        col1, col2 = st.columns([1, 2])  # Device chart narrower, trend wider
        
        with col1:
            create_device_chart(df_clean)
        
        with col2:
            st.markdown("**📈 Daily Performance Trend**")
            
            # Compact controls in a row
            trend_col1, trend_col2 = st.columns(2)
            with trend_col1:
                trend_kpi_options = ['Impressions', 'Clicks', 'CTR']
                if metrics['cost'] > 0:
                    trend_kpi_options.extend(['CPC'])
                if metrics['conversions'] > 0:
                    trend_kpi_options.extend(['CPA'])
                if metrics['starts'] > 0:
                    trend_kpi_options.extend(['VCR'])
                
                saved_trend_kpi = st.session_state.kpi_settings.get('trend_kpi', 'Impressions')
                trend_index = trend_kpi_options.index(saved_trend_kpi) if saved_trend_kpi in trend_kpi_options else 0
                
                selected_trend_kpi = st.selectbox(
                    "Select KPI for trend:",
                    options=trend_kpi_options,
                    index=trend_index,
                    help="Choose which metric to show",
                    key="trend_kpi_selectbox"
                )
                
                if selected_trend_kpi != st.session_state.kpi_settings.get('trend_kpi'):
                    st.session_state.kpi_settings['trend_kpi'] = selected_trend_kpi
            
            with trend_col2:
                days_back = st.number_input(
                    "Days to show:",
                    min_value=1,
                    max_value=365,
                    value=14,
                    step=1,
                    help="Number of days to include"
                )
            
            create_enhanced_daily_trend(df_clean, selected_trend_kpi, days_back)
"""

# ============================================================================
# ADDITIONAL: Line Item Table with CPA (add this new function if you want the table)
# ============================================================================

def create_line_item_performance_table(df_clean, metrics):
    """Create Line Item performance table with conversions and CPA like Looker"""
    if 'Line Item' not in df_clean.columns:
        return
    
    agg_dict = {'Impressions': 'sum', 'Clicks': 'sum'}
    
    cost_columns = ['Total Media Cost (Advertiser Currency)', 'Media Cost', 'Cost', 'Spend']
    cost_col = None
    for col in cost_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            agg_dict[col] = 'sum'
            cost_col = col
            break
    
    conversion_columns = ['Total Conversions', 'Conversions', 'Post-Click Conversions', 'Post-View Conversions']
    conv_col = None
    for col in conversion_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            agg_dict[col] = 'sum'
            conv_col = col
            break
    
    if not cost_col or not conv_col:
        return
    
    line_item_data = df_clean.groupby('Line Item').agg(agg_dict).reset_index()
    
    # Calculate CPA
    line_item_data['Total_Conversions'] = line_item_data[conv_col]
    line_item_data['CPA'] = (line_item_data[cost_col] / line_item_data[conv_col]).round(2)
    line_item_data['CPA'] = line_item_data['CPA'].replace([float('inf'), -float('inf')], 0).fillna(0)
    
    # Sort by conversions
    line_item_data = line_item_data.sort_values('Total_Conversions', ascending=False)
    
    # Display
    st.markdown("### Line Item Performance")
    
    # Summary metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CPA", f"£{metrics['cpa']:,.2f}")
    with col2:
        st.metric("Media Cost", f"£{metrics['cost']:,.2f}")
    with col3:
        st.metric("Total Conversions", f"{metrics['conversions']:,}")
    
    # Table
    display_df = line_item_data[['Line Item', 'Total_Conversions', 'CPA']].head(6)
    display_df.columns = ['Line_Item', 'Total_Conversions', 'CPA']
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ============================================================================
# ADDITIONAL: Creative Size Donut Chart (add if you want this)
# ============================================================================

def create_creative_size_chart(df_clean):
    """Create creative size distribution donut chart"""
    if 'Creative Size' not in df_clean.columns:
        return
    
    size_data = df_clean.groupby('Creative Size').agg({'Impressions': 'sum'}).reset_index()
    size_data = size_data.sort_values('Impressions', ascending=False).head(5)
    
    colors = ['#4285f4', '#f4b400', '#9c27b0', '#34a853', '#ea4335']
    
    fig = go.Figure(data=[go.Pie(
        labels=size_data['Creative Size'],
        values=size_data['Impressions'],
        hole=0.5,
        textinfo='percent',
        textposition='outside',
        marker=dict(colors=colors, line=dict(color='#ffffff', width=2))
    )])
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
        margin=dict(l=10, r=100, t=10, b=10),
        height=220
    )
    
    st.plotly_chart(fig, use_container_width=True)
