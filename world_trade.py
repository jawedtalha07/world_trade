import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Global Trade and Climate Impact Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def generate_trade_data():
    """Generates synthetic data for trade volume, costs, emissions, and risk scores (2015-2025)."""
    years = np.arange(2015, 2026)
    
   
    sea_base = np.linspace(10, 14, len(years)) * 1000
    air_base = np.linspace(0.1, 0.15, len(years)) * 1000
    
    sea_volume = (sea_base + np.random.normal(0, 100, len(years))).round(2)
    air_volume = (air_base + np.random.normal(0, 5, len(years))).round(2)

    sea_cost_base = np.linspace(200, 350, len(years))
    air_cost_base = sea_cost_base * np.random.uniform(10, 15, len(years))
    
    sea_cost_spike = np.zeros(len(years))
    sea_cost_spike[5:8] = [300, 450, 200] 
    sea_cost = (sea_cost_base + sea_cost_spike + np.random.normal(0, 20, len(years))).round(2)
    air_cost = (air_cost_base + np.random.normal(0, 100, len(years))).round(2)
    
    
    EMISSION_FACTOR_SEA = 0.005 
    EMISSION_FACTOR_AIR = 0.5 
    sea_emissions = (sea_volume * EMISSION_FACTOR_SEA * np.random.uniform(0.9, 1.1, len(years))).round(2)
    air_emissions = (air_volume * EMISSION_FACTOR_AIR * np.random.uniform(0.9, 1.1, len(years))).round(2)
    
    risk_base = np.linspace(30, 60, len(years))
    sea_risk_score = (risk_base + np.random.normal(0, 5, len(years))).clip(30, 75).round(0)
    air_risk_score = (risk_base * 0.9 + np.random.normal(0, 5, len(years))).clip(25, 65).round(0)
    
    df_trade = pd.DataFrame({
        'Year': years,
        'Sea_Volume_GT': sea_volume,
        'Air_Volume_GT': air_volume,
        'Sea_Cost_USD_Per_Ton': sea_cost,
        'Air_Cost_USD_Per_Ton': air_cost,
        'Sea_Emissions_MT': sea_emissions,
        'Air_Emissions_MT': air_emissions,
        'Sea_Risk_Score': sea_risk_score,
        'Air_Risk_Score': air_risk_score,
    })
    df_trade['Total_Trade_GT'] = df_trade['Sea_Volume_GT'] + df_trade['Air_Volume_GT']
    df_trade['Total_Emissions_MT'] = df_trade['Sea_Emissions_MT'] + df_trade['Air_Emissions_MT']
    return df_trade

@st.cache_data
def get_route_data():
    """Defines a comprehensive set of simulated global trade routes."""
    routes = [
        {"Mode": "Sea Freight", "Source": "Asia (Shanghai)", "Target": "Europe (Rotterdam)", "Color": "blue", "Weight": 3.0,
         "Lat": [31.23, 1.3, 12.7, 30.0, 51.92], "Lon": [121.47, 103.8, 45.0, 32.5, 4.48]},
        {"Mode": "Sea Freight", "Source": "Asia (China)", "Target": "North America (LA)", "Color": "blue", "Weight": 2.5,
         "Lat": [34.05, 35.0, 33.77], "Lon": [108.97, -150.0, -118.29]},
        {"Mode": "Air Freight", "Source": "Asia (HKG)", "Target": "Europe (FRA)", "Color": "red", "Weight": 1.5,
         "Lat": [22.31, 50.11], "Lon": [114.21, 8.68]},
        {"Mode": "Air Freight", "Source": "US (Memphis)", "Target": "Asia (Shanghai)", "Color": "red", "Weight": 1.5,
         "Lat": [35.15, 31.23], "Lon": [-89.98, 121.47]},
        {"Mode": "Sea Freight", "Source": "South America (Santos)", "Target": "Africa (Durban)", "Color": "darkgreen", "Weight": 1.5,
         "Lat": [-23.96, -30.0, -29.88], "Lon": [-46.33, 0.0, 31.05]},
        {"Mode": "Sea Freight", "Source": "Australia (Port Hedland)", "Target": "Asia (Qingdao)", "Color": "teal", "Weight": 2.0,
         "Lat": [-20.32, 25.0, 36.1], "Lon": [118.57, 130.0, 120.38]},
        {"Mode": "Air Freight", "Source": "Asia (Taipei)", "Target": "Europe (Munich)", "Color": "fuchsia", "Weight": 1.0,
         "Lat": [25.03, 48.14], "Lon": [121.56, 11.58]},
        {"Mode": "Sea Freight", "Source": "N. America (Vancouver)", "Target": "S. America (Valparaiso)", "Color": "indigo", "Weight": 1.8,
         "Lat": [49.28, 0.0, -33.05], "Lon": [-123.12, -80.0, -71.61]},
    ]
    return pd.DataFrame(routes)

@st.cache_data
def get_heatwave_data():
    """Defines major simulated heatwave events (based on real-world events 2020-2025)."""
    heatwaves = [
        {"Year": 2021, "Location": "Western N. America", "Lat": 47.0, "Lon": -120.0, "Severity": "Severe", "Description": "Record high temperatures across Pacific Northwest."},
        {"Year": 2023, "Location": "Southern Europe/Med", "Lat": 40.0, "Lon": 15.0, "Severity": "High", "Description": "Extensive heatwaves (Cerberus/Charon) impacting Italy, Greece, Spain."},
        {"Year": 2024, "Location": "SE Asia", "Lat": 10.0, "Lon": 100.0, "Severity": "High", "Description": "Early monsoon/record temperatures in Vietnam/Thailand."},
        {"Year": 2025, "Location": "Brazil/Amazon", "Lat": -10.0, "Lon": -55.0, "Severity": "Medium", "Description": "Drought and severe heat impacting agricultural trade."},
    ]
    return pd.DataFrame(heatwaves)

@st.cache_data
def get_risk_zone_data():
    """Defines critical geopolitical or climate risk zones for route analysis."""
    risk_zones = [
        {"Zone": "Suez Canal/Red Sea", "Lat": 27.9, "Lon": 32.5, "Type": "Geopolitical/Climate", "Severity_Weight": 1.5},
        {"Zone": "Panama Canal/Dry Zone", "Lat": 9.0, "Lon": -79.8, "Type": "Climate/Logistical", "Severity_Weight": 1.2},
        {"Zone": "North Atlantic Hurricane Zone", "Lat": 30.0, "Lon": -60.0, "Type": "Climate", "Severity_Weight": 1.1},
        {"Zone": "Hormuz Strait", "Lat": 26.5, "Lon": 56.5, "Type": "Geopolitical", "Severity_Weight": 1.4},
    ]
    return pd.DataFrame(risk_zones)

def get_region(port_name):
    """Categorizes port names into broad geopolitical regions for analysis."""
    if 'Asia' in port_name or 'China' in port_name or 'Korea' in port_name or 'Taipei' in port_name: return 'Asia-Pacific'
    if 'Europe' in port_name or 'Rotterdam' in port_name or 'Munich' in port_name: return 'Europe'
    if 'America' in port_name or 'LA' in port_name or 'NY' in port_name or 'Memphis' in port_name: return 'North/Central America'
    if 'S. America' in port_name or 'Santos' in port_name or 'Valparaiso' in port_name: return 'South America'
    if 'Africa' in port_name or 'Lagos' in port_name or 'Durban' in port_name: return 'Africa'
    if 'Middle East' in port_name or 'Dubai' in port_name: return 'Middle East'
    return 'Other'

@st.cache_data
def calculate_all_data():
    """Executes all data generation and initial calculations once."""
    df_trade = generate_trade_data()
    df_routes = get_route_data()
    df_heatwaves = get_heatwave_data()
    df_risk_zones = get_risk_zone_data()
    
    risk_summary = calculate_route_risk(df_routes, df_risk_zones)
    df_route_risk = risk_summary['df_route_risk']

    return df_trade, df_routes, df_heatwaves, df_risk_zones, df_route_risk, risk_summary

@st.cache_data
def calculate_route_risk(df_routes, df_risk_zones):
    """Simulates route risk based on proximity to defined risk zones (proxy calculation)."""
    route_risk_data = []
    for _, route in df_routes.iterrows():
        route_risk_score = 0
        is_sea = route['Mode'] == 'Sea Freight'
        for _, zone in df_risk_zones.iterrows():
            route_lat_min, route_lat_max = min(route['Lat']), max(route['Lat'])
            route_lon_min, route_lon_max = min(route['Lon']), max(route['Lon'])
            
            if (zone['Lat'] >= route_lat_min and zone['Lat'] <= route_lat_max) or \
               (zone['Lon'] >= route_lon_min and zone['Lon'] <= route_lon_max):
                route_risk_score += zone['Severity_Weight'] * (15 if is_sea else 5)
        
        route_risk_score = min(100, route_risk_score * 5 + route['Weight'] * 5)
        
        route_risk_data.append({
            'Route': f"{route['Source']} -> {route['Target']}",
            'Mode': route['Mode'],
            'Weight': route['Weight'],
            'Simulated_Route_Risk': round(route_risk_score, 0)
        })
    
    df_route_risk = pd.DataFrame(route_risk_data)
    total_sea_weight = df_route_risk[df_route_risk['Mode'] == 'Sea Freight']['Weight'].sum()
    high_risk_sea_weight = df_route_risk[(df_route_risk['Mode'] == 'Sea Freight') & (df_route_risk['Simulated_Route_Risk'] > 50)]['Weight'].sum()
    total_air_weight = df_route_risk[df_route_risk['Mode'] == 'Air Freight']['Weight'].sum()
    high_risk_air_weight = df_route_risk[(df_route_risk['Mode'] == 'Air Freight') & (df_route_risk['Simulated_Route_Risk'] > 50)]['Weight'].sum()
    
    risk_summary = {
        'Sea_Risk_Exposure_Pct': (high_risk_sea_weight / total_sea_weight) * 100 if total_sea_weight > 0 else 0,
        'Air_Risk_Exposure_Pct': (high_risk_air_weight / total_air_weight) * 100 if total_air_weight > 0 else 0,
        'df_route_risk': df_route_risk
    }
    return risk_summary

df_trade, df_routes, df_heatwaves, df_risk_zones, df_route_risk, risk_summary = calculate_all_data()


def render_trade_overview_page(df_trade, df_filtered, total_trade_volume, avg_annual_risk, risk_delta, delta_start_year):
    """Renders the KPI, Volume, and Cost Comparison charts."""
    st.title("📊 Trade Volume & Cost Overview")
    st.markdown("### Comparing Maritime Cargo and Air Freight (2015-2025 - *Simulated Data*)")

    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4) 
    
    total_sea_volume = df_filtered['Sea_Volume_GT'].sum()
    avg_sea_cost = df_filtered['Sea_Cost_USD_Per_Ton'].mean()
    avg_air_cost = df_filtered['Air_Cost_USD_Per_Ton'].mean()
    cost_ratio = avg_air_cost / avg_sea_cost if avg_sea_cost and avg_sea_cost != 0 else 0

    with metric_col1:
        st.metric(
            label="Total Sea Freight Volume (GT)",
            value=f"{total_sea_volume:,.0f}",
            delta=f"{(total_sea_volume / total_trade_volume) * 100:.1f}% of Total Volume" if total_trade_volume else "0.0% of Total Volume"
        )
    with metric_col2:
        st.metric(
            label="Avg. Air vs. Sea Cost Ratio",
            value=f"{cost_ratio:.1f}x Higher",
            delta=f"Air is {cost_ratio-1:.1f} times more expensive per ton." if cost_ratio > 1 else "Costs are balanced."
        )
    with metric_col3:
        latest_sea_cost = df_trade.loc[df_trade['Year'].idxmax(), 'Sea_Cost_USD_Per_Ton']
        latest_air_cost = df_trade.loc[df_trade['Year'].idxmax(), 'Air_Cost_USD_Per_Ton']
        cost_diff_latest = latest_air_cost - latest_sea_cost
        st.metric(
            label=f"Cost Differential (2025 Projection)",
            value=f"USD {cost_diff_latest:,.0f} / Ton",
            delta=f"Sea Freight: ${latest_sea_cost:,.0f}/Ton (Cost efficiency baseline)"
        )
    with metric_col4:
        st.metric(
            label=f"Avg. Annual Supply Chain Risk Score (0-100)",
            value=f"{avg_annual_risk:.1f}",
            delta=f"Trend since {delta_start_year}: {risk_delta:.1f} pts" if risk_delta else "No change in trend." ,
            delta_color="inverse"
        )

    st.markdown("---")


    st.subheader("Annual Trade Volume and Cost Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        
        df_volume = df_filtered.melt(
            id_vars=['Year'],
            value_vars=['Sea_Volume_GT', 'Air_Volume_GT'],
            var_name='Mode',
            value_name='Volume (GT)'
        )
        df_volume['Mode'] = df_volume['Mode'].str.replace('_Volume_GT', '')
        fig_volume = px.line(df_volume, x='Year', y='Volume (GT)', color='Mode', markers=True,
                             title='Sea (Cargo Ships) vs. Air Trade Volume',
                             color_discrete_map={'Sea': '#1f77b4', 'Air': '#ff7f0e'})
        fig_volume.update_layout(xaxis_title="Year", yaxis_title="Volume (Gigatons)", height=400)
        st.plotly_chart(fig_volume, use_container_width=True)

    with col2:
       
        df_cost = df_filtered.melt(
            id_vars=['Year'],
            value_vars=['Sea_Cost_USD_Per_Ton', 'Air_Cost_USD_Per_Ton'],
            var_name='Mode',
            value_name='Cost (USD/Ton)'
        )
        df_cost['Mode'] = df_cost['Mode'].str.replace('_Cost_USD_Per_Ton', '')
        fig_cost = px.bar(df_cost, x='Year', y='Cost (USD/Ton)', color='Mode', barmode='group',
                          title='Sea vs. Air Freight Cost per Ton',
                          color_discrete_map={'Sea': '#1f77b4', 'Air': '#ff7f0e'})
        fig_cost.update_layout(xaxis_title="Year", yaxis_title="Cost (USD/Ton)", height=400)
        st.plotly_chart(fig_cost, use_container_width=True)


    st.subheader("Yearly Average Freight Cost Heatmap")
    df_cost_matrix = df_trade.set_index('Year')[['Sea_Cost_USD_Per_Ton', 'Air_Cost_USD_Per_Ton']]
    fig_heatmap = px.imshow(
        df_cost_matrix.T,
        x=df_cost_matrix.index,
        y=df_cost_matrix.columns.str.replace('_Cost_USD_Per_Ton', ''),
        color_continuous_scale='Inferno',
        aspect="auto", text_auto=".0f", title='Cost Volatility and Trend by Mode (USD/Ton)', height=300
    )
    fig_heatmap.update_layout(xaxis_title="Year", yaxis_title="Shipping Mode", coloraxis_colorbar=dict(title="Cost (USD/Ton)"))
    st.plotly_chart(fig_heatmap, use_container_width=True)


def render_risk_analysis_page(df_filtered, df_route_risk, risk_summary):
    """Renders the Risk and Emissions charts."""
    st.title("🛡️ Risk and Sustainability Analysis")
    st.markdown("Quantifying climate vulnerability and environmental footprint across transport modes.")

    col_risk, col_emissions = st.columns(2)

    with col_risk:
        st.subheader("Simulated Route Risk Breakdown")
        st.markdown(f"**High-Risk Exposure (Routes with Risk > 50):** Sea: **{risk_summary['Sea_Risk_Exposure_Pct']:.1f}%** | Air: **{risk_summary['Air_Risk_Exposure_Pct']:.1f}%**")

        fig_risk = px.bar(
            df_route_risk.sort_values(by='Simulated_Route_Risk', ascending=True),
            y='Route',
            x='Simulated_Route_Risk',
            color='Mode',
            orientation='h',
            title='Individual Route Risk Scores (Proximity to Chokepoints/Climate Zones)',
            color_discrete_map={'Sea Freight': 'darkorange', 'Air Freight': 'lightcoral'},
            height=450
        )
        fig_risk.update_layout(xaxis_title="Risk Score (0-100)", yaxis_title="Trade Route")
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_emissions:
        st.subheader("Annual CO2 Emissions by Mode (Metric Tons)")

        df_emissions = df_filtered.melt(
            id_vars=['Year'],
            value_vars=['Sea_Emissions_MT', 'Air_Emissions_MT'],
            var_name='Mode',
            value_name='Emissions (MT CO2)'
        )
        df_emissions['Mode'] = df_emissions['Mode'].str.replace('_Emissions_MT', ' Freight')
        
        fig_emissions = px.line(
            df_emissions, x='Year', y='Emissions (MT CO2)', color='Mode', markers=True,
            title='Environmental Footprint: Sea vs. Air CO2',
            color_discrete_map={'Sea Freight': 'darkgreen', 'Air Freight': 'black'},
            log_y=True 
        )
        fig_emissions.update_layout(xaxis_title="Year", yaxis_title="Emissions (Metric Tons CO2) - LOG SCALE", height=450)
        st.plotly_chart(fig_emissions, use_container_width=True)
        st.caption("Note: Y-axis is on a logarithmic scale to capture the vast difference in emissions per unit of volume.")
    
    st.markdown("---")
    
    
    st.header("📈 Simulated Trade Flow Breakdown by Region")
    
    region_weights = {}
    avg_total_trade = df_trade['Total_Trade_GT'].mean()

    for index, row in df_routes.iterrows():
        source_region = get_region(row['Source'])
        target_region = get_region(row['Target'])
        weight_value = row['Weight'] * avg_total_trade
        region_weights[source_region] = region_weights.get(source_region, 0) + weight_value
        region_weights[target_region] = region_weights.get(target_region, 0) + weight_value

    df_regions = pd.DataFrame(list(region_weights.items()), columns=['Region', 'Simulated_Trade_Flow_Weight'])
    df_regions = df_regions.sort_values(by='Simulated_Trade_Flow_Weight', ascending=True)

    fig_region = px.bar(
        df_regions, y='Region', x='Simulated_Trade_Flow_Weight', orientation='h', color='Region',
        title='Relative Simulated Trade Flow Importance by Region',
        labels={'Simulated_Trade_Flow_Weight': 'Relative Flow Weight (Scaled GT)'}, height=450
    )
    fig_region.update_layout(yaxis_title="Global Region")
    st.plotly_chart(fig_region, use_container_width=True)


def render_global_maps_page(df_routes, df_heatwaves_filtered):
    """Renders the Trade Route Map and the Climate Impact Map."""
    st.title("🌍 Global Trade and Climate Visualization")
    
    st.header("🚢 Comprehensive Global Trade Routes (Globe View)")
    st.markdown("A diversified network of simulated trade routes. Thickness reflects relative importance.")

    ports_data = []
    for index, row in df_routes.iterrows():
        ports_data.append({'Name': row['Source'], 'Lat': row['Lat'][0], 'Lon': row['Lon'][0], 'Mode': row['Mode']})
        ports_data.append({'Name': row['Target'], 'Lat': row['Lat'][-1], 'Lon': row['Lon'][-1], 'Mode': row['Mode']})
    df_ports = pd.DataFrame(ports_data).drop_duplicates(subset=['Name', 'Lat', 'Lon'])
    
    port_color_map = {'Sea Freight': 'DarkBlue', 'Air Freight': 'DarkRed'}

    fig_routes = px.scatter_geo(
        df_ports, lat="Lat", lon="Lon", hover_name="Name", color="Mode",
        color_discrete_map=port_color_map, projection="orthographic", 
        title="Comprehensive Global Trade Routes: Major & Specialized Flows"
    )
    fig_routes.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='Black')))

    for index, row in df_routes.iterrows():
        fig_routes.add_trace(go.Scattergeo(
            lat=row['Lat'], lon=row['Lon'], mode='lines', 
            line=dict(width=row['Weight'] * 0.5, color=row['Color'], dash='dot'),
            name=f"{row['Mode']} ({row['Source']} -> {row['Target']})", showlegend=False,
            hoverinfo='text',
            text=f"Route: {row['Source']} to {row['Target']} | Type: {row['Mode']} | Relative Weight: {row['Weight']}"
        ))

    fig_routes.update_layout(
        geo=dict(showland=True, landcolor="rgb(243, 243, 243)", showocean=True, oceancolor="rgb(180, 220, 240)",
                 showcountries=True, countrycolor="rgb(100, 100, 100)", countrywidth=0.5, showcoastlines=True,
                 coastlinecolor="rgb(100, 100, 100)", projection_type='orthographic'),
        margin={"r":0,"t":50,"l":0,"b":0}, height=600
    )
    st.plotly_chart(fig_routes, use_container_width=True)

    st.markdown("---")


    st.header("🔥 Climate Impact: Heatwaves and Trade Volume")
    st.markdown("Map showing locations of major heatwave events. Use the sidebar filter to analyze specific years.")

   
    df_heatwaves_merged = pd.merge(
        df_heatwaves_filtered,
        df_trade[['Year', 'Total_Trade_GT']],
        on='Year',
        how='left'
    )

    max_trade = df_heatwaves_merged['Total_Trade_GT'].max()
    if not df_heatwaves_merged.empty and max_trade > 0:
        df_heatwaves_merged['Marker_Size'] = (df_heatwaves_merged['Total_Trade_GT'] / max_trade) * 30
    else:
        df_heatwaves_merged['Marker_Size'] = 10

    fig_heatwaves = px.scatter_mapbox(
        df_heatwaves_merged, lat="Lat", lon="Lon", hover_name="Location",
        hover_data={'Year': True, 'Severity': True, 'Description': True, 'Total_Trade_GT': ':,.0f'},
        color="Severity", size="Marker_Size", color_discrete_map={'Severe': 'darkred', 'High': 'orange', 'Medium': 'gold'},
        zoom=1.5, height=550, title='Global Heatwaves (2020-2025) vs. Annual Trade Volume'
    )

    fig_heatwaves.update_layout(
        mapbox_style="carto-darkmatter",
        margin={"r":0,"t":25,"l":0,"b":0},
        legend_title="Heatwave Severity"
    )
    st.plotly_chart(fig_heatwaves, use_container_width=True)


st.sidebar.header("Page Selector")
page_selection = st.sidebar.radio(
    "Choose Analysis Focus:",
    ["Trade Overview & Cost", "Risk & Sustainability", "Global Maps & Climate"],
    index=0
)

st.sidebar.header("Dashboard Filters")
selected_year_range = st.sidebar.slider(
    "Select Year Range for Charts",
    min_value=2015,
    max_value=2025,
    value=(2015, 2025)
)

selected_map_years = st.sidebar.multiselect(
    "Filter Climate Map by Year(s)",
    options=df_heatwaves['Year'].unique().tolist(),
    default=df_heatwaves['Year'].unique().tolist()
)


df_filtered = df_trade[(df_trade['Year'] >= selected_year_range[0]) & (df_trade['Year'] <= selected_year_range[1])]
df_heatwaves_filtered = df_heatwaves[df_heatwaves['Year'].isin(selected_map_years)]

total_trade_volume = df_filtered['Total_Trade_GT'].sum()

avg_annual_risk, risk_delta, delta_start_year = 0.0, 0.0, selected_year_range[0]
if not df_filtered.empty:
    avg_annual_risk = df_filtered[['Sea_Risk_Score', 'Air_Risk_Score']].mean().mean()
    latest_risk_series = df_filtered.iloc[df_filtered['Year'].argmax()]
    latest_risk = latest_risk_series[['Sea_Risk_Score', 'Air_Risk_Score']].mean()
    earliest_risk_series = df_filtered.iloc[df_filtered['Year'].argmin()]
    earliest_risk = earliest_risk_series[['Sea_Risk_Score', 'Air_Risk_Score']].mean()
    risk_delta = latest_risk - earliest_risk
    delta_start_year = df_filtered['Year'].min()

if page_selection == "Trade Overview & Cost":
    render_trade_overview_page(df_trade, df_filtered, total_trade_volume, avg_annual_risk, risk_delta, delta_start_year)
elif page_selection == "Risk & Sustainability":
    render_risk_analysis_page(df_filtered, df_route_risk, risk_summary)
elif page_selection == "Global Maps & Climate":
    
    render_global_maps_page(df_routes, df_heatwaves_filtered)
