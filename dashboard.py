import streamlit as st
import pandas as pd
from scipy.stats import linregress
from plotly.subplots import make_subplots
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="ENVECON 105: CO₂ Emissions Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background: linear-gradient(135deg, #1a2a6c, #b21f1f, #1a2a6c);
    }
    .main-title {
        font-size: 3.5rem !important;
        text-align: center;
        background: linear-gradient(135deg, #1a2a6c, #b21f1f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 15px;
        border-bottom: 2px solid #1a2a6c;
    }
    .section-title {
        font-size: 2.2rem !important;
        border-left: 5px solid #1a2a6c;
        padding-left: 15px;
        margin-top: 30px;
        margin-bottom: 20px;
        color: #1a2a6c;
    }
    .metric-card {
        border-radius: 10px;
        padding: 15px;
        background: #f0f2f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem !important;
        font-weight: 700;
        text-align: center;
        color: #1a2a6c;
    }
    .metric-label {
        text-align: center;
        font-size: 1.1rem;
        color: #555;
    }
    .plot-container {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    .footer {
        text-align:center; 
        color:#666; 
        padding-top:20px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data functions with caching
@st.cache_data
def load_global_data():
    # Load global data
    data_dir = Path('data')
    wrangled_data = pd.read_csv(data_dir / 'wrangled' / 'wrangled_data.csv')
    
    # Filter for relevant indicators
    global_co2 = wrangled_data[
        (wrangled_data['Indicator'] == 'Emissions') & 
        (wrangled_data['Label'] == 'CO2 Emissions (Metric Tons)')
    ]
    
    # Prepare top 10 emitters data
    top_10 = global_co2[global_co2['Year'] == 2014].nlargest(10, 'Value')['Country'].tolist()
    top_10_data = global_co2[
        (global_co2['Country'].isin(top_10)) & 
        (global_co2['Year'] >= 1900)
    ]
    
    # Prepare US data for correlation analysis
    us_data = wrangled_data[
        (wrangled_data['Country'] == 'United States') & 
        (wrangled_data['Year'] >= 1980) & 
        (wrangled_data['Year'] <= 2014)
    ]
    us_pivot = us_data.pivot_table(index='Year', columns='Indicator', values='Value').reset_index()
    
    # Prepare faceted data
    facet_data = wrangled_data[
        ~wrangled_data['Indicator'].isin(['Disasters', 'Temperature']) &
        (wrangled_data['Year'] >= 1960)
    ].copy()
    facet_data['Region'] = np.where(
        facet_data['Country'] == 'United States',
        'United States',
        'Rest of World'
    )
    
    return {
        'global_co2': global_co2,
        'top_10_data': top_10_data,
        'us_data': us_pivot,
        'facet_data': facet_data
    }

@st.cache_data
def load_india_data():
    # Load India data
    data_dir = Path('data')
    india_data = pd.read_csv(data_dir / 'output' / 'india_processed.csv')
    
    # Load top countries data
    co2_df = pd.read_csv(data_dir / 'CO2_emission.csv')
    year_cols = [col for col in co2_df.columns if col.isdigit()]
    co2_df[year_cols] = co2_df[year_cols].apply(pd.to_numeric, errors='coerce')
    
    # Calculate top 10 average emitters from 2010-2020
    top_countries = (
        co2_df.set_index('country')[['2010', '2011', '2012', '2013', '2014', 
                                     '2015', '2016', '2017', '2018', '2019', '2020']]
        .mean(axis=1)
        .nlargest(10)
        .index
        .tolist()
    )
    
    # Filter and reshape top countries data
    top_co2 = co2_df[co2_df['country'].isin(top_countries)]
    top_co2 = top_co2.melt(id_vars='country', var_name='Year', value_name='CO2')
    top_co2['Year'] = pd.to_numeric(top_co2['Year'], errors='coerce')
    top_co2 = top_co2.dropna(subset=['Year'])
    top_co2 = top_co2[top_co2['Year'] >= 1960]
    
    # Prepare faceted data for India
    facet_df = india_data.melt(
        id_vars='Year', 
        value_vars=['CO2', 'GDP', 'Energy', 'Temperature', 'Earthquakes'],
        var_name='Metric'
    )
    
    # Calculate per capita metrics (using World Bank population data)
    # Population in millions: 1960=450, 2022=1417
    india_data['Population'] = np.linspace(450, 1417, len(india_data))
    india_data['CO2_per_capita'] = india_data['CO2'] / india_data['Population']
    india_data['Energy_per_capita'] = india_data['Energy'] / india_data['Population']
    
    # Prepare decade analysis
    india_data['Decade'] = (india_data['Year'] // 10) * 10
    decade_data = india_data.groupby('Decade').agg({
        'CO2': 'mean',
        'GDP': 'mean',
        'Energy': 'mean',
        'Temperature': 'mean',
        'Earthquakes': 'sum'
    }).reset_index()
    
    return {
        'india_data': india_data,
        'top_co2': top_co2,
        'facet_df': facet_df,
        'decade_data': decade_data
    }

# Load data
global_data = load_global_data()
india_data = load_india_data()

# Dashboard title
st.markdown('<h1 class="main-title">ENVECON 105: Global CO₂ Emissions Analysis</h1>', unsafe_allow_html=True)

# Sidebar for navigation
analysis_type = st.sidebar.radio("Select Analysis Type:", 
                                ["Global Analysis", "India Analysis"], 
                                index=0)

# Tabs for additional analysis
if analysis_type == "Global Analysis":
    tab1, tab2, tab3 = st.tabs(["Overview", "Regional Comparison", "Deep Analysis"])
else:
    tab1, tab2, tab3 = st.tabs(["Overview", "Decadal Trends", "Correlation Analysis"])

# Global Analysis Section
if analysis_type == "Global Analysis":
    with tab1:
        # Section title
        st.markdown('<h2 class="section-title">Global CO₂ Emissions Analysis</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_emissions = global_data["global_co2"].groupby("Year")["Value"].sum().max()
            st.markdown(f'<div class="metric-card"><div class="metric-label">Peak Global Emissions</div>'
                        f'<div class="metric-value">{total_emissions/1e6:,.1f}M tons</div></div>', 
                        unsafe_allow_html=True)
        
        with col2:
            us_emissions = global_data["us_data"]["Emissions"].mean()
            st.markdown(f'<div class="metric-card"><div class="metric-label">Avg US Emissions</div>'
                        f'<div class="metric-value">{us_emissions/1000:,.0f}K tons</div></div>', 
                        unsafe_allow_html=True)
        
        with col3:
            correlation = global_data["us_data"][["Emissions", "Temperature"]].corr().iloc[0,1]
            st.markdown(f'<div class="metric-card"><div class="metric-label">US Correlation</div>'
                        f'<div class="metric-value">{correlation:.3f}</div></div>', 
                        unsafe_allow_html=True)
        
        with col4:
            growth_rates = global_data["global_co2"].groupby("Year")["Value"].sum().pct_change().dropna()
            avg_growth = growth_rates.mean() * 100
            st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Growth Rate</div>'
                        f'<div class="metric-value">{avg_growth:.2f}%</div></div>', 
                        unsafe_allow_html=True)
        
        # Plot 1: Global CO2 Emissions Over Time
        st.markdown('<h3>Global CO₂ Emissions Over Time (1751-2014)</h3>', unsafe_allow_html=True)
        fig1 = px.line(
            global_data["global_co2"].groupby("Year")["Value"].sum().reset_index(),
            x="Year", 
            y="Value",
            labels={"Value": "Emissions (Metric Tons)"},
            template="plotly_white"
        )
        fig1.update_layout(
            hovermode="x unified",
            xaxis_title="Year",
            yaxis_title="Emissions (Metric Tons)",
            height=500
        )
        fig1.add_vrect(x0=1973, x1=1975, fillcolor="gray", opacity=0.2, annotation_text="Oil Crisis")
        fig1.add_vrect(x0=2008, x1=2009, fillcolor="purple", opacity=0.2, annotation_text="Financial Crisis")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Plot 2: Top 10 Emitting Countries
        st.markdown('<h3>Top 10 CO₂ Emission-producing Countries (1900-2014)</h3>', unsafe_allow_html=True)
        fig2 = px.line(
            global_data["top_10_data"],
            x="Year",
            y="Value",
            color="Country",
            labels={"Value": "Emissions (Metric Tons)"},
            template="plotly_white"
        )
        fig2.update_layout(
            hovermode="x unified",
            xaxis_title="Year",
            yaxis_title="Emissions (Metric Tons)",
            height=500,
            legend_title="Country"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="section-title">Regional Comparison Analysis</h2>', unsafe_allow_html=True)
        
        # Plot 3: Faceted Regional Comparison
        st.markdown('<h3>Regional Comparison of Indicators</h3>', unsafe_allow_html=True)
        
        # Create subplots with independent y-axes
        fig3 = make_subplots(
            rows=3, cols=2,
            subplot_titles=("US CO₂", "Rest of World CO₂", 
                            "US GDP", "Rest of World GDP",
                            "US Energy", "Rest of World Energy"),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]],
            vertical_spacing=0.1
        )
        
        # Get unique indicators
        indicators = global_data["facet_data"]["Indicator"].unique()
        regions = global_data["facet_data"]["Region"].unique()
        
        # Add traces for each indicator and region
        row = 1
        for indicator in indicators:
            for region in regions:
                data = global_data["facet_data"][
                    (global_data["facet_data"]["Indicator"] == indicator) & 
                    (global_data["facet_data"]["Region"] == region)
                ]
                
                fig3.add_trace(
                    go.Scatter(
                        x=data["Year"],
                        y=data["Value"],
                        name=f"{region} {indicator}",
                        mode="lines",
                        line=dict(width=3)
                    ),
                    row=row, col=1 if region == "United States" else 2
                )
            row += 1
        
        # Update layout
        fig3.update_layout(
            height=900,
            showlegend=False,
            template="plotly_white",
            title="Regional Comparison of Indicators",
            yaxis1_title="CO₂ Emissions",
            yaxis3_title="GDP",
            yaxis5_title="Energy",
            xaxis1_title="Year",
            xaxis2_title="Year",
            xaxis3_title="Year",
            xaxis4_title="Year",
            xaxis5_title="Year",
            xaxis6_title="Year"
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Plot 4: Emissions Intensity Heatmap
        st.markdown('<h3>Top 10 Emitters: Emissions Intensity (1960-2014)</h3>', unsafe_allow_html=True)
        
        # Prepare data for heatmap
        heatmap_data = global_data["top_10_data"].pivot_table(
            index="Country", 
            columns="Year", 
            values="Value"
        ).fillna(0)
        
        # Create heatmap
        fig4 = px.imshow(
            np.log1p(heatmap_data),
            labels=dict(x="Year", y="Country", color="Log(Emissions)"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            aspect="auto",
            color_continuous_scale="Viridis"
        )
        fig4.update_layout(
            height=600,
            xaxis_title="Year",
            yaxis_title="Country"
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="section-title">Deep Analysis: US Emissions</h2>', unsafe_allow_html=True)
        
        # Plot 5: US Emissions vs Temperature
        st.markdown('<h3>US Emissions vs Temperature (1980-2014)</h3>', unsafe_allow_html=True)
        fig5 = px.scatter(
            global_data["us_data"],
            x="Emissions",
            y="Temperature",
            trendline="ols",
            labels={
                "Emissions": "Emissions (Metric Tons)",
                "Temperature": "Temperature (Fahrenheit)"
            },
            template="plotly_white",
            trendline_color_override="red"
        )
        
        # Calculate regression statistics
        slope, intercept, r_value, p_value, std_err = linregress(
            global_data["us_data"]["Emissions"].dropna(),
            global_data["us_data"]["Temperature"].dropna()
        )
        
        # Add annotation
        fig5.add_annotation(
            x=0.05,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"R² = {r_value**2:.3f}<br>p-value = {p_value:.3e}",
            showarrow=False,
            bgcolor="black",
            bordercolor="red",
            borderwidth=1
        )
        
        fig5.update_layout(height=500)
        st.plotly_chart(fig5, use_container_width=True)
        
        # Additional analysis: Scaled comparison
        st.markdown('<h3>Scaled Comparison of US Emissions and Temperature</h3>', unsafe_allow_html=True)
        
        # Scale data
        us_data_scaled = global_data["us_data"].copy()
        us_data_scaled["Scaled_Emissions"] = (us_data_scaled["Emissions"] - us_data_scaled["Emissions"].mean()) / us_data_scaled["Emissions"].std()
        us_data_scaled["Scaled_Temperature"] = (us_data_scaled["Temperature"] - us_data_scaled["Temperature"].mean()) / us_data_scaled["Temperature"].std()
        
        # Create dual-axis plot
        fig6 = go.Figure()
        
        # Add emissions trace
        fig6.add_trace(go.Scatter(
            x=us_data_scaled["Year"],
            y=us_data_scaled["Scaled_Emissions"],
            name="Scaled Emissions",
            line=dict(color="#1a2a6c", width=3)
        ))
        
        # Add temperature trace
        fig6.add_trace(go.Scatter(
            x=us_data_scaled["Year"],
            y=us_data_scaled["Scaled_Temperature"],
            name="Scaled Temperature",
            line=dict(color="#e63946", width=3),
            yaxis="y2"
        ))
        
        # Layout configuration
        fig6.update_layout(
            xaxis_title="Year",
            yaxis=dict(
                title=dict(
                    text="Scaled Emissions",
                    font=dict(color="#1a2a6c")
                ),
                tickfont=dict(color="#1a2a6c")
            ),
            yaxis2=dict(
                title=dict(
                    text="Scaled Temperature",
                    font=dict(color="#e63946")
                ),
                tickfont=dict(color="#e63946"),
                overlaying="y",
                side="right"
            ),
            height=500,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        
        st.plotly_chart(fig6, use_container_width=True)

# India Analysis Section
else:
    with tab1:
        # Section title
        st.markdown('<h2 class="section-title">India CO₂ Emissions Analysis</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # Convert from thousand metric tons to metric tons
            latest_co2 = india_data["india_data"]["CO2"].iloc[-1] * 1000
            st.markdown(f'<div class="metric-card"><div class="metric-label">Latest CO₂ Emissions</div>'
                        f'<div class="metric-value">{latest_co2:,.0f} tons</div></div>', 
                        unsafe_allow_html=True)
        
        with col2:
            max_temp = india_data["india_data"]["Temperature"].max()
            st.markdown(f'<div class="metric-card"><div class="metric-label">Max Temperature</div>'
                        f'<div class="metric-value">{max_temp:.1f} °C</div></div>', 
                        unsafe_allow_html=True)
        
        with col3:
            correlation = india_data["india_data"][["CO2", "Temperature"]].corr().iloc[0,1]
            st.markdown(f'<div class="metric-card"><div class="metric-label">Emissions-Temp Correlation</div>'
                        f'<div class="metric-value">{correlation:.3f}</div></div>', 
                        unsafe_allow_html=True)
        
        with col4:
            start_emissions = india_data["india_data"]["CO2"].iloc[0]
            end_emissions = india_data["india_data"]["CO2"].iloc[-1]
            growth_rate = ((end_emissions / start_emissions) ** (1/62) - 1) * 100
            st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Annual Growth</div>'
                        f'<div class="metric-value">{growth_rate:.2f}%</div></div>', 
                        unsafe_allow_html=True)
        
        # Plot 1: India's CO2 Emissions Over Time
        st.markdown('<h3>India\'s CO₂ Emissions (1960-2022)</h3>', unsafe_allow_html=True)
        fig1 = px.line(
            india_data["india_data"],
            x="Year",
            y="CO2",
            labels={"CO2": "Emissions (Thousand Metric Tons)"},
            template="plotly_white"
        )
        fig1.update_layout(
            hovermode="x unified",
            xaxis_title="Year",
            yaxis_title="Emissions (Thousand Metric Tons)",
            height=500
        )
        fig1.add_vrect(x0=1973, x1=1975, fillcolor="gray", opacity=0.2, annotation_text="Oil Crisis")
        fig1.add_vrect(x0=1991, x1=1993, fillcolor="blue", opacity=0.2, annotation_text="Economic Reforms")
        fig1.add_vrect(x0=2020, x1=2021, fillcolor="red", opacity=0.2, annotation_text="COVID-19")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Plot 2: Top 10 Emitting Countries Comparison
        st.markdown('<h3>Top 10 CO₂ Emitting Countries Comparison (1960-2020)</h3>', unsafe_allow_html=True)
        fig2 = px.line(
            india_data["top_co2"],
            x="Year",
            y="CO2",
            color="country",
            labels={"CO2": "Emissions (Thousand Metric Tons)", "country": "Country"},
            template="plotly_white"
        )
        fig2.update_layout(
            hovermode="x unified",
            xaxis_title="Year",
            yaxis_title="Emissions (Thousand Metric Tons)",
            height=500,
            legend_title="Country"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="section-title">Decadal Trends Analysis</h2>', unsafe_allow_html=True)
        
        # Plot 3: Faceted Indicators
        st.markdown('<h3>India\'s Environmental and Economic Indicators (1960-2022)</h3>', unsafe_allow_html=True)
        
        # Create subplots with independent y-axes
        fig3 = make_subplots(
            rows=3, cols=2,
            subplot_titles=("CO₂ Emissions", "GDP", "Energy", "Temperature", "Earthquakes"),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, None]],
            vertical_spacing=0.1
        )
        
        # Add traces for each metric
        metrics = india_data["facet_df"]["Metric"].unique()
        row = 1
        col = 1
        for metric in metrics:
            data = india_data["facet_df"][india_data["facet_df"]["Metric"] == metric]
            
            fig3.add_trace(
                go.Scatter(
                    x=data["Year"],
                    y=data["value"],
                    name=metric,
                    mode="lines",
                    line=dict(width=3)
                ),
                row=row, col=col
            )
            
            # Update axis titles
            if row == 1 and col == 1:
                fig3.update_yaxes(title_text="Thousand Metric Tons", row=row, col=col)
            elif row == 1 and col == 2:
                fig3.update_yaxes(title_text="GDP", row=row, col=col)
            elif row == 2 and col == 1:
                fig3.update_yaxes(title_text="Energy", row=row, col=col)
            elif row == 2 and col == 2:
                fig3.update_yaxes(title_text="°C", row=row, col=col)
            elif row == 3 and col == 1:
                fig3.update_yaxes(title_text="Earthquakes", row=row, col=col)
            
            # Move to next position
            col += 1
            if col > 2:
                col = 1
                row += 1
        
        # Update layout
        fig3.update_layout(
            height=900,
            showlegend=False,
            template="plotly_white",
            title="India's Environmental and Economic Indicators",
            xaxis1_title="Year",
            xaxis2_title="Year",
            xaxis3_title="Year",
            xaxis4_title="Year",
            xaxis5_title="Year"
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Plot 4: Decade Analysis
        st.markdown('<h3>Decadal Averages (1960-2020)</h3>', unsafe_allow_html=True)
        
        # Create subplots
        fig4 = make_subplots(rows=2, cols=2, subplot_titles=("CO₂ Emissions", "GDP Growth", "Energy Use", "Temperature"))
        
        # Add traces
        fig4.add_trace(
            go.Bar(x=india_data["decade_data"]["Decade"], y=india_data["decade_data"]["CO2"], name="CO2"),
            row=1, col=1
        )
        
        fig4.add_trace(
            go.Bar(x=india_data["decade_data"]["Decade"], y=india_data["decade_data"]["GDP"], name="GDP"),
            row=1, col=2
        )
        
        fig4.add_trace(
            go.Bar(x=india_data["decade_data"]["Decade"], y=india_data["decade_data"]["Energy"], name="Energy"),
            row=2, col=1
        )
        
        fig4.add_trace(
            go.Scatter(x=india_data["decade_data"]["Decade"], y=india_data["decade_data"]["Temperature"], 
                      name="Temperature", mode="lines+markers", line=dict(color="red", width=3)),
            row=2, col=2
        )
        
        # Update layout
        fig4.update_layout(
            height=700,
            showlegend=False,
            template="plotly_white"
        )
        
        st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="section-title">Correlation Analysis</h2>', unsafe_allow_html=True)
        
        # Plot 5: India's Emissions vs Temperature
        st.markdown('<h3>India: CO₂ Emissions vs Temperature</h3>', unsafe_allow_html=True)
        
        # Create figure with regression line
        fig5 = px.scatter(
            india_data["india_data"],
            x="CO2",
            y="Temperature",
            trendline="ols",
            labels={
                "CO2": "Emissions (Thousand Metric Tons)",
                "Temperature": "Temperature (°C)"
            },
            template="plotly_white",
            trendline_color_override="red"
        )
        
        # Calculate regression statistics
        slope, intercept, r_value, p_value, std_err = linregress(
            india_data["india_data"]["CO2"].dropna(),
            india_data["india_data"]["Temperature"].dropna()
        )
        
        # Add regression stats annotation
        fig5.add_annotation(
            x=0.05,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"R² = {r_value**2:.3f}<br>p-value = {p_value:.3e}",
            showarrow=False,
            bgcolor="black",
            bordercolor="red",
            borderwidth=1
        )
        
        # Add event annotations
        events = {
            1973: {'x': india_data["india_data"].loc[india_data["india_data"]['Year'] == 1973, 'CO2'].values[0],
                   'y': india_data["india_data"].loc[india_data["india_data"]['Year'] == 1973, 'Temperature'].values[0],
                   'text': '1973: Oil Crisis'},
            1991: {'x': india_data["india_data"].loc[india_data["india_data"]['Year'] == 1991, 'CO2'].values[0],
                   'y': india_data["india_data"].loc[india_data["india_data"]['Year'] == 1991, 'Temperature'].values[0],
                   'text': '1991: Economic Reforms'},
            2020: {'x': india_data["india_data"].loc[india_data["india_data"]['Year'] == 2020, 'CO2'].values[0],
                   'y': india_data["india_data"].loc[india_data["india_data"]['Year'] == 2020, 'Temperature'].values[0],
                   'text': '2020: COVID-19'}
        }
        
        for year, event in events.items():
            fig5.add_annotation(
                x=event['x'],
                y=event['y'],
                text=event['text'],
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
        
        fig5.update_layout(height=500)
        st.plotly_chart(fig5, use_container_width=True)
        
        # Additional analysis: Per Capita Metrics
        st.markdown('<h3>Per Capita Metrics Analysis</h3>', unsafe_allow_html=True)
        
        # Create subplots
        fig6 = make_subplots(rows=1, cols=2, subplot_titles=("CO₂ per Capita", "Energy per Capita"))
        
        # Add CO2 per capita trace
        fig6.add_trace(
            go.Scatter(
                x=india_data["india_data"]["Year"], 
                y=india_data["india_data"]["CO2_per_capita"],
                name="CO2 per Capita", 
                line=dict(color="#1a2a6c", width=3)
            ),
            row=1, col=1
        )
        
        # Add energy per capita trace
        fig6.add_trace(
            go.Scatter(
                x=india_data["india_data"]["Year"], 
                y=india_data["india_data"]["Energy_per_capita"],
                name="Energy per Capita", 
                line=dict(color="#e63946", width=3)
            ),
            row=1, col=2
        )
        
        # Update layout
        fig6.update_layout(
            height=500,
            showlegend=False,
            template="plotly_white",
            xaxis_title="Year",
            yaxis_title="Metric Tons per Capita",
            xaxis2_title="Year",
            yaxis2_title="kg Oil Equivalent per Capita"
        )
        
        st.plotly_chart(fig6, use_container_width=True)

# Footer
st.markdown("---")
st.markdown('<div class="footer">ENVECON 105: Data Tools for Sustainability and the Environment | UC Berkeley | August 2025</div>', unsafe_allow_html=True)