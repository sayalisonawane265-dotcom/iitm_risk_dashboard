import os
import sys
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import Search
import branca.colormap as cm
from branca.element import Template, MacroElement
import streamlit as st
from streamlit_folium import st_folium
import plotly.express as px
from scipy import stats
import warnings
import rioxarray

# ==========================================
# 1. ENVIRONMENT & GLOBAL PLATFORM PLUGINS
# ==========================================
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="IMD District Rainfall & Risk Platform")

# Central file path targets for local network server assets
DATA_FOLDER ="Rainfall_30_yrs_data/"
SHP_PATH = "dist_shape_fromJSON/dist2023.shp"
TIF_PATH = "ind_pop_2023_CN_1km_R2025A_UA_v1.tif"

# Risk Palette for Tab 4
RISK_PALETTE = {
    "VERY HIGH": "#7030a0", "HIGH": "#ff0000",
    "MEDIUM": "#ffc000", "LOW": "#ffff00", "NEGLIGIBLE": "#ffffff"
}

# Cached vector boundary data ingestion engine
@st.cache_data
def load_static_layers(shape_path):
    dist_base = gpd.read_file(shape_path).to_crs(epsg=4326)
    dist_base['geometry'] = dist_base['geometry'].simplify(0.005)
    return dist_base

# Cached Population extraction engine (Moved from old code to keep dashboard fast)
@st.cache_data
def load_population_data(shape_path, tif_path):
    districts = gpd.read_file(shape_path).to_crs(epsg=4326)
    districts['geometry'] = districts['geometry'].simplify(0.01)
    
    try:
        with xr.open_dataset(tif_path, engine="rasterio") as pop_ds:
            pop_raster = pop_ds.squeeze().band_data.load()
            pop_raster.rio.write_crs("epsg:4326", inplace=True)
            
            pop_list = []
            for _, r in districts.iterrows():
                clip = pop_raster.rio.clip([r.geometry], "epsg:4326", all_touched=True)
                val = float(np.nansum(clip.values))
                pop_list.append(val)
        
        districts['total_pop'] = pop_list
        return districts[['district_n', 'total_pop']]
    except Exception as e:
        # Fallback if TIFF is missing during testing
        districts['total_pop'] = 0.0
        return districts[['district_n', 'total_pop']]

districts_base = load_static_layers(SHP_PATH)
population_base = load_population_data(SHP_PATH, TIF_PATH)


# ==========================================
# 2. SELECTION HELPER FUNCTION ARRAYS
# ==========================================
def get_standard_category(departure):
    if departure >= 60: return "Large Excess"
    elif 20 <= departure < 60: return "Excess"
    elif -19 <= departure < 20: return "Normal"
    elif -59 <= departure < -20: return "Deficient"
    else: return "Large Deficient"

def get_probability_level(current_rain, historical_series):
    if len(historical_series) == 0: return 1
    percentile = stats.percentileofscore(historical_series, current_rain)
    if percentile > 98: return 4    
    elif percentile > 90: return 3  
    elif percentile > 75: return 2  
    else: return 1                  

def get_severity_level(current_rain):
    if current_rain >= 100: return 4     
    elif 50 <= current_rain < 100: return 3 
    elif 15 <= current_rain < 50: return 2  
    else: return 1                          

def get_risk_score(p_level, s_level):
    matrix = {
        4: {1: "#FFFF00", 2: "#FFA500", 3: "#FF0000", 4: "#FF0000"},
        3: {1: "#92D050", 2: "#FFFF00", 3: "#FFA500", 4: "#FF0000"},
        2: {1: "#92D050", 2: "#92D050", 3: "#FFFF00", 4: "#FFA500"},
        1: {1: "#92D050", 2: "#92D050", 3: "#92D050", 4: "#FFFF00"} 
    }
    return matrix[p_level][s_level]


# ==========================================
# 3. GLOBAL SIDEBAR MASTER FORM FRAMEWORK
# ==========================================
# Modified to standard sidebar (removed form) to allow dynamic hiding of End Date
with st.sidebar:
    st.subheader("Data Selection")

    if os.path.exists(DATA_FOLDER):
        available_files = [f for f in os.listdir(DATA_FOLDER) if f.startswith("RF25_ind") and f.endswith("_rfp25.nc")]
        detected_years = sorted(list(set([f.split("ind")[1].split("_")[0] for f in available_files])))
    else:
        detected_years = [str(y) for y in range(1995, 2025)]

    selected_year = st.selectbox("Select Year:", detected_years, index=0)

    st.markdown("---")
    st.subheader("Time Configuration")
    time_mode = st.selectbox("Select Time View:", ["Single Day", "Date Range"], index=0)
    
    st.markdown("---")
    st.markdown("Date Selection Framework:")
    start_d = st.date_input("Start / Target Date:", value=datetime.date(int(selected_year), 6, 1))
    
    # Dynamically display End Date only if Date Range is selected
    end_d = start_d
    if time_mode == "Date Range":
        end_d = st.date_input("End Date (Active for Range View):", value=datetime.date(int(selected_year), 6, 7))

    submit_button = st.button("Generate Analytics", use_container_width=True)

NC_PATH = os.path.join(DATA_FOLDER, f"RF25_ind{selected_year}_rfp25.nc")


# ==========================================
# 4. EXECUTION CONTROLLER GUARD ENGINE
# ==========================================
if submit_button or 'execution_state' in st.session_state:
    st.session_state.execution_state = True 

    if not os.path.exists(NC_PATH):
        st.error(f"NetCDF dataset for year {selected_year} not found at location: {NC_PATH}")
        st.stop()

    ds = xr.open_dataset(NC_PATH)
    ds.rio.write_crs("epsg:4326", inplace=True)

    file_start = pd.to_datetime(ds.TIME.min().values).date()
    file_end = pd.to_datetime(ds.TIME.max().values).date()

    if time_mode == "Single Day":
        target_date = max(min(start_d, file_end), file_start)
        date_label = pd.to_datetime(target_date).strftime('%d-%b-%Y')
        data_slice = ds.RAINFALL.sel(TIME=pd.to_datetime(target_date), method='nearest')
    else:
        bounded_start = max(min(start_d, file_end), file_start)
        bounded_end = max(min(end_d, file_end), file_start)
        if bounded_start > bounded_end:
            bounded_start, bounded_end = bounded_end, bounded_start
            
        date_label = f"Avg: {pd.to_datetime(bounded_start).strftime('%d-%b')} to {pd.to_datetime(bounded_end).strftime('%d-%b-%Y')}"
        data_slice = ds.RAINFALL.sel(TIME=slice(pd.to_datetime(bounded_start), pd.to_datetime(bounded_end))).mean(dim='TIME')

    df = districts_base.copy().reset_index(drop=True)
    results = []

    for poly in df['geometry']:
        try:
            clipped = data_slice.rio.clip([poly], ds.rio.crs, all_touched=True)
            val = float(np.nanmean(clipped.values))
            results.append(val if np.isfinite(val) else 0.0)
        except:
            results.append(0.0)
    df['daily_avg'] = results
    p25, p95, p99 = np.percentile(df['daily_avg'], [25, 95, 99])


    # ==========================================
    # 5. PROFESSIONAL INTERFACE TAB INFRASTRUCTURE
    # ==========================================
    st.title("District Rainfall & Risk Analytics System")
    st.sidebar.markdown(f"Active File: \n`RF25_ind{selected_year}_rfp25.nc`")
    st.markdown("---")

    # Added the 4th tab here
    tab_dashboard, tab_climatology, tab_risk_matrix, tab_pop_risk = st.tabs([
        "Real-time Dashboard", 
        "Climatology Analysis", 
        "Interactive Risk Matrix",
        "Population Risk Exposure"
    ])


    # ==========================================
    # TAB 1: REAL-TIME COUNTRY DASHBOARD 
    # ==========================================
    with tab_dashboard:
        st.markdown(f"### National Rainfall Summary: {date_label}")
        summary_df = pd.DataFrame({
            "Metric": ["Min", "Max", "Avg", "25th Perc.", "95th Perc.", "99th Perc."],
            "Value": [f"{df.daily_avg.min():.2f} mm", f"{df.daily_avg.max():.2f} mm", f"{df.daily_avg.mean():.2f} mm", 
                      f"{p25:.2f} mm", f"{p95:.2f} mm", f"{p99:.2f} mm"]
        }).set_index("Metric")
        st.dataframe(summary_df.T, use_container_width=True)

        color_map = {
            "Extremely Heavy": "#7030a0", "Very Heavy": "#ff0000", "Heavy": "#ffc000",
            "Moderate": "#ffff00", "Light": "#92d050", "Very Light": "#c6efce", "No Rain": "#ffffff"
        }

        t1 = df[df['daily_avg'] >= 204.4].nlargest(5, 'daily_avg')
        t2 = df[(df['daily_avg'] >= 115.6) & (df['daily_avg'] < 204.4)].nlargest(5, 'daily_avg')
        t3 = df[(df['daily_avg'] >= 64.5) & (df['daily_avg'] < 115.6)].nlargest(5, 'daily_avg')
        t4 = df[(df['daily_avg'] >= 15.6) & (df['daily_avg'] < 64.5)].nlargest(5, 'daily_avg')
        t5 = df[(df['daily_avg'] >= 2.5) & (df['daily_avg'] < 15.6)].nlargest(5, 'daily_avg')
        t6 = df[(df['daily_avg'] >= 0.1) & (df['daily_avg'] < 2.5)].nlargest(5, 'daily_avg')
        t7 = df[df['daily_avg'] < 0.1].head(5)
        tiered_df = pd.concat([t1, t2, t3, t4, t5, t6, t7])[['district_n', 'daily_avg']].round(2)

        col_graph, col_table = st.columns([1.1, 0.9])

        with col_graph:
            fig, ax1 = plt.subplots(figsize=(10, 5.5))
            sns.histplot(df['daily_avg'], bins=40, kde=True, color='#74add1', ax=ax1)
            ax1.axvspan(0, p25, color='gray', alpha=0.25, label='Bottom 25%')
            ax1.axvline(p25, color='gray', lw=1, ls='--')
            ax1.axvline(p95, color='orange', lw=2, ls='--', label='95th Perc.')
            ax1.axvline(p99, color='red', lw=2.5, ls='-', label='99th Perc. (Extreme)')
            
            trans = ax1.get_xaxis_transform()
            ax1.text(p25, 0.75, f' 25%: {p25:.1f}', color='dimgray', transform=trans, fontweight='bold')
            ax1.text(p95, 0.85, f' 95%: {p95:.1f}', color='orange', transform=trans, fontweight='bold')
            ax1.text(p99, 0.92, f' 99%: {p99:.1f}', color='red', transform=trans, fontweight='bold')
            ax1.set_title("Rainfall Distribution Analysis", fontsize=14, fontweight='bold')
            ax1.legend()
            st.pyplot(fig)
            plt.close()

        with col_table:
            st.markdown("**District Stratification Tier Highlights**")
            def style_rows_by_tier(val):
                if val >= 204.4: c = color_map["Extremely Heavy"]
                elif val >= 115.6: c = color_map["Very Heavy"]
                elif val >= 64.5: c = color_map["Heavy"]
                elif val >= 15.6: c = color_map["Moderate"]
                elif val >= 2.5: c = color_map["Light"]
                elif val >= 0.1: c = color_map["Very Light"]
                else: c = color_map["No Rain"]
                text_color = "white" if (val >= 115.6 or val >= 204.4) else "black"
                return f'background-color: {c}; color: {text_color}; font-weight: bold;'

            st.dataframe(tiered_df.style.map(style_rows_by_tier, subset=['daily_avg']), use_container_width=True, height=380)

        st.markdown("### Spatial Choropleth Hazard Map Representation")
        m = folium.Map(location=[22.0, 78.0], zoom_start=5, tiles=None, height=600)
        imd_cmap = cm.StepColormap(
            colors=[color_map["No Rain"], color_map["Very Light"], color_map["Light"], color_map["Moderate"], 
                    color_map["Heavy"], color_map["Very Heavy"], color_map["Extremely Heavy"]],
            index=[0, 0.1, 2.5, 15.6, 64.5, 115.6, 204.4, max(df.daily_avg.max(), 205)],
            caption="Rainfall (mm)"
        )
        df.explore(m=m, column='daily_avg', cmap=imd_cmap, tooltip=['district_n', 'daily_avg'], style_kwds=dict(color="black", weight=0.3, fillOpacity=0.8))
        imd_cmap.show = False
        m.add_child(imd_cmap)

        legend_html = '''
        <div style="position: fixed; top: 120px; right: 30px; width: 220px; height: 260px; 
                    background-color: white; border:2px solid grey; z-index:9999; font-size:12px;
                    padding: 12px; border-radius: 5px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3); 
                    font-family: Arial; color: black !important;">
        <b style="font-size:13px; color: black !important;">Rainfall Categories</b><br><br>
        <i style="background:%s;width:15px;height:15px;display:inline-block;vertical-align:middle;margin-right:5px;"></i> <span style="color: black !important;">> 204.4 (Ext. Heavy)</span><br>
        <i style="background:%s;width:15px;height:15px;display:inline-block;vertical-align:middle;margin-right:5px;"></i> <span style="color: black !important;">115.6 - 204.4 (Very Heavy)</span><br>
        <i style="background:%s;width:15px;height:15px;display:inline-block;vertical-align:middle;margin-right:5px;"></i> <span style="color: black !important;">64.5 - 115.5 (Heavy)</span><br>
        <i style="background:%s;width:15px;height:15px;display:inline-block;vertical-align:middle;margin-right:5px;"></i> <span style="color: black !important;">15.6 - 64.4 (Moderate)</span><br>
        <i style="background:%s;width:15px;height:15px;display:inline-block;vertical-align:middle;margin-right:5px;"></i> <span style="color: black !important;">2.5 - 15.5 (Light)</span><br>
        <i style="background:%s;width:15px;height:15px;display:inline-block;vertical-align:middle;margin-right:5px;"></i> <span style="color: black !important;">0.1 - 2.4 (Very Light)</span><br>
        <i style="background:%s;width:15px;height:15px;display:inline-block;vertical-align:middle;margin-right:5px;border:1px solid #ddd"></i> <span style="color: black !important;">0.0 (No Rain)</span>
        </div>
        ''' % (color_map["Extremely Heavy"], color_map["Very Heavy"], color_map["Heavy"], color_map["Moderate"], color_map["Light"], color_map["Very Light"], color_map["No Rain"])
        m.get_root().html.add_child(folium.Element(legend_html))

        for child in m._children.values():
            if isinstance(child, folium.features.GeoJson):
                Search(layer=child, geom_type='Polygon', placeholder="Find District...", search_label='district_n').add_to(m)
                break
        st_folium(m, width=1300, height=600, returned_objects=[], key="pure_widgets_map")


    # ==========================================
    # TAB 2: CLIMATOLOGY ANALYSIS PLATFORM
    # ==========================================
    with tab_climatology:
        st.header("Long-term Climatology Baseline Comparison")
        st.markdown("Analyze multi-decadal chronological baseline deviations down to specific targeted operational district layers.")
        
        sorted_districts_list = sorted(df['district_n'].unique().tolist())
        target_dist = st.selectbox("Select Target District for Climatology:", sorted_districts_list, key="local_climatology_dist")

        gdf_layer = districts_base.to_crs(epsg=4326)
        geom = gdf_layer[gdf_layer['district_n'] == target_dist].geometry.values[0]
        
        all_nc_files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith('.nc')])
        start_day_of_year = pd.to_datetime(start_d).timetuple().tm_yday
        end_day_of_year = pd.to_datetime(end_d).timetuple().tm_yday if time_mode == "Date Range" else start_day_of_year
        
        yearly_climatology_totals = []
        for nc_f in all_nc_files:
            file_year_str = nc_f.split('_ind')[1].split('_')[0]
            loop_file_path = os.path.join(DATA_FOLDER, nc_f)
            with xr.open_dataset(loop_file_path) as loop_ds:
                loop_ds = loop_ds.rio.write_crs("epsg:4326")
                loop_ds = loop_ds.rio.set_spatial_dims(x_dim="LONGITUDE", y_dim="LATITUDE")
                period_subset = loop_ds.RAINFALL.sel(TIME=((loop_ds.TIME.dt.dayofyear >= start_day_of_year) & (loop_ds.TIME.dt.dayofyear <= end_day_of_year)))
                if len(period_subset.TIME) > 0:
                    try:
                        clipped_raster = period_subset.rio.clip([geom], "epsg:4326", all_touched=True)
                        computed_mean = float(np.nanmean(clipped_raster.values))
                        yearly_climatology_totals.append({
                            'Year': int(file_year_str), 
                            'Rainfall': round(computed_mean, 2) if np.isfinite(computed_mean) else 0.0
                        })
                    except:
                        yearly_climatology_totals.append({'Year': int(file_year_str), 'Rainfall': 0.0})

        climatology_df = pd.DataFrame(yearly_climatology_totals).sort_values(by="Year").reset_index(drop=True)
        long_period_average = climatology_df['Rainfall'].mean()

        if long_period_average > 0:
            climatology_df['Departure'] = ((climatology_df['Rainfall'] - long_period_average) / long_period_average * 100).round(2)
        else:
            climatology_df['Departure'] = climatology_df['Rainfall'].apply(lambda x: 100.0 if x > 0 else 0.0)
        
        climatology_df['Classification'] = climatology_df['Departure'].apply(get_standard_category)
        
        reference_year_row = climatology_df[climatology_df['Year'] == int(selected_year)]
        observed_value = reference_year_row['Rainfall'].values[0] if not reference_year_row.empty else 0.0
        departure_value = reference_year_row['Departure'].values[0] if not reference_year_row.empty else 0.0

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Long Period Average (LPA)", f"{round(long_period_average, 2)} mm")
        metric_col2.metric(f"Observation ({selected_year})", f"{observed_value} mm")
        metric_col3.metric("Rainfall Departure Relative to Baseline", f"{departure_value}%", delta=f"{departure_value}%")

        st.markdown("---")
        st.subheader("30-Year Chronological Deviation Profile")
        
        climatology_color_discrete_map = {
            "Large Excess": "#0000FF", "Excess": "#00B0F0", "Normal": "#92D050", "Deficient": "#FFFF00", "Large Deficient": "#FF0000"
        }

        plotly_trend_fig = px.bar(
            climatology_df, x="Year", y="Rainfall", template="simple_white",
            color="Classification", color_discrete_map=climatology_color_discrete_map,
            category_orders={"Classification": ["Large Excess", "Excess", "Normal", "Deficient", "Large Deficient"]}
        )
        plotly_trend_fig.add_hline(y=long_period_average, line_dash="dash", line_color="black", annotation_text="LPA Reference Baseline Line")
        plotly_trend_fig.update_xaxes(type='linear', tickmode='linear', dtick=1, range=[climatology_df['Year'].min() - 0.5, climatology_df['Year'].max() + 0.5])
        st.plotly_chart(plotly_trend_fig, use_container_width=True)

        st.info(f"""
        **Automated Analytical Diagnosis:** Within the filtered timeframe, **{target_dist}** logged an observed value of **{observed_value} mm** versus a multi-decadal baseline LPA of **{round(long_period_average, 2)} mm**. 
        This is evaluated as a net departure shift of **{departure_value}%**, placing the performance inside the formal **{get_standard_category(departure_value)}** envelope.
        """)


    # ==========================================
    # TAB 3: INTERACTIVE RISK MATRIX PLATFORM
    # ==========================================
    with tab_risk_matrix:
        st.header("Interactive Hazard Risk Matrix Verification")
        st.markdown("Plots extreme value probabilities alongside current absolute volume triggers following structural contingency logic matrices.")
        
        target_dist_risk = st.selectbox("Select Target District for Risk Matrix Verification:", sorted_districts_list, key="local_risk_dist")

        historical_rainfall_series_list = climatology_df['Rainfall'].tolist()
        historical_percentile_rank_score = stats.percentileofscore(historical_rainfall_series_list, observed_value) if len(historical_rainfall_series_list) > 0 else 0
        
        probability_level_index = get_probability_level(observed_value, historical_rainfall_series_list)
        severity_level_index = get_severity_level(observed_value)

        # Reconstructed the cutoff code perfectly here
        matrix_column_headers = ["S1: Low", "S2: Mod", "S3: High", "S4: Ext"]
        matrix_row_headers = ["P4: Rare", "P3: Unlikely", "P2: Possible", "P1: Likely"]
        static_color_hex_grid = [
            ["#FFFF00", "#FFA500", "#FF0000", "#FF0000"], 
            ["#92D050", "#FFFF00", "#FFA500", "#FF0000"],
            ["#92D050", "#92D050", "#FFFF00", "#FFA500"],
            ["#92D050", "#92D050", "#92D050", "#FFFF00"]
        ]
        
        st.markdown("#### 4x4 Probability vs Severity Weighting")
        
        # Plotly representation of the explicit hex grid for clean label reading
        fig_matrix = px.imshow(
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], 
            x=matrix_column_headers, 
            y=matrix_row_headers,
            color_continuous_scale="Viridis"
        )
        
        # Override colors to match strict hex formatting
        fig_matrix.update_traces(
            type='heatmap',
            z=[[0]*4]*4, 
            colorscale=[[0, 'white'], [1, 'white']], showscale=False,
            hoverinfo='none'
        )
        
        for i, row in enumerate(matrix_row_headers):
            for j, col in enumerate(matrix_column_headers):
                border_color = "black"
                line_width = 1
                # Highlight active grid sector
                if (4 - i) == probability_level_index and (j + 1) == severity_level_index:
                    border_color = "blue"
                    line_width = 4
                    # Replaced "TARGET" with "✖" here
                    fig_matrix.add_annotation(x=j, y=i, text="<b>✖</b>", showarrow=False, font=dict(color="black", size=24))
                
                fig_matrix.add_shape(
                    type="rect", x0=j-0.5, x1=j+0.5, y0=i-0.5, y1=i+0.5,
                    fillcolor=static_color_hex_grid[i][j], line=dict(color=border_color, width=line_width)
                )

        st.plotly_chart(fig_matrix, use_container_width=True)
        
        # Drag-down option for test procedure / mathematical background
        with st.expander("Mathematical Procedure & Risk Matrix Calculation"):
            st.latex(r"Risk\_Score = f(P_{percentile}, S_{absolute})")
            st.markdown("""
            **Methodology:**
            1. **Probability Level (P1-P4):** Evaluated by calculating the percentile rank of the observed daily rainfall against the 30-year historical baseline distribution:
               * $P1: \leq 75^{th}$ percentile
               * $P2: 75^{th} - 90^{th}$ percentile
               * $P3: 90^{th} - 98^{th}$ percentile
               * $P4: > 98^{th}$ percentile
            
            2. **Severity Level (S1-S4):** Assigned using fixed absolute volumetric (mm) thresholds:
               * $S1: < 15$ mm (Low)
               * $S2: 15 - 50$ mm (Moderate)
               * $S3: 50 - 100$ mm (High)
               * $S4: \geq 100$ mm (Extreme)
            
            3. **Matrix Intersection:** The resulting coordinates $(P_x, S_y)$ are mapped to the 4x4 logic grid to yield the final categorical risk assessment.
            """)


    # ==========================================
    # TAB 4: POPULATION RISK EXPOSURE (NEW)
    # ==========================================
    with tab_pop_risk:
        st.header("Vulnerability & Population Risk Overlay")
        st.markdown("Identifies high-priority districts by combining absolute rainfall hazard severity with human population exposure metrics.")
        
        # The new selectbox specific to this tab
        analysis_var = st.selectbox("Select Analysis Variable:", ["Population Exposure", "Temperature Baseline"], index=0)
        
        if analysis_var == "Population Exposure":
            # Merge the current active dataframe (which has daily_avg) with the cached population data
            tab4_df = pd.merge(df, population_base, on='district_n', how='left')
            
            # --- OLD RISK LOGIC APPLIED TO NEW DATA ---
            RAIN_MAX_VAL = 200.0   
            POP_MAX_LOG = 7.0      
            
            tab4_df['r_norm'] = (tab4_df['daily_avg'] / RAIN_MAX_VAL).clip(0, 1) 
            tab4_df['log_p'] = np.log10(tab4_df['total_pop'] + 1) 
            tab4_df['p_norm'] = (tab4_df['log_p'] / POP_MAX_LOG).clip(0, 1) 
            tab4_df['risk_index'] = (tab4_df['r_norm'] * tab4_df['p_norm']).round(2) 

            def assign_tier(row):
                if row['daily_avg'] < 2.5: return "NEGLIGIBLE" 
                score = row['risk_index']
                if score >= 0.60: return "VERY HIGH"
                if score >= 0.40: return "HIGH"
                if score >= 0.15: return "MEDIUM"
                return "LOW"
            
            tab4_df['risk_level'] = tab4_df.apply(assign_tier, axis=1) 
            
            # Recreate Floating Map Legend for Tab 4
            legend_html_tab4 = """
            {% macro html(this, kwargs) %}
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 200px; height: 180px; 
                        background-color: white; color: black; border: 3px solid #333; 
                        z-index: 9999; font-size: 14px; padding: 12px; border-radius: 8px; 
                        font-family: 'Arial', sans-serif; font-weight: bold;
                        box-shadow: 4px 4px 10px rgba(0,0,0,0.5);">
                <p style="margin-top:0; margin-bottom:12px; border-bottom: 2px solid #333; font-size: 15px;">Risk Index Score</p>
                <div style="margin-bottom: 5px;"><i class="fa fa-square fa-1x" style="color:#7030a0"></i> Very High (> 0.60)</div>
                <div style="margin-bottom: 5px;"><i class="fa fa-square fa-1x" style="color:#ff0000"></i> High (0.40 - 0.60)</div>
                <div style="margin-bottom: 5px;"><i class="fa fa-square fa-1x" style="color:#ffc000"></i> Medium (0.15 - 0.40)</div>
                <div style="margin-bottom: 5px;"><i class="fa fa-square fa-1x" style="color:#ffff00"></i> Low (< 0.15)</div>
                <div><i class="fa fa-square fa-1x" style="color:#ffffff; border:1px solid #ccc"></i> Negligible (< 2.5mm)</div>
            </div>
            {% endmacro %}
            """

            # Initialize Folium map
            m_pop = folium.Map(location=[22.5, 78.5], zoom_start=5, tiles=None)
            
            tooltip = folium.GeoJsonTooltip(
                fields=['district_n', 'avg_rain', 'risk_index', 'risk_level'], 
                aliases=['District:', 'Rain(mm):', 'Index Score:', 'Risk Level:'],
                localize=True
            )

            # Map the dataframe to GeoJSON formatting
            geo_df = tab4_df.copy()
            geo_df['avg_rain'] = geo_df['daily_avg'].round(2)

            folium.GeoJson(
                geo_df,
                style_function=lambda x: {
                    'fillColor': RISK_PALETTE.get(x['properties']['risk_level'], 'white'),
                    'color': 'black', 'weight': 0.2, 'fillOpacity': 0.8
                },
                tooltip=tooltip
            ).add_to(m_pop)
            
            macro = MacroElement()
            macro._template = Template(legend_html_tab4)
            m_pop.get_root().add_child(macro)
            
            # Render map
            st_folium(m_pop, width=1100, height=600, returned_objects=[], key="pop_risk_map")
            
            # Display detailed data table
            st.subheader(f"Verification Table: {analysis_var} ({selected_year})")
            display_df = tab4_df[['district_n', 'daily_avg', 'total_pop', 'risk_index', 'risk_level']].sort_values('risk_index', ascending=False)
            display_df.rename(columns={'daily_avg': 'avg_rain'}, inplace=True)
            st.dataframe(display_df, use_container_width=True)
            
            # Drag-down option for test procedure / mathematical background
            with st.expander("Mathematical Procedure & Population Risk Index Calculation"):
                st.latex(r"Risk\_Index = \left( \frac{Rain_{avg}}{200.0} \right)_{norm} \times \left( \frac{\log_{10}(Total\_Pop + 1)}{7.0} \right)_{norm}")
                st.markdown("""
                **1. Hazard Normalization:** District average rainfall is divided by a maximum threshold (200.0 mm) and capped at 1.0.
                **2. Exposure Normalization:** Total population is transformed using a base-10 logarithm to manage high variance, divided by a maximum log-scale limit (7.0), and capped at 1.0.
                **3. Multiplicative Index:** The normalized hazard and exposure values are multiplied to generate a unified risk score ranging from 0.0 to 1.0.
                """)
            
        else:
            st.info("Temperature logic is pending future integration.")
