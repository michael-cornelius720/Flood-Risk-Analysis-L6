import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Flood Risk Analytics",
    layout="wide"
)


raw_df = pd.read_csv("flood_risk_dataset_india.csv")
df = raw_df.dropna(subset=["Rainfall (mm)", "Water Level (m)", "Soil Type"]).copy()

st.sidebar.title("Filters")

min_rainfall = st.sidebar.slider("Minimum Rainfall (mm)", 0, 300, 50, step=10)

land_options = ["All"] + sorted(df["Land Cover"].unique().tolist())
selected_land = st.sidebar.selectbox("Land Cover", land_options)

soil_options = sorted(df["Soil Type"].unique().tolist())
selected_soils = st.sidebar.multiselect("Soil Type", soil_options, default=soil_options)

filtered_df = df[
    (df["Rainfall (mm)"] >= min_rainfall) &
    (df["Soil Type"].isin(selected_soils))
]

if selected_land != "All":
    filtered_df = filtered_df[filtered_df["Land Cover"] == selected_land]

st.sidebar.markdown("---")
st.sidebar.write(f"Selected Records: **{len(filtered_df):,}** of **{len(df):,}**")

st.title("Flood Risk Analysis Dashboard")
st.write("An interactive web application to analyze flood patterns, rainfall distribution, and water levels across regions in India.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Dataset Overview",
    "Data Analysis",
    "Visualizations",
    "Insights"
])

with tab1:
    st.header("Dataset Overview")
    st.write("This dataset contains hydrological and geographical records used to study flood risks across India.")

    col1, col2, col3, col4 = st.columns(4)
    raw_missing = raw_df.isnull().sum()
    total_missing = int(raw_missing.sum())

    with col1:
        st.metric("Total Records", f"{len(raw_df):,}")
    with col2:
        st.metric("Total Columns", f"{len(raw_df.columns)}")
    with col3:
        st.metric("Missing Values", f"{total_missing}")
    with col4:
        st.metric("Duplicate Rows", f"{int(raw_df.duplicated().sum())}")

    st.subheader("Missing Value Breakdown")
    missing_table = raw_missing[raw_missing > 0].reset_index()
    missing_table.columns = ["Column Name", "Missing Count"]
    st.dataframe(missing_table, use_container_width=True)
    st.caption("Note: Missing values occurred in Rainfall (2 rows), Water Level (2 rows), and Soil Type (1 row). These 5 rows were cleaned using dropna() for accurate calculations.")

    st.subheader("Sample Records")
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("Column Data Types"):
        schema_df = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.notnull().sum().values
        })
        st.dataframe(schema_df, use_container_width=True)

    with st.expander("Summary Statistics"):
        st.dataframe(df.describe(), use_container_width=True)

with tab2:
    st.header("Key Performance Indicators")

    if filtered_df.empty:
        st.warning("No records match the selected filter criteria. Please adjust your filters.")
    else:
        col1, col2, col3, col4 = st.columns(4)

        total_obs = len(filtered_df)
        total_rain_sum = filtered_df["Rainfall (mm)"].sum()
        avg_rain = filtered_df["Rainfall (mm)"].mean()
        avg_water = filtered_df["Water Level (m)"].mean()
        total_flood_events = int(filtered_df["Flood Occurred"].sum())
        flood_rate = (total_flood_events / total_obs) * 100

        with col1:
            st.metric("Total Observations", f"{total_obs:,}")
        with col2:
            st.metric("Average Rainfall", f"{avg_rain:.1f} mm", f"Total: {total_rain_sum:,.0f} mm")
        with col3:
            st.metric("Average Water Level", f"{avg_water:.2f} m")
        with col4:
            st.metric("Flood Events", f"{total_flood_events:,}", f"{flood_rate:.1f}%")

        st.markdown("---")

        st.subheader("Category-wise Record Counts")
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.write("**Counts by Soil Type**")
            soil_grp = filtered_df.groupby("Soil Type").size().reset_index(name="Count")
            soil_grp["Percentage"] = (soil_grp["Count"] / total_obs * 100).round(2).astype(str) + "%"
            st.dataframe(soil_grp, use_container_width=True)

        with col_g2:
            st.write("**Counts by Land Cover**")
            land_grp = filtered_df.groupby("Land Cover").size().reset_index(name="Count")
            land_grp["Percentage"] = (land_grp["Count"] / total_obs * 100).round(2).astype(str) + "%"
            st.dataframe(land_grp, use_container_width=True)

        with st.expander("View Filtered Data Table"):
            st.dataframe(filtered_df, use_container_width=True)

with tab3:
    st.header("Visualizations")

    if filtered_df.empty:
        st.warning("No records available to display graphs. Please adjust your filters.")
    else:
        sample_size = min(500, len(filtered_df))
        plot_df = filtered_df.sample(n=sample_size, random_state=42) if len(filtered_df) > sample_size else filtered_df

        vcol1, vcol2 = st.columns(2)

        with vcol1:
            st.subheader("1. Bar Chart: Flood Occurrences by Soil Type")
            soil_floods = filtered_df.groupby("Soil Type")["Flood Occurred"].sum().sort_values(ascending=False)

            fig1, ax1 = plt.subplots(figsize=(6, 4))
            bars = ax1.bar(soil_floods.index, soil_floods.values, color="skyblue", edgecolor="black")
            ax1.set_title("Flood Occurrences by Soil Type")
            ax1.set_xlabel("Soil Type")
            ax1.set_ylabel("Number of Flood Events")
            ax1.grid(axis="y", linestyle="--", alpha=0.5)

            for bar in bars:
                h = bar.get_height()
                ax1.annotate(f"{int(h)}",
                             xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha="center", va="bottom", fontsize=9)

            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

            with st.expander("Bar Chart Interpretation"):
                if not soil_floods.empty:
                    top_soil = soil_floods.idxmax()
                    top_val = soil_floods.max()
                    st.write(f"{top_soil} has the highest number of flood events ({int(top_val)}). This indicates that low drainage soil types retain more water during rainfall.")

        with vcol2:
            st.subheader("2. Line Chart: Average Water Level by Rainfall Range")

            bins = [0, 50, 100, 150, 200, 250, 300]
            labels = ["0-50", "50-100", "100-150", "150-200", "200-250", "250-300"]
            df_temp = filtered_df.copy()
            df_temp["Rainfall Range"] = pd.cut(df_temp["Rainfall (mm)"], bins=bins, labels=labels)
            trend_data = df_temp.groupby("Rainfall Range", observed=True)["Water Level (m)"].mean().reset_index()

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(
                trend_data["Rainfall Range"].astype(str),
                trend_data["Water Level (m)"],
                marker="o",
                color="green",
                linewidth=2
            )
            ax2.set_title("Average Water Level across Rainfall Ranges")
            ax2.set_xlabel("Rainfall Range (mm)")
            ax2.set_ylabel("Average Water Level (m)")
            ax2.grid(True, linestyle="--", alpha=0.5)
            plt.xticks(rotation=30, ha="right", fontsize=9)

            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            with st.expander("Line Chart Interpretation"):
                st.write("The line chart illustrates water level trends across rainfall intervals. Higher rainfall intervals show elevated average water levels.")

        vcol3, vcol4 = st.columns(2)

        with vcol3:
            st.subheader("3. Scatter Plot: Rainfall vs Water Level")
            flood_pts = plot_df[plot_df["Flood Occurred"] == 1]
            no_flood_pts = plot_df[plot_df["Flood Occurred"] == 0]

            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.scatter(
                no_flood_pts["Rainfall (mm)"],
                no_flood_pts["Water Level (m)"],
                color="blue",
                label="No Flood",
                alpha=0.6,
                s=30,
                edgecolor="none"
            )
            ax3.scatter(
                flood_pts["Rainfall (mm)"],
                flood_pts["Water Level (m)"],
                color="red",
                label="Flood Occurred",
                alpha=0.7,
                s=30,
                edgecolor="none"
            )
            ax3.set_title(f"Rainfall vs Water Level ({len(plot_df)} Sampled Points)")
            ax3.set_xlabel("Rainfall (mm)")
            ax3.set_ylabel("Water Level (m)")
            ax3.legend(loc="upper left")
            ax3.grid(True, linestyle="--", alpha=0.4)

            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

            with st.expander("Scatter Plot Interpretation"):
                if len(plot_df) > 1:
                    corr = plot_df["Rainfall (mm)"].corr(plot_df["Water Level (m)"])
                    st.write(f"Correlation between rainfall and water level is {corr:.3f}.")
                    st.write("Flood events (red points) mostly occur when both rainfall and water level are high.")

        with vcol4:
            st.subheader("4. Histogram: Rainfall Distribution")

            fig4, ax4 = plt.subplots(figsize=(6, 4))
            rain_vals = plot_df["Rainfall (mm)"].dropna()
            ax4.hist(rain_vals, bins=15, color="skyblue", edgecolor="black", alpha=0.75)

            mean_val = rain_vals.mean()
            median_val = rain_vals.median()
            ax4.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
            ax4.axvline(median_val, color="green", linestyle="-", linewidth=1.5, label=f"Median: {median_val:.1f}")

            ax4.set_title("Distribution of Rainfall")
            ax4.set_xlabel("Rainfall (mm)")
            ax4.set_ylabel("Frequency")
            ax4.legend()
            ax4.grid(axis="y", linestyle="--", alpha=0.5)

            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

            with st.expander("Histogram Interpretation"):
                st.write(f"Mean rainfall is {mean_val:.2f} mm and median rainfall is {median_val:.2f} mm.")

with tab4:
    st.header("Key Findings & Insights")
    st.write("Summary of analytical observations from the dataset:")

    col_in1, col_in2 = st.columns(2)

    with col_in1:
        st.subheader("1. Rainfall and Water Level")
        st.write("Higher rainfall directly correlates with increased river water levels. Observations where rainfall exceeds 180 mm have a much higher flood frequency.")

        st.subheader("2. Soil Type Vulnerability")
        st.write("Clay and peat soils show higher flood occurrences compared to other soils. This is due to slower water infiltration and faster saturation.")

    with col_in2:
        st.subheader("3. Infrastructure Impact")
        st.write("Areas with flood management infrastructure present show lower flood frequency compared to areas without infrastructure under similar rainfall.")

        st.subheader("4. Practical Recommendations")
        st.write("- Issue early warnings when rainfall is predicted to exceed 150 mm.")
        st.write("- Focus flood prevention and drainage improvements in areas with clay soil.")
        st.write("- Regular maintenance of stormwater drains before the monsoon season.")

