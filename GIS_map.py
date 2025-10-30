import pandas as pd
import plotly.graph_objects as go
import gradio as gr

# Load county-level outcome data
map_df = pd.read_csv("[GIS]ca_counties_outcome.csv")

# The 15 outcome metrics you want to visualize
OUTCOME_METRICS = [
    'Ancillary_Access_Rate',
    'Education_and_Skills_Dev_Rate',
    'Employment_Rate',
    'Engagement_Rate',
    'Exits_With_Earnings',
    'Family_Stabilization_to_WTW_Eng',
    'First_Activity_Rate',
    'OCAT_Appraisal_to_Next_Activity',
    'OCAT_Timeliness_Rate',
    'PostCWEmployment',
    'Reentry_After_Exit_with_Earning',
    'Reentry',
    'Sanction_Rate',
    'Sanction_Resolution_Rate',
    'Orientation_Attendance_2024'
]

COUNTIES = sorted(map_df['County'].unique())

# Plot callback with optional county filter
def plot_outcome_map(metric, selected_counties):
    df = map_df.copy()
    if selected_counties:
        df = df[df['County'].isin(selected_counties)]

    color_map = {
    "low": "#d62728",     # red → poor outcome
    "medium": "#ff7f0e",  # orange → moderate
    "high": "#2ca02c"     # green → favorable
}
    fig = go.Figure()

    for label in df[metric].dropna().unique():
        mask = df[metric] == label
        fig.add_trace(go.Scattermapbox(
            lat=df.loc[mask, 'INTPTLAT'],
            lon=df.loc[mask, 'INTPTLON'],
            text=df.loc[mask, 'County'],
            mode='markers',
            name=label.title(),
            marker=go.scattermapbox.Marker(
                size=8,
                color=color_map.get(label.lower(), '#636EFA')
            ),
            hovertemplate=(
                "<b>County</b>: %{text}<br>"
                f"<b>Category</b>: {label.title()}"
            )
        ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center={"lat": 37.5, "lon": -119.5},
            zoom=5.2,  # Or try zoom=4.5
            style="open-street-map"
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"{metric.replace('_',' ').title()} Classification by County"
    )

    return fig