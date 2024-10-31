import dash
import polars as pl
from dash import dcc, html, Input, Output, State, _dash_renderer, dash_table
import plotly.graph_objects as go
import dash_mantine_components as dmc
from datetime import datetime
from scipy import stats
import numpy as np
from dash_ag_grid import AgGrid
import base64
import io
import pandas as pd

_dash_renderer._set_react_version("18.2.0")



def determine_groupby_method(df, x_col, y_col, color_col, aggr_method):
    if color_col == "None":
        match aggr_method:
            case "Sum":
                grouped_df = df.group_by([x_col]).agg(pl.sum(y_col))
            case "Mean":
                grouped_df = df.group_by([x_col]).agg(pl.mean(y_col))
            case "Min":
                grouped_df = df.group_by([x_col]).agg(pl.min(y_col))
            case "Max":
                grouped_df = df.group_by([x_col]).agg(pl.max(y_col))
            case "Count":
                grouped_df = df.group_by([x_col]).agg(pl.count(y_col))
            case _:
                raise ValueError(f"Unsupported aggregation method: {aggr_method}")
    else:
        match aggr_method:
            case "Sum":
                grouped_df = df.group_by([x_col, color_col]).agg(pl.sum(y_col))
            case "Mean":
                grouped_df = df.group_by([x_col, color_col]).agg(pl.mean(y_col))
            case "Min":
                grouped_df = df.group_by([x_col, color_col]).agg(pl.min(y_col))
            case "Max":
                grouped_df = df.group_by([x_col, color_col]).agg(pl.max(y_col))
            case "Count":
                grouped_df = df.group_by([x_col, color_col]).agg(pl.count(y_col))
            case _:
                raise ValueError(f"Unsupported aggregation method: {aggr_method}")
    return grouped_df

# Initialize Dash app
app = dash.Dash(__name__)

# Define available aggregation methods
aggregation_methods = ["Sum", "Mean", "Min", "Max", "Count"]

# Layout of the app
app.layout = dmc.MantineProvider(
    children=[
        html.H1("Dynamic Chart with Rolling Window"),
        dmc.NumberInput(
            id="row-limit",
            label="Number of Rows to Read",
            value=100000,
            min=1,
            max=1000000,
            step=1000,
        ),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            # Allow multiple files to be uploaded
            multiple=False,
        ),
        dcc.Store(id="stored-data"),
        html.Div(id="upload-status-message"),
        html.Div(
            [
                dmc.Card(
                    withBorder=True,
                    shadow="sm",
                    radius="md",
                    children=dmc.CardSection(
                        children=[
                            dmc.Grid(
                                children=[
                                    dmc.GridCol(
                                        children=dmc.Select(
                                            id="x-data-type-dropdown",
                                            data=[
                                                {"label": "All", "value": "all"},
                                                {"label": "Date", "value": "date"},
                                                {
                                                    "label": "Categorical",
                                                    "value": "categorical",
                                                },
                                                {
                                                    "label": "Numerical",
                                                    "value": "numerical",
                                                },
                                            ],
                                            value="date",
                                            label="X-axis Data Type",
                                        ),
                                        span=4,
                                    ),
                                    dmc.GridCol(
                                        children=dmc.Select(
                                            id="y-data-type-dropdown",
                                            data=[
                                                {"label": "All", "value": "all"},
                                                {"label": "Date", "value": "date"},
                                                {
                                                    "label": "Categorical",
                                                    "value": "categorical",
                                                },
                                                {
                                                    "label": "Numerical",
                                                    "value": "numerical",
                                                },
                                            ],
                                            value="numerical",
                                            label="Y-axis Data Type",
                                        ),
                                        span=4,
                                    ),
                                    dmc.GridCol(
                                        children=dmc.Select(
                                            id="color-data-type-dropdown",
                                            data=[
                                                {"label": "All", "value": "all"},
                                                {"label": "Date", "value": "date"},
                                                {
                                                    "label": "Categorical",
                                                    "value": "categorical",
                                                },
                                                {
                                                    "label": "Numerical",
                                                    "value": "numerical",
                                                },
                                            ],
                                            value="categorical",
                                            label="Z-axis Data Type",
                                        ),
                                        span=4,
                                    ),
                                    dmc.GridCol(
                                        children=dmc.Select(
                                            id="x-axis-dropdown",
                                            data=[],
                                            value=None,
                                            label="X-axis",
                                        ),
                                        span=4,
                                    ),
                                    dmc.GridCol(
                                        children=dmc.Select(
                                            id="y-axis-dropdown",
                                            data=[],
                                            value=None,
                                            label="Y-axis",
                                        ),
                                        span=4,
                                    ),
                                    dmc.GridCol(
                                        children=dmc.Select(
                                            id="color-dropdown",
                                            data=[{"label": "None", "value": "None"}],
                                            value="None",
                                            label="Z-axis",
                                        ),
                                        span=4,
                                    ),
                                    dmc.GridCol(
                                        children=dmc.Select(
                                            id="aggregation-dropdown",
                                            data=[
                                                {"label": method, "value": method}
                                                for method in aggregation_methods
                                            ],
                                            value="Sum",
                                            label="Aggregation method",
                                        ),
                                        span=4,
                                    ),
                                    dmc.GridCol(
                                        children=dmc.NumberInput(
                                            id="window-size-input",
                                            label="Rolling Window Size",
                                            value=7,
                                            min=1,
                                            step=1,
                                        ),
                                        span=4,
                                    ),
                                    dmc.GridCol(
                                        children=dmc.Button(
                                            "Generate Chart",
                                            id="generate-button",
                                            n_clicks=0,
                                        ),
                                        span=4,
                                    ),
                                ]
                            ),
                            dmc.Space(h=10),
                        ]
                    ),
                ),
            ]
        ),
        # Primary line/bar chart
        dcc.Graph(id="bar-chart"),
        # New boxplot chart for daily distribution
        dcc.Graph(id="boxplot-chart"),
        html.Div(id="breach-table-container"),
    ]
)


# Callback to process uploaded file and store content in dcc.Store
@app.callback(
    [Output('stored-data', 'data'), Output('upload-status-message', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('row-limit', 'value')
)
def handle_file_upload(contents, filename, row_limit):
    if contents is None:
        return None, "No file uploaded yet."
    
    # Decode the uploaded file contents
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        # Determine file type and read content
        if 'csv' in filename:
            # Read limited rows from CSV
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), nrows=row_limit)
        elif 'parquet' in filename:
            # Read limited rows from Parquet
            df = pd.read_parquet(io.BytesIO(decoded)).head(row_limit)
        else:
            return None, "Unsupported file format. Please upload a CSV or Parquet file."

        # Store data as JSON-compatible dictionary
        data_store = df.to_dict('records')
        return data_store, "File uploaded successfully!"

    except Exception as e:
        return None, f"There was an error processing the file: {str(e)}"


# Update axis dropdown options based on selected data types
# Update axis dropdown options based on selected data types
@app.callback(
    [
        Output("x-axis-dropdown", "data"),
        Output("y-axis-dropdown", "data"),
        Output("color-dropdown", "data"),
        Output("color-dropdown", "value"),
    ],
    [
        Input("stored-data", "data"),
        State("x-data-type-dropdown", "value"),
        State("y-data-type-dropdown", "value"),
        State("color-data-type-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def update_axis_dropdowns(data_store, x_data_type, y_data_type, color_data_type):
    # Convert the stored data back to a Polars DataFrame
    if data_store:
        df = pl.from_pandas(pd.DataFrame.from_records(data_store))

        # Helper function to check if a column is a date based on multiple formats
        def is_date_column(column):
            date_formats = [
                "%B-%Y",
                "%b-%Y",
                "%Y-%m-%d",
                "%d-%m-%Y",
                "%m/%d/%Y",
                "%Y/%m/%d",
                "%d %b %Y",
                "%d %B %Y",
            ] # Extendable list of formats

            for date_format in date_formats:
                try:
                    non_null_values = column.drop_nulls()
                    if non_null_values.is_empty():
                        continue
                    # Attempt parsing to check date validity
                    parsed_dates = [
                        datetime.strptime(value, date_format) for value in non_null_values
                    ]
                    return date_format  # Return matching format if parsing successful
                except ValueError:
                    continue
            return None

        # Identify columns by type
        date_columns = [col for col in df.columns if is_date_column(df[col].cast(str))]

        categorical_columns = [
            col for col in df.columns if df[col].dtype in [pl.Utf8, pl.Categorical]
        ]
        numerical_columns = [
            col for col in df.columns if df[col].dtype in [pl.Float64, pl.Int64, pl.Int32, pl.Float32]
        ]

        # Convert detected date columns to datetime format using identified format
        for col in date_columns:
            date_format = is_date_column(df[col].cast(str))
            if date_format:
                df = df.with_columns(
                    pl.col(col).str.strptime(pl.Date, format=date_format).alias(col)
                )

        # Helper function to generate dropdown options based on the data type
        def get_columns_by_type(data_type):
            options_map = {
                "date": date_columns,
                "categorical": categorical_columns,
                "numerical": numerical_columns,
                "all": df.columns,
            }
            # Default to empty list if data_type not in options_map
            return [{"label": col, "value": col} for col in options_map.get(data_type, [])]

        # Generate dropdown options for each axis
        x_options = get_columns_by_type(x_data_type)
        y_options = get_columns_by_type(y_data_type)
        color_options = get_columns_by_type(color_data_type) + [{"label": "None", "value": "None"}]

        return x_options, y_options, color_options,"None"
    else: 
        return [],[],["None"],"None" 




# Callback to update the charts based on user selections
@app.callback(
    [
        Output("bar-chart", "figure"),
        Output("boxplot-chart", "figure"),
        Output("breach-table-container", "children"),
    ],
    Input("generate-button", "n_clicks"),
    State("x-axis-dropdown", "value"),
    State("y-axis-dropdown", "value"),
    State("color-dropdown", "value"),
    State("aggregation-dropdown", "value"),
    State("window-size-input", "value"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def update_chart(n_clicks, x_col, y_col, color_col, agg_method, window_size, data_store):
    df = pl.from_pandas(pd.DataFrame.from_records(data_store))
    # Figure for the main chart (line/bar with rolling mean)
    fig_main = go.Figure()

    # Figure for the boxplot chart
    fig_boxplot = go.Figure()

    # Create boxplot for each date, with color by `color_col` if specified
    if color_col != "None":
        # Create boxplots grouped by `color_col`
        unique_colors = df[color_col].unique().to_list()
        for color_value in unique_colors:
            filtered_df = df.filter(pl.col(color_col) == color_value)
            fig_boxplot.add_trace(
                go.Box(
                    x=filtered_df[x_col].to_list(),
                    y=filtered_df[y_col].to_list(),
                    name=str(color_value),
                    boxmean=True,
                )
            )
    else:
        # Create boxplots for each date without color distinction
        fig_boxplot.add_trace(
            go.Box(
                x=df[x_col].to_list(),
                y=df[y_col].to_list(),
                name="All Data Points",
                boxmean=True,
            )
        )

    # Primary chart with line/bar chart logic remains the same as before
    if color_col == "None":
        grouped_df = determine_groupby_method(df, x_col, y_col, color_col, agg_method)
        filtered_df = grouped_df.sort(x_col)

        # Define rolling window parameters
        confidence_level = 0.95
        degrees_freedom = window_size - 1
        if degrees_freedom <= 0:
            degrees_freedom = (
                1  # Prevent degrees of freedom from being zero or negative
            )
        t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)

        # Compute rolling statistics step by step
        filtered_df = filtered_df.with_columns(
            pl.col(y_col).rolling_mean(window_size).alias("Rolling Mean")
        )
        filtered_df = filtered_df.with_columns(
            pl.col(y_col).rolling_std(window_size).alias("Rolling Std")
        )
        filtered_df = filtered_df.with_columns(
            (pl.col("Rolling Std") / np.sqrt(window_size)).alias("Rolling SEM")
        )
        filtered_df = filtered_df.with_columns(
            (t_critical * pl.col("Rolling SEM")).alias("Margin of Error")
        )
        filtered_df = filtered_df.with_columns(
            (pl.col("Rolling Mean") - 2 * pl.col("Margin of Error")).alias(
                "CI Lower"
            )
        )
        filtered_df = filtered_df.with_columns(
            (pl.col("Rolling Mean") + 2 * pl.col("Margin of Error")).alias(
                "CI Upper"
            )
        )

        fig_main.add_trace(
            go.Scatter(
                x=filtered_df[x_col].to_list(),
                y=filtered_df[y_col].to_list(),
                mode="lines",
            )
        )

        # Plot the rolling mean
        fig_main.add_trace(
            go.Scatter(
                x=filtered_df[x_col].to_list(),
                y=filtered_df["Rolling Mean"].to_list(),
                mode="lines",
                line=dict(color="red"),
                name=f"{window_size}-Period Rolling Mean",
            )
        )

        # Plot the confidence interval
        fig_main.add_trace(
            go.Scatter(
                x=filtered_df[x_col].to_list() + filtered_df[x_col].to_list()[::-1],
                y=filtered_df["CI Upper"].to_list()
                + filtered_df["CI Lower"].to_list()[::-1],
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name="Confidence Interval",
            )
        )

    else:
        grouped_df = determine_groupby_method(df, x_col, y_col, color_col, agg_method)
        unique_color_values = grouped_df[color_col].unique().to_list()

        for color_value in unique_color_values:
            filtered_group = grouped_df.filter(pl.col(color_col) == color_value).sort(
                x_col
            )


            fig_main.add_trace(
                go.Scatter(
                    x=filtered_group[x_col].to_list(),
                    y=filtered_group[y_col].to_list(),
                    mode="lines",
                    name=str(color_value),
                )
            )


    fig_main.update_layout(
        barmode="overlay",
        title=f"{'Line'} Chart of {y_col} vs {x_col} with {agg_method} aggregation",
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode="x unified",
    )

    fig_boxplot.update_layout(
        title="Boxplot of Daily Distributions",
        xaxis_title=x_col,
        yaxis_title=y_col,
        boxmode="group",
    )
    # Check for breaches in rolling mean
    breaches = filtered_df.filter(
        (pl.col("Rolling Mean") < pl.col("CI Lower"))
        | (pl.col("Rolling Mean") > pl.col("CI Upper"))
    )
    # Format breaches for display in AG Grid
    breach_data = (
        breaches.select([x_col, "Rolling Mean", "CI Lower", "CI Upper"])
        .to_pandas()
        .to_dict("records")
    )

    # Define column definitions for AG Grid
    column_defs = [
        {"headerName": x_col, "field": x_col, "sortable": True, "filter": True},
        {
            "headerName": "Rolling Mean",
            "field": "Rolling Mean",
            "sortable": True,
            "filter": True,
        },
        {
            "headerName": "CI Lower",
            "field": "CI Lower",
            "sortable": True,
            "filter": True,
        },
        {
            "headerName": "CI Upper",
            "field": "CI Upper",
            "sortable": True,
            "filter": True,
        },
    ]

    # Create AG Grid component
    breach_table = AgGrid(
        columnDefs=column_defs,
        rowData=breach_data,
        style={"height": "400px", "width": "100%"},
        dashGridOptions={"pagination": True, "paginationPageSize": 10},
    )

    return fig_main, fig_boxplot, breach_table


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
