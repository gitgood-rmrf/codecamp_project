import dash
import polars as pl
from dash import dcc, html, Input, Output, State, _dash_renderer
import plotly.graph_objects as go
import dash_mantine_components as dmc
from datetime import datetime
from scipy import stats
import numpy as np

_dash_renderer._set_react_version("18.2.0")

# Load data on startup
file_path = "/home/eric/codecamp/accepted_2007_to_2018Q4.parquet"
df = pl.read_parquet(file_path)

# Define multiple date formats to check
date_formats = ["%B-%Y", "%b-%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d", "%d %b %Y", "%d %B %Y"]

# Function to check if a column can be converted to a date using multiple formats
def is_date_column(column):
    for date_format in date_formats:
        try:
            non_null_values = column.cast(str).drop_nulls()
            if non_null_values.is_empty():
                continue
            parsed_dates = [datetime.strptime(value, date_format) for value in non_null_values]
            return date_format  # Return the matching date format if successful
        except ValueError:
            continue
    return None

# Automatically detect column types
date_columns = [col for col in df.columns if is_date_column(df[col].cast(str))]
categorical_columns = [col for col in df.columns if df[col].dtype in [pl.Utf8, pl.Categorical]]
numerical_columns = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Int64, pl.Int32, pl.Float32]]

# Detect date columns and convert them to datetime format
for col in date_columns:
    date_format = is_date_column(df[col].cast(str))
    if date_format:
        df = df.with_columns(
            pl.col(col).str.strptime(pl.Date, format=date_format).alias(col)
        )

# Initialize Dash app
app = dash.Dash(__name__)

# Define available aggregation methods
aggregation_methods = ["Sum", "Mean", "Min", "Max", "Count"]

# Layout of the app
app.layout = dmc.MantineProvider(children=[
    html.H1("Dynamic Chart with Rolling Window"),

    # Data type selectors for X, Y, and Color axes
    html.Div([
        dmc.Select(
            id='x-data-type-dropdown',
            data=[
                {'label': 'All', 'value': 'all'},
                {'label': 'Date', 'value': 'date'},
                {'label': 'Categorical', 'value': 'categorical'},
                {'label': 'Numerical', 'value': 'numerical'}
            ],
            value='date',
            label="X-axis Data Type"
        ),
        dmc.Select(
            id='y-data-type-dropdown',
            data=[
                {'label': 'All', 'value': 'all'},
                {'label': 'Date', 'value': 'date'},
                {'label': 'Categorical', 'value': 'categorical'},
                {'label': 'Numerical', 'value': 'numerical'}
            ],
            value='numerical',
            label="Y-axis Data Type"
        ),
        dmc.Select(
            id='color-data-type-dropdown',
            data=[
                {'label': 'All', 'value': 'all'},
                {'label': 'Date', 'value': 'date'},
                {'label': 'Categorical', 'value': 'categorical'},
                {'label': 'Numerical', 'value': 'numerical'}
            ],
            value='categorical',
            label="Color Variable Data Type"
        )
    ]),

    # Axis selectors
    dmc.Select(id='x-axis-dropdown', data=[], value=None, label="X-axis"),
    dmc.Select(id='y-axis-dropdown', data=[], value=None, label="Y-axis"),
    dmc.Select(
        id='color-dropdown',
        data=[{'label': 'None', 'value': 'None'}],
        value='None',
        label="Color variable"
    ),

    # Aggregation method selector
    dmc.Select(
        id='aggregation-dropdown',
        data=[{'label': method, 'value': method} for method in aggregation_methods],
        value='Sum',
        label="Aggregation method"
    ),

    # Rolling window size input
    dmc.NumberInput(
        id='window-size-input',
        label='Rolling Window Size',
        value=7,
        min=1,
        step=1
    ),

    # Generate button and graph
    dmc.Button("Generate Chart", id="generate-button", n_clicks=0),
    dcc.Graph(id='bar-chart')
])

# Update axis dropdown options based on selected data types
@app.callback(
    [Output('x-axis-dropdown', 'data'),
     Output('y-axis-dropdown', 'data'),
     Output('color-dropdown', 'data')],
    [Input('x-data-type-dropdown', 'value'),
     Input('y-data-type-dropdown', 'value'),
     Input('color-data-type-dropdown', 'value')]
)
def update_axis_dropdowns(x_data_type, y_data_type, color_data_type):
    # Helper function to get column options based on data type
    def get_columns_by_type(data_type):
        if data_type == 'date':
            return [{'label': col, 'value': col} for col in date_columns]
        elif data_type == 'categorical':
            return [{'label': col, 'value': col} for col in categorical_columns]
        elif data_type == 'numerical':
            return [{'label': col, 'value': col} for col in numerical_columns]
        else:  # 'all'
            return [{'label': col, 'value': col} for col in df.columns]

    x_options = get_columns_by_type(x_data_type)
    y_options = get_columns_by_type(y_data_type)
    color_options = get_columns_by_type(color_data_type) + [{'label': 'None', 'value': 'None'}]

    return x_options, y_options, color_options

# Callback to update the chart based on user selections
@app.callback(
    Output('bar-chart', 'figure'),
    Input('generate-button', 'n_clicks'),
    State('x-axis-dropdown', 'value'),
    State('y-axis-dropdown', 'value'),
    State('color-dropdown', 'value'),
    State('aggregation-dropdown', 'value'),
    State('window-size-input', 'value'),
    prevent_initial_call=True
)
def update_chart(n_clicks, x_col, y_col, color_col, agg_method, window_size):
    fig = go.Figure()

    # Determine if x_col is a date column
    is_date_column_flag = x_col in date_columns

    # Select the aggregation method based on user choice
    if color_col == "None":
        if agg_method == "Sum":
            grouped_df = df.group_by([x_col]).agg(pl.sum(y_col))
        elif agg_method == "Mean":
            grouped_df = df.group_by([x_col]).agg(pl.mean(y_col))
        elif agg_method == "Min":
            grouped_df = df.group_by([x_col]).agg(pl.min(y_col))
        elif agg_method == "Max":
            grouped_df = df.group_by([x_col]).agg(pl.max(y_col))
        elif agg_method == "Count":
            grouped_df = df.group_by([x_col]).agg(pl.count(y_col))
        else:
            raise ValueError(f"Unsupported aggregation method: {agg_method}")

        filtered_df = grouped_df.sort(x_col)

        if is_date_column_flag:
            # Define rolling window parameters
            confidence_level = 0.95
            degrees_freedom = window_size - 1
            if degrees_freedom <= 0:
                degrees_freedom = 1  # Prevent degrees of freedom from being zero or negative
            t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)

            # Compute rolling statistics step by step
            filtered_df = filtered_df.with_columns(
                pl.col(y_col).rolling_mean(window_size).alias('Rolling Mean')
            )
            filtered_df = filtered_df.with_columns(
                pl.col(y_col).rolling_std(window_size).alias('Rolling Std')
            )
            filtered_df = filtered_df.with_columns(
                (pl.col('Rolling Std') / np.sqrt(window_size)).alias('Rolling SEM')
            )
            filtered_df = filtered_df.with_columns(
                (t_critical * pl.col('Rolling SEM')).alias('Margin of Error')
            )
            filtered_df = filtered_df.with_columns(
                (pl.col('Rolling Mean') - 2*pl.col('Margin of Error')).alias('CI Lower')
            )
            filtered_df = filtered_df.with_columns(
                (pl.col('Rolling Mean') + 2* pl.col('Margin of Error')).alias('CI Upper')
            )

            # Plot the original data
            fig.add_trace(
                go.Scatter(
                    x=filtered_df[x_col].to_list(),
                    y=filtered_df[y_col].to_list(),
                    mode='lines',
                    name='Original Data'
                )
            )

            # Plot the rolling mean
            fig.add_trace(
                go.Scatter(
                    x=filtered_df[x_col].to_list(),
                    y=filtered_df['Rolling Mean'].to_list(),
                    mode='lines',
                    line=dict(color='red'),
                    name=f'{window_size}-Period Rolling Mean'
                )
            )

            # Plot the confidence interval
            fig.add_trace(
                go.Scatter(
                    x=filtered_df[x_col].to_list() + filtered_df[x_col].to_list()[::-1],
                    y=filtered_df['CI Upper'].to_list() + filtered_df['CI Lower'].to_list()[::-1],
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    name='Confidence Interval'
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=filtered_df[x_col].to_list(),
                    y=filtered_df[y_col].to_list(),
                )
            )
    else:
        # Handle cases where color_col is specified
        if agg_method == "Sum":
            grouped_df = df.group_by([x_col, color_col]).agg(pl.sum(y_col))
        elif agg_method == "Mean":
            grouped_df = df.group_by([x_col, color_col]).agg(pl.mean(y_col))
        elif agg_method == "Min":
            grouped_df = df.group_by([x_col, color_col]).agg(pl.min(y_col))
        elif agg_method == "Max":
            grouped_df = df.group_by([x_col, color_col]).agg(pl.max(y_col))
        elif agg_method == "Count":
            grouped_df = df.group_by([x_col, color_col]).agg(pl.count(y_col))
        else:
            raise ValueError(f"Unsupported aggregation method: {agg_method}")

        unique_color_values = grouped_df[color_col].unique().to_list()

        for color_value in unique_color_values:
            filtered_group = grouped_df.filter(pl.col(color_col) == color_value).sort(x_col)

            if is_date_column_flag:
                # Plot each group as a separate line
                fig.add_trace(
                    go.Scatter(
                        x=filtered_group[x_col].to_list(),
                        y=filtered_group[y_col].to_list(),
                        mode='lines',
                        name=str(color_value)
                    )
                )
            else:
                fig.add_trace(
                    go.Bar(
                        x=filtered_group[x_col].to_list(),
                        y=filtered_group[y_col].to_list(),
                        name=str(color_value)
                    )
                )

    fig.update_layout(
        barmode='group' if not is_date_column_flag else 'overlay',
        title=f"{'Line' if is_date_column_flag else 'Bar'} Chart of {y_col} vs {x_col} with {agg_method} aggregation",
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode='x unified' if is_date_column_flag else 'x'
    )

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
