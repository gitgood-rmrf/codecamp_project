import dash
import polars as pl
from dash import dcc, html, Input, Output, State, _dash_renderer
import plotly.graph_objects as go
import dash_mantine_components as dmc
_dash_renderer._set_react_version("18.2.0")

# Load data on startup
file_path = "/home/eric/codecamp/accepted_2007_to_2018Q4.csv"
columns_id = ["id"]
columns_fact = ["loan_amnt", "int_rate", "num_bc_tl"]
columns_cat = ["addr_state", "application_type", "grade", "home_ownership", "initial_list_status", "purpose", "term"]
columns_to_keep = columns_id + columns_fact + columns_cat
df = pl.read_csv(file_path, ignore_errors=True, n_rows=10000)

# Automatically detect column types
date_columns = [col for col in df.columns if pl.Date in df.schema[col].__class__.__bases__]
categorical_columns = [col for col in df.columns if df[col].dtype in [pl.Utf8, pl.Categorical]]
numerical_columns = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Int64, pl.Int32, pl.Float32]]

# Initialize Dash app
app = dash.Dash(__name__)

# Define available aggregation methods
aggregation_methods = ["Sum", "Mean", "Min", "Max", "Count"]

# Layout of the app
app.layout = dmc.MantineProvider(children=[
    html.H1("Dynamic Bar Chart with Dash and Plotly"),
    
    # Data type selectors for X, Y, and Color axes
    html.Div([
        dmc.Select(
            id='x-data-type-dropdown',
            data=[{'label': 'All', 'value': 'all'}, {'label': 'Date', 'value': 'date'}, 
                  {'label': 'Categorical', 'value': 'categorical'}, {'label': 'Numerical', 'value': 'numerical'}],
            value='all',
            label="X-axis Data Type"
        ),
        dmc.Select(
            id='y-data-type-dropdown',
            data=[{'label': 'All', 'value': 'all'}, {'label': 'Date', 'value': 'date'}, 
                  {'label': 'Categorical', 'value': 'categorical'}, {'label': 'Numerical', 'value': 'numerical'}],
            value='all',
            label="Y-axis Data Type"
        ),
        dmc.Select(
            id='color-data-type-dropdown',
            data=[{'label': 'All', 'value': 'all'}, {'label': 'Date', 'value': 'date'}, 
                  {'label': 'Categorical', 'value': 'categorical'}, {'label': 'Numerical', 'value': 'numerical'}],
            value='all',
            label="Color Variable Data Type"
        )
    ]),
    
    # Axis selectors
    dmc.Select(id='x-axis-dropdown', data=[], value=None, label="X-axis"),
    dmc.Select(id='y-axis-dropdown', data=[], value=None, label="Y-axis"),
    dmc.Select(id='color-dropdown', data=[{'label': 'None', 'value': 'None'}], value='None', label="Color variable"),
    
    # Aggregation method selector
    dmc.Select(
        id='aggregation-dropdown',
        data=[{'label': method, 'value': method} for method in aggregation_methods],
        value='Sum',
        label="Aggregation method"
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

# Callback to update the bar chart based on user selections
@app.callback(
    Output('bar-chart', 'figure'),
    Input('generate-button', 'n_clicks'),
    State('x-axis-dropdown', 'value'),
    State('y-axis-dropdown', 'value'),
    State('color-dropdown', 'value'),
    State('aggregation-dropdown', 'value')
)
def update_chart(n_clicks, x_col, y_col, color_col, agg_method):
    fig = go.Figure()

    # Determine the aggregation function based on the selected method
    if color_col == "None":
        match agg_method:
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
                raise ValueError(f"Unsupported aggregation method: {agg_method}")

        filtered_df = grouped_df.sort(x_col)
        fig.add_trace(
            go.Bar(
                x=filtered_df[x_col].to_list(),
                y=filtered_df[y_col].to_list(),
            )
        )
    else:
        match agg_method:
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
                raise ValueError(f"Unsupported aggregation method: {agg_method}")

        for color_value in grouped_df[color_col].unique():
            filtered_df = grouped_df.filter(pl.col(color_col) == color_value).sort(x_col)
            fig.add_trace(
                go.Bar(
                    x=filtered_df[x_col].to_list(),
                    y=filtered_df[y_col].to_list(),
                    name=str(color_value)
                )
            )
    
    fig.update_layout(
        barmode='group',
        title=f"Bar Chart of {y_col} vs {x_col} with {agg_method} aggregation",
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
