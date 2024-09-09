import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from flask import Flask
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize Flask server
server = Flask(__name__)
# Initialize Dash app
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.CYBORG])

# Define layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.NavbarSimple(
        children=[
            dbc.NavLink("Home", href="/"),
            dbc.NavLink("Plotly Visualization1", href="/page-1"),
            dbc.NavLink('Plotly Visualization2', href="/page-2"),
            dbc.NavLink('Plotly Visualization3', href="/page-3")
        ],
        brand="Dash App",
        brand_href="/",
        color="primary",
        dark=True,
    ),
    html.Div(id='page-content')
])
# Define page routing
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout()
    if pathname == '/page-2':
        return create_scatter_graph()
    if pathname == '/page-3':
        return animation_visual()
    return home_layout()

df = pd.read_csv('F:\\Dash_app\\data\\train.csv')
df_2 = pd.read_csv('https://raw.githubusercontent.com/yankev/testing/master/datasets/nycflights.csv')
df_2 = df_2.drop(df_2.columns[[0]], axis=1)

# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

def create_dropdown_options(columns):
    return [{'label': '', 'value': ''}] + [{'label': col, 'value': col} for col in columns]

trace1 = go.Histogram(x=df_2['arr_delay'], opacity=0.75, name='Arrival Delays')
trace2 = go.Histogram(x=df_2['dep_delay'], opacity=0.75, name='Departure Delays')
g = go.FigureWidget(data=[trace1, trace2],
                    layout=go.Layout(
                        title=dict(text='NYC Flight Database'),
                        barmode='overlay'
                    ))
# Layout for Home page
def home_layout():
    
    return html.Div([
        html.Div(
            className='top-navbar-container',
            children = [
                html.H1('Dash Visualization', className='text-center'),
            ],
        ),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.H5('Select Visualization'),
                                    dcc.Dropdown(
                                        options=create_dropdown_options(df.columns),
                                        value='',
                                        id='x-dropdown',
                                        clearable=False,
                                        style={'margin-bottom': '10px'}
                                    ),
                                    dcc.Dropdown(
                                        options=create_dropdown_options(df.columns),
                                        value='',
                                        id='y-dropdown',
                                        clearable=False,
                                        style={'margin-bottom': '10px'}
                                    ),
                                    dcc.Dropdown(
                                        options=[
                                            {'label': 'Histogram', 'value': 'histogram'},
                                            {'label': 'Scatter Plot', 'value': 'scatter'},
                                            {'label': 'Line Chart', 'value': 'line'}
                                        ],
                                        value='histogram',
                                        id='chart-type-dropdown',
                                        clearable=False,
                                        style={'margin-bottom': '10px'}
                                    ),
                                
                                    dbc.Button('Show Dataset', id='show-dataset-btn', n_clicks=0, className="me-1"),
                                    dbc.Button('Clear', id='clear-btn', n_clicks=0, className="me-1"),
                                ],
                                className='sidebar'
                            ),
                            width=2,
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.Graph(id='chart',style={'height': '600px', 'width': '50%'}),
                                    html.Hr(),
                                    html.Div(id='table-container'),
                                    dcc.Checklist(
                                        id='use-date-checklist',
                                        options=[{'label': 'Date', 'value': 'use_date'}],
                                        value=['use_date'],
                                        inline= True
                                    ),
                                    html.Label('Month:'),
                                    dcc.Slider(
                                        id='month-slider',
                                        min=1,
                                        max=12,
                                        step=1,
                                        marks={i: str(i) for i in range(1, 13)},
                                        value=1,
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                    html.Label('Origin Airport:'),
                                    dcc.Dropdown(
                                        id='carrier-dropdown',
                                        options=[{'label': c, 'value': c} for c in df_2['carrier'].unique()],
                                        value='',
                                        style={'margin-bottom': '10px', 'width': '30%'}
                                    ),
                                    html.Label('Airline:'),
                                    dcc.Dropdown(
                                        id='origin-dropdown',
                                        options=[{'label': o, 'value': o} for o in df_2['origin'].unique()],
                                        value='',
                                        style={'margin-bottom': '10px', 'width': '30%'}
                                    ),
                                    dcc.Graph(id='nyc-flight-graph', figure=g)
                                ],
                                className='content',
                            ),
                            width=10,
                        ),
                    ]
                ),
            ],
            fluid=True,
        ),
    ])

@app.callback(
    Output('chart', 'figure'),
    [
        Input('x-dropdown', 'value'),
        Input('y-dropdown', 'value'),
        Input('chart-type-dropdown', 'value'),
    ]
)
def update_chart(x_axis, y_axis, chart_type):
    
    colors = px.colors.qualitative.Plotly
    
    if not x_axis:
        return go.Figure()
    
    if x_axis == '' or y_axis == '':
        return go.Figure()
    
    if chart_type == 'histogram':
        
        fig = px.histogram(df, x=x_axis,y=y_axis,title=f'Histogram of {x_axis} vs {y_axis}')
    
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis
        )
    elif chart_type == 'scatter':
        if not y_axis:
            return go.Figure()  
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f'Scatter Plot of {x_axis} vs {y_axis}')
    
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis
        )
    elif chart_type == 'line':
        if not y_axis:
            return go.Figure()  
        fig = px.line(df, x=x_axis, y=y_axis, title=f'Line Chart of {x_axis} vs {y_axis}')
        
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis
        )
    else:
        fig = go.Figure() 
    return fig

@app.callback(
    Output('nyc-flight-graph', 'figure'),
    [
        Input('month-slider', 'value'),
        Input('carrier-dropdown', 'value'),
        Input('origin-dropdown', 'value'),
        Input('use-date-checklist', 'value')
    ]
)
def update_histogram(month, carrier, origin, use_date):
    filter_list = (df_2['month'] == month) & (df_2['carrier'] == carrier) & (df_2['origin'] == origin)
    if not origin or not carrier:
        return go.Figure()

    if 'use_date' in use_date:
        filter_list = (df_2['month'] == month) & (df_2['carrier'] == carrier) & (df_2['origin'] == origin)
    else:
        filter_list = (df_2['carrier'] == carrier) & (df_2['origin'] == origin)
    
    temp_df = df_2[filter_list]
    trace1 = go.Histogram(x=temp_df['arr_delay'], opacity=0.75, name='Arrival Delays')
    trace2 = go.Histogram(x=temp_df['dep_delay'], opacity=0.75, name='Departure Delays')
    
    fig = go.Figure(data=[trace1, trace2])
    fig.update_layout(
        title='NYC Flight Delays',
        barmode='overlay',
        xaxis_title='Delay in Minutes',
        yaxis_title='Number of Delays'
    )
    return fig

@app.callback(
    Output('table-container', 'children'),
    [Input('show-dataset-btn', 'n_clicks'),
    Input('x-dropdown', 'value'),
    Input('y-dropdown', 'value')]
)
def update_table_visibility(n_clicks, x_axis, y_axis):
    # Handle None case and convert n_clicks to integer if needed
    if n_clicks is None:
        n_clicks = 0
    
    if n_clicks % 2 == 0:
        # When dataset is hidden
        return None
    
    # Determine columns to display
    if x_axis and y_axis:
        columns = [{'name': col, 'id': col} for col in [x_axis, y_axis]]
        data = df[[x_axis, y_axis]].to_dict('records')
    else:
        columns = [{'name': col, 'id': col} for col in df.columns]
        data = df.to_dict('records')

    return dash_table.DataTable(
        id='data-table',
        columns=columns,
        data=data,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        page_size=10,  # Number of rows per page
    )


@app.callback(
    [
        Output('x-dropdown', 'value'),
        Output('y-dropdown', 'value'),
        Output('chart-type-dropdown', 'value'),
        Output('show-dataset-btn', 'n_clicks'),
    ],
    [Input('clear-btn', 'n_clicks')]
)
   
def clear_all(n_clicks):
        if n_clicks > 0:
            return '', '', 'histogram', 0
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Layout for Page 1 with Plotly Visualization
def page_1_layout():
   
    # Load dataset
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/volcano.csv")

    # Create figure
    fig = go.Figure()

    # Add surface trace
    fig.add_trace(go.Surface(z=df.values.tolist(), colorscale="Viridis"))

    # Update plot sizing
    fig.update_layout(
        width=800,
        height=500,
        autosize=False,
        margin=dict(t=0, b=0, l=0, r=0),
        template="plotly_white",
    )

    # Update 3D scene options
    fig.update_scenes(
        aspectratio=dict(x=1, y=1, z=0.7),
        aspectmode="manual"
    )

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=["type", "surface"],
                        label="3D Surface",
                        method="restyle"
                    ),
                    dict(
                        args=["type", "heatmap"],
                        label="Heatmap",
                        method="restyle"
                    )
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )
    # Add annotation
    fig.update_layout(
        annotations=[
            dict(text="Trace type:  ", showarrow=False,
            x=0, y=1.085, yref="paper", align="left")
        ]
    )

    fig2 = go.Figure()

    # Add surface trace
    fig2.add_trace(go.Heatmap(z=df.values.tolist(), colorscale="Viridis"))

    # Update plot sizing
    fig2.update_layout(
        width=800,
        height=500,
        autosize=False,
        margin=dict(t=100, b=0, l=0, r=0),
    )

    # Update 3D scene options
    fig2.update_scenes(
        aspectratio=dict(x=1, y=1, z=0.7),
        aspectmode="manual"
    )

    # Add dropdowns
    button_layer_1_height = 1.17
    fig2.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=["colorscale", "Viridis"],
                        label="Viridis",
                        method="restyle"
                    ),
                    dict(
                        args=["colorscale", "Cividis"],
                        label="Cividis",
                        method="restyle"
                    ),
                    dict(
                        args=["colorscale", "Blues"],
                        label="Blues",
                        method="restyle"
                    ),
                    dict(
                        args=["colorscale", "Greens"],
                        label="Greens",
                        method="restyle"
                    ),
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=button_layer_1_height,
                yanchor="top"
            ),
            dict(
                buttons=list([
                    dict(
                        args=["reversescale", False],
                        label="False",
                        method="restyle"
                    ),
                    dict(
                        args=["reversescale", True],
                        label="True",
                        method="restyle"
                    )
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.37,
                xanchor="left",
                y=button_layer_1_height,
                yanchor="top"
            ),
            dict(
                buttons=list([
                    dict(
                        args=[{"contours.showlines": False, "type": "contour"}],
                        label="Hide lines",
                        method="restyle"
                    ),
                    dict(
                        args=[{"contours.showlines": True, "type": "contour"}],
                        label="Show lines",
                        method="restyle"
                    ),
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.58,
                xanchor="left",
                y=button_layer_1_height,
                yanchor="top"
            ),
        ]
    )

    fig2.update_layout(
        annotations=[
            dict(text="colorscale", x=0, xref="paper", y=1.15, yref="paper",
                                align="left", showarrow=False),
            dict(text="Reverse<br>Colorscale", x=0.25, xref="paper", y=1.17,
                                yref="paper", showarrow=False),
            dict(text="Lines", x=0.54, xref="paper", y=1.15, yref="paper",
                                showarrow=False)
        ])

    return html.Div([
        html.H4("3D surface and heatmap", className='text-center'),
        dbc.Row(
        [
            dbc.Col(
                dcc.Graph(
                    figure=fig,
                    style={
                        'height': '80vh',  # Adjust height to fit your design
                        'width': '100%'    # Adjust width to fit your design
                    }
                ),
                width=6
            ),
            dbc.Col(
                dcc.Graph(
                    figure=fig2,
                    style={
                        'height': '80vh',  # Adjust height to fit your design
                        'width': '100%'    # Adjust width to fit your design
                    }
                ),
                width=6
            )
        ]
    )
])

def create_scatter_graph():
    np.random.seed(1)

    x0 = np.random.normal(2, 0.4, 400)
    y0 = np.random.normal(2, 0.4, 400)
    x1 = np.random.normal(3, 0.6, 600)
    y1 = np.random.normal(6, 0.4, 400)
    x2 = np.random.normal(4, 0.2, 200)
    y2 = np.random.normal(4, 0.4, 200)

    # Create figure
    fig = go.Figure()

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=x0,
            y=y0,
            mode="markers",
            marker=dict(color="DarkOrange")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x1,
            y=y1,
            mode="markers",
            marker=dict(color="Crimson")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x2,
            y=y2,
            mode="markers",
            marker=dict(color="RebeccaPurple")
        )
    )

    # Add buttons that add shapes
    cluster0 = [dict(type="circle",
                                xref="x", yref="y",
                                x0=min(x0), y0=min(y0),
                                x1=max(x0), y1=max(y0),
                                line=dict(color="DarkOrange"))]
    cluster1 = [dict(type="circle",
                                xref="x", yref="y",
                                x0=min(x1), y0=min(y1),
                                x1=max(x1), y1=max(y1),
                                line=dict(color="Crimson"))]
    cluster2 = [dict(type="circle",
                                xref="x", yref="y",
                                x0=min(x2), y0=min(y2),
                                x1=max(x2), y1=max(y2),
                                line=dict(color="RebeccaPurple"))]

    fig.update_layout(
        updatemenus=[
            dict(buttons=list([
                dict(label="None",
                    method="relayout",
                    args=["shapes", []]),
                dict(label="Cluster 0",
                    method="relayout",
                    args=["shapes", cluster0]),
                dict(label="Cluster 1",
                    method="relayout",
                    args=["shapes", cluster1]),
                dict(label="Cluster 2",
                    method="relayout",
                    args=["shapes", cluster2]),
                dict(label="All",
                    method="relayout",
                    args=["shapes", cluster0 + cluster1 + cluster2])
            ]),
            )
        ],
        plot_bgcolor='rgba(0,0,0,0)',
    )
    # Update remaining layout properties
    fig.update_layout(
        title_text="Highlight Clusters",
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
    )
    x = np.linspace(0, 10, 100)
    y = 2.5 * x + np.random.normal(0, 1, size=x.shape)
    # Create a DataFrame
    dt = pd.DataFrame({'x': x, 'y': y})
    # Create a Plotly scatter plot with a trend line
    fg = px.scatter(dt, x='x', y='y', trendline='ols')

    return html.Div([
        html.H4(" ", className='text-center'),
        dbc.Row(
        [
            dbc.Col(
                dcc.Graph(
                    figure=fig,
                    style={
                        'height': '80vh',  # Adjust height to fit your design
                        'width': '100%'    # Adjust width to fit your design
                    }
                ),
            ),
            dbc.Col(
                 dcc.Graph(
                    id='scatter-plot',
                    figure=fg,
                    style={
                        'height': '80vh',  
                        'width': '100%'    
                    }
                ),
                width=6
            )
        ]
        )
    ]
    )

@app.callback(
    Output("graph", "figure"), 
    Input("selection", "value"))

def display_animated_graph(selection):
    df = px.data.gapminder() 
    animations = {
        'GDP - Scatter': px.scatter(
            df, x="gdpPercap", y="lifeExp", animation_frame="year", 
            animation_group="country", size="pop", color="continent", 
            hover_name="country", log_x=True, size_max=55, 
            range_x=[100,100000], range_y=[25,90]),
        'Population - Bar': px.bar(
            df, x="continent", y="pop", color="continent", 
            animation_frame="year", animation_group="country", 
            range_y=[0,4000000000]),
    }
    return animations[selection]

def animation_visual():
    return html.Div([
    html.H4('Animated GDP and population over decades'),
    html.P("Select an animation:"),
    dcc.RadioItems(
        id='selection',
        options=["GDP - Scatter", "Population - Bar"],
        value='GDP - Scatter',
    ),
    dcc.Loading(dcc.Graph(id="graph"), type="cube")
    ])

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=80)
