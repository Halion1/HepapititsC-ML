import pandas as pd
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
from Funciones import recode_category
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from scipy.stats import shapiro, spearmanr, mannwhitneyu

# Define a function to perform the Shapiro-Wilk test for normality on a list of numerical columns
def perform_shapiro_wilk_tests(df, numerical_columns):
    # Dictionary to hold the p-values
    p_values = {}

    # Perform Shapiro-Wilk test for each numerical column
    for column in numerical_columns:
        stat, p = shapiro(df[column])
        # Format p-value to only have 3 decimal points
        p_values[column] = f"{p:.3f}"

    # Convert the p-values dictionary to a DataFrame for display
    p_values_df = pd.DataFrame(list(p_values.items()), columns=['Variable', 'P-Value'])

    # Ensure the 'P-Value' column is of float type to reflect the formatting (if needed)
    p_values_df['P-Value'] = p_values_df['P-Value'].astype(float)

    # Return the DataFrame with the results
    return p_values_df

numerical_columns = ['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

# Define the function to compute Spearman Rank Correlation for pairs
def compute_spearman_correlation(df, pairs):
    results = []
    for x, y in pairs:
        # Calculate Spearman Rank Correlation
        spearman_corr, p_value = spearmanr(df[x], df[y])

        # Round the Spearman correlation coefficient and p-value to 3 decimal places
        spearman_corr = round(spearman_corr, 3)
        p_value = round(p_value, 3)

        # Determine if the result is significant
        alpha = 0.05
        result = "Reject" if p_value < alpha else "Fail to reject"
        hypothesis = f"{result} the null hypothesis - There is {'a significant' if p_value < alpha else 'no significant'} relationship between {x} and {y}"

        # Store the results in a list
        results.append({
            'Pair': f'{x} and {y}',
            'Spearman Rank Correlation': spearman_corr,
            'P-value': p_value,
            'Hypothesis Test Result': hypothesis
        })

    # Convert the list to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# Define the pairs for which you want to compute Spearman Rank Correlation
pairs_to_test = [('PROT', 'CREA'), ('ALP', 'AST')]

# Define a function to perform the Mann-Whitney U test for a list of variable combinations
def perform_mannwhitneyu_tests(df, category_column, variable_columns):
    results = []
    for variable in variable_columns:
        group0 = df[df[category_column] == 0][variable]
        group1 = df[df[category_column] == 1][variable]

        # Check if either group is empty
        if group0.empty or group1.empty:
            print(f"Skipping {variable} as one of the groups is empty")
            continue

        # Perform Mann-Whitney U Test
        u_statistic, p_value = mannwhitneyu(group0, group1, alternative='two-sided')

        # Interpret the results
        alpha = 0.05  # Significance level
        result = "Reject H0" if p_value < alpha else "Fail to Reject H0"

        # Append the results to the results list
        results.append({
            'Variable': variable,
            'U Statistic': u_statistic,
            'P-Value': p_value,
            'Result': result
        })

    return pd.DataFrame(results)

variable_columns = ['ALT', 'ALB', 'BIL', 'CHOL']

variable_colors = {
    'Age': '#1f77b4',   # Blue
    'ALB': '#ff7f0e',   # Orange
    'ALP': '#2ca02c',   # Green
    'ALT': '#d62728',   # Red
    'AST': '#9467bd',   # Purple
    'BIL': '#8c564b',   # Brown
    'CHE': '#e377c2',   # Pink
    'CHOL': '#7f7f7f',  # Gray
    'CREA': '#bcbd22',  # Olive
    'GGT': '#17becf',   # Cyan
    'PROT': '#e7ba52'   # Gold
}

# Load the Excel file
hepatitis_data = pd.read_excel('hepatitisC.xlsx')
hepatitis_data = hepatitis_data.dropna()

variable_ranges = {
    'Age': (19, 77),
    'ALB': (14.9, 82.2),
    'ALP': (11.3, 416.6),
    'ALT': (0.9, 325.3),
    'AST': (10.6, 324),
    'BIL': (0.8, 254),
    'CHE': (1.42, 16.41),
    'CHOL': (1.43, 9.67),
    'CREA': (8, 107.9),
    'GGT': (4.5, 650),
    'PROT': (44.8, 90)
}

# Assuming df is your dataframe and you have the necessary functions defined
shapiro_wilk_results = perform_shapiro_wilk_tests(hepatitis_data, numerical_columns)
spearman_results_table = compute_spearman_correlation(hepatitis_data, pairs_to_test)
mannwhitneyu_results = perform_mannwhitneyu_tests(hepatitis_data, 'Category', variable_columns)

# Aplicar los rangos para filtrar
for variable, (low, high) in variable_ranges.items():
    hepatitis_data = hepatitis_data[(hepatitis_data[variable] >= low) & (hepatitis_data[variable] <= high)]

hepatitis_data['Sex'] = hepatitis_data['Sex'].map({'m': 0, 'f': 1})

# Columnas numéricas
numerical_columns = ['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
df_numerical = hepatitis_data[numerical_columns]

# Calcular la matriz de correlación
correlation_matrix = df_numerical.corr()

# Aplicar la función
hepatitis_data['Category'] = hepatitis_data['Category'].apply(recode_category)

# Preparing data for the plots
# Age Distribution by Category
age_dist_fig = px.histogram(hepatitis_data, x='Age', color='Category', barmode='overlay')

# Gender Distribution in Each Category
gender_dist_fig = px.bar(hepatitis_data.groupby(['Category', 'Sex']).size().reset_index(name='Count'),
                         x='Category', y='Count', color='Sex', barmode='group')

# Liver Enzyme Levels (ALT, AST) by Category
enzyme_levels_fig = px.box(hepatitis_data, x='Category', y=['ALT', 'AST'],
                            color='Category', notched=True)

stats = hepatitis_data.describe()
# Convert the descriptive statistics DataFrame to a format suitable for dash_table
stats.reset_index(inplace=True)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the app
app.layout = dbc.Container([
    html.H1('Hepatitis C Dashboard'),

    dbc.Tabs([
        dbc.Tab([

            # Add the DataTable component to display the descriptive statistics
            dash_table.DataTable(
                id='table',
                columns=[
                    {"name": i, "id": i, "type": "numeric",
                     "format": {"specifier": ".3f"},  # Format specifier for 3 decimal places
                     "presentation": "plain",  # Use 'plain' for numbers without any special formatting
                    }
                    for i in stats.columns
                ],
                data=stats.to_dict('records'),
                style_cell={'textAlign': 'center'},  # Center text alignment for all cells
            ),

            # Dropdown for variable selection
            dcc.Dropdown(
                id='variable-dropdown',
                options=[{'label': i, 'value': i} for i in hepatitis_data.columns],
                value='Age'  # default value
            ),

            # Histogram graph
            dcc.Graph(id='histogram-graph'),
        ], label='Histogram'),

        dbc.Tab([
            html.H2('Pruebas Estadísticas'),

            html.H3('Shapiro-Wilk Test'),
            dash_table.DataTable(
                data=shapiro_wilk_results.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in shapiro_wilk_results.columns],
                style_cell={'textAlign': 'center'}  # Center text alignment for all cells
            ),

            html.H3('Spearman Rank Correlation'),
            dash_table.DataTable(
                data=spearman_results_table.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in spearman_results_table.columns],
                style_cell={'textAlign': 'center'}  # Center text alignment for all cells
            ),

            html.H3(),
            dash_table.DataTable(
                data=mannwhitneyu_results.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in mannwhitneyu_results.columns],
                style_cell={'textAlign': 'center'}  # Center text alignment for all cells
            )
        ], label='Estadisticas'),

        dbc.Tab([
            # Heatmap graph
            dcc.Graph(id='heatmap-graph', figure=go.Figure(
                go.Heatmap(
                    z=correlation_matrix,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    colorscale='Viridis'  # Use a valid colorscale
                )
            ).update_layout(title='Correlation Heatmap', xaxis_nticks=36, yaxis_nticks=36))
        ], label='Heatmap'),
    ])
])

# Callback for updating the histogram
@app.callback(
    Output('histogram-graph', 'figure'),
    [Input('variable-dropdown', 'value')]
)
def update_graph(selected_variable):
    # Get color for the selected variable
    color = variable_colors.get(selected_variable, '#1f77b4')  # Default color if not found

    # Generate the histogram
    fig = go.Figure(data=[go.Histogram(x=hepatitis_data[selected_variable], marker_color=color)])

    # Update layout
    fig.update_layout(
        title=f'Histogram of {selected_variable}',
        xaxis_title=selected_variable,
        yaxis_title='Frequency',
        template='plotly_white',
        font=dict(size=14),
        autosize=True
    )

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)