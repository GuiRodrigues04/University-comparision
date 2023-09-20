import pandas as pd
import numpy as np
from dash import dash, html, dash_table, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px

university_rank_df = pd.read_csv('World University Rankings 2023.csv')
university_rank_df = university_rank_df.dropna()
university_rank_df['No of student'] = university_rank_df['No of student'].str.replace(",", ".").astype(float)

# colors

legend_color = 'grey'
graphs1_color = ['#790f8f', '#410f26']
graphs2_color = ['#f93c3c', '#a50101']

# components

# title
title = html.H2(children='University comparison', style={"textAlign": 'center'})

# dropdown1
choose_country1 = html.Div(children='Choose country1', style={'color': legend_color})
drop1 = dcc.Dropdown(university_rank_df['Location'].unique(),
                     value='United Kingdom',
                     id='drop1',
                     style={'color': 'Black'})

# dropdown2
choose_country2 = html.Div(children='Choose country 2', style={'color': legend_color})
drop2 = dcc.Dropdown(university_rank_df['Location'].unique(),
                     value='United States',
                     id='drop2',
                     style={'color': 'Black'})

# dropdown3
choose_university1 = html.Div(children='Choose university 1', style={'color': legend_color})
drop3 = dcc.Dropdown(
    ['Mean of all'] + list(university_rank_df[university_rank_df['Location'] == 'United Kingdom']['Name of University']),
    value='Mean of all',
    id='drop3',
    style={'color': 'Black'})
# dropdown4
choose_university2 = html.Div(children='Choose university 2', style={'color': legend_color})
drop4 = dcc.Dropdown(
    ['Mean of all'] + list(university_rank_df[university_rank_df['Location'] == 'United States']['Name of University']),
    value='Mean of all',
    id='drop4',
    style={'color': 'Black'})

# Score title
Score_title1 = html.H3(children='Scores 1', style={"textAlign": "center"}, id='scoretitle1')
Score_title2 = html.H3(children='Scores 2', style={"textAlign": "center"}, id='scoretitle2')

# Scores and  legends
overall_legend1 = html.Div(children='Overall', style={'color': legend_color})
overall_score1 = html.H4(children='0.0', id='overall1')

teaching_legend1 = html.Div(children='Teaching', style={'color': legend_color})
teaching_score1 = html.H4(children='0.0', id='teaching1')

s = html.Div(children='.', style={'color': '#0f2537'})

research_legend1 = html.Div(children='Research', style={'color': legend_color})
research_score1 = html.H4(children='0.0', id='research1')

citations_legend1 = html.Div(children='Citation', style={'color': legend_color})
citations_score1 = html.H4(children='0.0', id='citation1')

industry_income_legend1 = html.Div(children='Industry income', style={'color': legend_color})
industry_income_score1 = html.H4(children='0.0', id='industry1')

international_outlook_legend1 = html.Div(children='International outlook', style={'color': legend_color})
international_outlook_score1 = html.H4(children='0.0', id='international1')

overall_legend2 = html.Div(children='Overall', style={'color': legend_color, "textAlign": 'right'})
overall_score2 = html.H4(children='0.0', id='overall2', style={"textAlign": 'right'})

teaching_legend2 = html.Div(children='Teaching', style={'color': legend_color, "textAlign": 'right'})
teaching_score2 = html.H4(children='0.0', id='teaching2', style={"textAlign": 'right'})

research_legend2 = html.Div(children='Research', style={'color': legend_color, "textAlign": 'right'})
research_score2 = html.H4(children='0.0', id='research2', style={"textAlign": 'right'})

citations_legend2 = html.Div(children='Citation', style={'color': legend_color, "textAlign": 'right'})
citations_score2 = html.H4(children='0.0', id='citation2', style={"textAlign": 'right'})

industry_income_legend2 = html.Div(children='Industry income', style={'color': legend_color, "textAlign": 'right'})
industry_income_score2 = html.H4(children='0.0', id='industry2', style={"textAlign": 'right'})

international_outlook_legend2 = html.Div(children='International outlook',
                                         style={'color': legend_color, "textAlign": 'right'})
international_outlook_score2 = html.H4(children='0.0', id='international2', style={"textAlign": 'right'})

# graphs


graph1 = px.pie(values=[0, 1], width=260, height=350, title='piegraph', names=['yet', 'nothing'])
graph1.update_layout(margin=dict(
    l=0,
    r=0,
    b=0,
    t=0,
    pad=0),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    paper_bgcolor="rgba(0,0,0,0)")

gender_graph1_dcc = dcc.Graph(figure=graph1, id='gender graph1')
international_graph1_dcc = dcc.Graph(figure=graph1, id='international graph1')

gender_graph2_dcc = dcc.Graph(figure=graph1, id='gender graph2')
international_graph2_dcc = dcc.Graph(figure=graph1, id='international graph2')

#### Callback

@callback(
    [Output(component_id='drop3', component_property='options'),
     Output(component_id='drop4', component_property='options')],
    [Input(component_id='drop1', component_property='value'),
     Input(component_id='drop2', component_property='value')]
)
def sla(drop1, drop2):
    filter1 = university_rank_df['Location'] == drop1
    filter2 = university_rank_df['Location'] == drop2

    universities1 = ['Mean of all'] + list(university_rank_df[filter1]['Name of University'])
    universities2 = ['Mean of all'] + list(university_rank_df[filter2]['Name of University'])

    return universities1, universities2

@callback(
    [Output(component_id='scoretitle1', component_property='children'),
     Output(component_id='scoretitle2', component_property='children'),
     Output(component_id='overall1', component_property='children'),
     Output(component_id='overall2', component_property='children'),
     Output(component_id='teaching1', component_property='children'),
     Output(component_id='teaching2', component_property='children'),
     Output(component_id='research1', component_property='children'),
     Output(component_id='research2', component_property='children'),
     Output(component_id='citation1', component_property='children'),
     Output(component_id='citation2', component_property='children'),
     Output(component_id='industry1', component_property='children'),
     Output(component_id='industry2', component_property='children'),
     Output(component_id='international1', component_property='children'),
     Output(component_id='international2', component_property='children'),
     Output(component_id='gender graph1', component_property='figure'),
     Output(component_id='gender graph2', component_property='figure'),
     Output(component_id='international graph1', component_property='figure'),
     Output(component_id='international graph2', component_property='figure')
     ],
    [Input(component_id='drop1', component_property='value'),
     Input(component_id='drop2', component_property='value'),
     Input(component_id='drop3', component_property='value'),
     Input(component_id='drop4', component_property='value')]
)
def country_mean(country1, country2, university1, university2):


    # treating the overall column
    def mean_overall(df):
        overall_mean = []
        for i in df['OverAll Score']:
            if len(i) == 4:
                overall_mean.append(float(i))
            else:
                overall_mean.append((float(i[0:4]) + float(i[5:])) / 2)
        return np.array(overall_mean)


    graphs1_color = ['#790f8f', '#410f26']
    graphs2_color = ['#f93c3c', '#a50101']

    titlescore1, titlescore2 = '*', '*'

    # Scores and  legends
    international_ratio1, international_ratio1_1 = 1, 1
    male_ratio1, female_ratio1 = 1, 1

    international_ratio2, international_ratio2_1 = 1, 1
    male_ratio2, female_ratio2 = 1, 1

    overall_score1 = '*'

    teaching_score1 = '*'

    research_score1 = '*'

    citation_score1 = '*'

    Industry_Income_Score1 = '*'

    International_Outlook_Score1 = '*'

    overall_score2 = '*'

    teaching_score2 = '*'

    research_score2 = '*'

    citation_score2 = '*'

    Industry_Income_Score2 = '*'

    International_Outlook_Score2 = '*'

    # graphs


    graph1 = px.pie(values=[0, 1], width=260, height=350, title='piegraph', names=['yet', 'nothing'])
    graph1.update_layout(margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        paper_bgcolor="rgba(0,0,0,0)")

    gender_graph1 = graph1
    gender_graph2 = graph1

    international_graph1 = graph1
    international_graph2 = graph1

    # checking what we want to compare
    if university1 == 'Mean of all' and country1 != None and country2 != None and university1 != None and university2 != None:

        titlescore1 = f'{country1} mean Score'

        filter_country1 = university_rank_df['Location'] == country1

        df_country1 = university_rank_df[filter_country1]

        # getting the mean of overall score of each country
        overall_score1 = "%.1f" % mean_overall(df_country1).mean()

        # getting the mean of teatching score of each country
        teaching_score1 = "%.1f" % df_country1['Teaching Score'].mean()

        # getting the mean of research score of each country
        research_score1 = "%.1f" % df_country1['Research Score'].mean()

        # getting the mean of citations score of each country
        citation_score1 = "%.1f" % df_country1['Citations Score'].mean()

        # getting the mean of Industry Income score of each country
        Industry_Income_Score1 = "%.1f" % df_country1['Industry Income Score'].mean()

        # getting the mean of International Outlook score of each country
        International_Outlook_Score1 = "%.1f" % df_country1['International Outlook Score'].mean()

        # getting the total number of international students of each country and treating the column
        international_treated1 = df_country1['International Student'].str[::-1].str[1::].str[::-1].astype(float)
        international_ratio1 = (international_treated1 * 0.01 * df_country1['No of student']).sum()

        # getting the total number of non international studenst of each country
        international_ratio1_1 = df_country1['No of student'].sum() - international_ratio1

        # getting the total number of male and faemale students from country1
        female_ratio1 = (
                    df_country1['Female:Male Ratio'].str[0:2].astype(float) * 0.01 * df_country1['No of student']).sum()
        male_ratio1 = (df_country1['Female:Male Ratio'].str[4:].astype(float) * 0.01 * df_country1['No of student']).sum()

    elif country1 != None and country2 != None and university1 != None and university2 != None:
        # title score
        titlescore1 = f'{university1} Score'

        # filter
        university1_filter = university_rank_df['Name of University'] == university1

        # creating
        df_university1 = university_rank_df[university1_filter]

        # getting the mean of overall score of each country
        overall_score1 = "%.1f" % mean_overall(df_university1)[0]

        # getting the mean of teatching score of each country
        teaching_score1 = "%.1f" % float(df_university1['Teaching Score'])

        # getting the mean of research score of each country
        research_score1 = "%.1f" % float(df_university1['Research Score'])

        # getting the mean of citations score of each country
        citation_score1 = "%.1f" % float(df_university1['Citations Score'])

        # getting the mean of Industry Income score of each country
        Industry_Income_Score1 = "%.1f" % float(df_university1['Industry Income Score'])

        # getting the mean of International Outlook score of each country
        International_Outlook_Score1 = "%.1f" % float(df_university1['International Outlook Score'])

        # getting the total number of international students of each country and treating the column
        international_treated1 = df_university1['International Student'].str[::-1].str[1::].str[::-1].astype(float)
        international_ratio1 = float(international_treated1 * 0.01 * df_university1['No of student'])

        # getting the total number of non international studenst of each country
        international_ratio1_1 = float(df_university1['No of student'] - international_ratio1)

        # getting the total number of male and faemale students from country2
        female_ratio1 = float(df_university1['Female:Male Ratio'].str[0:2].astype(float))
        male_ratio1 = float(df_university1['Female:Male Ratio'].str[4:].astype(float))

    if university2 == 'Mean of all' and country1 != None and country2 != None and university1 != None and university2 != None:

        titlescore2 = f'{country2} mean Score'

        filter_country2 = university_rank_df['Location'] == country2

        df_country2 = university_rank_df[filter_country2]

        # getting the mean of overall score of each country
        overall_score2 = "%.1f" % mean_overall(df_country2).mean()

        # getting the mean of teatching score of each country
        teaching_score2 = "%.1f" % df_country2['Teaching Score'].mean()

        # getting the mean of research score of each country
        research_score2 = "%.1f" % df_country2['Research Score'].mean()

        # getting the mean of citations score of each country
        citation_score2 = "%.1f" % df_country2['Citations Score'].mean()

        # getting the mean of Industry Income score of each country
        Industry_Income_Score2 = "%.1f" % df_country2['Industry Income Score'].mean()

        # getting the mean of International Outlook score of each country
        International_Outlook_Score2 = "%.1f" % df_country2['International Outlook Score'].mean()

        # getting the total number of international students of each country and treating the column

        international_treated2 = df_country2['International Student'].str[::-1].str[1::].str[::-1].astype(float)
        international_ratio2 = (international_treated2 * 0.01 * df_country2['No of student']).sum()

        # getting the total number of non international studenst of each country
        international_ratio2_1 = df_country2['No of student'].sum() - international_ratio2

        # getting the total number of male and faemale students from country2
        female_ratio2 = (
                    df_country2['Female:Male Ratio'].str[0:2].astype(float) * 0.01 * df_country2['No of student']).sum()
        male_ratio2 = (df_country2['Female:Male Ratio'].str[4:].astype(float) * 0.01 * df_country2['No of student']).sum()


    elif country1 != None and country2 != None and university1 != None and university2 != None:

        titlescore2 = f'{university2} score'

        university2_filter = university_rank_df['Name of University'] == university2

        df_university2 = university_rank_df[university2_filter]

        # getting the mean of overall score of each country
        overall_score2 = "%.1f" % mean_overall(df_university2)[0]

        # getting the mean of teatching score of each country
        teaching_score2 = "%.1f" % float(df_university2['Teaching Score'])

        # getting the mean of research score of each country
        research_score2 = "%.1f" % float(df_university2['Research Score'])

        # getting the mean of citations score of each country
        citation_score2 = "%.1f" % float(df_university2['Citations Score'])

        # getting the mean of Industry Income score of each country
        Industry_Income_Score2 = "%.1f" % float(df_university2['Industry Income Score'])

        # getting the mean of International Outlook score of each country
        International_Outlook_Score2 = "%.1f" % float(df_university2['International Outlook Score'])

        # getting the total number of international students of each country and treating the column

        international_treated2 = df_university2['International Student'].str[::-1].str[1::].str[::-1].astype(float)
        international_ratio2 = float(international_treated2 * 0.01 * df_university2['No of student'])

        # getting the total number of non international studenst of each country
        international_ratio2_1 = float(df_university2['No of student'] - international_ratio2)

        # getting the total number of male and faemale students from country2
        female_ratio2 = float(df_university2['Female:Male Ratio'].str[0:2].astype(float))
        male_ratio2 = float(df_university2['Female:Male Ratio'].str[4:].astype(float))
        ####  Building all graphs

        # Gender pie graph

    gender_graph2 = px.pie(values=[male_ratio2, female_ratio2],
                           names=['Male', 'Female'],
                           color_discrete_sequence=graphs2_color)

    gender_graph1 = px.pie(values=[male_ratio1, female_ratio1],
                           names=['Male', 'Female'],
                           color_discrete_sequence=graphs1_color)

    # International students pie graph
    international_graph2 = px.pie(values=[international_ratio2, international_ratio2_1],
                                  names=['Foreigns students', 'Local students'],
                                  color_discrete_sequence=graphs2_color,
                                  )
    international_graph1 = px.pie(values=[international_ratio1, international_ratio1_1],
                                  names=['Foreigns students', 'Local students'],
                                  color_discrete_sequence=graphs1_color,
                                  )

    graphs = [international_graph1, gender_graph1, international_graph2, gender_graph2]
    for chart in graphs:
        chart.update_layout(margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            font=dict(
                color="#dadaff"
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            title_font_family="Times New Roman",
            title={
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 18}
            }
        )

    return (titlescore1, titlescore2, overall_score1, overall_score2, teaching_score1, teaching_score2,
            research_score1, research_score2, citation_score1, citation_score2, International_Outlook_Score1,
            International_Outlook_Score2, Industry_Income_Score1, Industry_Income_Score2, gender_graph1,
            gender_graph2, international_graph1, international_graph2)



#### Layout

app = dash.Dash(external_stylesheets=[dbc.themes.SUPERHERO])

app.layout = dbc.Container([
    dbc.Row([
        title,
    ]),
    dbc.Row([
        dbc.Col([
            choose_country1,
            drop1,
            choose_university1,
            drop3
        ]),
        dbc.Col([
            choose_country2,
            drop2,
            choose_university2,
            drop4
        ])
    ]),
    dbc.Row([
        s,
        dbc.Col([
            Score_title1
        ]),
        dbc.Col([
            Score_title2
        ])
    ]),
    dbc.Row([
        dbc.Col([
            overall_legend1,
            overall_score1,
            citations_legend1,
            citations_score1
        ]),
        dbc.Col([
            teaching_legend1,
            teaching_score1,
            industry_income_legend1,
            industry_income_score1
        ]),
        dbc.Col([
            research_legend1,
            research_score1,
            international_outlook_legend1,
            international_outlook_score1
        ]),
        dbc.Col([
            research_legend2,
            research_score2,
            international_outlook_legend2,
            international_outlook_score2
        ]),
        dbc.Col([
            teaching_legend2,
            teaching_score2,
            industry_income_legend2,
            industry_income_score2
        ]),
        dbc.Col([
            overall_legend2,
            overall_score2,
            citations_legend2,
            citations_score2
        ])
    ]),
    dbc.Row([
        dbc.Col([
            gender_graph1_dcc
        ]),
        dbc.Col([
            international_graph1_dcc
        ]),
        dbc.Col([
            gender_graph2_dcc
        ]),
        dbc.Col([
            international_graph2_dcc
        ])
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
