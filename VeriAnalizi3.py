import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State, ALL , MATCH
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import dash_table
from plotly.subplots import make_subplots
import numpy as np

class KesifselVeriAnalizi():
    
    def __init__(self,):
        self.df = None    
        self.filteredDf = None
    
    def getDF(self,):
        return self.df
    
    def setDF(self,df):
        self.df = df
        self.filteredDf = df
        
    def getFilteredDF(self,):
        return self.filteredDf

    def setFilteredDF(self,filteredDf):
        self.filteredDf = filteredDf
        
    def getObjNumCols(self,):

        self.categorical_columns = []
        self.numerical_columns = []
        
        for col in self.filteredDf.columns:
            colDtype = self.filteredDf[col].dtype
            if colDtype == 'O':
                self.categorical_columns.append(col)
            elif colDtype =='int64' or colDtype == 'float64' or colDtype == 'int32' or colDtype == 'float32':
                self.numerical_columns.append(col)
              
        return self.categorical_columns , self.numerical_columns
    
    


KVA = KesifselVeriAnalizi()
KVA.setDF(pd.read_csv("train.csv"))



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                             meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}],
                            suppress_callback_exceptions=True)


app.layout = dbc.Container([
    
    
    ############SATIR 1##################
    html.H1("Fibabank Keşifsel Veri Analizi",
                className='text-center text-dark'),

    
    
    
    
    ############SATIR 2##################
    dbc.Row([
        
        ############SÜTUN 1##################
        dbc.Col([
            html.Div([
                dcc.Tabs(id='filtretab1', value='tab-1', children=[
                    dcc.Tab(label='Sütun Çıkar', value='tab-1'),
                    dcc.Tab(label='Bakımda', value='tab-2'), #Veri Türünü Değiştir
                    dcc.Tab(label='Bakımda', value='tab-3'), #Filtrele
                ]),
                html.Div(id='tabs-content1',children=[])
                ]),
            
        ],width=3),
        
        
        
        ############SÜTUN 2##################
        dbc.Col([
            dash_table.DataTable(id='table',
                                page_size=10,
                                cell_selectable  = True,
                                virtualization=True,
                                ),
            ],width=9),
    ]),
    
    
    ############SATIR 3##################   
    html.Div([
                dcc.Tabs(id='filtretab2', value='tab-1', children=[
                    dcc.Tab(label='Özet İstatistik', value='tab-1'),
                    dcc.Tab(label='Özet Grafikler', value='tab-2'), #Özet Grafik
                    dcc.Tab(label='Hedef Değişken Görsel', value='tab-3'), #Hedef Değişken Görsel
                    dcc.Tab(label='Bakımda', value='tab-4'), #Hedef Değişken Analiz
                    dcc.Tab(label='Bakımda', value='tab-5'), #Yapay Zeka
                ]),
                
            ]),
    

   
    
    ###########SATIR 4#################
    #html.Br(id="break1"),
    dbc.Col(id='tabs-content2',children=[]),
    
    
    ###########SATIR 5#################
    html.Center(id='placeholder0'),

        
], fluid=True)





@app.callback(
    Output('tabs-content1', 'children'),
    Input('filtretab1', 'value'),
    State('tabs-content1','children')
)              
def filtretabUst(tab,children):
    if tab == 'tab-1':
        return [html.Center("Görselleştirilmeyecek Değişkenleri Seç"),
            dcc.Dropdown(id='gorsellestirilmeyecek-degiskenler', 
                         options = [{'label': x, 'value': x} for x in sorted(KVA.getDF().columns)],
                         value = [],
                         multi=True,
                         persistence=True)]          
    
    """
    elif tab == 'tab-2':
        df = KVA.getFilteredDF()

        returnwidgets = []
        for i,j in enumerate(df.columns):
            text = html.Center("{} :".format(j))
            drop = dcc.Dropdown(id={'type' : 'verituru-degistirilecek-degiskenler{}'.format(j),
                                    'index': i}, 
                         options = [{'label': x, 'value': x} for x in ["Nümerik","Kategorik"]],
                         value = [],
                         multi=False,
                         persistence=True,
                         style={"width":"200px","text-align": "center"}) 
            row = dbc.Row([text,drop])
            children.append(row)
        
        return dbc.Col(returnwidgets,style={"overflow": "scroll", "height": "200px"})"""
                             
                             
            
    
@app.callback(
    [Output('table', 'data'),
     Output('table', 'columns')],
    [Input('gorsellestirilmeyecek-degiskenler', "value")])
def show_tab1(gD):
    df = KVA.getDF()
    if gD == [] :
        KVA.setFilteredDF(df)
        return df.to_dict('records') , [{'id': c, 'name': c} for c in df.columns]

    df = df.drop(columns=gD)
    KVA.setFilteredDF(df)
    return df.to_dict('records') , [{'id': c, 'name': c} for c in df.columns]


"""
@app.callback(
    [Output('placeholder1', 'title')],
    [Input({'type':'verituru-degistirilecek-degiskenler{}'.format(i) , 'index':ALL} , "value")for i in KVA.getFilteredDF().columns])
def show_tab2(*args):
    print("asasas")
    #print(KVA.getFilteredDF().columns)
    help_dict = {}
    for i,j in enumerate(KVA.getDF().columns):
        #print(i is None)
        help_dict[j] = args[i]
    
    #print(help_dict)
    #df = KVA.getFilteredDF()
    #if gD == [] :
        #return df.to_dict('records') , [{'id': c, 'name': c} for c in df.columns]

    #df = df.drop(columns=gD)
    #KVA.setFilteredDF(df)
    #return df.to_dict('records') , [{'id': c, 'name': c} for c in df.columns]
    
    return [""]"""


@app.callback(
    output = [Output('tabs-content2', 'children')],
    inputs=[Input('filtretab2', 'value'),
            Input("table","columns")])              
def filtreVeriTipi(tab,columns):
    if tab == 'tab-1':
        df = KVA.getFilteredDF()
        
        veriTipi = pd.DataFrame(df.dtypes,columns=["Veri Tipi"]).replace(['object'],'Kategorik').replace(['float64'],'Nümerik').replace(["int64"],"Nümerik").replace(["int32"],"Nümerik").replace(["float32"],"Nümerik")
        veriTipi.reset_index(inplace=True)
        veriTipi.rename(columns={"index":"Sütun Adı"}, inplace=True)
        
        
        dashPVeriTipi = html.Center("Veri Türleri")
        dashVeriTipi = dash_table.DataTable(id='tab1table1',
                                data = veriTipi.to_dict('records') , 
                                columns = [{'id': c, 'name': c} for c in veriTipi.columns],
                                page_size=10,
                                cell_selectable  = True,
                                virtualization=False)
        
        
        dashNA = pd.DataFrame(df.isnull().sum(),columns=["Na Sayısı"]).reset_index().rename(columns={"index":"Sütun Adı"})
        dashPVeriTipi2 = html.Center("Boş Değer Sayıları")
        dashVeriTipi2 = dash_table.DataTable(id='tab1table2',
                                data = dashNA.to_dict('records') , 
                                columns = [{'id': c, 'name': c} for c in dashNA.columns],
                                page_size=10,
                                cell_selectable  = True,
                                virtualization=False)
        
        
        cat,num = KVA.getObjNumCols()
        cat_dfs = [df[j].value_counts().reset_index().rename(columns={"index":str(j),
                                                                      j:"Toplam"}) for j in cat]
        
        
        described = df.describe().reset_index().rename(columns={"index":"İstatistik"}).round(3)


        col1 = [dbc.Col([
                    dashPVeriTipi,dashVeriTipi
                    ])]
        col2 = [dbc.Col([
                    dashPVeriTipi2,dashVeriTipi2
                    ])]
        
        col3 = [dbc.Col([
                    html.Center("{} Değişkeni".format(j.columns[0])),
                    dash_table.DataTable(id='tab1catTable{}'.format(i),
                                data = j.to_dict('records') , 
                                columns = [{'id': c, 'name': c} for c in j.columns],
                                page_size=10,
                                cell_selectable  = True,
                                virtualization=False)
                    ]) 
                for i,j in enumerate(cat_dfs)]
        
        

        col4 = [dbc.Col([
                html.Center("Tanımlayıcı İstatistik"),
                dash_table.DataTable(id='tab1descrpTable',
                                data = described.to_dict('records'), 
                                columns = [{'id': c, 'name': c} for c in described.columns],
                                page_size=10,
                                cell_selectable  = True,
                                virtualization=False)
            ])]
        
        allcols = col1+col2+col3+col4
        row = dbc.Row(allcols)
        
        
        returnwidgets = [row]
        

        
        return returnwidgets
    
    
    elif tab == 'tab-2':
        cat,num = KVA.getObjNumCols()
        cols = []
        
        

        for index,i in enumerate(cat):
            text1 = html.Center("{}'nin Bar Chartı".format(i))
            graph1 = dcc.Graph(id={'type':'cat-graph-bar',
                                  'index':index},
                               className = i, 
                               figure={})
            
            col = dbc.Col([text1,graph1],width=6)
            cols.append(col)
            
            text2 = html.Center("{}'nin Pie Chartı".format(i))
            graph2 = dcc.Graph(id={'type':'cat-graph-pie',
                                  'index':index},
                               figure={})
            
            htmlp2 = html.P(id={'type':'html-2',
                                'index':index},
                            className=i)
            
            
            col = dbc.Col([text2,graph2,htmlp2],width=6)
            cols.append(col)
        

        for index,i in enumerate(num):
            text3 = html.Center("{}'nin Histogramı".format(i))

            slider3 = dcc.Slider(id={'type':'my-slider-tab2',
                                     'index':index},
                                 className = i,
                                 min=1,max=100,step=1,value=20,
                                 tooltip={"always_visible":True})

            graph3 = dcc.Graph(id={'type':'num-graph-hist',
                                   'index':index},
                              figure={})

            
            col = dbc.Col([text3,slider3,graph3],width=6)
            cols.append(col)
            
            
        returnwidget = [dbc.Row(cols)]
        return  returnwidget
    
    
    elif tab == 'tab-3':
        center1 = html.Center("Ondalık Gösterimi")
        ondalikSecimi = dcc.Slider(id='ondalik-secimi',
                                   min=0,max=3,step=1,value=2,
                                   marks={0: '0',1:'1',2:'2',3:'3'})    
        
        center2 = html.Center("Hedef Değişken Seçiniz")
        hedefDegisken = dcc.Dropdown(id='hedef-degisken-secimi', 
                         options = [{'label': x, 'value': x} for x in sorted(KVA.getFilteredDF().columns)],
                         value = [],
                         multi=False,
                         persistence=True)
        
        center3 = html.Center(".....")
        center4 = html.Center("....")
        center5 = html.Center("...")
        center6 = html.Center("..")
        center7 = html.Center(".")
        row = html.Row(id="tab-3-row",children=[])
        col = dbc.Col([center1,ondalikSecimi,center2,hedefDegisken,center3,center4,center5,center6,center7,row]
                      ,width={"offset":3,
                              "size":6})
        
        
        return [col]
    

#Input("hedef-degisken-secimi","value"),
#Input("ondalik-secimi","value"),
@app.callback(
    output= [Output("tab-3-row","children")],
    inputs=[Input("hedef-degisken-secimi","value"),
            Input("table","columns")]
    )    
def show_tab3Content(hedefDegisken,columns):
    returnWidget = []
    if hedefDegisken == None:
        return [""]
    
    
    cat,num = KVA.getObjNumCols()
    
    
    for index,i in enumerate(cat+num):
        if i == hedefDegisken:
            continue
        
        
        slider = dcc.Slider(id={'type':'my-slider-tab3',
                                     'index':index},
                            className = i,
                            min=1,max=100,step=1,value=20,
                            tooltip={"always_visible":True})
        
        
        dropdown = dcc.Dropdown(id={'type':'dropdown-tab3',
                                     'index':index}, 
                                 options = [{'label': x, 'value': x} for x in sorted(KVA.getFilteredDF()[hedefDegisken].unique())],
                                 value = [],
                                 multi=True,
                                 persistence=True)
        

        graph = dcc.Graph(id={'type':'num-graph-hist-tab3',
                              'index':index},
                          figure={})
        
        returnWidget.append(slider)
        returnWidget.append(dropdown)
        returnWidget.append(graph)
        
    col = dbc.Col(returnWidget)
        
    return [col] if len(returnWidget) > 1 else [""]
    
@app.callback(
    output= [Output({'type':'num-graph-hist-tab3' , 'index':MATCH}, 'figure')],
    inputs=[Input("hedef-degisken-secimi","value"),
            Input("ondalik-secimi","value"),
            Input({'type':'my-slider-tab3' , 'index':MATCH} , "value"),
            Input({'type':'my-slider-tab3' , 'index':MATCH} , "className"),
            Input({'type':'dropdown-tab3' , 'index':MATCH} , "value")]
    )       
def show_tab3Fig(hedefDegisken,ondalik,bins,sutun,lineSelection):
    
    cat,num = KVA.getObjNumCols()
    dataX = KVA.getFilteredDF()
    if sutun in num:
        i = sutun

        if i == hedefDegisken:
            return [{}]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        _, edges = pd.cut(dataX[i], bins=bins, retbins=True)
        
        if ondalik==0:
            labels = [f'{abs(edges[i]):.0f}-{edges[i+1]:.0f}' for i in range(len(edges)-1)]
        elif ondalik==1:
            labels = [f'{abs(edges[i]):.1f}-{edges[i+1]:.1f}' for i in range(len(edges)-1)]
        elif ondalik==2:
            labels = [f'{abs(edges[i]):.2f}-{edges[i+1]:.2f}' for i in range(len(edges)-1)]
        elif ondalik==3:
            labels = [f'{abs(edges[i]):.3f}-{edges[i+1]:.3f}' for i in range(len(edges)-1)]

        dataX["AnlikSutun"] = pd.cut(dataX[i] , bins=bins,labels=labels)
        forbarChart = pd.DataFrame(dataX.groupby(["AnlikSutun",hedefDegisken]).size(),columns=["Count"])
        forbarChart["{}".format(i)] = forbarChart.index.get_level_values(0)
        forbarChart["{}?".format(hedefDegisken)] = forbarChart.index.get_level_values(1)
        forbarChart = forbarChart.reset_index(drop=True)
        forbarChart["Perc"] = forbarChart["Count"]/forbarChart["Count"].sum()
        
        
        emptySeri = pd.Series()
        for uni in forbarChart[i].unique():
            Perc = forbarChart[forbarChart[i] == uni].Count / forbarChart[forbarChart[i] == uni].Count.sum()
            emptySeri = emptySeri.append(Perc,ignore_index=True)
        forbarChart["Perc2"] = emptySeri
        
        forbarChartx = forbarChart.groupby(by=[i]).sum().reset_index()
        
        figBar = px.bar(forbarChartx,
                       x="{}".format(i),
                       y="Count",
                       #color=forbarChart["{}?".format(self.hedef)].astype(str),
                       color_discrete_sequence=px.colors.qualitative.Set2)
    
        
        figLine = px.line(forbarChart,
                       x="{}".format(i),
                       y="Perc2",
                       color=forbarChart["{}?".format(hedefDegisken)].astype(str),
                       color_discrete_sequence=px.colors.qualitative.Dark2)
        
        
        
        lineIndex = []
        for line in lineSelection:
            lineIndex.append(np.where(forbarChart["{}?".format(hedefDegisken)].unique() == line))
        lineIndex = np.ravel(lineIndex)

        for figdata in lineIndex:
            fig.add_trace(figLine["data"][figdata],secondary_y=True)
            
        for figdata in range(len(figBar["data"])):
            fig.add_trace(figBar["data"][figdata])
            
        fig.update_layout(barmode="group" , title_text="{}-{}".format(i,hedefDegisken) ,
                          legend_title_text=hedefDegisken , yaxis2_tickformat = '.1%',
                          yaxis_title="Toplam Oran" , yaxis2_title="Oran" , xaxis_title="{}".format(i))

        return [fig]
    
    
    elif sutun in cat:
        i = sutun
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                
        forbarChart = pd.DataFrame(dataX.groupby([i,hedefDegisken]).size(),columns=["Count"])
        forbarChart["{}".format(i)] = forbarChart.index.get_level_values(0)
        forbarChart["{}?".format(hedefDegisken)] = forbarChart.index.get_level_values(1)
        forbarChart = forbarChart.reset_index(drop=True)
        
        forbarChart["Perc"] = forbarChart["Count"]/forbarChart["Count"].sum()
        
        emptySeri = pd.Series()
        for uni in forbarChart[i].unique():
            Perc = forbarChart[forbarChart[i] == uni].Count / forbarChart[forbarChart[i] == uni].Count.sum()
            emptySeri = emptySeri.append(Perc,ignore_index=True)
        forbarChart["Perc2"] = emptySeri
            
        forbarChartx = forbarChart.groupby(by=[i]).sum().reset_index()

        figBar = px.bar(forbarChartx,
                       x="{}".format(i),
                       y="Count", #Count yapılabilir
                       color_discrete_sequence=px.colors.qualitative.Set2)
        
        figLine = px.line(forbarChart,
                       x="{}".format(i),
                       y="Perc2",
                       color=forbarChart["{}?".format(hedefDegisken)].astype(str),
                       color_discrete_sequence=px.colors.qualitative.Dark2)
        

        
        lineIndex = []
        for line in lineSelection:
            lineIndex.append(np.where(forbarChart["{}?".format(hedefDegisken)].unique() == line))
        lineIndex = np.ravel(lineIndex)

        
        for figdata in lineIndex:
            fig.add_trace(figLine["data"][figdata],secondary_y=True)
        
        
        for figdata in range(len(figBar["data"])):
            fig.add_trace(figBar["data"][figdata])
        
        
            
        fig.update_layout(barmode = "relative", title_text="{}-{}".format(i,hedefDegisken) ,
                          legend_title_text=hedefDegisken , yaxis2_tickformat = '.1%',
                          yaxis_title="Toplam Oran" , yaxis2_title="Oran" , xaxis_title="{}".format(i))
    
        return [fig]
    
    return [{}]



@app.callback(
    output = Output({'type':'num-graph-hist' , 'index':MATCH}, 'figure'),
    inputs = [Input({'type':'my-slider-tab2' , 'index':MATCH} , "value"),
              Input({'type':'my-slider-tab2' , 'index':MATCH} , "className"),]
    )
def show_tab2num(value,className):
    df = KVA.getFilteredDF()

    fig = px.histogram(df,
                       x=str(className),
                       nbins=value,
                       marginal="box",
                       histfunc="count",title="{}'nin Histogramı".format(className))
    
    return fig


@app.callback(
    output = [Output({'type':'cat-graph-bar' , 'index':MATCH}, 'figure'),
              Output({'type':'cat-graph-pie' , 'index':MATCH}, 'figure')],
    inputs = [Input({'type':'html-2' , 'index':MATCH} , "className"),]
    )
def show_tab2cat(catName):
    df = KVA.getFilteredDF()
    catVal = df[catName].value_counts()
    
    if len(catVal.keys()) <= 10:
        fig1 = px.bar(df,
                      x=catVal.keys(),
                      y=catVal,
                      color=catVal.keys(),
                      text = ["%{}".format(round(j*100,2)) for j in catVal/catVal.sum()],
                      title = "{}'nin BarChartı".format(catName))
        
        
        fig2 = px.pie(df,
                      names=catVal.keys(),
                      values=catVal,
                      color=catVal.keys(),
                      title = "{}'nin PieChartı".format(catName))
    
    else:
        fig1 = px.bar(title = "Görselleştirelemedi. {} için çok fazla kategorik değişken var.".format(catName))
        
        fig2 = px.pie(title = "Görselleştirelemedi. {} için çok fazla kategorik değişken var.".format(catName))

    
    return fig1 , fig2



















if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False,port=3000)
    
    
    
    