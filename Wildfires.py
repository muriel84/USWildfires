# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 19:47:48 2022

@author: mlant
"""

import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objects as go
from scipy.stats import pearsonr 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

st.sidebar.title("Sommaire")

pages = ["Introduction", "Exploration du dataset", "Data Visualisation", "Modélisation", "Conclusion"]
page = st.sidebar.radio("Aller vers", pages)

#df = pd.read_csv(r"D:\Emploi - Formation\Datascientest\Element de cours\Projets\fire_data.csv", low_memory = False)

if page == pages[0]:
    st.title("Projet Wildfire")
    st.subheader("Introduction")
    #df = pd.read_csv(r"C:\Users\Florian & Lucile\Desktop\df_carte.csv")
    #fig = px.scatter_mapbox(df, 
                        #lat="latitude", 
                       # lon="longitude", 
                       # color="stat_cause_descr",
                        #color_continuous_scale=px.colors.cyclical.IceFire, 
                        #animation_frame="fire_year",
                        #hover_name="states_long",
                        #hover_data = {"state" : False, "latitude" : False, "longitude" : False},
                        #title = "La cause des feux de forêt entre 1992 et 2015",
                       #labels = {"fire_year" : "Année", "stat_cause_descr" : "Cause du départ du feu"},
                       #zoom=3)
    #fig.update_layout(mapbox_style="open-street-map")
    #fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})

    #st.plotly_chart(fig)
    
    st.image("carte_intro.png")

    st.write("\n")
    
    st.markdown("Le dataset est une **base de données spatiales des feux de forêt** ayant eu lieu aux **États-Unis** entre **1992**\
                et **2015**. Cette base de données comprend **1.88 million** d'enregistrement d'incendies géoréférencés,\
                représentant un total de **57 millions d'hectares brûlés** au cours d'une période de **24 ans**")
    
    st.markdown("Le dataset a été créé par le gouvernement américain. Nous considérons ces données comme **fiables**.")
    
    st.write("\n")
    
    st.subheader("Nos objectifs")
    
    st.markdown("- Comprendre l'évolution des feux de forêts aux États-Unis entre 1995 et 2015 grâce à la **Data Visualisation**.\n"
                "- Essayer de déterminer s'il est possible de construire un modèle de **Machine Learning** capable de prédire la superficie relative de la végétation brûlée.")
    
    
elif page == pages[1] :



    df = pd.read_csv("df_head.csv", index_col= 0)

    st.title("Exploration du dataset")

    st.header("Les premières informations du dataset")
    st.subheader("Aperçu du dataset")
    st.dataframe(df)
    st.markdown("- Nous avons **1 880 465 entrées pour 39 variables**.\n" 
                "- Les données sont composées de variables numériques et catégorielles.\n" 
                "- Il n'y a **pas de doublons.**\n"
                "- Il y a de **nombreuses données manquantes**.\n")

    st.write("\n")
    st.write("\n")

    st.subheader("Sélection des variables")

    col1, col2 = st.columns(2, gap = "medium")

    # Code

    df2 = pd.read_csv("df_nan.csv")


    with col1 :
        st.write(df2)

    with col2:
        st.markdown("- Suppression des variables sur la provenance des données.\n" 
                "- Suppression des variables redondantes.\n" 
                "- Suppression des variables contenant trop de valeurs manquantes.\n"
                "\n"
                "\n")
        
        
    st.write("\n")
    st.write("\n")

    st.subheader("Dataset après un premier nettoyage")


    df3 = pd.read_csv("df_head_2.csv", index_col= 0)

    st.dataframe(df3)

    # Saut de lignes


    st.write("\n")
    st.write("\n")

    st.subheader("Deux variables cibles potentielles")

    st.markdown("Nous avons **deux variables cibles** dans notre dataset qui peuvent nous permettrent de répondre à notre problématique :")
    st.markdown ("- **fire_size** qui donne une estimation du nombre d'hectares brulés dans le périmètre final de l'incendie.\n"
                 "- **fire_size_class**, variable qui divise la précédente en **7 catégories** :")
    st.markdown("> A : > 0 et <= 0.10 hectares  \n"
                "> B : 0.11 à 3.6 hectares  \n"
                "> C : 3.7 à 40 hectares  \n"
                "> D : 41 à 121 hectares  \n"
                "> E : 122 à 404 hectares  \n"
                "> F : 405 à 2023 hectares  \n"
                "> G : > 2023 hectares")
                    
    st.write("\n")
    st.write("\n")
             

    # Graphique de distribution des classes de feux

    st.image("classes_desequilibre.png")
        
    st.write("\n")

    st.markdown("Le jeu de données est **déséquilibré.**")

    # Saut de lignes

    st.write("\n")
    st.write("\n")

    st.subheader("Matrice de corrélation")

    st.image("matrice_corr.png")

    st.write("\n")

    st.write("Il y a très peu de corrélations entre la variable cible et les variables explicatives.\
             Il va falloir injecter de nouvelles données.")

    # Saut de lignes

    st.write("\n")
    st.write("\n")

    # Ajout sous-titre
    st.subheader("Ajouts de données")


    st.markdown ("- **Données météorologiques** (source : librairie meteostat)\n"
                 "- **Nombre de casernes de pompiers** disponibles par État (source : U.S Department of Homeland Security)\n"
                 "- **Distance entre le point de départ de feu et la caserne la plus proche** (source : U.S Department of Homeland Security)\n"
                 "- **Densité de population au niveau de chaque État** (source : Wikipédia)\n"
                 "- **La couverture forestière par État** (source : Wikipédia)\n")
if page==pages[2]:
    
    st.subheader('DataViz')
        
    df = pd.read_csv('clean_data.csv')
    
    # Mise en parallèle du nombre d'incendies et du nombre d'hectares brûlés
    st.subheader("Evolution globale sur la période")
 
    df_fire_size = df.groupby("fire_year")["fire_size"].sum().reset_index(name="Total_surface_brulee")
    df_fire_number = df.groupby("fire_year")["fod_id"].count().reset_index(name="nombre incendie")
    # Premier graphe (barplot représentant le nombre d'hectares brûlés)
    trace1 = go.Bar(
        x= df_fire_size["fire_year"],
        y= df_fire_size["Total_surface_brulee"],
        marker=dict(color='grey'),
        name="Nombre d'hectares brûlés") # titre légende
    # Deuxième graphe (lineplot représentant le nombre d'incendies)
    trace2 = go.Scatter(
        x=df_fire_number["fire_year"],
        y=df_fire_number["nombre incendie"],
        marker=dict(color='Red'),
        name="Nombre d'incendies") # titre légende

    fig = make_subplots(specs=[[{"secondary_y": True}]]) # On précise qu'il y aura un second axe Y
    fig.add_trace(trace1)
    fig.add_trace(trace2,secondary_y=True) # on précise qu'il s'agit du deuxième axe Y
    fig.update_yaxes(range=[0,120000], secondary_y=True)
    fig['layout'].update(height = 400, width = 800,xaxis=dict(tickangle=-70), template="simple_white")
    fig.update_layout( 
        title = {
                'text':"Nombre de feux de forêt et surfaces brûlées aux États-Unis entre 1992 et 2015",
                'y':0.9,
                'x':0.5,
                'xanchor' : 'center'
            }
    )
    # Titre de l'axe X
    fig.update_xaxes(title_text="Années")
    # Titre des axes Y
    fig.update_yaxes(title_text= "ha brûlés (M)", title_font_size= 12, secondary_y=False)
    fig.update_yaxes(title_text="Nb d'incendies (K)", title_font_size= 12, secondary_y=True)

    st.plotly_chart(fig)
        
    st.markdown("Les superficies brûlées annuellement ont presque doublées alors que le nombre d'incendies n'a augmenté que de 13% :\n"
                "- **1.5M** d’ha brûlés pour **72K** incendies en moyenne dans les années 90\n"
                "- **2.8M** d'ha brûlés pour **82K** incendies en moyenne dans les années 2000.")             
    #st.markdown("Descendons maintenant d’un niveau pour visualiser s’il existe ou non une saisonnalité des feux de forêt.")
    
    st.subheader('Analyse de la saisonnalité')
    
    st.markdown("Nous commencons par visualiser simplement le cumul d'incendies par mois")
    # Nombre d'incendies par mois
    fig3 = plt.figure(figsize=(15, 6))
    sns.countplot(x='month', palette='Spectral',data=df)
    plt.xlabel("Mois")
    plt.ylabel("Nombre d'incendies")
    plt.title("Nombre d'incendies par mois calendaire de 1992 à 2015")
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11],['Janvier','février','Mars','Avril','Mai','Juin','Juillet','Août','Septembre','Octobre','Novembre','Décembre'])
    st.pyplot(fig3)
    
    st.markdown("On observe **2 pics dans l'année** : en mars/ avril ainsi qu'en juillet/août, et une baisse à l'automne pour atteindre le niveau le plus bas en décembre. Le pic estival était attendu, mais pas celui du printemps.")
    st.markdown(" Etant donné la taille du territoire étudié, il peut y avoir des **disparités géographiques**, avec une saison des feux différente selon la localisation. Afin de vérifier cette hypothèse, nous regroupons les Etats selon les régions suivantes : WEST, MIDWEST, NORTHEAST, SOUTH et PACIFIC.")
    
    fig4 = plt.figure(figsize=(15,10)) 
    # Nombre d'incendies par mois par région
    plt.subplot(231)
    sns.countplot(x='month', palette='Spectral',data=df[df['region'] == 'WEST'])
    plt.xlabel(None)
    plt.ylabel("Nombre d'incendies")
    plt.title("WEST")

    plt.subplot(232)
    sns.countplot(x='month', palette='Spectral',data=df[df['region'] == 'MIDWEST'])
    plt.xlabel(None)
    plt.ylabel(None)
    plt.title("MIDWEST")

    plt.subplot(233)
    sns.countplot(x='month', palette='Spectral',data=df[df['region'] == 'NORTHEAST'])
    plt.xlabel(None)
    plt.ylabel(None)
    plt.title("NORTHEAST")

    plt.subplot(234)
    sns.countplot(x='month', palette='Spectral',data=df[df['region'] == 'SOUTH'])
    plt.xlabel(None)
    plt.ylabel("Nombre d'incendies")
    plt.title("SOUTH")

    plt.subplot(235)
    sns.countplot(x='month', palette='Spectral',data=df[df['region'] == 'PACIFIC'])
    plt.xlabel(None)
    plt.ylabel(None)
    plt.title("PACIFIC")
    
    st.pyplot(fig4)
    
    st.markdown("Avec ce découpage géographique, on constate qu'il n'y a qu’**une seule « saison des feux » par région**, mais qu’elle est diffère selon la localisation :")
    lst = ["Une saison courte dans les régions MIDWEST et NORTHEAST, de mars à mai ;", "Une saison plus longue en début d'année pour la région SOUTH, de février à avril. On constate de plus qu'il n'y a pas de « basse saison », le nombre de feu le reste de l'année demeurant important, y compris en hiver ;", "Une saison longue dans les régions WEST et PACIFIC, de mai à août / septembre."]
    s = ''
    for i in lst:
        s += "- " + i + "\n"
    st.markdown(s)
    
    st.markdown("Intéressons-nous maintenant aux feux de classe 6 et 7, les feux de taille très importante. Nous voulons voir s'ils surviennent durant ces pics de nombre d’incendie ou si leur distribution est différente :")
    
    # Nombre d'incendies par mois par région
    fig5 = plt.figure(figsize=(15,10))

    plt.subplot(231)
    sns.countplot(x='month', palette='Spectral',data=df[(df['region'] == 'WEST') & (df['fire_size_class'] == 7) | (df['fire_size_class'] == 6)])
    plt.xlabel(None)
    plt.ylabel("Nombre d'incendies")
    plt.title("WEST")

    plt.subplot(232)
    sns.countplot(x='month', palette='Spectral',data=df[(df['region'] == 'MIDWEST') & (df['fire_size_class'] == 7) | (df['fire_size_class'] == 6)])
    plt.xlabel(None)
    plt.ylabel(None)
    plt.title("MIDWEST")

    plt.subplot(233)
    sns.countplot(x='month', palette='Spectral',data=df[(df['region'] == 'NORTHEAST') & (df['fire_size_class'] == 7) | (df['fire_size_class'] == 6)])
    plt.xlabel(None)
    plt.ylabel(None)
    plt.title("NORTHEAST")

    plt.subplot(234)
    sns.countplot(x='month', palette='Spectral',data=df[(df['region'] == 'SOUTH') & (df['fire_size_class'] == 7) | (df['fire_size_class'] == 6)])
    plt.xlabel(None)
    plt.ylabel("Nombre d'incendies")
    plt.title("SOUTH")

    plt.subplot(235)
    sns.countplot(x='month', palette='Spectral',data=df[(df['region'] == 'PACIFIC') & (df['fire_size_class'] == 7) | (df['fire_size_class'] == 6)])
    plt.xlabel(None)
    plt.ylabel(None)
    plt.title("PACIFIC")
    
    st.pyplot(fig5)
    
    st.markdown("**Les feux de classe 6 et 7 ont lieux principalement en été et ce sur tout le territoire**. Si les graphiques pour les régions WEST et PACIFIC ne changent presque pas, ceux du MIDWEST, NORTHEAST et SOUTH eux sont très différents. Ces 3 régions ont donc davantage de feux en début d'année, mais ces derniers sont de taille plus réduite qu'en été.")
    
    st.markdown("Nous avons vu qu'il existe une « saison des feux », et que celle-ci est différente selon la localisation. Nous allons à présent nous intéresser plus finement à ces localisations.")
    
    st.subheader('Analyse spatiale')
    
    st.markdown("Regardons tout d'abord le nombre total d'incendies et d'hectares brûlés par Etat sur toute la période.")
    
    df_count_sum = df.groupby(['region','state']).agg({'fod_id':'count', 'fire_size': 'sum'}).reset_index()
    df_count_sum['fire_size'] /= 1000000
    df_count_sum['fod_id'] /= 1000
    df_count_sum = df_count_sum[df_count_sum['state'] != 'PR']
    df_count_sum['region'] = df_count_sum['region'].replace(to_replace='PACIFIC', value='PAC')
    fig6 = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig6.add_trace(
        go.Bar(x=[df_count_sum.region, df_count_sum.state], y=df_count_sum.fod_id, name="Nb. d'incendies", offsetgroup=1),
        secondary_y=False,
    )
    fig6.add_trace(
        go.Bar(x=[df_count_sum.region, df_count_sum.state], y=df_count_sum.fire_size, name="Ha brûlés", offsetgroup=2),
        secondary_y=True,
    )
    # Set x-axis title
    fig6.update_xaxes(title_text="Etat")
    # Set y-axes titles
    fig6.update_yaxes(title_text=" Ha brûlés (M)", range=[0,14], secondary_y=True)
    fig6.update_yaxes(title_text="Nombre d'incendies (K)", range=[0,200], tickvals = [0, 29, 57, 86, 114, 143, 171, 200], secondary_y=False)
    fig6.update_layout(barmode='group')
    fig6.update_layout( 
        title = {
                'text':"Cumul du nombre d'incendies et superficies brûlées par Etat de 1992 à 2015",
                'y':0.9,
                'x':0.5,
                'xanchor' : 'center'
            }
    )
    st.plotly_chart(fig6)

    
    st.markdown('On constate de grandes disparités entre les Etats, que ce soit en nombre d’incendies ou en superficies. Le total des superficies ne semble pas directement corrélé au nombre d’incendies. Voici quelques chiffres marquants illustrant cela :')
    lst = ['l’Alaska totalise à lui seul près de 25% des superficies brûlés pour moins de 1% des incendies ;', 'dans le même temps, la Géorgie, deuxième en nombre total d’incendies avec plus de 9%, ne représente qu’à peine 1% des superficies brûlées ;', 'la Californie est première en nombre d’incendies (10% du total) et 3ème en forêts brulées avec 9% des superficies.']
    s = ''
    for i in lst:
        s += "- " + i + "\n"
    st.markdown(s)
    
    st.markdown('En regroupant les données par région, ces disparité géographiques sont très visible :')
    
    df_region_count_sum = df[df['region']!='PR'].groupby(['region']).agg({'fod_id':'count', 'fire_size': 'sum'}).reset_index()
    df_region_count_sum['fire_size'] /= 1000000
    df_region_count_sum['fod_id'] /= 1000

    fig7 = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig7.add_trace(
        go.Bar(x=df_region_count_sum.region, y=df_region_count_sum.fod_id, name="Nb. d'incendies", offsetgroup=1),
        secondary_y=False,
    )
    fig7.add_trace(
        go.Bar(x=df_region_count_sum.region, y=df_region_count_sum.fire_size, name="Ha brûlés", offsetgroup=2),
        secondary_y=True,
    )

    # Add figure title
    fig7.update_layout( 
        title = {
                'text':"Cumul du nombre d'incendies et des superficies brûlées par région de 1992 à 2015",
                'y':0.9,
                'x':0.5,
                'xanchor' : 'center'
            }
    )

    # Set x-axis title
    fig7.update_xaxes(title_text="Région")

    # Set y-axes titles
    #
    fig7.update_yaxes(title_text=" Ha brûlés (M)", range=[0,32], secondary_y=True)
    fig7.update_yaxes(title_text="Nombre d'incendies (K)", range=[0,1000], tickvals = [0, 156, 312, 468, 625, 781, 937], secondary_y=False)
    fig7.update_layout(barmode='group')
    st.plotly_chart(fig7)
    
    lst = ["La région Pacific, dans laquelle l’Alaska est inclus, n'a qu'un nombre très limité d'incendies sur la période et se positionne en 2ème position des superficies parties en fumée avec 13 millions d’hectares ;","La région South compte le plus d’incendies (près de 1 millions en 24 ans), mais ne totalise « que » 10 millions d’hectare brûlés ;","La région WEST, qui inclue la Californie qui a fait la une des journaux ces dernières années avec des feux complètement hors de contrôle, totalise plus de 30 millions d’hectares brûlées pour 550 000 feux, soit 3 fois plus d’ha pour 2 fois moins d’incendie que la région SOUTH ;","Les régions MIDWEST ET NORTHEST ne sont pas très touchées, que ce soit en nombre d'incendies ou de superficies brûlées."]
    s = ''
    for i in lst:
        s += "- " + i + "\n"
    st.markdown(s)
    
    st.markdown("Visualisons sur une carte le cumul des incendies au fil du temps :")
    
    df_by_state = df.groupby(["state", "states_long","fire_year", ])["fod_id"].count().reset_index(name="Count")
    cum= []
    cum.insert(0,int(df_by_state.loc[0,['Count']]))

    for i in range(1,len(df_by_state)):
        if (df_by_state.loc[i,['state']].item() == df_by_state.loc[i-1,['state']]).item():
            cum.insert(i,(cum[i-1]+ int(df_by_state['Count'][i])))
        else:
            cum.insert(i,int(df_by_state['Count'][i]))
    df_by_state['cum']= cum
    
    fig8 = px.choropleth(data_frame = df_by_state, 
                locations ="state", 
                color = "cum", 
                locationmode="USA-states", 
                scope="usa", 
                animation_frame = "fire_year",
                hover_name="states_long",
                hover_data = {"state" : False},
                color_continuous_scale="YlOrRd",
                labels = {"fire_year" : "Année", "cum" : "Nombre d'incendies cumulés", "state" : "État"})
            
    fig8.update_layout(coloraxis_colorbar=dict(thickness=15))
    fig8.update_layout(title_text="Cumul du nombre de feux de forêt aux États-Unis entre 1992 et 2015", title_x=0.5)
    st.plotly_chart(fig8)

    st.markdown("Les chiffres et disparités géographiques sont très visible sur une carte : avec le cumul sur toute la période, les Etats du Sud et l'Ouest apparaissent majoritairement en rouge tandis que l’intérieur des terres semble épargné.")
    
    
    # Carte pour analyser l'évolution par classe de feu et par Etat du nombre d'incendies

    df_by_fsc = df.groupby(by=["fire_size_class", "state"])["fod_id"].count().reset_index(name='Count')

    fig9 = px.choropleth(data_frame = df_by_fsc, 
                        locations ="state", 
                        color = "Count", 
                        locationmode="USA-states", 
                        scope="usa", 
                        animation_frame = "fire_size_class",
                        hover_name="state",
                        color_continuous_scale="Bluered",
                    labels = {"fire_size_class" : "Classe de feu", "Count" : "Nombre d'incendies", "state" : "État"})
                
    fig9.update_layout(coloraxis_colorbar=dict(thickness=15))
    fig9.update_layout(title_text="Répartition des classes de feux par Etat", title_x=0.5)
    st.plotly_chart(fig9)
    
    st.markdown("L'ouest du pays compte le plus de feux de classes 4 à 7, et moins de feux de petite taille que le Sud. La Californie fait exception et apparait en rouge quelle que soit la classe de feux. Enfin, on visualise bien ici le « paradoxe » de l’Alaska qui ne compte que très peu de feux mais se classe pourtant loin devant en superficies brûlées : il n'apparait en rouge que pour les feux de classe 5 à 7.")
    
    st.subheader("Analyse des causes")
    
    df_by_cause = df.groupby(["fire_year", "stat_cause_regroup"])["fod_id"].count().reset_index(name="Count")

    fig10 = px.line(df_by_cause, 
                x="fire_year", 
                y="Count", 
                color = "stat_cause_regroup", 
                labels = {"fire_year" : "Année", "Count" : "Nombre d'incendies", "stat_cause_regroup" : "Causes"},
                markers=True)
    fig10.update_layout(title_text="Nombres d'incendies par causes et par année", title_x=0.5)
    st.plotly_chart(fig10)
    
    st.markdown("- Les incendies sont causés en très grande majorité par l'activité humaine directe et indirecte (Human et Infrastructure) jsuqu'en 2005 ;\n"
                "- Hausse inexpliquée des feux d’origine inconnue à partir de cette année (changement dans la méthode de classification, ou de méthodologie ?) ;\n"
                "- Les incendies de cause naturelle sont peu nombreux et leur nombre reste stable.")
    
    #st.markdown("Observons à présent la répartition des causes par ha brûlés.")
    
    df_by_cause_burn = df.groupby(["fire_year", "stat_cause_regroup"])["fire_size"].sum().reset_index(name="sum")

    fig11 = px.line(df_by_cause_burn, 
                x="fire_year", 
                y="sum", 
                color = "stat_cause_regroup", 
                labels = {"fire_year" : "Année", "sum" : "Ha brûlés", "stat_cause_regroup" : "Causes"},
                markers=True)
    fig11.update_layout(title_text="Ha brûlés par causes et par année", title_x=0.5)
    st.plotly_chart(fig11)
    
    st.markdown("Les incendies causés par des causes naturelles sont beaucoup plus dévastateurs. On peut supposer que l'origine étant non humaine, les feux peuvent apparaître dans des zones moins peuplées. Ils sont donc détectés plus tardivement et peu de moyens sont présents sur place pour les maîtriser.")
    
    st.markdown("Vérifions cette dernière hypothèse :")
    # Distance de la caserne de pompiers la plus proche
    st.markdown("Cause et distance des casernes de pompiers")
    df_cause_distancefirestation = df.groupby(['stat_cause_regroup']).agg({'closest_firestation_km': ['mean', 'median']})
    df_cause_distancefirestation.columns = df_cause_distancefirestation.columns.droplevel()
    df_cause_distancefirestation = df_cause_distancefirestation.reset_index(level=0)
    df_cause_distancefirestation
    
    st.markdown("Cause et densité de population")
    df_cause_pop = df.groupby(['stat_cause_regroup']).agg({'state_density': ['mean', 'median']})
    df_cause_pop.columns = df_cause_pop.columns.droplevel()
    df_cause_pop = df_cause_pop.reset_index(level=0)
    df_cause_pop
    
    st.markdown("Cette hypothèse semble se vérifier, la distance moyenne des casernes de pompiers étant environ 3 fois plus importante pour les départs de feux d’origine naturelle que pour les autres catégories, et la densité de population moyenne 1.5 à 2 fois plus faible.")
    
    #st.markdown("Pour terminer sur l'analyse des causes et de l'impact de la distance des casernes, nous calculons la distance moyenne de la caserne le plus proche pour chaque classe de feux (les classes A à G ont ici été numérotées de 1 à 7) :")
    #df_classe_distancefirestation = df.groupby(['fire_size_class']).agg({'closest_firestation_km': ['mean', 'min', 'max']})
    #df_classe_distancefirestation.columns = df_classe_distancefirestation.columns.droplevel()
    #df_classe_distancefirestation = df_classe_distancefirestation.reset_index(level=0)
    
    #st.markdown("La distance moyenne augmente avec la classe du feu. Statistiquement, nous avons un coefficient de corrélation de 91.6% et une p-value de 0.0037, ces deux variables sont donc fortement corrélées positivement.")
    
    st.subheader("Analyse de l'impact des température sur le nombre de départ de feux et de leur taille")   
    
    st.markdown("Pour terminer, nous allons faire une rapide analyse de l'impact des températures sur le nombre et sur la taille des incendies")
    
    df['daily_avg_temp_rounded'] = df['daily_avg_temp'].round()
    df['daily_max_temp_rounded'] = df['daily_max_temp'].round()
    df = df[(df['daily_avg_temp'] != 0) & (df['daily_min_temp'] != 0) & (df['daily_max_temp'] != 0)]
    df_full = df[df['daily_avg_temp'].notnull()]

    df_temp = df_full[['daily_avg_temp_rounded', 'fire_size']].groupby(by='daily_avg_temp_rounded').agg(['sum', 'mean', 'count']).add_prefix('fire_size_')
    df_temp.columns = df_temp.columns.droplevel()
    df_temp = df_temp.reset_index(level=0)

    x = df_temp['daily_avg_temp_rounded']
    y = df_temp['fire_size_mean']

    fig12, ax = plt.subplots(figsize=(20,15))

    # Graphique des ha brulés moyens selon la température moyenne de la journée
    plt.subplot(222)
    plt.plot('daily_avg_temp_rounded', 'fire_size_mean', data = df_temp)
    plt.xlabel("Température moyenne")
    plt.ylabel("Moyenne ha brulés")
    plt.title("Hectares brûlés moyens selon la température moyenne de la journée", loc='center');
    # Trendline calculation
    x = df_temp['daily_avg_temp_rounded']
    y = df_temp['fire_size_mean']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
    plt.text(10,120, 'p-value : ' + str(round(pearsonr(df_temp['daily_avg_temp_rounded'],df_temp['fire_size_mean'])[1],6)), color='red')

    # Graphique des nombres d'incendies selon la température moyenne de la journée
    plt.subplot(221)
    plt.plot('daily_avg_temp_rounded', 'fire_size_count', data = df_temp)
    plt.xlabel("Température moyenne")
    plt.ylabel("Nombre d'incendies")
    plt.title("Nombre de départs d'incendies selon la température moyenne de la journée", loc='center');
    # Trendline calculation
    x = df_temp['daily_avg_temp_rounded']
    y = df_temp['fire_size_count']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
    plt.text(-20,16000, 'p-value : ' + str(round(pearsonr(df_temp['daily_avg_temp_rounded'],df_temp['fire_size_count'])[1],6)), color='red')


    # Relation entre les hectares brulés et la température maximale
    plt.subplot(224)

    df_temp_max = df_full[['daily_max_temp_rounded', 'fire_size']].groupby(by='daily_max_temp_rounded').agg(['sum', 'mean', 'count']).add_prefix('fire_size_')
    df_temp_max.columns = df_temp_max.columns.droplevel()
    df_temp_max = df_temp_max.reset_index(level=0)
    x = df_temp_max['daily_max_temp_rounded']
    y = df_temp_max['fire_size_sum']

    plt.plot('daily_max_temp_rounded', 'fire_size_sum', data = df_temp_max)
    plt.xlabel("Température maximale")
    plt.ylabel("Total ha brulés (en millions)")
    plt.title("Hectares brûlés selon la température maximale de la journée", loc='center')
    # Tendance et p-value
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
    plt.text(-15,500000, 'p-value : ' + str(round(pearsonr(df_temp_max['daily_max_temp_rounded'],df_temp_max['fire_size_sum'])[1],6)), color='red')


    # Graphique des nombres d'incendies selon la température moyenne de la journée
    plt.subplot(223)

    df_temp_max_count = df_full[['daily_max_temp_rounded', 'fod_id']].groupby(by='daily_max_temp_rounded').agg(['count']).add_prefix('fod_id_')
    df_temp_max_count.columns = df_temp_max_count.columns.droplevel()
    df_temp_max_count = df_temp_max_count.reset_index(level=0)
    x = df_temp_max_count['daily_max_temp_rounded']
    y = df_temp_max_count['fod_id_count']

    plt.plot('daily_max_temp_rounded', 'fod_id_count', data = df_temp_max_count)
    plt.xlabel("Température journalière maximale")
    plt.ylabel("Nombre d'incendies")
    plt.title("Nombre de départs d'incendies selon la température maximale de la journée", loc='center');
    # Trendline calculation
    x = df_temp_max_count['daily_max_temp_rounded']
    y = df_temp_max_count['fod_id_count']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
    plt.text(-25,11000, 'p-value : ' + str(round(pearsonr(df_temp_max_count['daily_max_temp_rounded'],df_temp_max_count['fod_id_count'])[1],6)), color='red')

    st.pyplot(fig12)
    
    st.markdown("Au regard de ces graphiques, on peut affirmer que la température joue un rôle majeur dans le déclenchement des incendies et dans la taille des feux. Plus la température augmente, plus il y a d'incendies, et plus ces derniers sont importants. Le nombre d’incendies semble décroitre pour les température moyenne supérieures à 28 degrés. Ceci est explicable par la très rare occurrence de telles températures moyennes sur la période étudiée.")
   
if page==pages[3]:
   
   st.write('# Modélisation - Machine Learning')
    
   st.write('## Première étape exploratoire')
   st.write("Préparation du jeu de données:")
   st.image("scikitschema2.jpg")
   #st.write("### Premiers essais en régression et classification")
   #st.markdown("Pour rappel, nous avons le choix entre un traitement de Machine Learning de type **Régression** avec comme variable cible la taille des feux :« fire_size » ou de **Classification** avec comme variable cible la catégorie de taille des feux « fire_size_class »")
   #st.write("Dans un premier temps, des essais rapides ont été effectués :   \n "+"- en **régression** et **classification**  \n "+"-  avec **des algorithmes « simples »** (non optimisés) : régression linéaire, les arbres de décision,...  et   \n "+"- avec des **algorithmes d’optimisation** : SGD (Stochastic Gradient Descent), XGBoost, RandomForest,… ")
   st.write("Essais en régression et classification:")
   
   st.image("regression.jpg")
   st.write(" ")
   st.image("classification.jpg")

       
   st.write("- Nous poursuivons donc avec la **classification**.")
   
   st.write("- La prédiction s'effectue donc sur **les catégories de taille de feux**, numérotées de 1 à 7.")
   st.write("- Essais approfondis effectués avec les algorithmes d'arbres de décision, de SGD, de XGBoost et de RandomForest:")
   
   st.image("DecisionTree1a.jpg") 
      
   # st.write("#### Observations")
   st.write("-  Pour l'**arbre de décision**, le **'test score'** est moyen, nous observons un fort surapprentissage avec le **'train score'**. Le temps de traitement est cependant réduit par rapport aux autres algorithmes (266 s pour le SGD).")
   st.write("-   Concernant le **rapport de classification**, le constat est le même pour l'arbre de décision et pour tous les modèles testés: seules **les deux premières classes ont de bons résultats**, et les cinq autres classes sont très mal prédites. Cela vient du fort **déséquilibre du jeu de données**, déjà présenté lors de l'exploration. Il faut donc procéder à un rééchantillonnage, et bien tenir compte des métriques pour chaque classe.")

   st.write('-   Les modèles donnant un meilleur score pour le jeu test : **SGD, XGBoost** et **RandomForest**, qui correspondent tous à des algorithmes d’optimisation.')
  
   st.write('## Seconde étape: optimisation')
   
   st.markdown('### Ajout de données')
   st.write("Les faibles résultats nous ont amené à enrichir notre dataset de données complémentaires à ce stade.")
   col1, col2=st.columns(2)
   col1.write("La [couverture forestière](https://en.wikipedia.org/wiki/Forest_cover_by_state_and_territory_in_the_United_States) par état")
   col1.image("forest.jpg")
       
   col2.write("Le [nombre de casernes](https://hifld-geoplatform.opendata.arcgis.com/datasets/0ccaf0c53b794eb8ac3d3de6afdb3286_0/explore?location=40.553343%2C-120.631622%2C4.32) par état")
   col2.image("fire_station.jpg")
   
   st.markdown("###  Rééchantillonnage avec imblearn")
   st.write("- Via un **undersampling** car le jeu de données est très volumineux")
   st.write("- Voici les résultats des 2 modèles **les plus performants**:")
   st.image("RF2ag.jpg")
   with st.expander("matrice de confusion"):
       st.image("RF2amat.jpg")
   st.image("XGBoost2ag.jpg")
   with st.expander("matrice de confusion"):
       st.image("XGBoost2amat.jpg")
   
   st.write("-  Après rééchantillonnage, le **score moyen** diminue fortement.")
   st.write("-  **Rapport de classification** : Les deux premières classes présentent toujours les meilleurs résultats, mais le **rappel** des catégories minoritaires est cependant amélioré.")
   st.write("-  Les résultats sont similaires, mais XGBoost étant plus **rapide**, nous poursuivons avec ce modèle.")
   st.write("-  Difficile d'obtenir une bonne précision ET un bon rappel. Nous faisons le choix de cibler le **meilleur rappel** pour les catégories de feux de grandes tailles. En pratique, cela permettrait de détecter les conditions les plus favorables à un feu de taille critique, et ainsi de déployer un protocole de gestion du feu adapté.")
   
   st.write('### Optimisation des hyperparamètres de XGBoost')
      
   st.image("XGBoost3opta2.jpg")
   
   st.write("-  Très légère amélioriation du score du jeu de test")
   st.write("-  En revanche, forte **réduction du surapprentissage**")
   
   #st.write("###  Axe d'amélioriation: Réduction du nombre de catégories")
   st.write("- Constat à ce stade : Les résultats sont bons pour les feux de classes 1, 2 et 7 mais demeurent médiocres pour les classes 3 à 6.")
   st.write("Si le but d’une prédiction des classes de feux est d’identifier les feux de grandes tailles afin de déployer un dispositif approprié, il n’est pas nécessaire de détecter 7 catégories de feux. Nous proposons de **réduire le nombre de catégories à 3**.")
             
   st.write("## Axe d'amélioriation: Modélisation sur 3 catégories")  
   
   st.write("Les catégories de taille de feux à prédire ont été redéfinies de la manière suivante :")
   col1, col2, col3 = st.columns(3)
   with col1:
       st.write("##### Catégorie 1 : < à 1 acre (0.4 ha), soit la moitié de la surface d’un terrain de football.")
       st.image("foot.jpg")
   with col2:
       st.write("##### Catégorie 2 : 1-20 acres (0.4 - 8 ha), soit la Place de la Concorde")
       st.image("concorde.jpg")
   with col3:
       st.write("##### Catégorie 3 : > à 20 acres  ( > 8 ha)")
       st.image("image_feu2.jfif")
   
   st.write("Ce découpage est arbitraire. Celui-ci reste à être adapté selon l'objectif du modèle ou selon les protocoles spécifiques à la gestion des feux.")
   st.write("Le déséquilibre des catégories est réduit, mais un rééchantillonage reste nécessaire.")
   st.write("Mêmes observations qu'avec 7 classes : résultats similaires entre XGBoost et d'autres algorithmes d'optimisation. **XGBoost** demeure plus rapide, nous le conservons donc pour l'optimisation des hyperparamètres:")
   
   st.image("cat3xgba.jpg")
   st.write("- Le score moyen a été nettement amélioré, passant de 0.39 à 0.57.")
   st.write("- Pour la catégorie minoritaire, la précision reste faible mais le rappel est correct avec une valeur de 0.57.")
   st.write("- D'autres découpages de catégorie ont été testés, par exemple avec une troisième catégorie au delà de 100 ha, et présentent des résultats proches.")
   st.write ('##### Interprétation')
   st.write("Le modèle XGBoost est très performant mais est difficile à interpréter, souvent considéré comme une boîte noire. Il est cependant intéressant de connaître la façon dont l’algorithme est parvenu à ces conclusions. Nous allons utiliser le **package Shap** pour tenter de comprendre comment le modèle a fonctionné et quelles variables ont eu le plus de poids.")
   
   st.write("######  les 30 variables les plus importantes pour le modèle XGBoost en fonction des valeurs de Shap :")
   
   st.image("shap3cata.jpg")

   st.write("-  Dans l’ensemble, les données ayant le plus de poids correspondent à celles qui ont été **importées et rajoutées au dataset.**")
   
elif page == pages[4] :

    st.title("Conclusion")
    
    st.subheader("Pour améliorer le modèle :")
    
    st.markdown ("- **Ajustement des données importées**\n"
                 "- **Ajout de données supplémentaires** (type de végétation par territoire)\n"
                 "- **Nécessité d'un savoir métier plus important sur le sujet**")
                 
    st.write("\n")
    st.write("\n")
    
    st.image("Deerfire.jpg")
   
   

   
