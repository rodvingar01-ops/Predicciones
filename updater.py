import os
import pandas as pd
import requests
from datetime import datetime, timedelta

# Lee la clave de las variables secretas de GitHub
API_KEY = os.environ.get('API_SPORTS_KEY')
HEADERS = {"x-rapidapi-host": "v3.football.api-sports.io", "x-rapidapi-key": API_KEY}

def recalcular_elo(elo_local, elo_visita, goles_local, goles_visita, k=20):
    r_local = 10 ** (elo_local / 400)
    r_visita = 10 ** (elo_visita / 400)
    e_local = r_local / (r_local + r_visita)
    e_visita = r_visita / (r_local + r_visita)
    
    s_local = 1 if goles_local > goles_visita else (0.5 if goles_local == goles_visita else 0)
    s_visita = 1 if goles_visita > goles_local else (0.5 if goles_visita == goles_local else 0)
    
    nuevo_elo_local = elo_local + k * (s_local - e_local)
    nuevo_elo_visita = elo_visita + k * (s_visita - e_visita)
    return round(nuevo_elo_local, 2), round(nuevo_elo_visita, 2)

def actualizar_base():
    if not API_KEY:
        print("Error: No se encontró la API_SPORTS_KEY en las variables de entorno.")
        return

    df = pd.read_csv('Matches.csv', low_memory=False)
    ultima_fecha = pd.to_datetime(df['MatchDate']).max()
    
    fecha_inicio = (ultima_fecha + timedelta(days=1)).strftime('%Y-%m-%d')
    fecha_fin = datetime.today().strftime('%Y-%m-%d')
    
    if fecha_inicio > fecha_fin:
        print("La base de datos ya está al día.")
        return

    url = f"https://v3.football.api-sports.io/fixtures?from={fecha_inicio}&to={fecha_fin}&status=FT"
    response = requests.get(url, headers=HEADERS)
    datos = response.json()
    
    nuevos_registros = []
    
    for match in datos.get('response', []):
        equipo_l = match['teams']['home']['name']
        equipo_v = match['teams']['away']['name']
        gf_l = match['goals']['home']
        gf_v = match['goals']['away']
        
        if gf_l is None or gf_v is None:
            continue
            
        if gf_l > gf_v: res = 'H'
        elif gf_l < gf_v: res = 'A'
        else: res = 'D'
        
        try:
            elo_l_previo = df[df['HomeTeam'] == equipo_l]['HomeElo'].dropna().iloc[-1]
        except IndexError:
            elo_l_previo = 1500
            
        try:
            elo_v_previo = df[df['AwayTeam'] == equipo_v]['AwayElo'].dropna().iloc[-1]
        except IndexError:
            elo_v_previo = 1500

        nuevo_elo_l, nuevo_elo_v = recalcular_elo(elo_l_previo, elo_v_previo, gf_l, gf_v)
        
        nuevo_partido = {
            'Division': match['league']['name'],
            'MatchDate': match['fixture']['date'].split('T')[0],
            'HomeTeam': equipo_l,
            'AwayTeam': equipo_v,
            'FTHome': gf_l,
            'FTAway': gf_v,
            'FTResult': res,
            'HomeElo': nuevo_elo_l,
            'AwayElo': nuevo_elo_v
        }
        nuevos_registros.append(nuevo_partido)
        
    if nuevos_registros:
        df_nuevos = pd.DataFrame(nuevos_registros)
        df_nuevos = df_nuevos.reindex(columns=df.columns)
        df_final = pd.concat([df, df_nuevos], ignore_index=True)
        df_final.to_csv('Matches.csv', index=False)
        print(f"Base de datos actualizada con {len(df_nuevos)} partidos.")
    else:
        print("No hay partidos finalizados nuevos en este rango.")

if __name__ == "__main__":
    actualizar_base()