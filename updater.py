import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import time

API_KEY = os.environ.get('API_SPORTS_KEY')
HEADERS = {"x-rapidapi-host": "v3.football.api-sports.io", "x-rapidapi-key": API_KEY}

def recalcular_elo(elo_local, elo_visita, goles_local, goles_visita, k=20):
    r_local = 10 ** (elo_local / 400)
    r_visita = 10 ** (elo_visita / 400)
    e_local = r_local / (r_local + r_visita)
    e_visita = r_visita / (r_local + r_visita)
    
    s_local = 1 if goles_local > goles_visita else (0.5 if goles_local == goles_visita else 0)
    s_visita = 1 if goles_visita > goles_local else (0.5 if goles_visita == goles_local else 0)
    
    return round(elo_local + k * (s_local - e_local), 2), round(elo_visita + k * (s_visita - e_visita), 2)

def obtener_estadisticas(fixture_id):
    # Petición extra a la API para sacar los córners, tarjetas y remates de este partido específico
    url_stats = f"https://v3.football.api-sports.io/fixtures/statistics?fixture={fixture_id}"
    req = requests.get(url_stats, headers=HEADERS)
    data = req.json()
    
    stats_dict = {'HC': 0, 'AC': 0, 'HY': 0, 'AY': 0, 'HR': 0, 'AR': 0, 'HST': 0, 'AST': 0}
    
    if not data.get('response'):
        return stats_dict
        
    for team_data in data['response']:
        is_home = True # Asumimos local primero; lo validamos con el ID del equipo en producción
        # Para simplificar en este script, la API devuelve el Local en el índice 0 y Visita en el 1
        prefix = 'H' if data['response'].index(team_data) == 0 else 'A'
        
        for stat in team_data['statistics']:
            tipo = stat['type']
            valor = stat['value'] if stat['value'] is not None else 0
            
            if tipo == 'Corner Kicks': stats_dict[f'{prefix}C'] = valor
            elif tipo == 'Yellow Cards': stats_dict[f'{prefix}Y'] = valor
            elif tipo == 'Red Cards': stats_dict[f'{prefix}R'] = valor
            elif tipo == 'Shots on Goal': stats_dict[f'{prefix}ST'] = valor
            
    return stats_dict

def actualizar_base():
    if not API_KEY: return
    
    df = pd.read_csv('Matches.csv', low_memory=False)
    ultima_fecha = pd.to_datetime(df['MatchDate']).max()
    
    fecha_actual = ultima_fecha + timedelta(days=1)
    fecha_fin = datetime.today()
    
    if fecha_actual.date() > fecha_fin.date(): return
    
    nuevos_registros = []
    
    while fecha_actual.date() <= fecha_fin.date():
        fecha_str = fecha_actual.strftime('%Y-%m-%d')
        print(f"Consultando {fecha_str}...")
        
        url = f"https://v3.football.api-sports.io/fixtures?date={fecha_str}&status=FT"
        response = requests.get(url, headers=HEADERS)
        datos = response.json()
        
        for match in datos.get('response', []):
            if match['league']['id'] not in [39, 140, 135, 78, 61]: continue
                
            equipo_l = match['teams']['home']['name']
            equipo_v = match['teams']['away']['name']
            gf_l = match['goals']['home']
            gf_v = match['goals']['away']
            
            if gf_l is None or gf_v is None: continue
            res = 'H' if gf_l > gf_v else ('A' if gf_l < gf_v else 'D')
            
            try: elo_l_previo = df[df['HomeTeam'] == equipo_l]['HomeElo'].dropna().iloc[-1]
            except: elo_l_previo = 1500
            try: elo_v_previo = df[df['AwayTeam'] == equipo_v]['AwayElo'].dropna().iloc[-1]
            except: elo_v_previo = 1500

            nuevo_elo_l, nuevo_elo_v = recalcular_elo(elo_l_previo, elo_v_previo, gf_l, gf_v)
            
            # Obtener córners, tarjetas y tiros
            fixture_id = match['fixture']['id']
            stats = obtener_estadisticas(fixture_id)
            time.sleep(1) # Pequeña pausa para no saturar la API
            
            nuevo_partido = {
                'Division': match['league']['name'],
                'MatchDate': match['fixture']['date'].split('T')[0],
                'HomeTeam': equipo_l, 'AwayTeam': equipo_v,
                'FTHome': gf_l, 'FTAway': gf_v, 'FTResult': res,
                'HomeElo': nuevo_elo_l, 'AwayElo': nuevo_elo_v,
                **stats # Agrega las estadísticas al registro
            }
            nuevos_registros.append(nuevo_partido)
            
        fecha_actual += timedelta(days=1)
        
    if nuevos_registros:
        df_nuevos = pd.DataFrame(nuevos_registros)
        df_final = pd.concat([df, df_nuevos], ignore_index=True)
        # Rellenar vacíos por si acaso
        df_final.fillna(0, inplace=True)
        df_final.to_csv('Matches.csv', index=False)
        print("Base actualizada exitosamente.")

if __name__ == "__main__":
    actualizar_base()
