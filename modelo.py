import pandas as pd
from scipy.stats import poisson

# 1. Leer el archivo masivo de partidos
try:
    df = pd.read_csv('archive/matches.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('matches.csv', low_memory=False)

# 2. Promedios globales de la base de datos
promedio_liga_local = df['FTHome'].mean()
promedio_liga_visita = df['FTAway'].mean()

def analizar_partido(local, visitante):
    print(f"\n{'='*60}")
    print(f" 🤖 MODELO CUANTITATIVO PRO: {local.upper()} vs {visitante.upper()} ")
    print(f"{'='*60}")

    # --- HISTORIAL DIRECTO (H2H) ---
    h2h = df[((df['HomeTeam'] == local) & (df['AwayTeam'] == visitante)) |
             ((df['HomeTeam'] == visitante) & (df['AwayTeam'] == local))]
    victorias_local_h2h, victorias_visita_h2h, empates_h2h = 0, 0, 0
    for _, partido in h2h.iterrows():
        if partido['HomeTeam'] == local:
            if partido['FTHome'] > partido['FTAway']: victorias_local_h2h += 1
            elif partido['FTHome'] < partido['FTAway']: victorias_visita_h2h += 1
            else: empates_h2h += 1
        else:
            if partido['FTAway'] > partido['FTHome']: victorias_local_h2h += 1
            elif partido['FTAway'] < partido['FTHome']: victorias_visita_h2h += 1
            else: empates_h2h += 1

    print("--- 1. HISTORIAL DIRECTO (H2H) ---")
    if len(h2h) > 0:
        print(f"Victorias {local}: {victorias_local_h2h} | Empates: {empates_h2h} | Victorias {visitante}: {victorias_visita_h2h}\n")
    else:
        print("No hay registros previos entre estos equipos en la base de datos.\n")

    # --- ESTADO DE FORMA (ÚLTIMOS 10 PARTIDOS) ---
    def obtener_forma(equipo):
        ultimos_10 = df[(df['HomeTeam'] == equipo) | (df['AwayTeam'] == equipo)].tail(10)
        if ultimos_10.empty: return 0, 0, 0
        anotados, recibidos, puntos = 0, 0, 0
        for _, partido in ultimos_10.iterrows():
            if partido['HomeTeam'] == equipo:
                anotados += partido['FTHome']
                recibidos += partido['FTAway']
                if partido['FTHome'] > partido['FTAway']: puntos += 3
                elif partido['FTHome'] == partido['FTAway']: puntos += 1
            else:
                anotados += partido['FTAway']
                recibidos += partido['FTHome']
                if partido['FTAway'] > partido['FTHome']: puntos += 3
                elif partido['FTAway'] == partido['FTHome']: puntos += 1
        return anotados / 10, recibidos / 10, puntos

    gf_reciente_local, gc_reciente_local, pts_local = obtener_forma(local)
    gf_reciente_visita, gc_reciente_visita, pts_visita = obtener_forma(visitante)

    print("--- 2. MOMENTUM (ÚLTIMOS 10 PARTIDOS) ---")
    print(f"{local:<12} -> {pts_local}/30 pts | GF: {gf_reciente_local*10:.0f} | GC: {gc_reciente_local*10:.0f}")
    print(f"{visitante:<12} -> {pts_visita}/30 pts | GF: {gf_reciente_visita*10:.0f} | GC: {gc_reciente_visita*10:.0f}\n")

    # --- CÁLCULO DE GOLES BASE (xG) ---
    goles_favor_local = df[df['HomeTeam'] == local]['FTHome'].mean()
    goles_contra_local = df[df['HomeTeam'] == local]['FTAway'].mean()
    goles_favor_visita = df[df['AwayTeam'] == visitante]['FTAway'].mean()
    goles_contra_visita = df[df['AwayTeam'] == visitante]['FTHome'].mean()

    fuerza_ataque_local = goles_favor_local / promedio_liga_local if promedio_liga_local > 0 else 1
    fuerza_defensa_local = goles_contra_local / promedio_liga_visita if promedio_liga_visita > 0 else 1
    fuerza_ataque_visita = goles_favor_visita / promedio_liga_visita if promedio_liga_visita > 0 else 1
    fragilidad_defensiva_visita = goles_contra_visita / promedio_liga_local if promedio_liga_local > 0 else 1

    xg_local_hist = fuerza_ataque_local * fragilidad_defensiva_visita * promedio_liga_local
    xg_visita_hist = fuerza_ataque_visita * fuerza_defensa_local * promedio_liga_visita

    # --- INTEGRACIÓN DE RANKING ELO ---
    try:
        elo_local = df[df['HomeTeam'] == local]['HomeElo'].dropna().iloc[-1]
        elo_visita = df[df['AwayTeam'] == visitante]['AwayElo'].dropna().iloc[-1]
        elo_promedio = df['HomeElo'].mean()
        ajuste_elo_local = elo_local / elo_promedio
        ajuste_elo_visita = elo_visita / elo_promedio
    except (KeyError, IndexError):
        ajuste_elo_local, ajuste_elo_visita = 1, 1 

    # Mezcla de xG: 50% Histórico ELO ajustado, 50% Momentum reciente
    xg_local = ((xg_local_hist * ajuste_elo_local) * 0.5) + (gf_reciente_local * 0.5) if gf_reciente_local > 0 else xg_local_hist
    xg_visita = ((xg_visita_hist * ajuste_elo_visita) * 0.5) + (gf_reciente_visita * 0.5) if gf_reciente_visita > 0 else xg_visita_hist

    # --- ALGORITMO DIXON-COLES (CORRECCIÓN DE EMPATES) ---
    rho = -0.15 
    matriz_prob = []
    
    for i in range(6):
        for j in range(6):
            prob = poisson.pmf(i, xg_local) * poisson.pmf(j, xg_visita)
            if i == 0 and j == 0: prob *= max(0, 1 - rho * xg_local * xg_visita)
            elif i == 1 and j == 0: prob *= max(0, 1 + rho * xg_visita)
            elif i == 0 and j == 1: prob *= max(0, 1 + rho * xg_local)
            elif i == 1 and j == 1: prob *= max(0, 1 - rho)
            matriz_prob.append((i, j, prob))

    suma_total = sum(p for i, j, p in matriz_prob)
    prob_local, prob_empate, prob_visita = 0, 0, 0
    prob_under, prob_over, prob_btts_si, prob_btts_no = 0, 0, 0, 0
    
    for i, j, p in matriz_prob:
        p_norm = p / suma_total
        if i > j: prob_local += p_norm
        elif i == j: prob_empate += p_norm
        else: prob_visita += p_norm
        
        if i + j < 3: prob_under += p_norm
        else: prob_over += p_norm
        
        if i > 0 and j > 0: prob_btts_si += p_norm
        else: prob_btts_no += p_norm

    cuota_local = 1 / prob_local if prob_local > 0 else 0
    cuota_empate = 1 / prob_empate if prob_empate > 0 else 0
    cuota_visita = 1 / prob_visita if prob_visita > 0 else 0
    cuota_under = 1 / prob_under if prob_under > 0 else 0
    cuota_over = 1 / prob_over if prob_over > 0 else 0
    cuota_btts_si = 1 / prob_btts_si if prob_btts_si > 0 else 0
    cuota_btts_no = 1 / prob_btts_no if prob_btts_no > 0 else 0

    print("--- 3. MERCADOS PRINCIPALES Y DETECTOR +EV ---")
    print(f"Gana {local:<12} -> Prob: {prob_local*100:.1f}% | Cuota Justa: {cuota_local:.2f} 🟢 [APUESTA SI PAGA MÁS DE: {cuota_local+0.05:.2f}]")
    print(f"Empate {'':<12} -> Prob: {prob_empate*100:.1f}% | Cuota Justa: {cuota_empate:.2f} 🟢 [APUESTA SI PAGA MÁS DE: {cuota_empate+0.05:.2f}]")
    print(f"Gana {visitante:<12} -> Prob: {prob_visita*100:.1f}% | Cuota Justa: {cuota_visita:.2f} 🟢 [APUESTA SI PAGA MÁS DE: {cuota_visita+0.05:.2f}]\n")

    print(f"OVER 2.5 GOLES    -> Prob: {prob_over*100:.1f}% | Cuota Justa: {cuota_over:.2f} 🟢 [APUESTA SI PAGA MÁS DE: {cuota_over+0.05:.2f}]")
    print(f"UNDER 2.5 GOLES   -> Prob: {prob_under*100:.1f}% | Cuota Justa: {cuota_under:.2f} 🟢 [APUESTA SI PAGA MÁS DE: {cuota_under+0.05:.2f}]")
    print(f"AMBOS ANOTAN: SÍ  -> Prob: {prob_btts_si*100:.1f}% | Cuota Justa: {cuota_btts_si:.2f}")
    print(f"AMBOS ANOTAN: NO  -> Prob: {prob_btts_no*100:.1f}% | Cuota Justa: {cuota_btts_no:.2f}\n")

    # --- CÁLCULO DE ESTADÍSTICAS ---
    pick_tiros, pick_corners, pick_tarjetas = "", "", ""
    try:
        tiros_local = df[df['HomeTeam'] == local]['HomeTarget'].mean()
        tiros_visita = df[df['AwayTeam'] == visitante]['AwayTarget'].mean()
        total_tiros = tiros_local + tiros_visita
        
        corners_local = df[df['HomeTeam'] == local]['HomeCorners'].mean()
        corners_visita = df[df['AwayTeam'] == visitante]['AwayCorners'].mean()
        total_corners = corners_local + corners_visita
        
        tarj_local = df[df['HomeTeam'] == local]['HomeYellow'].mean() + df[df['HomeTeam'] == local]['HomeRed'].mean()
        tarj_visita = df[df['AwayTeam'] == visitante]['AwayYellow'].mean() + df[df['AwayTeam'] == visitante]['AwayRed'].mean()
        total_tarjetas = tarj_local + tarj_visita

        prob_over_85_tiros = 1 - sum([poisson.pmf(k, total_tiros) for k in range(9)])
        prob_over_95_corners = 1 - sum([poisson.pmf(k, total_corners) for k in range(10)])
        prob_over_45_tarj = 1 - sum([poisson.pmf(k, total_tarjetas) for k in range(5)])

        print("--- 4. MERCADOS DE ESTADÍSTICAS (BET BUILDER) ---")
        print(f"TIROS A PUERTA (Proyectados: {total_tiros:.1f})")
        print(f"  └ Más de 8.5 Tiros  -> Prob: {prob_over_85_tiros*100:.1f}% | Cuota Justa: {1/prob_over_85_tiros if prob_over_85_tiros>0 else 0:.2f}")
        
        print(f"CÓRNERS TOTALES (Proyectados: {total_corners:.1f})")
        print(f"  └ Más de 9.5 Córners -> Prob: {prob_over_95_corners*100:.1f}% | Cuota Justa: {1/prob_over_95_corners if prob_over_95_corners>0 else 0:.2f}")
        
        print(f"TARJETAS TOTALES (Proyectadas: {total_tarjetas:.1f})")
        print(f"  └ Más de 4.5 Tarjetas -> Prob: {prob_over_45_tarj*100:.1f}% | Cuota Justa: {1/prob_over_45_tarj if prob_over_45_tarj>0 else 0:.2f}\n")
        
        pick_tiros = "Más de 8.5 Tiros" if prob_over_85_tiros > 0.5 else "Menos de 8.5 Tiros"
        pick_corners = "Más de 9.5 Córners" if prob_over_95_corners > 0.5 else "Menos de 9.5 Córners"
        pick_tarjetas = "Más de 4.5 Tarjetas" if prob_over_45_tarj > 0.5 else "Menos de 4.5 Tarjetas"
        
    except KeyError:
        pass 

    # --- LÓGICA DE RECOMENDACIÓN AUTOMÁTICA ---
    if prob_local > prob_empate and prob_local > prob_visita: pick_1x2 = f"Gana {local}"
    elif prob_visita > prob_local and prob_visita > prob_empate: pick_1x2 = f"Gana {visitante}"
    else: pick_1x2 = "Empate"

    pick_goles = "+2.5 goles" if prob_over > prob_under else "-2.5 goles"
    pick_btts = "Ambos Anotan SÍ" if prob_btts_si > prob_btts_no else "Ambos Anotan NO"

    prob_1x = prob_local + prob_empate
    prob_x2 = prob_visita + prob_empate
    pick_seguro = f"{local} o Empate" if prob_1x > prob_x2 else f"{visitante} o Empate"

    print("--- 💡 VEREDICTO FINAL DEL ALGORITMO ---")
    print(f"🔥 PICK ESTÁNDAR: {pick_1x2}, {pick_goles} y {pick_btts}")
    print(f"🛡️ PICK SEGURO:   {pick_seguro} y {pick_goles}")
    
    if pick_tiros:
        print(f"📈 BET BUILDER:   {pick_tiros} | {pick_corners} | {pick_tarjetas}\n")

# 4. Bucle Interactivo con Buscador Inteligente
print("\n" + "="*60)
print(" INICIANDO MOTOR PREDICTIVO DE APUESTAS PRO (CON ELO Y EV+) ")
print("="*60)

equipos_validos = sorted(df['HomeTeam'].dropna().unique().tolist())

def buscar_equipo(nombre_ingresado):
    nombre_limpio = nombre_ingresado.strip().lower()
    traducciones = {
        "manchester united": "man united",
        "manchester city": "man city",
        "spurs": "tottenham"
    }
    if nombre_limpio in traducciones:
        nombre_limpio = traducciones[nombre_limpio]
    
    for equipo in equipos_validos:
        if equipo.lower() == nombre_limpio:
            return equipo
            
    coincidencias = [eq for eq in equipos_validos if nombre_limpio in eq.lower()]
    if len(coincidencias) == 1:
        return coincidencias[0]
    elif len(coincidencias) > 1:
        print(f"\n[?] Hay varios equipos parecidos a '{nombre_ingresado}':")
        for eq in coincidencias:
            print(f"    - {eq}")
        return None
    else:
        print(f"\n[ERROR] No se encontró nada parecido a '{nombre_ingresado}'.")
        return None

while True:
    print("\n(Escribe 'salir' para cerrar)")
    
    input_local = input("Ingresa el equipo LOCAL: ")
    if input_local.lower().strip() == 'salir': break
    equipo_local = buscar_equipo(input_local)
    if not equipo_local: continue
        
    input_visita = input("Ingresa el equipo VISITANTE: ")
    if input_visita.lower().strip() == 'salir': break
    equipo_visita = buscar_equipo(input_visita)
    if not equipo_visita: continue

    print(f"\n>> Procesando matemática avanzada: {equipo_local} vs {equipo_visita}...")
    try:
        analizar_partido(equipo_local, equipo_visita)
    except Exception as e:
        print(f"\n[ERROR DEL SISTEMA] Asegúrate de que los equipos tengan historial suficiente. Detalle: {e}")