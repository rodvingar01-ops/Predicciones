import streamlit as st
import pandas as pd
from scipy.stats import poisson

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Motor Predictivo PRO", page_icon="🤖", layout="wide")
st.title("🤖 Motor Predictivo de Apuestas PRO")
st.markdown("---")

# --- CARGA DE DATOS OPTIMIZADA ---
@st.cache_data
def cargar_datos():
    # Intenta buscar el archivo ZIP (compatible con Linux/GitHub y Windows)
    for nombre_zip in ['matches.zip', 'Matches.zip']:
        try:
            return pd.read_csv(nombre_zip, low_memory=False)
        except FileNotFoundError:
            continue
            
    # Respaldo por si se ejecuta localmente con el CSV suelto
    for nombre_csv in ['archive/matches.csv', 'Matches.csv', 'matches.csv']:
        try:
            return pd.read_csv(nombre_csv, low_memory=False)
        except FileNotFoundError:
            continue
            
    raise FileNotFoundError("No se encontró ningún archivo de datos (matches.zip o matches.csv).")

df = cargar_datos()

# Promedios globales
promedio_liga_local = df['FTHome'].mean()
promedio_liga_visita = df['FTAway'].mean()
equipos_validos = sorted(df['HomeTeam'].dropna().unique().tolist())

# --- INTERFAZ GRÁFICA (FRONTEND) ---
col1, col2 = st.columns(2)
with col1:
    equipo_local = st.selectbox("Selecciona el equipo LOCAL:", equipos_validos)
with col2:
    equipo_visita = st.selectbox("Selecciona el equipo VISITANTE:", equipos_validos, index=1)

# --- MOTOR MATEMÁTICO (BACKEND) ---
if st.button("📊 Analizar Partido", type="primary"):
    if equipo_local == equipo_visita:
        st.error("⚠️ Debes seleccionar dos equipos diferentes.")
    else:
        # 1. H2H
        h2h = df[((df['HomeTeam'] == equipo_local) & (df['AwayTeam'] == equipo_visita)) |
                 ((df['HomeTeam'] == equipo_visita) & (df['AwayTeam'] == equipo_local))]
        
        v_local, v_visita, empates = 0, 0, 0
        for _, partido in h2h.iterrows():
            if partido['HomeTeam'] == equipo_local:
                if partido['FTHome'] > partido['FTAway']: v_local += 1
                elif partido['FTHome'] < partido['FTAway']: v_visita += 1
                else: empates += 1
            else:
                if partido['FTAway'] > partido['FTHome']: v_local += 1
                elif partido['FTAway'] < partido['FTHome']: v_visita += 1
                else: empates += 1

        # 2. MOMENTUM (10 partidos)
        def obtener_forma(equipo):
            ultimos = df[(df['HomeTeam'] == equipo) | (df['AwayTeam'] == equipo)].tail(10)
            if ultimos.empty: return 0, 0, 0
            gf, gc, pts = 0, 0, 0
            for _, p in ultimos.iterrows():
                if p['HomeTeam'] == equipo:
                    gf += p['FTHome']; gc += p['FTAway']
                    if p['FTHome'] > p['FTAway']: pts += 3
                    elif p['FTHome'] == p['FTAway']: pts += 1
                else:
                    gf += p['FTAway']; gc += p['FTHome']
                    if p['FTAway'] > p['FTHome']: pts += 3
                    elif p['FTAway'] == p['FTHome']: pts += 1
            return gf/10, gc/10, pts

        gf_loc_10, gc_loc_10, pts_loc = obtener_forma(equipo_local)
        gf_vis_10, gc_vis_10, pts_vis = obtener_forma(equipo_visita)

        # 3. xG & ELO
        gf_loc_hist = df[df['HomeTeam'] == equipo_local]['FTHome'].mean()
        gc_loc_hist = df[df['HomeTeam'] == equipo_local]['FTAway'].mean()
        gf_vis_hist = df[df['AwayTeam'] == equipo_visita]['FTAway'].mean()
        gc_vis_hist = df[df['AwayTeam'] == equipo_visita]['FTHome'].mean()

        f_ataque_loc = gf_loc_hist / promedio_liga_local if promedio_liga_local > 0 else 1
        f_defensa_loc = gc_loc_hist / promedio_liga_visita if promedio_liga_visita > 0 else 1
        f_ataque_vis = gf_vis_hist / promedio_liga_visita if promedio_liga_visita > 0 else 1
        frag_defensa_vis = gc_vis_hist / promedio_liga_local if promedio_liga_local > 0 else 1

        xg_loc_hist = f_ataque_loc * frag_defensa_vis * promedio_liga_local
        xg_vis_hist = f_ataque_vis * f_defensa_loc * promedio_liga_visita

        try:
            elo_loc = df[df['HomeTeam'] == equipo_local]['HomeElo'].dropna().iloc[-1]
            elo_vis = df[df['AwayTeam'] == equipo_visita]['AwayElo'].dropna().iloc[-1]
            elo_prom = df['HomeElo'].mean()
            ajuste_elo_loc, ajuste_elo_vis = elo_loc / elo_prom, elo_vis / elo_prom
        except:
            ajuste_elo_loc, ajuste_elo_vis = 1, 1

        xg_loc = ((xg_loc_hist * ajuste_elo_loc) * 0.5) + (gf_loc_10 * 0.5) if gf_loc_10 > 0 else xg_loc_hist
        xg_vis = ((xg_vis_hist * ajuste_elo_vis) * 0.5) + (gf_vis_10 * 0.5) if gf_vis_10 > 0 else xg_vis_hist

        # 4. DIXON-COLES
        rho = -0.15
        m_prob = []
        for i in range(6):
            for j in range(6):
                p = poisson.pmf(i, xg_loc) * poisson.pmf(j, xg_vis)
                if i == 0 and j == 0: p *= max(0, 1 - rho * xg_loc * xg_vis)
                elif i == 1 and j == 0: p *= max(0, 1 + rho * xg_vis)
                elif i == 0 and j == 1: p *= max(0, 1 + rho * xg_loc)
                elif i == 1 and j == 1: p *= max(0, 1 - rho)
                m_prob.append((i, j, p))

        suma = sum(p for _, _, p in m_prob)
        p_loc, p_emp, p_vis, p_under, p_over, p_btts_s, p_btts_n = 0,0,0,0,0,0,0
        for i, j, p in m_prob:
            p_n = p / suma
            if i > j: p_loc += p_n
            elif i == j: p_emp += p_n
            else: p_vis += p_n
            if i + j < 3: p_under += p_n
            else: p_over += p_n
            if i > 0 and j > 0: p_btts_s += p_n
            else: p_btts_n += p_n

        c_loc = 1/p_loc if p_loc > 0 else 0
        c_emp = 1/p_emp if p_emp > 0 else 0
        c_vis = 1/p_vis if p_vis > 0 else 0
        c_un = 1/p_under if p_under > 0 else 0
        c_ov = 1/p_over if p_over > 0 else 0

        # 5. ESTADÍSTICAS SECUNDARIAS (Córners, Tiros, Tarjetas)
        pick_tiros, pick_corners, pick_tarjetas = "", "", ""
        try:
            t_loc = df[df['HomeTeam'] == equipo_local]['HomeTarget'].mean()
            t_vis = df[df['AwayTeam'] == equipo_visita]['AwayTarget'].mean()
            total_tiros = t_loc + t_vis
            
            corn_loc = df[df['HomeTeam'] == equipo_local]['HomeCorners'].mean()
            corn_vis = df[df['AwayTeam'] == equipo_visita]['AwayCorners'].mean()
            total_corners = corn_loc + corn_vis
            
            tarj_loc = df[df['HomeTeam'] == equipo_local]['HomeYellow'].mean() + df[df['HomeTeam'] == equipo_local]['HomeRed'].mean()
            tarj_vis = df[df['AwayTeam'] == equipo_visita]['AwayYellow'].mean() + df[df['AwayTeam'] == equipo_visita]['AwayRed'].mean()
            total_tarjetas = tarj_loc + tarj_vis

            prob_tiros = 1 - sum([poisson.pmf(k, total_tiros) for k in range(9)])
            prob_corners = 1 - sum([poisson.pmf(k, total_corners) for k in range(10)])
            prob_tarjetas = 1 - sum([poisson.pmf(k, total_tarjetas) for k in range(5)])

            pick_tiros = "Más de 8.5 Tiros a puerta" if prob_tiros > 0.5 else "Menos de 8.5 Tiros a puerta"
            pick_corners = "Más de 9.5 Córners" if prob_corners > 0.5 else "Menos de 9.5 Córners"
            pick_tarjetas = "Más de 4.5 Tarjetas" if prob_tarjetas > 0.5 else "Menos de 4.5 Tarjetas"
        except KeyError:
            pass 

        # --- DIBUJAR RESULTADOS EN PANTALLA ---
        st.subheader("📈 Análisis de Rendimiento")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Victorias H2H ({equipo_local})", v_local)
        c2.metric("Empates Históricos", empates)
        c3.metric(f"Victorias H2H ({equipo_visita})", v_visita)

        st.write("**Momentum (Últimos 10 partidos):**")
        st.info(f"**{equipo_local}**: {pts_loc}/30 pts | GF: {gf_loc_10*10:.0f} | GC: {gc_loc_10*10:.0f}")
        st.info(f"**{equipo_visita}**: {pts_vis}/30 pts | GF: {gf_vis_10*10:.0f} | GC: {gc_vis_10*10:.0f}")

        st.subheader("⚖️ Probabilidades y Cuotas (+EV)")
        res1, res2, res3 = st.columns(3)
        res1.success(f"**Gana {equipo_local}**\n\nProb: {p_loc*100:.1f}%\n\nCuota Justa: {c_loc:.2f}\n\n**APUESTA SI PAGA > {c_loc+0.05:.2f}**")
        res2.warning(f"**Empate**\n\nProb: {p_emp*100:.1f}%\n\nCuota Justa: {c_emp:.2f}\n\n**APUESTA SI PAGA > {c_emp+0.05:.2f}**")
        res3.error(f"**Gana {equipo_visita}**\n\nProb: {p_vis*100:.1f}%\n\nCuota Justa: {c_vis:.2f}\n\n**APUESTA SI PAGA > {c_vis+0.05:.2f}**")

        st.subheader("⚽ Goles Principales")
        g1, g2, g3 = st.columns(3)
        g1.metric("OVER 2.5", f"{p_over*100:.1f}%", f"Cuota > {c_ov+0.05:.2f}", delta_color="off")
        g2.metric("UNDER 2.5", f"{p_under*100:.1f}%", f"Cuota > {c_un+0.05:.2f}", delta_color="off")
        g3.metric("Ambos Anotan", "SÍ" if p_btts_s > p_btts_n else "NO", f"Prob SÍ: {p_btts_s*100:.1f}%", delta_color="off")

        if pick_tiros:
            st.subheader("🚩 Estadísticas (Bet Builder)")
            e1, e2, e3 = st.columns(3)
            e1.metric("Tiros a Puerta (Línea 8.5)", f"{total_tiros:.1f} Proyectados", f"{prob_tiros*100:.1f}% Prob Over", delta_color="off")
            e2.metric("Córners Totales (Línea 9.5)", f"{total_corners:.1f} Proyectados", f"{prob_corners*100:.1f}% Prob Over", delta_color="off")
            e3.metric("Tarjetas Totales (Línea 4.5)", f"{total_tarjetas:.1f} Proyectadas", f"{prob_tarjetas*100:.1f}% Prob Over", delta_color="off")

        # Recomendaciones
        p_1x2 = f"Gana {equipo_local}" if p_loc > p_emp and p_loc > p_vis else (f"Gana {equipo_visita}" if p_vis > p_loc and p_vis > p_emp else "Empate")
        p_goles = "+2.5 goles" if p_over > p_under else "-2.5 goles"
        p_btts = "Ambos Anotan SÍ" if p_btts_s > p_btts_n else "Ambos Anotan NO"
        p_seguro = f"{equipo_local} o Empate" if (p_loc + p_emp) > (p_vis + p_emp) else f"{equipo_visita} o Empate"

        st.markdown("---")
        st.subheader("💡 Veredicto Final del Algoritmo")
        st.markdown(f"- 🔥 **PICK ESTÁNDAR:** {p_1x2}, {p_goles} y {p_btts}")
        st.markdown(f"- 🛡️ **PICK SEGURO:** {p_seguro} y {p_goles}")
        if pick_tiros:
            st.markdown(f"- 📈 **BET BUILDER:** {pick_tiros} | {pick_corners} | {pick_tarjetas}")
