import os
import sys
from collections import defaultdict

from cg.api import AreaType, CardType, EnergyType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, to_observation_class

"""
Mega Lucario ex Deck
Intermediate Level
This deck battles by strategically switching between Mega Lucario ex as the main attacker, and Hariyama and Solrock as secondary attackers.
"""

# Load deck.csv in the dataset
file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(60):
    my_deck.append(int(csv[i]))

# Fetch card metadata database and create an ID-to-Card lookup table
all_card = all_card_data()
card_table = {c.cardId:c for c in all_card}

# Decklist
Makuhita = 673  # ×2
Hariyama = 674  # ×2
Lunatone = 675  # ×2
Solrock = 676  # ×3
Riolu = 677  # ×3
Mega_Lucario_ex = 678  # ×4
Dusk_Ball = 1102  # ×4
Switch = 1123  # ×2
Premium_Power_Pro = 1141  # ×4
Fighting_Gong = 1142  # ×4
Poke_Pad = 1152  # x4
Hero_Cape = 1159  # ×1
Boss_Orders = 1182  # ×2
Carmine = 1192  # ×4
Lillie_Determination = 1227  # ×4
Gravity_Mountain = 1252  # ×2
Basic_Fighting_Energy = 6  # ×13
Marnie_Impidimp = 646
Marnie_Morgrem = 647
Marnie_Grimmsnarl = 648
Alakazam_Abra = 741
Alakazam_Kadabra = 742
Alakazam_Alakazam = 743
Budew = 235


# ====== r20: gate CLASE B de r19 (E17_sw2900) portado sobre v2_combo ======
# Asimetria medida en 424 replays reales: 0 Megas cedidos wr .857 / 1 -> .514
# / 2 -> .000 (0W-54L). El Mega que cierra el turno ACTIVO y con PS bajo regala
# 3 premios. P(muere): .80 a <=100 PS, .62 a 101-200, .31 a 201-250.
# Todos los interruptores en False => byte-equivalente a v2_combo_fix.
_R19_ETIQUETA       = "E17_sw2900_sobre_combo"
_R19_B_ON           = True
_R19_B_HP           = 150
_R19_B_SWITCH       = True
_R19_B_RETREAT      = True
_R19_B_OPPRIZE_MAX  = 3
_R19_B_MYPRIZE_MIN  = 3
_R19_B_OPPRIZE_MIN  = 2
_R19_B_DIF_MAX      = 2
_R19_B_APAGA_ESPEJO = True
_R19_B_SW_SCORE     = 2900
_R19_B_RET_SOLO_DESMONTADO = True
_R19_CNT = {"gateable": 0, "disparo": 0, "pts_switch": 0, "pts_retreat": 0,
            "ejec_switch": 0, "ejec_retreat": 0}
# ==========================================================================



# ==================== XEROSIC portado desde v9x_xero =========================
# Aislamiento: este brazo es v9w EXACTO (895 lineas, sus 4 capas intactas) mas
# el mazo de v9x_xero (-1 Dusk Ball, -2 Carmine, +3 Xerosic) y SOLO el bloque de
# politica de la carta. v9x_xero era otra rama del linaje, 202 lineas mas corta
# y sin _V9V_A1/_V9M_P1/_V9V_G1/_V9V_A2, asi que su +26.7pp vs Alakazam mezclaba
# la carta con las capas. Esto separa las dos cosas.
#
# Xerosic's Machinations (1197, Supporter). CardImpl.h:3940
#   .conditionGreaterEqual(4, Enemy).targetHand()
#   .effect(ToTrash, Enemy).targetHand().select(CardUntil, 3).enemySelect()
# El motor solo la ofrece si la mano rival tiene >= 4; la deja en 3 exactas y no
# toca la nuestra. Contra Alakazam, Powerful Hand (1072) hace 20 x su mano, o
# sea que el cap queda en 60 mas lo que robe en su turno.
Xerosic = 1197
_XE_LETAL = 9000   # nos matan este turno con Powerful Hand: por encima de todo
_XE_ALTA = 3900    # gate Alakazam + swing gordo: por encima de Boss (3200)
_XE_MEDIA = 3150   # por encima de Lillie's (3100) y Carmine (3000), no de Boss
_XE_BAJA = 3050    # solo por encima de Carmine (3000)


def _xerosic_score(alkz, op_hand, my_active_hp):
    """Puntua jugar Xerosic. -1 = no la gastes."""
    if op_hand < 4:
        return -1
    swing = 20 * (op_hand - 3)
    if alkz:
        if my_active_hp is not None and 20 * op_hand >= my_active_hp:
            return _XE_LETAL
        if swing >= 60:
            return _XE_ALTA
        return _XE_MEDIA
    if swing >= 100:
        return _XE_MEDIA
    if swing >= 60:
        return _XE_BAJA
    return -1
# ============================================================================


def _grim_line_target_bonus(card_id: int) -> int:
    """Reward choking the two pre-evolutions before Grimmsnarl appears.

    A 650-point bonus puts a bare Impidimp (1720) or Morgrem (1880) above an
    energized Munkidori (1560; 1660 with a Tool) without overriding multi-Prize
    targets or every heavily invested attacker.
    """
    return 650 if card_id in (Marnie_Impidimp, Marnie_Morgrem) else 0


# (fn-evoarch) CORTE DE LINEA EVOLUTIVA, GATEADO A ARCHALUDON.
# Medido 10-ago: generalizar el bono +650 a las 457 pre-evoluciones cuesta
# -1.0/-2.6/-3.0 pp contra Grimmsnarl/espejo/Alakazam pero GANA +1.87/+1.69 pp
# contra Archaludon, porque Duraludon (169/839/992) es la unica pre-evolucion
# relevante que la lista cableada (Impidimp/Morgrem) no cubre. Aqui se cobra solo
# esa celda: mismo bono, gate de matchup al estilo _ALKZ. Fuera de la linea
# Archaludon el agente es byte-identico a x2 (verificable con pareado 0.0000).
_FN_EVOARCH = True
_ARCH_PRE_IDS = (169, 839, 992)                    # Duraludon
_ARCH_LINE_IDS = (169, 839, 992, 170, 190, 840)    # + Archaludon / Archaludon ex
_fn_arch_line = False


class AttackPlan:
    attacker = -1
    target = -1
    attack_index = -1
    remain_hp = -1
    energy = False
    ppp_need = 0


plan = AttackPlan()
pre_turn = 0
ability_used = False
ppp_spent = 0

# (eval-q) canal de salida del ranking heuristico + interruptor de modo
# busqueda: impide que las llamadas dentro del arbol muten el estado del turno
# real. Con search_mode=False el comportamiento es identico a la BASE.
_EXPORT = {"scores": None, "order": None, "search_mode": False}

# (v9m) ESPEJO Lucario: estado del gate para pokemon_score (se fija en cada
# llamada a agent(); False fuera del espejo => byte-por-byte v9a).
_v9m_espejo = False
_v9m_turno = 0
_V9M_RIOLU_BONO = 2000  # autopsia: KO Riolu (~1100+bono) debe superar chip al Mega (~2800)
# Interruptores de ablacion (True = parche activo); fuera del espejo todos son no-op.
# Ablacion pareada 80 semillas vs v9a (2026-08-06, out_espejo/ABL_*.json):
#   P1 solo:  pixiux +3.8 (z=1.28), sample +6.3 (z=2.18), aristo -3.8 (z=-1.51) — SOBREVIVE
#   P2 solo:  pixiux -1.9 (z=-1.75) — REGRESA, apagado
#   P3 solo:  0.0000 exacto en las 3 celdas espejo — NEUTRO inerte, apagado
#   Variantes P1 probadas y descartadas: bono 800/1400 (misma sangria aristo,
#   menos ganancia), corte a cero tras t6 (aristo -5.6, z=-2.24), bono solo-KO
#   (diluye sample a +3.8). Gana bono 2000 con mitad tras t6.
_V9M_P1 = True   # bono Riolu rival en pokemon_score
_V9M_P2 = False  # disciplina de Cape — regreso vs pixiux, apagado
_V9M_P3 = False  # chip de PPP al Mega rival — neutro exacto, apagado

# (v9v) VECINOS: 4 parches de la autopsia r6-vecinos (160 trazas sembradas,
# err=0). Cada uno con interruptor propio para ablacion pareada; todos en
# try/except con caida al comportamiento v9m original. Con los 4 en False el
# agente es byte-equivalente a v9m_espejo.
_V9V_G1 = False  # (v9w) APAGADO: medido pareado -0.6/-1.9/-1.9pp en sus 3 celdas
_V9V_G2 = False  # objetivo. El gradiente 0.217->0.921 de la autopsia era correlacion
                 # (consecuencia de partidas que ya iban bien), no palanca causal.
_V9V_A1 = True   # guard de exposicion del Mega en _ALKZ (53/72 muertes fueron eleccion propia)
_V9V_A2 = True   # caza Abra/Kadabra tambien en el ACTIVO (montaje t<=4 -> WR 0.278; caza t<=4 -> 0.615)
_V9V_IMP_BONO = 2000  # bono total al Impidimp pre-montaje (patron _V9M_RIOLU_BONO)
# (r14-rapid) La muestra activa nueva corrige la lectura inicial: Hariyama hizo
# 0.70 Wild Press por VICTORIA vs Marnie y la primera linea es productiva. Se
# conserva esa linea y se bloquea solo el SEGUNDO Makuhita redundante. Asi no se
# pagan dos objetivos blandos a Shadow Bullet/Munkidori sin quitar el atacante
# de un premio que aparece en las victorias actuales.
_R14_MARNIE_DUPLICATE_MAKUHITA_GUARD = False

# (r16-rapid) Salida de Lunatone activo bajo Itchy Pollen. En las batallas activas, los juegos
# perdidos acumulan mas del doble de dano sobre atacantes baratos (360 vs 174),
# y una derrota contra Dragapult dejo dos Mega Lucario cargados en banca hasta
# el turno 13 porque Budew bloqueaba Switch. Una energia en Lunatone paga su
# retirada de 1 y convierte una banca inaccesible en ataque al turno siguiente.
# El gate exacto de Budew evita la regresion observada con la version general.
_R16_BUDEW_ESCAPE = True
# Estado del gate pre-montaje para pokemon_score (se fija en cada agent();
# False fuera de la linea Marnie o con 648 ya en mesa).
_v9v_grim_pre = False


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    """Helper function to safely extract a Card or Pokemon object from specific zones."""
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def prize_count(pokemon: Pokemon) -> int:
    """Calculates how many Prize cards a Pokémon yields upon being Knocked Out, factoring in modifiers."""
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == 12:  # Legacy Energy
            count -= 1
    for card in pokemon.tools:
        if card.id == 1172 and "Lillie" in data.name:  # Lillie’s Pearl
            count -= 1
    return max(0, count)


def pokemon_score(pokemon: Pokemon) -> int:
    """Heuristically evaluates the tactical worth of targeting a specific Pokémon on the opponent's field."""
    data = card_table[pokemon.id]
    # (r11-19) peso de premios 1000 -> 2000. Curva dosis-respuesta monotona de 7
    # puntos y confirmacion en tramo disjunto: +1,59 pp (z=+4,87) exploracion y
    # +1,64 pp (z=+5,03) confirmacion sobre espejo/aristo/pixiux, n~1.600/brazo.
    # Panel de 8 celdas no saturadas, campo ponderado por censo +0,81 pp.
    score = prize_count(pokemon) * 2000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    
    id = pokemon.id
    score += _grim_line_target_bonus(id)
    try:
        if _FN_EVOARCH and _fn_arch_line and id in _ARCH_PRE_IDS:
            score += 650
    except Exception:
        pass
    # Squawkabilly ex, Noctowl, Fan Rotom, Archaludon ex
    if id == 144 or id == 322 or id == 323 or id == 337:
        score -= 200
    if id == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300
    score += pokemon.hp
    # (v9m-1) ESPEJO: la carrera de Riolus decide el matchup (WR 0.389 si
    # perdemos nuestro Riolu antes, 0.810 si cae el suyo; n=80 sembradas).
    # Bono para que el KO al Riolu rival supere el chip al Mega (~2800).
    # Decae a la mitad tras t6: en victorias nuestros drags al Mega son
    # mayormente t>=6 (21/29), asi el bono no colisiona con el remate.
    try:
        if _V9M_P1 and _v9m_espejo and id == Riolu:
            score += _V9M_RIOLU_BONO if _v9m_turno <= 6 else _V9M_RIOLU_BONO // 2
    except Exception:
        pass
    # (v9v-g1) CAZA IMPIDIMP PRE-MONTAJE: WR 0.217 con 0 Imps cazados vs 0.921
    # con >=2. El +650 plano de _grim_line_target_bonus se diluye: en t2-3 sin
    # KO limpio el score fraccional (x damage/hp) lo entierra y el plan apunta
    # al activo. Pre-montaje se escala el bono TOTAL del Impidimp a ~2000
    # (patron del bono espejo _V9M_RIOLU_BONO). 54/80 montajes SALTAN Morgrem
    # via Rare Candy: el blanco que importa es IMPIDIMP en t2-4. El gate se
    # apaga solo cuando aparece un 648 montado en mesa.
    try:
        if _V9V_G1 and _v9v_grim_pre and id == Marnie_Impidimp:
            score += _V9V_IMP_BONO - 650  # 650 ya sumados arriba -> total 2000
    except Exception:
        pass
    return score


def _heuristic_agent(obs_dict: dict) -> list[int]:
    """Main Agent Function.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.
    
    Returns:
        list[int]: A list of option index.
    """
    obs = to_observation_class(obs_dict)
    if obs.select == None:
        # In the initial selection, the obs.select is None, and it is necessary to return the deck.
        # The deck is a list of 60 card IDs.
        # The deck must comply with the Pokémon Trading Card Game rules.
        return my_deck
        
    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    my_prize = len(my_state.prize)

    # (v9a) MODO ALAKAZAM portado de al1 (r9-al1), politica pura, cero cambios
    # de mazo. Motivo: el barrido Xerosic demostro que CUALQUIER hueco de mazo
    # anti-Alakazam se paga -2/-3pp en banda 1000+ (dilucion; xe2a lo confirmo
    # con gate solo-Alakazam). Esta es la via sin dilucion. Gate: linea
    # 741/742/743 rival en mesa o descarte.
    try:
        _ALKZ = (any(c is not None and c.id in (741, 742, 743)
                     for c in list(op_state.active) + list(op_state.bench))
                 or any(c.id in (741, 742, 743) for c in op_state.discard))
    except Exception:
        _ALKZ = False

    # (fn-evoarch) gate de matchup Archaludon: linea rival en mesa o descarte.
    global _fn_arch_line
    try:
        _fn_arch_line = (any(c is not None and c.id in _ARCH_LINE_IDS
                             for c in list(op_state.active) + list(op_state.bench))
                         or any(c.id in _ARCH_LINE_IDS for c in op_state.discard))
    except Exception:
        _fn_arch_line = False

    # (xero) PS restantes de nuestro activo: lo usa _xerosic_score para saber si
    # Powerful Hand nos remata este turno con la mano que el rival YA tiene.
    try:
        _a = my_state.active[0] if my_state.active else None
        _my_active_hp = _a.hp if _a is not None else None
    except Exception:
        _my_active_hp = None

    # (v9m) GATE ESPEJO: linea Riolu/Mega Lucario ex (677/678) rival en mesa o
    # descarte — mismo patron que _ALKZ. Fuera del espejo todo parche v9m es
    # no-op y el agente es identico a v9a (verificable con ceros pareados).
    global _v9m_espejo
    global _v9m_turno
    try:
        _ESPEJO = (any(c is not None and c.id in (Riolu, Mega_Lucario_ex)
                       for c in list(op_state.active) + list(op_state.bench))
                   or any(c.id in (Riolu, Mega_Lucario_ex) for c in op_state.discard))
    except Exception:
        _ESPEJO = False
    _v9m_espejo = _ESPEJO
    _v9m_turno = state.turn

    # (v9v) GATE MARNIE: linea Impidimp/Morgrem/Grimmsnarl (646/647/648) rival
    # en mesa o descarte — mismo patron que _ALKZ/_ESPEJO. _GRIM_PRE ademas
    # exige que NO haya ningun Grimmsnarl (648) YA MONTADO en mesa: es la
    # ventana de caza que importa (t2-4); montado el 648, los parches G1 se
    # apagan solos y queda el comportamiento v9m.
    global _v9v_grim_pre
    try:
        _op_mesa = [c for c in list(op_state.active) + list(op_state.bench)
                    if c is not None]
        _MARN = (any(c.id in (Marnie_Impidimp, Marnie_Morgrem, Marnie_Grimmsnarl)
                     for c in _op_mesa)
                 or any(c.id in (Marnie_Impidimp, Marnie_Morgrem, Marnie_Grimmsnarl)
                        for c in op_state.discard))
        _GRIM_PRE = _MARN and not any(c.id == Marnie_Grimmsnarl for c in _op_mesa)
    except Exception:
        _MARN = False
        _GRIM_PRE = False
    _v9v_grim_pre = _GRIM_PRE

    # (v9v-a2) Alakazam (743) ya montado en mesa rival: con el montado, cazar
    # Abra/Kadabra activos ya no rompe el motor de robo. Fallo cerrado a True
    # (= sin bono) para que el except nunca active el parche.
    try:
        _ALAK_MONTADO = any(c is not None and c.id == Alakazam_Alakazam
                            for c in list(op_state.active) + list(op_state.bench))
    except Exception:
        _ALAK_MONTADO = True

    if context == SelectContext.IS_FIRST:
        # (v5-f) IR PRIMERO. Ver gen.py para la justificacion completa y la
        # atribucion (aristophanivan/improved-probabilistic-agent, LB 959,4,
        # mismo mazo que el nuestro, contesta YES).
        # SetupProc.h:249  SetYesNoSelect(state, IsFirst, 0)  -> solo pregunta
        # al jugador 0; SetupProc.h:227-233  YES => firstPlayer = selectPlayer.
        for i, o in enumerate(select.option):
            if o.type == OptionType.YES:
                return [i]

    global plan
    global pre_turn
    global ability_used
    global ppp_spent
    if pre_turn != state.turn:
        pre_turn = state.turn
        plan = AttackPlan()
        ability_used = False
        ppp_spent = 0
            
    field_counts = defaultdict(int)  # Number of cards per card ID on the Bench and in the Active Spot
    hand_counts = defaultdict(int)  # Number of cards per card ID in hand
    discard_counts = defaultdict(int)  # Number of cards per card ID in discard pile

    attacker1 = False
    attacker2 = False
    for card in my_state.active + my_state.bench:
        if card == None:
            continue
        field_counts[card.id] += 1
        if card.id == Makuhita or card.id == Hariyama:
            if len(card.energies) >= 3:
                attacker2 = True
        elif card.id == Riolu or card.id == Mega_Lucario_ex:
            if len(card.energies) >= 2:
                attacker1 = True

    for card in my_state.hand:
        hand_counts[card.id] += 1

    for card in my_state.discard:
        discard_counts[card.id] += 1

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id
            
    can_attack = False
    if context == SelectContext.MAIN:
        can_switch = False
        can_op_switch = False
        can_use_mega_brave = False
        for o in select.option:
            if o.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card.id == Switch:
                    can_switch = True
                elif card.id == Boss_Orders:
                    can_op_switch = True
            elif o.type == OptionType.EVOLVE:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card.id == Hariyama:
                    can_op_switch = True
            elif o.type == OptionType.RETREAT:
                can_switch = True
            elif o.type == OptionType.ATTACK:
                can_attack = True
                if o.attackId == 983:  # Mega Brave
                    can_use_mega_brave = True
        
        my_cards = [my_state.active[0]]
        for pokemon in my_state.bench:
            my_cards.append(pokemon)
        op_cards = [op_state.active[0]]
        for pokemon in op_state.bench:
            op_cards.append(pokemon)

        # (R17-asiento) ERA 'state.turn >= 2'. Ese umbral saltaba el constructor
        # del plan EXACTAMENTE en el turno 1 del motor, que es el turno de
        # apertura del jugador que va PRIMERO: yendo primeros jugabamos nuestras
        # ~7,1 decisiones MAIN de apertura sin plan (energia sin destino, y
        # Switch/PPP/Boss/retirada vetados por plan.attacker == -1), mientras que
        # yendo segundos nuestro turno de apertura es el 2 y el plan si se
        # construia. Medido 13-ago: +0,387 pp de campo (z +5,31) en dos tramos
        # disjuntos de 1.500 semillas. Control de signo (umbral 3, que quita el
        # plan tambien al segundo): -1,32 pp.
        if state.turn >= 1:
            best_score = -1
            for i, my_pokemon in enumerate(my_cards):
                if i != 0 and not can_switch:
                    break
                for a in range(2):
                    energy_required = 0
                    base_damage = 0
                    base_score = 0
                    if my_pokemon.id == Mega_Lucario_ex:
                        if a == 0:
                            energy_required = 1
                            base_damage = 130
                            base_score += 60 * min(3, discard_counts[Basic_Fighting_Energy])
                        else:
                            energy_required = 2
                            base_damage = 270
                        if my_prize == 2 or my_prize == 3:
                            base_score -= 500
                        elif _ALKZ and 20 * op_state.handCount >= my_pokemon.hp:
                            # (v9a-a) guarda de exposicion: si el Powerful Hand
                            # potencial (20 x mano rival; contadores, ignoran
                            # W/R y Cape) ya alcanza el HP restante de este
                            # Lucario, no lo planifiques como atacante.
                            base_score -= 500
                        else:
                            # (v9v-a1) lookahead del guard: la mano rival crece
                            # ~+2.5/turno con Psychic Draw, asi que el guard
                            # v9a-a llega un turno tarde. Dispara ya con
                            # 20*(mano+4) >= hp restante del Mega.
                            try:
                                if (_V9V_A1 and _ALKZ
                                        and 20 * (op_state.handCount + 4)
                                        >= my_pokemon.hp):
                                    base_score -= 500
                            except Exception:
                                pass
                    elif a == 1:
                        break
                    elif my_pokemon.id == Hariyama:
                        energy_required = 3
                        base_damage = 210
                        if _ALKZ:
                            # (v9a-b) atacante primario del matchup: Wild Press
                            # 210-30(res)=180 >= 140 OHKOea a Alakazam y solo
                            # regala 1 premio si cae (vs 3 del Mega ex).
                            base_score += 250
                    elif my_pokemon.id == Makuhita:
                        for o in select.option:
                            if o.type == OptionType.EVOLVE:
                                index = o.inPlayIndex
                                if o.inPlayArea == AreaType.BENCH:
                                    index += 1
                                if index == i:
                                    break
                        else:
                            break
                        base_score -= 100
                        energy_required = 3
                        base_damage = 210
                        if _ALKZ:
                            # (v9a-b) misma linea: evolucionar a Hariyama y
                            # pegar Wild Press este turno
                            base_score += 250
                    elif my_pokemon.id == Solrock:
                        if field_counts[Lunatone] >= 1:
                            energy_required = 1
                            base_damage = 70
                    
                    if base_damage <= 0:
                        continue
                    
                    more_energy = False
                    energy_count = len(my_pokemon.energies)
                    if a == 1 and i == 0 and energy_count >= 2 and not can_use_mega_brave:
                        break
                    if energy_count < energy_required:
                        if hand_counts[Basic_Fighting_Energy] >= 1 and not state.energyAttached:
                            energy_count += 1
                            if energy_count < energy_required:
                                continue
                            else:
                                more_energy = True
                        else:
                            continue

                    for j, op_pokemon in enumerate(op_cards):
                        if j != 0 and not can_op_switch:
                            break
                        data = card_table[op_pokemon.id]
                        # (v4-c) INMUNIDAD A {ex}. El motor pone damage=0 cuando el
                        # defensor tiene NoDamageEnemyExAttack y el atacante es
                        # isEx() = Ex o MegaEx (SetProperty.h:393, Card.h:121).
                        # Cartas con .effectMe(NoDamageEnemyExAttack) en CardImpl.h:
                        #   345 Crustle "Mysterious Rock Inn"
                        #   330 Sylveon "Safeguard"
                        # OJO: en cg.api el flag `ex` es False para Mega Lucario ex
                        # (678); solo `megaEx` es True. Hacen falta los dos.
                        if (op_pokemon.id == 345 or op_pokemon.id == 330
                                or (stadium_id == 1247
                                    and not (data.ex or data.megaEx))):
                            _mine = card_table[my_pokemon.id]
                            if _mine.ex or _mine.megaEx:
                                continue
                        # (conf-ppx) PREMIUM POWER PRO COMO RECURSO.
                        # Barrido sobre el numero de copias gastables ESTE turno
                        # (las de la mano mas las ya jugadas, porque el bonus ya
                        # realizado no esta en base_damage). Se coge el MINIMO que
                        # convierte el ataque en KO; si ninguno mata, cero copias.
                        # Orden del motor: +30 por copia (paso 2, solo contra el
                        # ACTIVO y solo atacante {F}), luego debilidad/resistencia
                        # (pasos 3-4), luego el -30 de Full Metal Lab (paso 5).
                        # (4) Cosmic Beam de Solrock ignora debilidad y resistencia
                        _cop_tot = hand_counts[Premium_Power_Pro] + ppp_spent
                        _cop_use = 0
                        damage = None
                        for _cop in range(_cop_tot + 1):
                            _dmg = base_damage + 30 * _cop
                            if my_pokemon.id != Solrock:
                                if data.weakness == EnergyType.FIGHTING:
                                    _dmg *= 2
                                elif data.resistance == EnergyType.FIGHTING:
                                    _dmg -= 30
                            if stadium_id == 1244 and data.energyType == EnergyType.METAL:
                                _dmg -= 30
                            if _dmg < 0:
                                _dmg = 0
                            if damage is None:
                                damage = _dmg
                            if _dmg >= op_pokemon.hp:
                                damage = _dmg
                                _cop_use = _cop
                                break
                        # (v9m-3) ESPEJO: chip de PPP al Mega rival SIN KO
                        # inmediato — el patron con el que pixiux remato 26/58
                        # de nuestros Megas (chip acumulado + base 130). Solo si
                        # el gasto lo deja a <=130 (remate barato del turno
                        # siguiente; con Cape: 270+2x30=330 deja 460-330=130,
                        # rompe 270+130=400<460). Gate: >=2 PPP en mano para no
                        # quemar la ultima copia sin premio.
                        try:
                            if (_V9M_P3 and _ESPEJO and _cop_use == 0
                                    and damage < op_pokemon.hp
                                    and op_pokemon.hp - damage > 130
                                    and op_pokemon.id == Mega_Lucario_ex
                                    and hand_counts[Premium_Power_Pro] >= 2):
                                for _cop in range(1, _cop_tot + 1):
                                    _dmg = base_damage + 30 * _cop
                                    if my_pokemon.id != Solrock:
                                        if data.weakness == EnergyType.FIGHTING:
                                            _dmg *= 2
                                        elif data.resistance == EnergyType.FIGHTING:
                                            _dmg -= 30
                                    if stadium_id == 1244 and data.energyType == EnergyType.METAL:
                                        _dmg -= 30
                                    if _dmg < 0:
                                        _dmg = 0
                                    if op_pokemon.hp - _dmg <= 130:
                                        damage = _dmg
                                        _cop_use = _cop
                                        break
                        except Exception:
                            pass
                        prize = 0
                        score = pokemon_score(op_pokemon)
                        ko = op_pokemon.hp <= damage
                        if ko:
                            prize = prize_count(op_pokemon)
                        else:
                            score *= damage / op_pokemon.hp
                        score += base_score

                        # (v9a-c) romper su motor de robo: Abra/Kadabra como
                        # blanco de arrastre SOLO si el plan lo remata (el Boss
                        # que no remata es whiff; cada evolucion suya = 2-3
                        # robos = 40-60 de Powerful Hand futuro). Abra 50 y
                        # Kadabra 80 mueren a Aura Jab 100 pese a resistencia.
                        if _ALKZ and j != 0 and ko and op_pokemon.id in (741, 742):
                            score += 300
                        # (v9v-a2) caza tambien en el ACTIVO: montaje t<=4 ->
                        # WR 0.278 vs caza t<=4 -> 0.615. El bono v9a-c solo
                        # cubria arrastres (j!=0); con Abra/Kadabra ACTIVO
                        # rival (30 partidas con Abra activo t1-3) el remate
                        # gratis rompe el mismo motor de robo. Solo si el plan
                        # lo remata y no hay 743 ya montado. Aura Jab 100 mata
                        # Abra 50 y Kadabra 80 pese a resistencia.
                        try:
                            if (_V9V_A2 and _ALKZ and j == 0 and ko
                                    and op_pokemon.id in (Alakazam_Abra,
                                                          Alakazam_Kadabra)
                                    and not _ALAK_MONTADO):
                                score += 300
                        except Exception:
                            pass

                        # (3) tambien se gana si el rival se queda sin Pokemon
                        if my_prize <= prize or (ko and j == 0 and len(op_state.bench) == 0):
                            score = 50000
                        
                        if i == 0:
                            score += 220
                        score += energy_count
                        if best_score < score:
                            best_score = score
                            plan.attacker = i
                            plan.target = j
                            plan.attack_index = a
                            plan.remain_hp = op_pokemon.hp - damage
                            plan.energy = more_energy
                            plan.ppp_need = _cop_use
    
    # La opcion PLAY desaparece cuando Budew u otro efecto bloquea objetos. Esta
    # lectura distingue un Switch realmente util de una copia atrapada en mano.
    _switch_playable_now = False
    try:
        _switch_playable_now = any(
            o.type == OptionType.PLAY
            and get_card(obs, AreaType.HAND, o.index, my_index).id == Switch
            for o in select.option
        )
    except Exception:
        _switch_playable_now = False

    # Attach energy score
    def energy_score(pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        score = 8000
        if active:
            score += 10
        if pokemon.id == Makuhita or pokemon.id == Hariyama:
            if pokemon.id == Hariyama:
                score += 1
            # (r6-as3) E3 gated por ARQUETIPO: muro en mesa rival (Dwebble/
            # Crustle/Great Tusk) o desgaste generico. Cubre las derrotas
            # donde el muro toma premio temprano y _GRIND no dispara.
            _muro = any(c.id in (344, 345, 58)
                        for c in op_state.active + op_state.bench)
            if energy_count < 3 and (_muro or _GRIND):
                score += 100
            elif energy_count < 3 and _ALKZ:
                # (v9a-b) priorizar la linea Hariyama en el gate Alakazam
                score += 100
            if attacker2:
                score -= 50
        elif pokemon.id == Lunatone:
            # Lunatone sigue siendo el ultimo destino de energia cuando esta en
            # banca. Activo y sin energia, en cambio, la primera energia compra
            # una retirada completa. Exigir banca evita gastar el recurso si no
            # existe ningun Pokemon al cual promover.
            if (_R16_BUDEW_ESCAPE and active and energy_count == 0
                    and not _switch_playable_now
                    and op_state.active and op_state.active[0].id == Budew
                    and any(c is not None for c in my_state.bench)):
                score += 250
            else:
                score -= 100
        elif pokemon.id == Solrock:
            if energy_count < 1:
                score += 20
            else:
                score -= 100
        elif pokemon.id == Riolu or pokemon.id == Mega_Lucario_ex:
            if pokemon.id == Mega_Lucario_ex:
                score += 1
            if energy_count < 2:
                score += 100
            if attacker1:
                score -= 50
        return score

    # (v4-s) modo estancamiento: pasado el turno 8 el rival no ha tomado ni un
    # premio, o sea no me esta matando; la unica forma de perder es el mazo.
    # (resucita-v2 / fabrica-v2 13-ago-2026) APAGADO. Medido: encenderlo cuesta
    # +1,172 pp de winrate ponderado por censo en el panel medible (z +10,63,
    # dos tramos disjuntos, 6.000 semillas/celda). El coste esta entero en los
    # cuatro vetos al motor de robo que _GRIND activaba (`desveto` +1,144 pp);
    # el bono de energia bajo _GRIND no aporta nada (`grind_ener` +0,057).
    # Caveat declarado: en Kangaskhan (celda saturada, wr_base 0,997-0,999)
    # apagarlo cuesta ~-0,45 pp de winrate.
    # expresion original, conservada en comentario para auditoria:
    #   _GRIND = (state.turn >= 8 and len(op_state.prize) >= 6)
    # ---- r21: _GRIND re-activado SOLO bajo muro en mesa ----------------
    # Medido 14-ago: apagar _GRIND del todo (grind_off) vale +1,172 pp en el
    # panel PERO -9,60 pp (z=-8,04, 183v/375r) contra Crustle, que el panel de
    # la fabrica v2 no tenia. La causa es la que el comentario original de v4-s
    # ya decia: contra un rival que no te mata, la unica forma de perder es el
    # MAZO, y _GRIND existia para vetar el motor de robo en ese escenario.
    # Muro = Dwebble(344) / Crustle(345) / Great Tusk(58) en mesa rival, la
    # misma deteccion que usa el gate de energia de r6-as3 mas abajo.
    # Fuera del muro sigue APAGADO: ahi es donde vale los +1,172 pp.
    _R21_GRIND_MURO = True
    # OJO: la banca puede contener None (lo detecto valida_runtime.py el 14-ago;
    # sin la guarda esto revienta con AttributeError en el runtime de Kaggle).
    _r21_muro = any(c is not None and c.id in (344, 345, 58)
                    for c in list(op_state.active) + list(op_state.bench))
    _GRIND = (_R21_GRIND_MURO and _r21_muro
              and state.turn >= 8 and len(op_state.prize) >= 6)

    _R19_IDX_SW = []
    _R19_IDX_RET = []
    _R19_FIRE = False
    _R19_MONTADO = False
    try:
        _r19_act = my_state.active[0] if my_state.active else None
        if (_r19_act is not None and _r19_act.id == Mega_Lucario_ex
                and context == SelectContext.MAIN):
            # retreatCost del Mega = 2 = coste de Mega Brave: retirarlo tira el
            # montaje. Switch (item) rota conservando las energias.
            _R19_MONTADO = len(_r19_act.energies) >= 2
            _r19_opz = len(op_state.prize)
            _r19_banca = any(c is not None and c.id != Mega_Lucario_ex
                             for c in my_state.bench)
            if _r19_act.hp <= _R19_B_HP and _r19_banca and plan.attacker <= 0:
                _R19_CNT["gateable"] += 1
                _R19_FIRE = (
                    _R19_B_ON
                    and _r19_opz <= _R19_B_OPPRIZE_MAX
                    and my_prize >= _R19_B_MYPRIZE_MIN
                    and _r19_opz >= _R19_B_OPPRIZE_MIN
                    and (my_prize - _r19_opz) <= _R19_B_DIF_MAX
                    and not (_R19_B_APAGA_ESPEJO and _ESPEJO))
                if _R19_FIRE:
                    _R19_CNT["disparo"] += 1
    except Exception:
        _R19_FIRE = False

    # Iterate over every possible option and assign a heuristic score.
    scores = []  # Score for each action
    for o in select.option:
        score = 0  # The default and baseline score is 0.
        if o.type == OptionType.NUMBER:
            score = o.number  # e.g., for "draw X cards"
        elif o.type == OptionType.YES:
            score = 1  # Prefer "Yes"
            # (2) rechaza el arrastre de Hariyama si el plan apunta al activo
            if context == SelectContext.ACTIVATE and plan.target == 0:
                cc = select.contextCard
                if cc != None and cc.id == Hariyama:
                    score = -1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card != None:
                energy_count = 0
                if isinstance(card, Pokemon):
                    energy_count = len(card.energies)
                if context == SelectContext.SWITCH or context == SelectContext.TO_ACTIVE:
                    # Selection of the Pokémon to send to the Active Spot
                    if o.playerIndex == my_index:
                        score += energy_count * 2
                        if o.index == plan.attacker - 1:
                            score += 100
                        if card.id == Mega_Lucario_ex:
                            # (v9v-a1) guard de exposicion al promover: 53/72
                            # muertes del Mega a Powerful Hand fueron eleccion
                            # propia (este +20 era el mayor bono y NO tenia
                            # guard en _ALKZ). Si 20*(mano rival+4) alcanza el
                            # hp restante, invierte a malus: que prefiera
                            # promover Hariyama/Makuhita (Hariyama regala 1
                            # premio y aguanta 8 manos de 150; en victorias el
                            # PH pega 34 veces a cuerpos baratos y caen 14).
                            # Excepcion: si el Mega promovido REMATA este turno
                            # (es el atacante del plan y el plan deja KO),
                            # dejalo con su bono original.
                            _v9v_mega_malus = False
                            try:
                                if (_V9V_A1 and _ALKZ
                                        and 20 * (op_state.handCount + 4)
                                        >= card.hp
                                        and not (o.index == plan.attacker - 1
                                                 and plan.remain_hp <= 0)):
                                    _v9v_mega_malus = True
                            except Exception:
                                _v9v_mega_malus = False
                            if _v9v_mega_malus:
                                score -= 20
                            elif my_prize == 2 or my_prize == 3:
                                score += 8
                            else:
                                score += 20
                        elif card.id == Hariyama and energy_count >= 2:
                            score += 15
                        elif card.id == Makuhita and energy_count >= 2:
                            score += 10
                        elif card.id == Solrock:
                            score += 5
                        elif card.id == Riolu:
                            score += 4
                        # (v9v-a1) Lunatone como activo en _ALKZ t2-4: patron
                        # perdedor puro (19 derrotas vs 3 victorias). Malus
                        # para que quede detras de cualquier otro cuerpo.
                        try:
                            if (_V9V_A1 and _ALKZ and card.id == Lunatone
                                    and 2 <= state.turn <= 4):
                                score -= 20
                        except Exception:
                            pass
                    else:
                        if o.index == plan.target - 1:
                            score += 100
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    # Prioritize playing Riolu if going first, and Solrock if going second.
                    if card.id == Solrock:
                        if state.firstPlayer == my_index:
                            score = 2
                        else:
                            score = 4
                    elif card.id == Riolu:
                        score = 3
                    elif card.id == Makuhita:
                        score = 1
                elif context == SelectContext.TO_HAND:
                    score = 200 - hand_counts[card.id] * 100
                    if card.id == Makuhita:
                        if (_R14_MARNIE_DUPLICATE_MAKUHITA_GUARD and _MARN
                                and field_counts[Makuhita] + field_counts[Hariyama] >= 1):
                            score = -500
                        elif field_counts[card.id] >= 1:
                            score -= 10
                        else:
                            score += 10
                    elif card.id == Hariyama:
                        if field_counts[Makuhita] >= 1:
                            score += 20
                        else:
                            score -= 20
                    elif card.id == Lunatone:
                        if field_counts[card.id] >= 1:
                            score -= 250
                        else:
                            score += 60
                    elif card.id == Solrock:
                        if field_counts[card.id] >= 1:
                            score -= 250
                        else:
                            score += 50
                    elif card.id == Riolu:
                        if field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 2:
                            score -= 150
                        elif field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 1:
                            score -= 3
                        else:
                            score += 40
                    elif card.id == Mega_Lucario_ex:
                        if field_counts[Riolu] >= 1:
                            score += 40
                        else:
                            score -= 15
                    elif card.id == Basic_Fighting_Energy:
                        if not ability_used or not state.energyAttached:
                            score += 30
                        else:
                            score -= 1
                elif context == SelectContext.ATTACH_FROM:
                    score = energy_score(card, o.area == AreaType.ACTIVE)
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[card.id]
            if data.cardType == CardType.POKEMON:
                score = 20000
                if (card.id == Makuhita and _R14_MARNIE_DUPLICATE_MAKUHITA_GUARD
                        and _MARN
                        and field_counts[Makuhita] + field_counts[Hariyama] >= 1):
                    score = -1
                elif card.id == Lunatone or card.id == Solrock:
                    if field_counts[card.id] >= 1:
                        score = -1
                elif card.id == Riolu:
                    if field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 2:
                        score = -1
            else:
                score = 10000
                # (v4-s) en modo estancamiento los buscadores queman mazo
                if _GRIND and card.id in (Dusk_Ball, Poke_Pad, Fighting_Gong):
                    score = -1
                if card.id == Switch:
                    if plan.attacker <= 0:
                        if _R19_FIRE and _R19_B_SWITCH:
                            score = _R19_B_SW_SCORE
                            _R19_CNT["pts_switch"] += 1
                            _R19_IDX_SW.append(len(scores))
                        else:
                            score = -1
                    else:
                        score = 6000
                elif card.id == Premium_Power_Pro:
                    if can_attack:
                        # (conf-ppx) jugarla solo si el plan la cuenta
                        score = 5000 if plan.ppp_need > ppp_spent else -1
                    elif (not state.supporterPlayed and hand_counts[Carmine] > 0
                            and hand_counts[Lillie_Determination] == 0):
                        score = 3050
                    else:
                        score = -1
                elif card.id == Boss_Orders:
                    if plan.target >= 1:
                        score = 3200
                        # (v9v-g1/a2) Boss para REMATAR arrastres que rompen el
                        # motor rival. Evidencia: en las 23 partidas con caza=0
                        # siempre hubo ventana y en 12 teniamos Boss en mano;
                        # jugamos 0.6 Boss/partida. Solo si el plan REMATA
                        # (remain_hp <= 0): NO quemar los Boss en Imps que el
                        # plan no remata. Sube sobre Lillie (3100) y cualquier
                        # otro supporter para ganar el slot del turno.
                        try:
                            _bt = op_cards[plan.target]
                            if plan.remain_hp <= 0:
                                if (_V9V_G1 and _GRIM_PRE
                                        and _bt.id in (Marnie_Impidimp,
                                                       Marnie_Morgrem)):
                                    score = 3500
                                elif (_V9V_A2 and _ALKZ and not _ALAK_MONTADO
                                        and _bt.id in (Alakazam_Abra,
                                                       Alakazam_Kadabra)):
                                    score = 3500
                        except Exception:
                            pass
                    else:
                        score = -1
                elif card.id == Xerosic:
                    # (xero) disrupcion asimetrica; compite por el UNICO hueco
                    # de Supporter del turno.
                    try:
                        score = _xerosic_score(_ALKZ, op_state.handCount,
                                               _my_active_hp)
                    except Exception:
                        score = -1
                elif card.id == Carmine:
                    score = 3000
                    # (1) freno al agotamiento de mazo
                    if my_state.deckCount <= 12 or _GRIND:
                        score = -1
                elif card.id == Lillie_Determination:
                    score = 3100
                    if (my_state.deckCount + len(my_state.hand) <= 10
                            or (_GRIND and len(my_state.hand) < 7)):
                        score = -1
                elif card.id == Gravity_Mountain:
                    # (v4-g) PUERTO S2. Bajarla solo si HACE ALGO:
                    #   (a) hay estadio rival en juego -> lo elimina;
                    #   (b) hay un Stage 2 rival en juego -> le quita 30 PS
                    #       (CardImpl.h:4155 MaxHpChange eVal -30 TargetType::Stage2).
                    # Nuestro mazo no lleva ningun Stage 2 (Mega Lucario ex es
                    # Stage 1), asi que el -30 es unilateral.
                    _op_s2 = False
                    for _p in list(op_state.active) + list(op_state.bench):
                        if _p is None:
                            continue
                        _d = card_table.get(_p.id)
                        if _d is not None and _d.stage2:
                            _op_s2 = True
                            break
                    if stadium_id == Gravity_Mountain:
                        score = -1
                    elif stadium_id != 0 or _op_s2:
                        score = 10000
                    else:
                        score = -1
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card.id == Hero_Cape:
                score = 7000
                if pokemon.id == Riolu:
                    score += 100
                elif pokemon.id == Mega_Lucario_ex:
                    score += 200
                # (v9m-2) ESPEJO: disciplina de Cape. En derrotas la Cape se
                # fue 5 veces a Solrock/Makuhita (basura); en victorias 29/46
                # al Mega. Nunca a la morralla (retener en mano) y prioridad
                # al Riolu LIDER en t1-2: 70+100=170 bloquea el beam-snipe de
                # t2 (70+3xPPP=160<170), obliga a quemar 4 PPP o esperar.
                try:
                    if _V9M_P2 and _ESPEJO:
                        if pokemon.id in (Solrock, Makuhita, Lunatone):
                            score = -1
                        elif (pokemon.id == Riolu and state.turn <= 2
                                and o.inPlayArea == AreaType.ACTIVE):
                            score += 300
                except Exception:
                    pass
                # (v9v-g2) Cape al Mega, gateada a linea MARNIE (no al espejo:
                # ahi la disciplina de Cape regresaba -1.9pp y sigue apagada).
                # Grimmsnarl two-shotea al Mega: sin Cape mueren 36/36; con
                # Cape 360 < 440 exige 3er golpe. Cape al Mega -> WR 0.763; a
                # otro -> 0.650; sin jugar -> 0.545: prioridad fuerte al Mega
                # y nunca a la morralla (retener en mano vale mas que Solrock/
                # Makuhita/Lunatone; Riolu conserva su +100 porque la Cape
                # sube con la evolucion al Mega).
                try:
                    if _V9V_G2 and _MARN:
                        if pokemon.id == Mega_Lucario_ex:
                            score += 300
                        elif pokemon.id in (Solrock, Makuhita, Lunatone):
                            score = -1
                except Exception:
                    pass
            else:
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
                if o.inPlayArea == AreaType.ACTIVE:
                    if plan.attacker == 0 and plan.energy:
                        score += 200
                else:
                    if plan.attacker == 1 + o.inPlayIndex and plan.energy:
                        score += 200
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 9000 + len(pokemon.energies)
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == 1267:  # Lumiose City
                score = 1
            elif card.id == Lunatone and (my_state.deckCount <= 10 or _GRIND):
                # (1) Lunar Cycle quema 3 cartas de mazo
                score = -1
            else:
                score = 30000
        elif o.type == OptionType.RETREAT:
            if plan.attacker >= 1:
                score = 2000
            elif (_R19_FIRE and _R19_B_RETREAT
                    and not (_R19_B_RET_SOLO_DESMONTADO and _R19_MONTADO)):
                score = 2500
                _R19_CNT["pts_retreat"] += 1
                _R19_IDX_RET.append(len(scores))
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            score = 1000
            if plan.attack_index == 1:
                if o.attackId == 983:  # Mega Brave
                    score += 100
            else:
                if o.attackId != 983:
                    score += 100

        scores.append(score)

    # Select in descending order of score
    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    _EXPORT["scores"] = scores
    _EXPORT["order"] = desc_indices
    if context == SelectContext.MAIN and not _EXPORT["search_mode"]:
        o = select.option[desc_indices[0]]
        if o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == Lunatone:
                ability_used = True
        elif o.type == OptionType.PLAY:
            _pc = get_card(obs, AreaType.HAND, o.index, my_index)
            if _pc is not None and _pc.id == Premium_Power_Pro:
                ppp_spent += 1
    _r19_pick = desc_indices[0] if desc_indices else -1
    if _r19_pick in _R19_IDX_SW:
        _R19_CNT["ejec_switch"] += 1
    if _r19_pick in _R19_IDX_RET:
        _R19_CNT["ejec_retreat"] += 1
    return desc_indices[:select.maxCount]


# ============================================================================
# CAPA Q  —  destilacion de la BUSQUEDA en la familia de PROMOCION (ctx3/ctx4)
#
# Punto de partida medido (fabrica-20260813/busqueda-limpia):
#   - el arbol PIMC corre limpio (begin_fail=0 en 34.760 search_begin)
#   - con la hoja CONSTANTE produce 0 overrides  => TODO lo decide la eval
#   - las evals disponibles miden -2,879 pp (posicional) y +0,029 pp (v.probada)
#
# Hipotesis de wangyuou (research/intel_20260810/discusiones.md:102), nunca
# probada en este repo: V(s) no sirve para decidir; hay que aprender Q(s,a)
# ramificando CONTRAFACTUALMENTE con la Search API.
#
# Esta capa hace las dos mitades:
#   _Q_COLLECT=True  -> en cada decision ctx3/ctx4 ramifica CADA candidato con
#                       search_begin/search_step y hace ROLLOUT HASTA EL FINAL
#                       con la politica actual. Escribe (rasgos, resultado) a
#                       JSONL. NO altera la partida viva (devuelve siempre None).
#   _Q_COLLECT=False -> usa el modelo destilado, o como POLITICA directa
#                       (_Q_MODE="policy") o como EVAL DE HOJA del mismo arbol
#                       (_Q_MODE="leaf").
# ============================================================================
import ctypes as _ctypes
import json as _json
import time as _time
import random as _random
from collections import Counter as _Counter

# ENDURECIDO PARA ENVIO: estos dos imports solo hacen falta en modo `leaf` y en
# modo coleccion (el arbol PIMC). El brazo que se envia corre en modo `policy`,
# que NUNCA llama a la busqueda, asi que si el runtime no los expone la capa Q
# sigue funcionando entera. Sin la guarda, un fallo de import aqui mataria el
# main.py al cargarlo. Probado: con _Q_MODE="policy" el delta contra el brazo
# sin guarda es 0,000000 exacto (ver json/nulohard_*.json).
try:
    from cg import api as _cgapi
except Exception:                       # pragma: no cover
    _cgapi = None
try:
    from cg.sim import lib as _cglib
except Exception:                       # pragma: no cover
    _cglib = None

# ------------------------------- interruptores -------------------------------
_Q_ON = True            # gate maestro (el brazo `off` lo pone a False)
_Q_COLLECT = False      # modo generacion de dataset
_Q_MODE = "policy"  # "policy" | "leaf"
_Q_CTX = (4,)  # contextos donde actua (3=SWITCH, 4=TO_ACTIVE tras KO)
_Q_MARGIN = 0.02  # margen de override en unidades de ventaja del modelo
_Q_PLACEBO = False      # True = calcula todo y TIRA la decision (nulo fuerte)
_Q_LEAF_OLD = False     # ablacion: eval de hoja POSICIONAL vieja (no el modelo)
_Q_LEAF_CONST = False   # ablacion: eval de hoja CONSTANTE
_Q_FIJO = False         # ablacion FIJA: ignora el estado, solo la id de carta
_Q_SHUFFLE = False      # ablacion PLACEBO-MODELO: pesos barajados

# --------------------------- parametros de coleccion -------------------------
_QC_NDET = 4            # determinizaciones por decision
_QC_MAXROLL = 900       # tope de pasos por rollout (censura si se pasa)
_QC_MAXCAND = 6
_QC_OUT = os.environ.get("QCOL_OUT")          # fichero JSONL de salida
_QC_RIVDECK = None      # deck.csv REAL del rival: determinizacion ORACULO
# El oraculo SOLO existe si la variable de entorno esta puesta (coleccion
# offline). En el bundle no hay env: cae a la determinizacion ciega de r11.
_qrd = os.environ.get("QCOL_RIVDECK")
if _qrd:
    try:
        _QC_RIVDECK = [int(x) for x in open(_qrd).read().split() if x.strip()]
    except Exception:
        _QC_RIVDECK = None

# --------------------------- parametros del arbol ----------------------------
_QT_NDET = 3
_QT_KOPP = 2
_QT_MAXSUB = 40
_QT_KSTEPS = 700
_QT_GUARD_S = 20.0
DUMMY_BASIC = 646
DUMMY_ENERGY = 6

_RNG = _random.Random(20260813)
_ONE = _ctypes.c_int * 1
_MY_DECK = my_deck

QSTATS = {
    "seen3": 0, "seen4": 0, "cand3": 0, "cand4": 0,
    "fire3": 0, "fire4": 0,          # decisiones donde el modelo pudo decidir
    "chg3": 0, "chg4": 0,            # decisiones donde CAMBIO el argmax
    "skip": 0, "fail": 0, "no_sbi": 0,
    "begin_ok": 0, "begin_fail": 0, "steps": 0,
    "rollouts": 0, "roll_term": 0, "roll_censor": 0, "roll_steps": 0,
    "records": 0, "decisions": 0, "ms": 0.0, "games": 0,
    "leaf_evals": 0, "considered": 0, "overrides": 0, "discarded": 0,
}

# ------------------------- tabla de ataques (una vez) ------------------------
try:
    _ATK = {}
    for _a in _cgapi.all_attack():
        _ATK[_a.attackId] = (int(_a.damage or 0), len(_a.energies or []))
except Exception:
    _ATK = {}


def _atk_info(cid, n_en):
    """(dmg_asequible_max, dmg_max, coste_minimo, deficit) para una carta."""
    d = card_table.get(cid)
    if d is None:
        return 0, 0, 9, 9
    best_now = 0
    best_all = 0
    cost_min = 9
    for aid in (d.attacks or []):
        dm, co = _ATK.get(aid, (0, 9))
        if dm > best_all:
            best_all = dm
        if co < cost_min:
            cost_min = co
        if n_en >= co and dm > best_now:
            best_now = dm
    return best_now, best_all, cost_min, max(0, cost_min - n_en)


# ============================== RASGOS (s,a) =================================
# ORDEN CONGELADO. Se calculan sobre el dict crudo (sin dataclasses) para que
# coleccion y runtime usen EXACTAMENTE el mismo codigo.
QFEAT_NAMES = [
    "c_en", "c_hp", "c_maxhp", "c_hpfrac", "c_retreat", "c_prize", "c_stage",
    "c_is_mega", "c_is_riolu", "c_is_solrock", "c_is_lunatone", "c_is_hariyama",
    "c_is_makuhita", "c_tools", "c_can_atk", "c_dmg_now", "c_dmg_full",
    "c_deficit", "c_ko_opp", "c_survives", "c_dmgratio", "c_evolved",
    "h_same_id", "h_mega", "h_riolu", "h_energy", "h_switch", "h_count",
    "g_turn", "g_my_prz", "g_op_prz", "g_prz_diff", "g_bench", "g_op_bench",
    "g_deck", "o_hp", "o_en", "o_maxhp", "o_prize", "o_dmg", "g_benchhp",
    "g_ctx", "g_nopts", "s_heur", "s_heur_rank", "s_heur_dif",
]
QN = len(QFEAT_NAMES)


def _pk_of(state, opt, me_i):
    """Pokemon dict al que apunta una opcion CARD de ctx3/ctx4."""
    try:
        if opt.get("playerIndex") != me_i:
            return None
        ar = opt.get("area")
        ix = opt.get("index")
        ps = state["players"][me_i]
        if ar == 5:      # BENCH
            return ps["bench"][ix]
        if ar == 4:      # ACTIVE
            return (ps["active"] or [None])[ix]
    except Exception:
        return None
    return None


def _qfeats(state, sel, me_i, scores, order, ctx):
    """Devuelve (lista_de_vectores, lista_de_indices_validos)."""
    opts = sel["option"]
    me = state["players"][me_i]
    op = state["players"][1 - me_i]
    hand = me.get("hand") or []
    h_cnt = _Counter(c["id"] for c in hand)
    n_en_hand = sum(v for k, v in h_cnt.items()
                    if card_table.get(k) is not None and card_table[k].cardType in (5, 6))
    oa = (op["active"] or [None])[0]
    if oa:
        o_hp = oa["hp"]; o_max = oa["maxHp"]; o_en = len(oa["energies"])
        od = card_table.get(oa["id"])
        o_prz = 3 if (od and od.megaEx) else 2 if (od and od.ex) else 1
        o_dmg, _oa2, _oc, _od2 = _atk_info(oa["id"], o_en)
    else:
        o_hp = o_max = o_en = 0; o_prz = 1; o_dmg = 0
    bench_hp = sum(p["hp"] for p in me["bench"] if p)
    rank = {}
    for r, i in enumerate(order or []):
        rank[i] = r
    smax = max(scores) if scores else 0.0
    vecs, idxs = [], []
    for i, o in enumerate(opts):
        pk = _pk_of(state, o, me_i)
        if pk is None:
            continue
        cid = pk["id"]
        d = card_table.get(cid)
        n_en = len(pk["energies"])
        hp = pk["hp"]; mhp = max(1, pk["maxHp"])
        dmg_now, dmg_all, cost_min, defi = _atk_info(cid, n_en)
        prz = 3 if (d and d.megaEx) else 2 if (d and d.ex) else 1
        stage = 2 if (d and d.stage2) else 1 if (d and d.stage1) else 0
        v = [
            float(n_en), float(hp), float(mhp), hp / mhp,
            float(d.retreatCost if d else 0), float(prz), float(stage),
            1.0 if cid == 678 else 0.0, 1.0 if cid == 677 else 0.0,
            1.0 if cid == 676 else 0.0, 1.0 if cid == 675 else 0.0,
            1.0 if cid == 674 else 0.0, 1.0 if cid == 673 else 0.0,
            float(len(pk["tools"])), 1.0 if dmg_now > 0 else 0.0,
            float(dmg_now), float(dmg_all), float(defi),
            1.0 if (dmg_now >= o_hp and o_hp > 0) else 0.0,
            1.0 if hp > o_dmg else 0.0,
            (o_dmg / hp) if hp else 9.0,
            float(len(pk["preEvolution"])),
            float(h_cnt.get(cid, 0)), float(h_cnt.get(678, 0)),
            float(h_cnt.get(677, 0)), float(n_en_hand), float(h_cnt.get(1123, 0)),
            float(len(hand)),
            float(state.get("turn") or 0), float(len(me["prize"])),
            float(len(op["prize"])), float(len(op["prize"]) - len(me["prize"])),
            float(len(me["bench"])), float(len(op["bench"])),
            float(me["deckCount"]), float(o_hp), float(o_en), float(o_max),
            float(o_prz), float(o_dmg), float(bench_hp),
            float(ctx), float(len(opts)),
            float(scores[i]) if scores else 0.0,
            float(rank.get(i, len(opts))),
            float((scores[i] - smax)) if scores else 0.0,
        ]
        vecs.append(v)
        idxs.append(i)
    return vecs, idxs


# ============================ MODELO DESTILADO ===============================
# Se sustituye por el exportado en tiempo de construccion del brazo.
QTREES = []          # lista de arboles: (feat[], thr[], left[], right[], val[])
QBASE = 0.0
QW = None            # alternativa lineal (lista de QN+1 pesos)


def q_pred(v):
    """Ventaja estimada del candidato. Puro Python, sin dependencias."""
    if _Q_FIJO:
        # ablacion FIJA: solo la identidad de la carta, sin leer el tablero
        return (v[7] * 3.0 + v[11] * 2.0 + v[12] * 1.0 + v[9] * 0.5 + v[8] * 0.25)
    s = QBASE
    for tr in QTREES:
        f, th, lf, rt, va = tr
        n = 0
        while f[n] >= 0:
            n = lf[n] if v[f[n]] <= th[n] else rt[n]
        s += va[n]
    if QW is not None:
        s += QW[QN]
        for k in range(QN):
            s += QW[k] * v[k]
    return s


# ============================== motor de busqueda ============================
def _step(sid, choice):
    n = len(choice)
    arr = _ONE(choice[0]) if n == 1 else (_ctypes.c_int * n)(*choice)
    bs = _cglib.SearchStep(_cgapi.agent_ptr, sid, arr, n)
    QSTATS["steps"] += 1
    d = _json.loads(bs)
    if d["error"] != 0:
        return None, None
    st = d["state"]
    return st["searchId"], st["observation"]


def _policy(cur):
    sel = cur.get("select")
    if not sel:
        return None
    opts = sel["option"]
    n = len(opts)
    if n == 0:
        return None
    lo = max(1, sel["minCount"])
    if n <= 1 or lo >= n:
        return list(range(min(max(lo, 1), n)))
    _EXPORT["search_mode"] = True
    try:
        ch = _heuristic_agent(cur)
    except Exception:
        ch = list(range(lo))
    finally:
        _EXPORT["search_mode"] = False
    return ch or list(range(lo))


def _leaf_eval(cur, me_i):
    QSTATS["leaf_evals"] += 1
    if _Q_LEAF_CONST:
        return 0.0
    if cur is None:
        return 0.0
    state = cur.get("current")
    if state is None:
        return 0.0
    res = state.get("result")
    if res is not None and res >= 0:
        return 1e7 if res == me_i else (0.0 if res == 2 else -1e7)
    if _Q_LEAF_OLD:
        me = state["players"][me_i]; op = state["players"][1 - me_i]
        my_hp = my_en = op_hp = op_en = 0
        for p in me["active"] + me["bench"]:
            if p:
                my_hp += p["hp"]; my_en += len(p["energies"])
        for p in op["active"] + op["bench"]:
            if p:
                op_hp += p["hp"]; op_en += len(p["energies"])
        no_active = 0 if (me["active"] and me["active"][0]) else 1
        return (1000.0 * (len(op["prize"]) - len(me["prize"]))
                + (my_hp - op_hp) + 5.0 * (my_en - op_en) - 4000.0 * no_active)
    # ---- EVAL DE HOJA APRENDIDA: el modelo puntua el propio activo ----
    me = state["players"][me_i]
    act = (me["active"] or [None])[0]
    if act is None:
        return -1e6
    fake_opt = {"playerIndex": me_i, "area": 4, "index": 0}
    fake_sel = {"option": [fake_opt], "minCount": 1, "maxCount": 1}
    try:
        vs, ix = _qfeats(state, fake_sel, me_i, [0.0], [0], 4)
    except Exception:
        return 0.0
    if not vs:
        return 0.0
    prz = 1000.0 * (len(state["players"][1 - me_i]["prize"]) - len(me["prize"]))
    return prz + 100.0 * q_pred(vs[0])


# --------------------------- determinizacion ---------------------------------
def _visible_mine(state, me_i):
    me = state["players"][me_i]
    seen = _Counter()
    for c in me["hand"] or []:
        seen[c["id"]] += 1
    for c in me["discard"]:
        seen[c["id"]] += 1
    for c in me["prize"]:
        if c is not None:
            seen[c["id"]] += 1
    for p in me["active"] + me["bench"]:
        if not p:
            continue
        seen[p["id"]] += 1
        for c in p["energyCards"] + p["tools"] + p["preEvolution"]:
            seen[c["id"]] += 1
    for c in state.get("stadium") or []:
        if c and c.get("playerIndex") == me_i:
            seen[c["id"]] += 1
    return seen


def _visible_op(state, op_i):
    op = state["players"][op_i]
    seen, etype = _Counter(), _Counter()
    for c in op["discard"]:
        seen[c["id"]] += 1
    for p in op["active"] + op["bench"]:
        if not p:
            continue
        seen[p["id"]] += 1
        for c in p["energyCards"] + p["tools"] + p["preEvolution"]:
            seen[c["id"]] += 1
        for e in p["energies"]:
            etype[int(e)] += 1
    for c in op["prize"]:
        if c is not None:
            seen[c["id"]] += 1
    for c in state.get("stadium") or []:
        if c and c.get("playerIndex") == op_i:
            seen[c["id"]] += 1
    return seen, etype


def _determinize(state, me_i, rng):
    me = state["players"][me_i]
    op_i = 1 - me_i
    op = state["players"][op_i]
    seen = _visible_mine(state, me_i)
    remain = []
    for cid, n in _Counter(_MY_DECK).items():
        remain.extend([cid] * max(0, n - seen.get(cid, 0)))
    n_prize_hidden = sum(1 for c in me["prize"] if c is None)
    need = me["deckCount"] + n_prize_hidden
    if len(remain) < need:
        remain += [DUMMY_ENERGY] * (need - len(remain))
    rng.shuffle(remain)
    your_deck = remain[:me["deckCount"]]
    fill = iter(remain[me["deckCount"]:need])
    your_prize = [c["id"] if c is not None else next(fill, DUMMY_ENERGY)
                  for c in me["prize"]]
    op_seen, etype = _visible_op(state, op_i)
    if _QC_RIVDECK:                      # ORACULO: mazo real del rival
        pool = []
        for cid, n in _Counter(_QC_RIVDECK).items():
            pool.extend([cid] * max(0, n - op_seen.get(cid, 0)))
    else:
        etop = max(etype.items(), key=lambda x: x[1])[0] if etype else DUMMY_ENERGY
        top_card = max(op_seen.items(), key=lambda x: x[1])[0] if op_seen else None
        pool = ([top_card] * 30 if top_card else []) + [etop] * 30 + [DUMMY_BASIC] * 8
    n_op_prize_hidden = sum(1 for c in op["prize"] if c is None)
    op_need = op["deckCount"] + n_op_prize_hidden + op["handCount"]
    if len(pool) < op_need:
        pool += [DUMMY_ENERGY] * (op_need - len(pool))
    rng.shuffle(pool)
    opponent_deck = pool[:op["deckCount"]]
    off = op["deckCount"]
    fop = iter(pool[off:off + n_op_prize_hidden])
    opponent_prize = [c["id"] if c is not None else next(fop, DUMMY_ENERGY)
                      for c in op["prize"]]
    off += n_op_prize_hidden
    opponent_hand = pool[off:off + op["handCount"]]
    opponent_active = [DUMMY_BASIC] if (op["active"] and op["active"][0] is None) else []
    return dict(your_deck=your_deck, your_prize=your_prize,
                opponent_deck=opponent_deck, opponent_prize=opponent_prize,
                opponent_hand=opponent_hand, opponent_active=opponent_active)


# ------------------- snapshot del estado de la heuristica --------------------
def _save_heur_state():
    return (dict(plan.__dict__), pre_turn, ability_used, ppp_spent,
            _v9m_espejo, _v9m_turno, _v9v_grim_pre, _fn_arch_line)


def _restore_heur_state(s):
    global pre_turn, ability_used, ppp_spent
    global _v9m_espejo, _v9m_turno, _v9v_grim_pre, _fn_arch_line
    plan.__dict__.clear()
    plan.__dict__.update(s[0])
    pre_turn, ability_used, ppp_spent = s[1], s[2], s[3]
    _v9m_espejo, _v9m_turno, _v9v_grim_pre = s[4], s[5], s[6]
    _fn_arch_line = s[7]


# ============================ ROLLOUT HASTA EL FINAL =========================
def _rollout(sid, cur, me_i, cap):
    """Juega hasta terminal con la politica actual. 1/0/0.5 o None si censura."""
    for k in range(cap):
        cs = cur.get("current")
        if cs is None:
            return None, k
        r = cs.get("result")
        if r is not None and r >= 0:
            return (1.0 if r == me_i else (0.5 if r == 2 else 0.0)), k
        if not cur.get("select"):
            return None, k
        ch = _policy(cur)
        if not ch:
            return None, k
        nsid, ncur = _step(sid, ch)
        if nsid is None:
            return None, k
        sid, cur = nsid, ncur
    return None, cap


# ============================ MODO COLECCION =================================
_QC_FH = None
_QC_GAME = [0]
_QC_DEC = [0]


def _collect(obs_dict, sel, state, ctx, me_i, scores, order):
    global _QC_FH
    if _QC_FH is None:
        _QC_FH = open(_QC_OUT, "a")
    vecs, idxs = _qfeats(state, sel, me_i, scores, order, ctx)
    if len(idxs) < 2:
        QSTATS["skip"] += 1
        return
    if len(idxs) > _QC_MAXCAND:
        rk = {i: r for r, i in enumerate(order)}
        pares = sorted(zip(idxs, vecs), key=lambda t: rk.get(t[0], 99))[:_QC_MAXCAND]
        idxs = [p[0] for p in pares]
        vecs = [p[1] for p in pares]
    if obs_dict.get("search_begin_input") is None:
        QSTATS["no_sbi"] += 1
        return
    obs_cls = to_observation_class(obs_dict)
    acc = {i: 0.0 for i in idxs}
    nev = {i: 0 for i in idxs}
    snap = _save_heur_state()
    began = False
    _QC_DEC[0] += 1
    try:
        for _d in range(_QC_NDET):
            hidden = _determinize(state, me_i, _RNG)
            try:
                root = _cgapi.search_begin(obs_cls, **hidden)
                QSTATS["begin_ok"] += 1
                began = True
            except Exception:
                QSTATS["begin_fail"] += 1
                return
            rsid = root.searchId
            for idx in idxs:
                sid1, cur = _step(rsid, [idx])
                if sid1 is None:
                    continue
                y, ns = _rollout(sid1, cur, me_i, _QC_MAXROLL)
                QSTATS["rollouts"] += 1
                QSTATS["roll_steps"] += ns
                if y is None:
                    QSTATS["roll_censor"] += 1
                    continue
                QSTATS["roll_term"] += 1
                acc[idx] += y
                nev[idx] += 1
            try:
                _cgapi.search_end()
            except Exception:
                pass
            began = False
        ok = [i for i in idxs if nev[i] > 0]
        if len(ok) < 2:
            return
        rec = {
            "g": _QC_GAME[0], "d": _QC_DEC[0], "ctx": ctx,
            "top": order[0] if order else -1,
            "cand": [{"i": i, "q": acc[i] / nev[i], "n": nev[i],
                      "f": [round(x, 4) for x in vecs[idxs.index(i)]]}
                     for i in ok],
        }
        _QC_FH.write(_json.dumps(rec) + "\n")
        QSTATS["records"] += len(ok)
        QSTATS["decisions"] += 1
    except Exception:
        QSTATS["fail"] += 1
    finally:
        if began:
            try:
                _cgapi.search_end()
            except Exception:
                pass
        _restore_heur_state(snap)


# ============================ ARBOL CON HOJA Q ===============================
class _Bud:
    __slots__ = ("left", "dead")

    def __init__(self, k, dl):
        self.left = k; self.dead = dl

    def ok(self):
        return self.left > 0 and _time.monotonic() < self.dead

    def spend(self):
        self.left -= 1


def _complete_turn(sid, cur, owner, bud):
    for _ in range(_QT_MAXSUB):
        if not bud.ok():
            return sid, cur
        cs = cur.get("current")
        if cs is None or (cs.get("result") is not None and cs["result"] >= 0):
            return sid, cur
        if cs["yourIndex"] != owner or not cur.get("select"):
            return sid, cur
        ch = _policy(cur)
        if not ch:
            return sid, cur
        bud.spend()
        nsid, ncur = _step(sid, ch)
        if nsid is None:
            return sid, cur
        sid, cur = nsid, ncur
    return sid, cur


def _advance_forced(sid, cur, owner, bud, limit=8):
    for _ in range(limit):
        if not bud.ok():
            break
        cs = cur.get("current"); sel = cur.get("select")
        if (cs is None or not sel or cs["yourIndex"] != owner or sel["context"] == 0
                or (cs.get("result") is not None and cs["result"] >= 0)):
            break
        ch = _policy(cur)
        if not ch:
            break
        bud.spend()
        nsid, ncur = _step(sid, ch)
        if nsid is None:
            break
        sid, cur = nsid, ncur
    return sid, cur


def _tree_decide(obs_dict, sel, state, ctx, me_i, scores, order):
    """Mismo arbol de busqueda-limpia, con la eval de hoja intercambiable."""
    n = len(sel["option"])
    cand = list(order[:_QC_MAXCAND])
    if len(cand) < 2:
        return None
    if obs_dict.get("search_begin_input") is None:
        QSTATS["no_sbi"] += 1
        return None
    obs_cls = to_observation_class(obs_dict)
    t0 = _time.monotonic()
    bud = _Bud(_QT_KSTEPS, t0 + _QT_GUARD_S)
    acc = {i: 0.0 for i in cand}
    nev = {i: 0 for i in cand}
    snap = _save_heur_state()
    began = False
    try:
        for _d in range(_QT_NDET):
            if not bud.ok():
                break
            hidden = _determinize(state, me_i, _RNG)
            try:
                root = _cgapi.search_begin(obs_cls, **hidden)
                QSTATS["begin_ok"] += 1
                began = True
            except Exception:
                QSTATS["begin_fail"] += 1
                return None
            rsid = root.searchId
            for idx in cand:
                if not bud.ok():
                    break
                bud.spend()
                sid1, cur = _step(rsid, [idx])
                if sid1 is None:
                    continue
                sid1, cur = _complete_turn(sid1, cur, me_i, bud)
                cs = cur.get("current")
                if (cs is None or (cs.get("result") is not None and cs["result"] >= 0)
                        or cs["yourIndex"] == me_i or not cur.get("select")):
                    acc[idx] += _leaf_eval(cur, me_i); nev[idx] += 1
                    continue
                sid1, cur = _advance_forced(sid1, cur, 1 - me_i, bud)
                cs = cur.get("current"); sl = cur.get("select")
                if cs is None or not sl or sl["context"] != 0 or cs["yourIndex"] == me_i:
                    acc[idx] += _leaf_eval(cur, me_i); nev[idx] += 1
                    continue
                op_ch = _policy(cur)
                worst = None
                for k in range(_QT_KOPP):
                    if not bud.ok():
                        break
                    bud.spend()
                    pick = op_ch[0] if (k == 0 and op_ch) else k
                    if pick >= len(sl["option"]):
                        break
                    sid2, cur2 = _step(sid1, [pick])
                    if sid2 is None:
                        continue
                    sid2, cur2 = _complete_turn(sid2, cur2, 1 - me_i, bud)
                    sid2, cur2 = _advance_forced(sid2, cur2, me_i, bud, limit=6)
                    v = _leaf_eval(cur2, me_i)
                    worst = v if worst is None else min(worst, v)
                if worst is None:
                    worst = _leaf_eval(cur, me_i)
                acc[idx] += worst; nev[idx] += 1
            try:
                _cgapi.search_end()
            except Exception:
                pass
            began = False
        top = order[0]
        ntop = nev.get(top, 0)
        if ntop == 0:
            return None
        ev = [i for i in cand if nev[i] == ntop]
        avg = {i: acc[i] / nev[i] + 1e-6 * scores[i] for i in ev}
        best = max(ev, key=lambda i: avg[i])
        QSTATS["considered"] += 1
        if best == top:
            return None
        QSTATS["overrides"] += 1
        return best
    except Exception:
        QSTATS["fail"] += 1
        return None
    finally:
        if began:
            try:
                _cgapi.search_end()
            except Exception:
                pass
        _restore_heur_state(snap)
        QSTATS["ms"] += (_time.monotonic() - t0) * 1000.0


# ============================== ENTRADA DEL AGENTE ===========================
def agent(obs_dict: dict) -> list[int]:
    base = _heuristic_agent(obs_dict)
    if not _Q_ON:
        return base
    try:
        sel = obs_dict.get("select")
        state = obs_dict.get("current")
        if not sel or state is None:
            return base
        ctx = sel.get("context")
        if ctx not in _Q_CTX:
            return base
        QSTATS["seen3" if ctx == 3 else "seen4"] += 1
        if sel.get("minCount", 1) > 1 or sel.get("maxCount", 1) != 1:
            return base
        n = len(sel["option"])
        if n < 2:
            return base
        scores = _EXPORT.get("scores")
        order = _EXPORT.get("order")
        if not order or scores is None or len(scores) != n:
            QSTATS["skip"] += 1
            return base
        me_i = state["yourIndex"]
        if _Q_COLLECT:
            _collect(obs_dict, sel, state, ctx, me_i, scores, order)
            return base
        if _Q_MODE == "leaf":
            pick = _tree_decide(obs_dict, sel, state, ctx, me_i, scores, order)
            QSTATS["fire3" if ctx == 3 else "fire4"] += 1
            if pick is None or _Q_PLACEBO:
                if pick is not None:
                    QSTATS["discarded"] += 1
                return base
            QSTATS["chg3" if ctx == 3 else "chg4"] += 1
            return [pick]
        # -------- politica destilada: argmax del modelo, sin buscar --------
        vecs, idxs = _qfeats(state, sel, me_i, scores, order, ctx)
        if len(idxs) < 2:
            QSTATS["skip"] += 1
            return base
        QSTATS["cand3" if ctx == 3 else "cand4"] += 1
        qs = [q_pred(v) for v in vecs]
        QSTATS["fire3" if ctx == 3 else "fire4"] += 1
        best_k = max(range(len(idxs)), key=lambda k: (qs[k], -idxs[k]))
        best = idxs[best_k]
        top = order[0]
        if best == top:
            return base
        if top in idxs:
            if qs[best_k] - qs[idxs.index(top)] < _Q_MARGIN:
                return base
        if _Q_PLACEBO:
            QSTATS["discarded"] += 1
            return base
        QSTATS["chg3" if ctx == 3 else "chg4"] += 1
        return [best]
    except Exception:
        QSTATS["fail"] += 1
        return base


# ------------------------- volcado de contadores -----------------------------
_QTAG = os.environ.get("Q_ARM", 'q4m')


def _flush():
    try:
        if _QC_FH is not None:
            _QC_FH.flush(); _QC_FH.close()
    except Exception:
        pass
    try:
        print("[QSTATS] %s %s" % (_QTAG, _json.dumps(QSTATS)), file=sys.stderr, flush=True)
    except Exception:
        pass
    f = os.environ.get("Q_STATS_FILE")
    if f:
        try:
            s = dict(QSTATS); s["tag"] = _QTAG; s["pid"] = os.getpid()
            with open(f, "a") as fh:
                fh.write(_json.dumps(s) + "\n")
        except Exception:
            pass


if _Q_ON:
    import atexit as _atexit
    _atexit.register(_flush)


# ---- MODELO DESTILADO (generado por entrena.py) ----
"""Modelo Q(s,a) destilado de la busqueda. Generado por entrena.py.
NO editar a mano. Puro Python: sin numpy/sklearn en runtime."""
QINTER = [('c_is_mega', 'h_mega'), ('c_is_riolu', 'h_mega'), ('c_is_riolu', 'g_turn'), ('c_is_mega', 'g_turn'), ('c_is_hariyama', 'g_turn'), ('c_is_mega', 'o_dmg'), ('c_is_riolu', 'o_dmg'), ('c_is_solrock', 'o_dmg'), ('c_en', 'g_turn'), ('c_en', 'o_hp'), ('c_hp', 'o_dmg'), ('c_can_atk', 'o_hp'), ('c_dmg_now', 'o_hp'), ('c_ko_opp', 'o_prize'), ('c_survives', 'o_dmg'), ('c_prize', 'g_my_prz'), ('c_prize', 'g_op_prz'), ('c_is_mega', 'g_my_prz'), ('c_is_mega', 'g_op_prz'), ('c_is_mega', 'g_ctx'), ('c_is_riolu', 'g_ctx'), ('c_en', 'g_ctx'), ('c_hpfrac', 'o_dmg'), ('c_retreat', 'g_turn'), ('c_deficit', 'g_turn'), ('c_is_solrock', 'g_turn'), ('c_is_lunatone', 'g_turn'), ('c_dmg_full', 'o_hp'), ('c_is_mega', 'g_bench'), ('c_en', 'h_energy')]
QFIJO = {678: 0.038652076396596534, 677: 0.00044516885126524847, 676: 0.025929826774302108, 675: -0.029391582061019286, 674: -0.015353065070391326, 673: -0.020028929403929398}
_IX = {n: i for i, n in enumerate(QFEAT_NAMES)}


def _expande(v):
    out = list(v)
    for a, b in QINTER:
        out.append(v[_IX[a]] * v[_IX[b]])
    return out
QTREES = [
  ([70, 34, 73, -1, -1, 39, -1, -1, 34, 70, -1, -1, 34, -1, -1],[0.0, 10.0, 37800.0, 0.0, 0.0, 60.0, 0.0, 0.0, 12.0, 18.0, 0.0, 0.0, 19.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.00404002178056085, 0.008796281325788196, 0.0, 0.001277122419526887, 0.0030889748549323016, 0.0, 0.0, -0.002143530419880034, -0.00526515713932195, 0.0, -0.0018007694196611277, -0.0006188080572005782]),
  ([15, 28, 39, -1, -1, 17, -1, -1, 51, 39, -1, -1, 41, -1, -1],[10.0, 9.0, 50.0, 0.0, 0.0, 1.0, 0.0, 0.0, 210.0, 10.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00047438058263451225, -0.0013770328752114715, 0.0, -0.0006424573669131275, -0.003659899809523056, 0.0, 0.0, 0.0012417431358052512, 0.002855081321924655, 0.0, 0.012673501627879964, 0.00589808056865274]),
  ([0, 30, 61, -1, -1, 30, -1, -1, 30, 63, -1, -1, 51, -1, -1],[0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.007855056352274084, -0.0019426802125327327, 0.0, -0.0018315534302336476, -0.0005781846392261405, 0.0, 0.0, 0.002658606416222472, 0.010608647888404486, 0.0, 0.0013391179764008772, 0.005898053889568518]),
  ([70, 38, 26, -1, -1, 26, -1, -1, 45, 38, -1, -1, 63, -1, -1],[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0020418783730605976, -0.00016494331182549343, 0.0, 0.00615974951830674, -0.00010747888190770564, 0.0, 0.0, -0.001278486754681296, -0.00368394270342803, 0.0, 0.0003742482105055164, 0.0020633264316347595]),
  ([15, 62, 70, -1, -1, 34, -1, -1, 62, 63, -1, -1, 60, -1, -1],[50.0, 1.0, 15.0, 0.0, 0.0, 8.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 210.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0007170798982698483, -0.005876128642754138, 0.0, -0.003281447518594318, -0.0005079986042350384, 0.0, 0.0, 0.0029294276711242526, 0.00992609670176457, 0.0, 0.0013038208786017266, 0.006801701392811631]),
  ([43, 30, 29, -1, -1, 37, -1, -1, 30, 43, -1, -1, 37, -1, -1],[5.0, 1.0, 3.0, 0.0, 0.0, 320.0, 0.0, 0.0, 1.0, 107.0, 0.0, 0.0, 320.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00590830014474474, -0.0005282808514730965, 0.0, -0.000560409027496394, -0.0018784066037288685, 0.0, 0.0, 0.004867270077312367, 0.017409019703880212, 0.0, 0.0008823767932752462, 0.0036513160286201583]),
  ([43, 62, 45, -1, -1, 45, -1, -1, 26, 62, -1, -1, 39, -1, -1],[5.0, 1.0, -8.0, 0.0, 0.0, -12.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 130.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.005931513869110402, 5.354206073397308e-05, 0.0, -0.0012690416311635724, 0.00011808031322543728, 0.0, 0.0, 0.004329300796557472, 0.001439123274405629, 0.0, 3.051980430351962e-05, -0.0010740103537087496]),
  ([57, 45, 34, -1, -1, 61, -1, -1, 31, 34, -1, -1, 35, -1, -1],[0.0, -6.0, 14.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 14.0, 0.0, 0.0, 320.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0031619625116476716, -0.0008914250586090044, 0.0, 0.0003213622662097036, 0.0024152434329779856, 0.0, 0.0, 0.005189460102307351, 0.0016279440751444593, 0.0, 0.00019451070395837847, 0.0026956334032157995]),
  ([15, 30, 70, -1, -1, 20, -1, -1, 31, 26, -1, -1, 57, -1, -1],[30.0, 1.0, 18.0, 0.0, 0.0, 0.3889, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 320.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0002921039808959025, -0.005641119491001973, 0.0, -0.00020454651395579543, -0.0010089568992498446, 0.0, 0.0, 0.004255558061957513, -0.00035095062251091054, 0.0, 0.0002181841129430232, 0.004270763454439614]),
  ([58, 45, 26, -1, -1, 64, -1, -1, 26, 38, -1, -1, 20, -1, -1],[3500.0, -6.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.625, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0014662393693754873, 0.00041441083514508247, 0.0, 0.0003480080943978798, 0.0018800198616560275, 0.0, 0.0, 0.0014870408266186577, 0.005179070176275884, 0.0, -0.0004325459479010131, -0.0019424145083899787]),
  ([43, 30, 74, -1, -1, 35, -1, -1, 30, 2, -1, -1, 57, -1, -1],[2.0, 1.0, 2.0, 0.0, 0.0, 320.0, 0.0, 0.0, 1.0, 150.0, 0.0, 0.0, 320.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.004533792599220921, 0.005385708871836234, 0.0, -0.0004992658899364642, -0.0021358582603656184, 0.0, 0.0, 0.0019136772271998196, 0.008641845093574725, 0.0, 0.0004779747936962317, 0.0032733982937138504]),
  ([15, 26, 20, -1, -1, 28, -1, -1, 26, 38, -1, -1, 69, -1, -1],[50.0, 0.0, 0.6364, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 6.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0003378680828564705, -0.001781942772745446, 0.0, -0.00015256859722215176, 0.0005743156114843573, 0.0, 0.0, 0.0015575985250802966, 0.005825518811977054, 0.0, 0.0014529572042841952, -0.000899183269038913]),
  ([15, 34, 26, -1, -1, 20, -1, -1, 26, 59, -1, -1, 73, -1, -1],[30.0, 12.0, 0.0, 0.0, 0.0, 0.3889, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 86400.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.003413921716912678, 0.0006435093854785209, 0.0, -1.8619472991238206e-05, -0.0006453643223349669, 0.0, 0.0, 0.0016474600348740777, 0.00846004184954751, 0.0, -0.0005093604617405652, -0.0029524113403557282]),
  ([44, 34, 59, -1, -1, 56, -1, -1, 34, 31, -1, -1, 26, -1, -1],[1.0, 10.0, 1.0, 0.0, 0.0, 4200.0, 0.0, 0.0, 14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.002494223629982693, 0.008489079642918599, 0.0, 0.0002130804707291971, 0.0013571177243691538, 0.0, 0.0, -0.003966640481618548, -0.000690515849929455, 0.0, -0.0008213860146774915, 0.0005598981553518756]),
  ([17, 31, 51, -1, -1, 51, -1, -1, 62, 31, -1, -1, 62, -1, -1],[0.0, 1.0, 10.0, 0.0, 0.0, 200.0, 0.0, 0.0, 1.0, -2.0, 0.0, 0.0, 2.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0012041483630283035, 0.0032308930406280308, 0.0, -0.0003195085823748897, 0.005840889962079139, 0.0, 0.0, -0.0016992725103081021, -0.0056482846342744905, 0.0, -0.001272890065581475, -0.0001638307856057146]),
  ([43, 28, 52, -1, -1, 17, -1, -1, 43, 8, -1, -1, 56, -1, -1],[4.0, 11.0, 60.0, 0.0, 0.0, 1.0, 0.0, 0.0, 107.0, 0.0, 0.0, 0.0, 4200.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0006742719268218333, 0.0008668917727228186, 0.0, -0.000596466785173992, -0.002895781738848628, 0.0, 0.0, 0.000726728420534009, -0.0007020155769786568, 0.0, 0.0006254482040778455, 0.005253165656093569]),
  ([45, 20, 63, -1, -1, 34, -1, -1, 70, 34, -1, -1, 47, -1, -1],[-20.0, 0.0909, 5.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 14.0, 0.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00013611202871139928, -0.003835043574253903, 0.0, -0.0032003778615738642, -0.0011051963951523936, 0.0, 0.0, 0.0027641822444453893, 0.0007960118561026105, 0.0, -4.5057674170759914e-05, 0.002076345395506806]),
  ([44, 38, 56, -1, -1, 21, -1, -1, 38, 9, -1, -1, 44, -1, -1],[1.0, 2.0, 7700.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.00043134988581582643, 0.001219072824364748, 0.0, 0.0013746314076939377, 0.0037470927774054977, 0.0, 0.0, -0.0006313839568403827, 0.0006689596289169222, 0.0, -0.0006251372207854194, -0.0030372484993249293]),
  ([43, 34, 30, -1, -1, 34, -1, -1, 30, 43, -1, -1, 68, -1, -1],[5.0, 12.0, 1.0, 0.0, 0.0, 18.0, 0.0, 0.0, 1.0, 107.0, 0.0, 0.0, 200.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0048898523480257125, -0.001845750871507088, 0.0, -0.0011218598899196692, -0.00026896631471072426, 0.0, 0.0, 0.0032175066885410334, 0.014551833598708631, 0.0, 0.0003415678205114223, 0.002256280174543145]),
  ([58, 34, 70, -1, -1, 13, -1, -1, 34, 48, -1, -1, 61, -1, -1],[3500.0, 15.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 7.0, 0.0, 0.0, 12.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0003415718682766182, -0.00229160214590255, 0.0, -0.00014479817661391904, 0.0023200202182928485, 0.0, 0.0, 0.002515942377369377, -0.0021397491346940526, 0.0, 0.0003079437255995033, 0.0018298091882884384]),
  ([43, 31, 35, -1, -1, 35, -1, -1, 31, 43, -1, -1, 57, -1, -1],[2.0, 1.0, 80.0, 0.0, 0.0, 320.0, 0.0, 0.0, 1.0, 107.0, 0.0, 0.0, 320.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.001949896634533993, -0.0005874531431618183, 0.0, 0.00042330782855449215, -0.0028351025583460312, 0.0, 0.0, 0.0006227581831139421, 0.003233089353370489, 0.0, -0.0003879858034891608, 0.002500419051426093]),
  ([15, 62, 66, -1, -1, 30, -1, -1, 30, 62, -1, -1, 35, -1, -1],[30.0, 2.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 320.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0017674439000271618, 0.0019367443230953693, 0.0, 0.012358645447774425, -0.00018984134181293904, 0.0, 0.0, 0.0016960351936310455, 0.007385893761297543, 0.0, 0.00045236422212508435, 0.00240290642312823]),
  ([45, 26, 19, -1, -1, 34, -1, -1, 34, 26, -1, -1, 61, -1, -1],[-20.0, 0.0, 0.0, 0.0, 0.0, 19.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.002286570967454857, -0.0005441999373966441, 0.0, 0.001551756683035786, 0.0005484809086834312, 0.0, 0.0, 0.002948439163331918, -0.0004220425777776923, 0.0, 0.00030808537379883465, 0.0017579999512716568]),
  ([44, 30, 49, -1, -1, 39, -1, -1, 30, 9, -1, -1, 39, -1, -1],[1.0, 1.0, 9.0, 0.0, 0.0, 200.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0012635442915230305, 0.006836115754457645, 0.0, 0.00016721578845905552, 0.0016339620971678132, 0.0, 0.0, -0.0020485341663860697, 0.0013320097452717999, 0.0, 2.4520937222251537e-05, -0.0016640889599766893]),
  ([51, 75, 68, -1, -1, 53, -1, -1, 51, 49, -1, -1, 49, -1, -1],[10.0, 1.0, 50.0, 0.0, 0.0, 10.0, 0.0, 0.0, 210.0, 9.0, 0.0, 0.0, 5.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00011077685042205424, -0.000782693741260921, 0.0, 0.00035748322757028127, 0.0017053505293225118, 0.0, 0.0, 0.0007033716195854202, 0.0022520423002927227, 0.0, 0.010588703746681518, 0.0031484965857900196]),
  ([15, 34, 10, -1, -1, 38, -1, -1, 34, 37, -1, -1, 38, -1, -1],[50.0, 10.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 15.0, 230.0, 0.0, 0.0, 2.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.001075792724486776, -0.0031791386756391572, 0.0, -6.064587183074445e-05, -0.0008660029543573542, 0.0, 0.0, 0.0009053270076048583, 0.002666522413259668, 0.0, -3.257792132616931e-05, 0.0016270496906005433]),
  ([58, 37, 61, -1, -1, 61, -1, -1, 48, 37, -1, -1, 0, -1, -1],[3500.0, 320.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 320.0, 0.0, 0.0, 1.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0006444612227843149, 4.5341397203075395e-05, 0.0, -0.0026276818538866643, -3.018751036607923e-05, 0.0, 0.0, 0.0005213663090797649, 0.0016903353214915975, 0.0, -0.003656046745540084, 0.0008407961108839216]),
  ([54, 34, 30, -1, -1, 34, -1, -1, 30, 49, -1, -1, 33, -1, -1],[0.0, 15.0, 2.0, 0.0, 0.0, 21.0, 0.0, 0.0, 1.0, 11.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0026576678014506847, -0.0001745956849679979, 0.0, -0.0005415098175420588, 3.820803952426399e-05, 0.0, 0.0, 0.0019460519816201316, 0.00802162946397069, 0.0, -0.00018779198409662364, 0.0007326148917102323]),
  ([51, 43, 68, -1, -1, 68, -1, -1, 43, 51, -1, -1, 51, -1, -1],[10.0, 0.0, 50.0, 0.0, 0.0, 180.0, 0.0, 0.0, 107.0, 200.0, 0.0, 0.0, 210.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -9.176266426174628e-05, -0.0010487124930388682, 0.0, 6.565167679955775e-05, 0.0010627509057956, 0.0, 0.0, 0.0004739952439370598, 0.00249570498491538, 0.0, 0.003493560694982103, 0.007129129180680814]),
  ([43, 34, 40, -1, -1, 39, -1, -1, 59, 34, -1, -1, 43, -1, -1],[2.0, 10.0, 600.0, 0.0, 0.0, 40.0, 0.0, 0.0, 2.0, 15.0, 0.0, 0.0, 107.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0010041295816790967, -0.003376601017190081, 0.0, 3.6127837571401295e-05, -0.0007679587509685514, 0.0, 0.0, 0.001089595419861564, 0.0002496294380649894, 0.0, 0.002088978153129733, 0.012993225846872356]),
  ([15, 62, 47, -1, -1, 38, -1, -1, 62, 2, -1, -1, 38, -1, -1],[30.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 250.0, 0.0, 0.0, 2.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0021448381317557335, 0.004863535088267735, 0.0, 7.933529063117813e-05, -0.0009799754876610549, 0.0, 0.0, 0.001366293104958474, 0.004759896269611717, 0.0, -9.3055976695696e-05, 0.002550777758406622]),
  ([70, 26, 51, -1, -1, 51, -1, -1, 26, 39, -1, -1, 28, -1, -1],[2.0, 0.0, 210.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 7.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0006525767677511346, 0.004488695987596364, 0.0, -0.0008757084350731575, -0.004490466036082107, 0.0, 0.0, -3.987582495378628e-05, -0.0009693579814770237, 0.0, 0.00019285865321012714, 0.00125186518048924]),
  ([62, 54, 31, -1, -1, 0, -1, -1, 59, 68, -1, -1, 31, -1, -1],[1.0, 7.0, -2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 200.0, 0.0, 0.0, -3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00033333917016056206, -0.004288311387749207, 0.0, 0.00041782503605455136, 0.006835385277790228, 0.0, 0.0, 0.00016565036726003457, -0.00043151211590308867, 0.0, -0.0018328429177755865, 0.0037102802580418703]),
  ([72, 49, 71, -1, -1, 34, -1, -1, 34, 68, -1, -1, 34, -1, -1],[9.0, 9.0, 8.0, 0.0, 0.0, 8.0, 0.0, 0.0, 15.0, 147.276, 0.0, 0.0, 23.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -5.519924202792407e-05, 0.0007021532825652527, 0.0, 0.002588306727417362, 0.0005274644256560695, 0.0, 0.0, -0.0015429391490211226, -0.0035121015834357534, 0.0, -0.000461203914967417, 0.0009481621402370739]),
  ([43, 66, 48, -1, -1, 36, -1, -1, 39, 28, -1, -1, 43, -1, -1],[20.0, 3.0, 7.0, 0.0, 0.0, 2.0, 0.0, 0.0, 210.0, 4.0, 0.0, 0.0, 107.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00024914136487179205, -0.0026147405263163568, 0.0, 0.0014914685920276378, -6.984462572003338e-05, 0.0, 0.0, 0.0018042536974345951, 0.00043649945526825855, 0.0, 0.0012060330304615203, 0.00734681008130723]),
  ([72, 1, 17, -1, -1, 64, -1, -1, 28, 17, -1, -1, 41, -1, -1],[7.0, 340.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 12.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0003551862050695827, -5.085543716237418e-05, 0.0, 0.007208670330566096, 0.0008922796773197327, 0.0, 0.0, 0.0017698839845637152, -0.0006032393283287225, 0.0, -0.0015004894761801858, -0.003962490539496653]),
  ([45, 62, 9, -1, -1, 39, -1, -1, 62, 41, -1, -1, 54, -1, -1],[-17.0, 1.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 4.0, 3.0, 0.0, 0.0, 7.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.003946005610843936, 0.001920254346969255, 0.0, -0.00015287046130825127, -0.0022564326109475365, 0.0, 0.0, 0.004564830630576055, 0.0005593552758592739, 0.0, 0.0004869395541252823, -0.00033907730758431573]),
  ([59, 15, 26, -1, -1, 26, -1, -1, 26, 62, -1, -1, -1],[2.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1],[0.0, 0.0, 0.0, -0.00038033874591734447, 0.0007920838584969984, 0.0, 0.0006996258629368168, -0.0014721629260647149, 0.0, 0.0, 0.007903860158759768, 0.0031066875270429816, -0.0030303918109715953]),
  ([45, 31, 61, -1, -1, 73, -1, -1, 31, 74, -1, -1, 57, -1, -1],[-20.0, 0.0, 1.0, 0.0, 0.0, 16000.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.007583574650761924, -0.000655855741113129, 0.0, 0.0004208418943543445, -0.0004650266981549906, 0.0, 0.0, 0.0004426542161504469, 0.002379260831991992, 0.0, 0.000459914880647023, -0.0006040139031968314]),
  ([44, 34, 16, -1, -1, 66, -1, -1, 34, 26, -1, -1, 56, -1, -1],[2.0, 12.0, 50.0, 0.0, 0.0, 3.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 4500.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0008210718833144704, 0.001563795249545741, 0.0, 9.984939216894699e-05, 0.0008072881704976945, 0.0, 0.0, -0.004628535994496824, 0.0025271235621130345, 0.0, 0.00020350514371885085, -0.0010825084206576617]),
  ([62, 66, 16, -1, -1, 39, -1, -1, 30, 65, -1, -1, 69, -1, -1],[1.0, 3.0, 50.0, 0.0, 0.0, 210.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 14.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.003100969487657111, -0.0003034803316700352, 0.0, 0.004705888201823171, 0.0005572150015213222, 0.0, 0.0, 0.010196284797106669, 0.0036343811783726434, 0.0, 0.0001702093942609501, -0.00036935357535096284]),
  ([57, 31, 38, -1, -1, 58, -1, -1, 31, 58, -1, -1, 57, -1, -1],[50.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 62100.0, 0.0, 0.0, 320.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00024284341057627097, -0.0008735373765376098, 0.0, 0.0007634998838008341, -0.0008289716032620387, 0.0, 0.0, 0.000812582159853976, 0.0025303404566415417, 0.0, -0.0005900192694797995, 0.0011529476780716985]),
  ([72, 30, 65, -1, -1, 73, -1, -1, 30, 29, -1, -1, 73, -1, -1],[9.0, 1.0, 0.0, 0.0, 0.0, 86400.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 20300.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.00016093158294927572, 0.004234540748746134, 0.0, -8.01769376830029e-05, 0.0012925337854846192, 0.0, 0.0, -0.005782994463391078, 0.0007241324531541224, 0.0, 0.00019665080660615829, -0.008650185530799312]),
  ([75, 62, 47, -1, -1, 64, -1, -1, 39, 62, -1, -1, 36, -1, -1],[1.0, 1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 10.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0016178638366958883, 0.004661321264766522, 0.0, -4.5746947949208664e-05, -0.001118439201731163, 0.0, 0.0, 0.001122596280591324, -0.0002593814063631124, 0.0, 0.0023661936964578995, 0.0006977915515664323]),
  ([71, 62, 40, -1, -1, 2, -1, -1, 62, 22, -1, -1, 70, -1, -1],[8.0, 1.0, 500.0, 0.0, 0.0, 340.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0011164251312977187, -0.0025679791320736022, 0.0, -9.061410452847937e-05, 0.0007961844953767715, 0.0, 0.0, 0.000520806135203224, 0.002322722678034344, 0.0, -0.0011376298507582685, 0.0009189299435817907]),
  ([51, 26, 17, -1, -1, 17, -1, -1, 26, 63, -1, -1, -1],[200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1],[0.0, 0.0, 0.0, 0.00037274594714778214, -0.00033893623042755775, 0.0, -0.0012269629549561708, 0.0009404673784033528, 0.0, 0.0, 0.00486200594789404, 0.0009728068655316623, -0.0036537822811794434]),
  ([45, 62, 26, -1, -1, 56, -1, -1, 56, 67, -1, -1, 43, -1, -1],[-20.0, 1.0, 0.0, 0.0, 0.0, 4500.0, 0.0, 0.0, 4500.0, 3.0, 0.0, 0.0, 107.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.004704024397418129, 0.003551100146878404, 0.0, 0.00010825407572672381, -0.000832659244317306, 0.0, 0.0, 0.0001289633888511541, -0.0005158569057042345, 0.0, 0.00038515428362900556, 0.0028846453307369573]),
  ([59, 75, 69, -1, -1, 30, -1, -1, 34, -1, 32, -1, -1],[2.0, 3.0, 11.0, 0.0, 0.0, 1.0, 0.0, 0.0, 8.0, 0.0, 4.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, -1, 11, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 10, -1, 12, -1, -1],[0.0, 0.0, 0.0, 8.645996315762929e-05, -0.0002657033395982934, 0.0, 0.0038211304733286908, 0.00017988357957192337, 0.0, 0.010557789570742958, 0.0, -0.0016631124279079868, 0.0025031559698188615]),
  ([44, 62, 49, -1, -1, 68, -1, -1, 62, 68, -1, -1, 68, -1, -1],[2.0, 4.0, 9.0, 0.0, 0.0, 30.0, 0.0, 0.0, 1.0, 200.0, 0.0, 0.0, 200.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0004209962356383375, 0.004271698006753768, 0.0, -0.00022738337076944816, 0.00037630908470815164, 0.0, 0.0, -0.005292492934250284, 0.00023469277959963888, 0.0, 1.4463226998803159e-05, -0.0017386004466729645]),
  ([62, 66, 1, -1, -1, 32, -1, -1, 30, 56, -1, -1, 62, -1, -1],[1.0, 3.0, 70.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 55800.0, 0.0, 0.0, 15.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0066558169760260015, -0.0010419508495968647, 0.0, 0.0014537926321105809, 0.003882464835823709, 0.0, 0.0, 0.004714236574914639, 0.00032700412919099513, 0.0, 7.715957240404191e-05, -0.0008837065452304585]),
  ([17, 30, 64, -1, -1, 28, -1, -1, 62, 35, -1, -1, 62, -1, -1],[1.0, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 1.0, 320.0, 0.0, 0.0, 4.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0009262657628766608, 0.003895759375313959, 0.0, 7.858213381360678e-05, -0.0005901937339069475, 0.0, 0.0, -0.0031100291287944576, 0.0022383510356782923, 0.0, -0.0004486409778481136, 0.00033093666758061946]),
  ([45, 44, 45, -1, -1, 39, -1, -1, 43, 70, -1, -1, 39, -1, -1],[-8.0, 2.0, -22.0, 0.0, 0.0, 200.0, 0.0, 0.0, 107.0, 4.0, 0.0, 0.0, 10.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0001518932682756517, 0.00039555374659127533, 0.0, -0.0003517385712147099, -0.002257679101629399, 0.0, 0.0, -5.321130271633504e-05, 0.0006568411906160664, 0.0, 0.00016981774998140188, 0.002684827288339179]),
  ([59, 73, 35, -1, -1, 74, -1, -1, 69, 42, -1, -1, -1],[2.0, 86400.0, 320.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 4.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1],[0.0, 0.0, 0.0, 4.880234027984357e-05, -0.0003741019860279861, 0.0, -0.0009606944205472756, 0.0023764543730430275, 0.0, 0.0, -0.0008151911457956639, 0.0038260503097484544, 0.010217913047181407]),
  ([45, 30, 70, -1, -1, 45, -1, -1, 34, 30, -1, -1, 55, -1, -1],[-20.0, 4.0, 6.0, 0.0, 0.0, -103.0, 0.0, 0.0, 14.0, 2.0, 0.0, 0.0, 140.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0003729722921425827, -0.0011311238226566428, 0.0, 0.00029430620872003104, -0.0009284709320693756, 0.0, 0.0, 0.0017310714476685624, -0.00016508016137186188, 0.0, 0.00031274827501062566, -0.00039298969389131]),
  ([9, 62, 40, -1, -1, 30, -1, -1, 32, 40, -1, -1, 62, -1, -1],[0.0, 1.0, 570.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 370.0, 0.0, 0.0, 4.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0002377398249353876, -0.0024438216769338233, 0.0, 0.003491774356322622, -0.0001363211489324273, 0.0, 0.0, -0.0008530176902251033, 0.00012211601846342013, 0.0, 0.0015819868198852533, 0.0001771768514245441]),
  ([44, 34, 64, -1, -1, 50, -1, -1, 30, 64, -1, -1, 36, -1, -1],[1.0, 8.0, 2.0, 0.0, 0.0, 10.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.002471949999951171, -0.0014134436596072901, 0.0, 0.00019005258620908167, -0.001490453422372186, 0.0, 0.0, -0.0006572727917287028, -0.0038243429732376655, 0.0, -7.680001926226507e-05, 0.0008280541244869513]),
  ([44, 17, 28, -1, -1, 28, -1, -1, 38, 28, -1, -1, 28, -1, -1],[2.0, 2.0, 9.0, 0.0, 0.0, 6.0, 0.0, 0.0, 2.0, 8.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -1.5897402098360027e-05, 0.0003237970249653501, 0.0, -0.00010091335582556376, 0.0017595838160497186, 0.0, 0.0, 0.0003260948454559155, -0.0006392959793425746, 0.0, 0.001280309065915105, -0.0019294823223133116]),
  ([1, 27, 54, -1, -1, 61, -1, -1, 27, 51, -1, -1, 40, -1, -1],[340.0, 8.0, 16.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 10.0, 0.0, 0.0, 710.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00010119786911768598, 0.0003980865984773453, 0.0, 0.0005830419211590995, -0.0007431114483705237, 0.0, 0.0, 0.0006630321938535939, 0.002734556067378272, 0.0, 0.0019161866635664824, -0.0005957260741737688]),
  ([43, 64, 64, -1, -1, 28, -1, -1, 19, -1, 51, -1, -1],[107.0, 1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, -1, 11, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 10, -1, 12, -1, -1],[0.0, 0.0, 0.0, -7.090382156054677e-05, 0.002428740660113984, 0.0, -0.008759879958564025, -0.000530839756051591, 0.0, 0.0061659313261164354, 0.0, -0.0003915037831018083, 0.0020895261821584827]),
  ([45, 20, 61, -1, -1, 47, -1, -1, 47, 32, -1, -1, 45, -1, -1],[-20.0, 0.2143, 15.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, -4.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 1.0621537136646744e-05, -0.0038073484395560594, 0.0, -0.0008258130962951155, -0.006050710048625099, 0.0, 0.0, -0.00013011043328684945, 0.0005072023252864719, 0.0, 0.003721659975605015, 0.0011701035804477962]),
  ([64, 64, 60, -1, -1, 60, -1, -1, 69, 60, -1, -1, 74, -1, -1],[1.0, 0.0, 70.0, 0.0, 0.0, 200.0, 0.0, 0.0, 12.0, 210.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 4.954387326435492e-05, -0.0016986260852921244, 0.0, 0.004120963242222992, -0.001610961057090135, 0.0, 0.0, 0.0004477729021249832, 0.005140364479286087, 0.0, -0.0021041246018845467, -0.0006333734552287368]),
  ([52, 43, 64, -1, -1, 39, -1, -1, 66, 73, -1, -1, 52, -1, -1],[30.0, 24.0, 1.0, 0.0, 0.0, 10.0, 0.0, 0.0, 3.0, 9600.0, 0.0, 0.0, 130.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -9.603464251401712e-06, -0.0005611123448074406, 0.0, -0.00030422453221117657, 0.001788478325011418, 0.0, 0.0, -7.042518548850793e-05, -0.005586724560401565, 0.0, 0.0029018305241633193, 0.0009951499389864397]),
  ([45, 26, 9, -1, -1, 70, -1, -1, 26, 57, -1, -1, 70, -1, -1],[-6.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 320.0, 0.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0006089543920860084, 0.0006869993841260117, 0.0, -0.0007217736212003612, 0.0014117923835748666, 0.0, 0.0, 0.0004963168257979598, 0.0021613587105330226, 0.0, -0.0018059416455738723, 0.00017318628638638073]),
  ([62, 66, 15, -1, -1, 47, -1, -1, 57, 62, -1, -1, 62, -1, -1],[1.0, 3.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 320.0, 6.0, 0.0, 0.0, 6.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0015699483296042678, 0.0006004119749610963, 0.0, 0.0007438120618474549, 0.005865309806308565, 0.0, 0.0, 8.173172818593455e-05, -0.0005507075865640595, 0.0, 3.2326231349467925e-05, 0.003762751967120651]),
  ([45, 68, 65, -1, -1, 53, -1, -1, 68, 53, -1, -1, 75, -1, -1],[-6.0, 50.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 50.0, 130.0, 0.0, 0.0, 1.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.00012076720359891678, -0.0007189012192369215, 0.0, -0.000844809614382319, 0.0006401140862884915, 0.0, 0.0, 8.13376508780042e-05, -0.006562241573305521, 0.0, 0.0003766647829731001, 0.0012656247794645412]),
  ([42, 57, 57, -1, -1, 53, -1, -1, 53, 58, -1, -1, 30, -1, -1],[4.0, 110.0, 50.0, 0.0, 0.0, 200.0, 0.0, 0.0, 60.0, 32400.0, 0.0, 0.0, 4.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -7.824969745712755e-05, 0.0006311785047887206, 0.0, -0.0006023837513680157, -0.002569484200012409, 0.0, 0.0, 7.69808835124942e-05, 0.0008402446180855125, 0.0, 0.001716802702255259, -0.00021356981075575776]),
  ([59, 45, 30, -1, -1, 0, -1, -1, 34, -1, 45, -1, -1],[2.0, -22.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 8.0, 0.0, -17.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, -1, 11, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 10, -1, 12, -1, -1],[0.0, 0.0, 0.0, -0.0021459783112897537, -0.0001419775934454069, 0.0, 0.00019577430220873768, -0.0009682361877915929, 0.0, 0.008621028624628739, 0.0, 0.007718325610541067, 0.0006795329771785077]),
  ([9, 1, 52, -1, -1, 26, -1, -1, 26, 27, -1, -1, 54, -1, -1],[0.0, 340.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 9.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00013688723687360694, 0.0005270603002871971, 0.0, 0.0010180597897278488, -0.0009572966599218697, 0.0, 0.0, 0.00039236312161381414, 0.0019304777248693834, 0.0, -0.0006300732220201774, -0.0021579638690820612]),
  ([45, 26, 58, -1, -1, 43, -1, -1, 43, 55, -1, -1, 26, -1, -1],[-12.0, 0.0, 3500.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 140.0, 0.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0005707293714272991, 0.0003115696000910535, 0.0, 0.0013037876527488479, -0.0006947864261601449, 0.0, 0.0, 0.000585668757486328, -0.0004801233388577432, 0.0, 0.00020046617050084248, -0.0015519178044153903]),
  ([47, 45, 64, -1, -1, 62, -1, -1, 62, 45, -1, -1, 36, -1, -1],[0.0, -8.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.0, -22.0, 0.0, 0.0, 2.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -3.678088091277235e-05, -0.001015533447720161, 0.0, 0.0006660733700549171, -4.666547950425529e-05, 0.0, 0.0, -0.00716674520037395, 0.007341537818482586, 0.0, 0.001129733270046954, -0.0004485575663776639]),
  ([45, 30, 49, -1, -1, 64, -1, -1, 70, 30, -1, -1, 8, -1, -1],[-8.0, 1.0, 9.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.001564101435400189, 0.005085704539552614, 0.0, -6.740624300958156e-05, -0.0009806501524523795, 0.0, 0.0, 0.0004962664432033725, -0.000162839197200836, 0.0, 0.000515664120585125, 0.0013369207748180055]),
  ([71, 34, 15, -1, -1, 57, -1, -1, 34, 26, -1, -1, 39, -1, -1],[8.0, 18.0, 30.0, 0.0, 0.0, 140.0, 0.0, 0.0, 18.0, 0.0, 0.0, 0.0, 210.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0006341651002502427, 0.000443595369584294, 0.0, 0.0002787653388825156, -0.0005653074582088347, 0.0, 0.0, 0.0019732563260354764, -0.0018103608538604958, 0.0, -0.0007512769152591741, 0.0007986141305171351]),
  ([59, 1, 31, -1, -1, 31, -1, -1, 43, 45, -1, -1, -1],[2.0, 340.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.0, -2.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1],[0.0, 0.0, 0.0, 0.0007350020819704602, -1.701627225109615e-05, 0.0, 0.002291900161356107, -0.0008959475510250485, 0.0, 0.0, 0.0034937055172994482, -0.001052045943115395, 0.009949000811508088]),
  ([47, 41, 55, -1, -1, 26, -1, -1, 41, 27, -1, -1, 27, -1, -1],[0.0, 3.0, 320.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 1.2124814561079198e-05, 0.0014933140009983994, 0.0, -0.0002295005231333173, 7.70124375226776e-05, 0.0, 0.0, 0.00281757811727528, -0.0018947163186950348, 0.0, -0.0015037534963746044, 0.001736230355426682]),
  ([8, 45, 26, -1, -1, 26, -1, -1, 35, 40, -1, -1, 45, -1, -1],[0.0, -8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 110.0, 370.0, 0.0, 0.0, -103.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00020037138317518114, 0.0007475248372389553, 0.0, 0.0004832484529464582, -0.00040292875893719276, 0.0, 0.0, -0.0010532739036015822, 0.000921786943823866, 0.0, -0.0020414972467325578, -0.00038146013871818594]),
  ([43, 37, 40, -1, -1, 35, -1, -1, 37, 53, -1, -1, 35, -1, -1],[24.0, 260.0, 790.0, 0.0, 0.0, 170.0, 0.0, 0.0, 210.0, 10.0, 0.0, 0.0, 240.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 6.065253056148501e-05, 0.0004365796947305289, 0.0, -0.0007674366015198653, 5.677675183223107e-05, 0.0, 0.0, -0.00023059472599125497, 0.00404402740426154, 0.0, 0.003463556160140685, 0.00026444510010133394]),
  ([53, 62, 64, -1, -1, 38, -1, -1, 32, 29, -1, -1, 29, -1, -1],[130.0, 9.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 4.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -2.4077919703228316e-05, -0.0022158479891353057, 0.0, 0.00011909021932460324, 0.003082582226659625, 0.0, 0.0, -0.001046357573387244, -0.0026407081098146334, 0.0, 0.0008333222816930114, -0.0008982907449018609]),
  ([52, 22, 20, -1, -1, 68, -1, -1, 28, -1, 28, -1, -1],[180.0, 1.0, 0.2, 0.0, 0.0, 200.0, 0.0, 0.0, 3.0, 0.0, 13.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, -1, 11, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 10, -1, 12, -1, -1],[0.0, 0.0, 0.0, 0.00012930852911286313, -0.00017357887546148542, 0.0, 0.0005843647776695918, 0.00393232254192955, 0.0, 0.0037156270496134596, 0.0, 0.0006968948498153012, -0.003953950904441744]),
  ([62, 17, 0, -1, -1, 39, -1, -1, 30, 65, -1, -1, 49, -1, -1],[1.0, 1.0, 3.0, 0.0, 0.0, 200.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 5.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -2.519910624732719e-05, 0.005103308552322102, 0.0, -0.003379382456596661, 0.0011038897085637917, 0.0, 0.0, 0.007364349256019773, 0.0012515048635001952, 0.0, 7.546074904143027e-05, -0.000427847725441617]),
  ([61, 38, 31, -1, -1, 53, -1, -1, 31, 29, -1, -1, 53, -1, -1],[3.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 3.0, 0.0, 0.0, 10.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0008640547735954669, 4.547990286425965e-06, 0.0, -0.0017874580654986321, 0.0003350066320897072, 0.0, 0.0, 0.0028908966518431307, 0.00010954662612808925, 0.0, -0.00019298317972016958, 0.001056181023633051]),
  ([1, 44, 31, -1, -1, 31, -1, -1, 31, 29, -1, -1, 20, -1, -1],[340.0, 2.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2955, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.00012082287521148116, -0.0003734807865291236, 0.0, -0.0005095516564815622, 0.0006662381616011449, 0.0, 0.0, 0.010662456272289154, 0.0013432906993562552, 0.0, -0.00016629282497566635, -0.0011939197849411454]),
  ([59, 67, 70, -1, -1, -1, 26, 29, -1, -1, -1],[2.0, 18.0, 18.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, -1, 7, 8, -1, -1, -1],[6, 5, 4, -1, -1, -1, 10, 9, -1, -1, -1],[0.0, 0.0, 0.0, -2.7735880253516956e-05, -0.00039753095962251713, -0.004701739293109784, 0.0, 0.0, 0.009812887371976256, 0.0031665961339601358, -0.004284299997286593]),
  ([69, 60, 25, -1, -1, -1, 73, 38, -1, -1, 73, -1, -1],[13.0, 210.0, 0.0, 0.0, 0.0, 0.0, 4200.0, 2.0, 0.0, 0.0, 16000.0, 0.0, 0.0],[1, 2, 3, -1, -1, -1, 7, 8, -1, -1, 11, -1, -1],[6, 5, 4, -1, -1, -1, 10, 9, -1, -1, 12, -1, -1],[0.0, 0.0, 0.0, 0.00031756199624172454, -2.9317734158981956e-07, 0.0045281898107411974, 0.0, 0.0, 0.0005006077533344544, -0.001372718634228728, 0.0, -0.0006480446060042054, -5.4713096506780936e-05]),
  ([20, 62, 16, -1, -1, 73, -1, -1, 18, 26, -1, -1, 26, -1, -1],[0.2143, 3.0, 210.0, 0.0, 0.0, 54600.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0005531028494642567, 0.00528323891817158, 0.0, 5.6152190184228053e-05, -0.0013478423864408955, 0.0, 0.0, -0.0004316207967575022, 0.00022866037437442412, 0.0, 0.0015513818633830715, -0.0018287680862147711]),
  ([52, 52, 60, -1, -1, 28, -1, -1, 41, 75, -1, -1, 44, -1, -1],[30.0, 10.0, 60.0, 0.0, 0.0, 4.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.00011037812865035836, -0.000427769895198748, 0.0, -0.0021742835760179668, 0.00018301799716766729, 0.0, 0.0, -0.0004855406031799182, -0.004414106631679808, 0.0, -0.00011589352269397791, 0.0016717529602710741]),
  ([44, 56, 51, -1, -1, 65, -1, -1, 25, 29, -1, -1, 56, -1, -1],[2.0, 7700.0, 10.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 4200.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -6.592947433965486e-06, -0.004835502473828994, 0.0, 0.0005743053107779772, -0.0001670435173738146, 0.0, 0.0, -0.0004016078737449628, 0.00033487888612569993, 0.0, -0.0003242879992218182, -0.001604316700458811]),
  ([45, 20, 64, -1, -1, 34, -1, -1, 34, 59, -1, -1, 54, -1, -1],[-20.0, 0.0909, 1.0, 0.0, 0.0, 14.0, 0.0, 0.0, 12.0, 2.0, 0.0, 0.0, 26.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.00038159465825120443, -0.0006831915767696596, 0.0, -0.0014708665228723447, -0.00029258342366066825, 0.0, 0.0, 0.0007603956622476338, 0.003768193504836036, 0.0, 0.00013221265566153026, -0.0011514281854975422]),
  ([43, 30, 44, -1, -1, 70, -1, -1, 30, 69, -1, -1, 69, -1, -1],[107.0, 4.0, 3.0, 0.0, 0.0, 12.0, 0.0, 0.0, 4.0, 14.0, 0.0, 0.0, 12.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -1.7256238050252022e-05, -0.0011219659228948098, 0.0, -2.584700321087051e-05, 0.0007241657761060635, 0.0, 0.0, 0.01488641211558206, 0.0026301158755368944, 0.0, 0.0009197766833080166, -0.0009641712561732838]),
  ([32, 54, 29, -1, -1, 29, -1, -1, 43, 38, -1, -1, 62, -1, -1],[4.0, 5.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0005258707444033123, 0.00015415392440519917, 0.0, -6.015333576522772e-05, -0.0010791215584353912, 0.0, 0.0, 0.0003170704501328365, -0.0013583661267474758, 0.0, 0.0025564250674558154, 0.00021752585921044467]),
  ([59, 0, 2, -1, -1, -1, 41, -1, 32, -1, -1],[2.0, 4.0, 340.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0],[1, 2, 3, -1, -1, -1, 7, -1, 9, -1, -1],[6, 5, 4, -1, -1, -1, 8, -1, 10, -1, -1],[0.0, 0.0, 0.0, -1.9249473014284718e-05, 0.0006753802475493778, -0.0035053850806497267, 0.0, 0.006104490258356656, 0.0, -0.0017015315008026253, 0.0022372554099228193]),
  ([45, 37, 62, -1, -1, 29, -1, -1, 32, 55, -1, -1, 37, -1, -1],[-22.0, 320.0, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 3.0, 380.0, 0.0, 0.0, 320.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0028034598774278195, 0.000120111978991787, 0.0, -0.0022103861802663855, -0.0006086270049905117, 0.0, 0.0, -0.0001492988484100518, -0.0015544991765691842, 0.0, 0.00014852326440781984, 0.0005917713537627]),
  ([44, 30, 73, -1, -1, 30, -1, -1, 30, 73, -1, -1, 56, -1, -1],[1.0, 1.0, 86400.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 16000.0, 0.0, 0.0, 4500.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.001924099845364831, -0.003400201976596661, 0.0, 0.00030345795458699406, -0.00024238790638499686, 0.0, 0.0, -0.0023482432786024357, 0.0005369173226589377, 0.0, 0.00022046149534790484, -0.0004280529660735018]),
  ([23, 45, 26, -1, -1, 26, -1, -1, 13, 71, -1, -1, 40, -1, -1],[0.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 760.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00010967824061308743, 0.0008341506463494118, 0.0, 0.0005607216173381527, -0.0003065882873125179, 0.0, 0.0, -0.00017911796879069444, 0.0007192699600018328, 0.0, 0.00255135519178458, -0.00019810910030058112]),
  ([45, 31, 45, -1, -1, 31, -1, -1, 31, 2, -1, -1, 54, -1, -1],[-17.0, -3.0, -40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 240.0, 0.0, 0.0, 16.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0026334594239613464, 0.0006237759959800816, 0.0, -0.0005644300619204865, 0.0001564125136552586, 0.0, 0.0, 0.00012159821166782575, 0.0008657009933015885, 0.0, -2.885006176034507e-05, -0.0012300734149213075]),
  ([32, 67, 70, -1, -1, 16, -1, -1, 44, 70, -1, -1, 28, -1, -1],[2.0, 3.0, 18.0, 0.0, 0.0, 210.0, 0.0, 0.0, 2.0, 10.0, 0.0, 0.0, 11.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0003440745450916709, 0.0011772191091423494, 0.0, -0.002788100936692496, -0.0007090995038202275, 0.0, 0.0, 2.7328116765271708e-06, 0.0006501892483413248, 0.0, -7.247741875452122e-05, -0.0009669515441083131]),
  ([59, 73, 31, -1, -1, 57, -1, -1, 31, 29, -1, -1, -1],[2.0, 9600.0, 0.0, 0.0, 0.0, 80.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1],[0.0, 0.0, 0.0, -0.00032395513708087964, 0.0002961592608952548, 0.0, 0.00029522744766572395, -0.0001536453396669408, 0.0, 0.0, 0.0036830763489943954, -0.00042153396579000464, -0.003498675830535321]),
  ([43, 31, 62, -1, -1, 17, -1, -1, 31, -1, 31, -1, -1],[24.0, 1.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, -1, 11, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 10, -1, 12, -1, -1],[0.0, 0.0, 0.0, -0.00013705981794166763, -0.0023464988967987033, 0.0, -0.0005525449504629029, 0.0005557646849154884, 0.0, -0.006046880871976147, 0.0, 0.0015700161659183325, -0.0006077200623077516]),
  ([69, 42, 57, -1, -1, 35, -1, -1, 57, 69, -1, -1, 26, -1, -1],[12.0, 3.0, 140.0, 0.0, 0.0, 260.0, 0.0, 0.0, 290.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -5.458841167469763e-05, -0.0014248314473576272, 0.0, 0.00010233093084217074, 0.0005749518481065597, 0.0, 0.0, -0.00044922472737475165, 5.808683666018318e-05, 0.0, -0.00030853629596273966, -0.0031096146377860108]),
  ([30, 74, 47, -1, -1, 26, -1, -1, 48, 58, -1, -1, 30, -1, -1],[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 11900.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00016370652688972188, 0.005044257966020237, 0.0, 0.006164125282590556, -0.0054052899875422136, 0.0, 0.0, 7.93809140553301e-05, -0.0002456579908108447, 0.0, -0.0014936656068382918, -0.0003534862890404263]),
  ([52, 41, 52, -1, -1, 49, -1, -1, 23, 22, -1, -1, 41, -1, -1],[180.0, 3.0, 70.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.00027272330861930546, -0.005317504379452076, 0.0, -8.595250507706517e-05, -0.0007847309722157286, 0.0, 0.0, 0.0003900284516385206, 0.004778238368548296, 0.0, -0.0011616641660450162, 0.003013262167327164]),
  ([66, 62, 3, -1, -1, 62, -1, -1, 68, 62, -1, -1, 2, -1, -1],[0.0, 1.0, 0.5455, 0.0, 0.0, 3.0, 0.0, 0.0, 180.0, 3.0, 0.0, 0.0, 80.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.005230192256661116, -0.00029647900243346623, 0.0, 0.0004386414604122501, -1.2212885167113466e-05, 0.0, 0.0, -0.0016281274047769128, -0.00014981804164061654, 0.0, 0.0011411386746776777, -0.002396714751079686]),
  ([23, 36, 36, -1, -1, 63, -1, -1, 57, 66, -1, -1, -1],[1.0, 5.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 320.0, 3.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1],[0.0, 0.0, 0.0, 2.450753340527703e-05, -0.0002567902834989547, 0.0, 0.0005272072174027798, -0.0019130768242289532, 0.0, 0.0, -0.0006912187034718568, 0.0007743156245431321, 0.0016760109882361957]),
  ([45, 31, 61, -1, -1, 72, -1, -1, 54, 28, -1, -1, 31, -1, -1],[-22.0, 0.0, 1.0, 0.0, 0.0, 10.0, 0.0, 0.0, 7.0, 12.0, 0.0, 0.0, -3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.005799276229864292, -0.00044520395886325765, 0.0, 0.0001272447714417084, 0.001083530347179607, 0.0, 0.0, 0.0003619858429451199, 0.0016671738311332131, 0.0, -0.0013278118501632546, -3.8065989650518126e-05]),
  ([49, 59, 26, -1, -1, 26, -1, -1, 30, 26, -1, -1, 30, -1, -1],[5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -4.818636398837613e-05, 0.0003337224431409292, 0.0, 0.004318592092167573, -0.003443295197385801, 0.0, 0.0, 0.003534219087373221, -0.004838141681712379, 0.0, -0.0013494613266944598, -0.00012188318923092768]),
  ([44, 30, 73, -1, -1, 62, -1, -1, 30, 73, -1, -1, 73, -1, -1],[1.0, 1.0, 86400.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 9600.0, 0.0, 0.0, 10200.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.001565291842067101, -0.0028455818809123646, 0.0, 0.0003929454146604825, -0.00016688046338758934, 0.0, 0.0, -0.0010403465851263824, -0.00015845244242451026, 0.0, -0.0001273319571387055, 0.00033413552570231626]),
  ([45, 31, 59, -1, -1, 43, -1, -1, 31, 29, -1, -1, 17, -1, -1],[-22.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0006122456949067624, 0.0031997645834240737, 0.0, 0.0003949982709302147, -0.00032396582024042875, 0.0, 0.0, 0.0022483522847356117, 0.0001243410458083534, 0.0, -0.0006803716218390409, 0.00024716528029707297]),
  ([45, 38, 63, -1, -1, 71, -1, -1, 63, 45, -1, -1, 64, -1, -1],[-8.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, -2.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 3.3111338959363245e-05, -0.0010647377078603461, 0.0, -0.0015013951408687316, 0.001057391207009132, 0.0, 0.0, 0.0002886781337962009, -0.0001313663354521008, 0.0, -0.002133508262139537, 0.001149077797343968]),
  ([3, 32, 20, -1, -1, 6, -1, -1, 40, 12, -1, -1, 32, -1, -1],[0.5455, 3.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 390.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00657137378715146, -0.0016628866161634292, 0.0, 0.0005004485779438612, -0.0016026286138175014, 0.0, 0.0, 0.0001013669725700069, 0.0011078633778938381, 0.0, -0.0003065214544688197, 9.015014934434858e-05]),
  ([1, 58, 69, -1, -1, -1, 62, 57, -1, -1, 57, -1, -1],[110.0, 32400.0, 30.0, 0.0, 0.0, 0.0, 3.0, 320.0, 0.0, 0.0, 320.0, 0.0, 0.0],[1, 2, 3, -1, -1, -1, 7, 8, -1, -1, 11, -1, -1],[6, 5, 4, -1, -1, -1, 10, 9, -1, -1, 12, -1, -1],[0.0, 0.0, 0.0, -2.9655269015102386e-05, -0.0020404622965728408, -0.002889719225929099, 0.0, 0.0, 0.0009124613434434883, -0.0022455925250501364, 0.0, -4.621655360391961e-05, 0.0013999655357285754]),
  ([45, 30, 45, -1, -1, 1, -1, -1, 30, 41, -1, -1, 53, -1, -1],[-8.0, 1.0, -121.0, 0.0, 0.0, 280.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 210.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.005104067306608557, -0.0004937137850442465, 0.0, 5.592115650942783e-05, -0.000742161590757521, 0.0, 0.0, 0.0035465253912159034, 0.0007160688249813106, 0.0, 0.00022588692309287814, -0.0013813128654304892]),
  ([0, 44, 32, -1, -1, 61, -1, -1, 23, 32, -1, -1, -1],[2.0, 2.0, 4.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1],[0.0, 0.0, 0.0, -8.831874619064136e-05, 0.00032688875783731667, 0.0, -0.0006302487275170837, -9.337228926762252e-06, 0.0, 0.0, -0.00340359699271871, -0.0006573332225400452, 0.0007793727354994589]),
  ([57, 23, 73, -1, -1, 72, -1, -1, 36, 57, -1, -1, 45, -1, -1],[290.0, 0.0, 5500.0, 0.0, 0.0, 9.0, 0.0, 0.0, 1.0, 320.0, 0.0, 0.0, -5.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0002468960505061774, 0.00018188844551713726, 0.0, -0.00011134670749124419, -0.000945603107298823, 0.0, 0.0, -0.0017113749191002732, 0.0013046529777954346, 0.0, -6.322324724639999e-05, -0.001006571087218508]),
  ([44, 30, 61, -1, -1, 70, -1, -1, 30, 61, -1, -1, 37, -1, -1],[1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 3.0, 0.0, 0.0, 150.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.001676758165617834, -6.03783002109889e-05, 0.0, -0.00019821973730133462, 0.0003050405712247023, 0.0, 0.0, -0.0009834723750798211, 5.9481090089953294e-05, 0.0, -0.00018717941722933002, 0.0004370503714042028]),
  ([51, 73, 38, -1, -1, 14, -1, -1, 26, 64, -1, -1, 64, -1, -1],[10.0, 54600.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 5.054277987704753e-05, -0.0003018221218481804, 0.0, -0.00018904753851480783, -0.0018615631534580604, 0.0, 0.0, 0.003853237224707855, 0.00045261534673585185, 0.0, -0.006647675633531933, -0.0007692526076507018]),
  ([62, 62, 15, -1, -1, 54, -1, -1, 68, 39, -1, -1, 39, -1, -1],[9.0, 6.0, 210.0, 0.0, 0.0, 10.0, 0.0, 0.0, 130.0, 10.0, 0.0, 0.0, 210.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00012620065001144246, 0.0007788527511791137, 0.0, -0.003602819959919863, -0.0005807211960794836, 0.0, 0.0, 2.14275125539479e-05, 0.0017043336752735675, 0.0, -0.0008417953642483975, 0.001987638753037332]),
  ([65, 51, 60, -1, -1, 30, -1, -1, 45, -1, 0, -1, -1],[3.0, 10.0, 50.0, 0.0, 0.0, 1.0, 0.0, 0.0, -40.0, 0.0, 2.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, -1, 11, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 10, -1, 12, -1, -1],[0.0, 0.0, 0.0, 7.542865268113934e-05, -0.0005319704286911585, 0.0, 0.007120561413523152, 0.00109476897088453, 0.0, -0.003514611105641093, 0.0, -0.00016405962805583697, -0.00232903195997212]),
  ([52, 37, 52, -1, -1, 74, -1, -1, 37, 26, -1, -1, 29, -1, -1],[60.0, 320.0, 10.0, 0.0, 0.0, 4.0, 0.0, 0.0, 150.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 2.1474803244110817e-05, -0.0006064653598257544, 0.0, -0.0004889170959581423, 0.0014215015112594995, 0.0, 0.0, 0.004392048209754179, -0.0011084169436645338, 0.0, -0.00016947154608851897, 0.0011207183860395776]),
  ([56, 31, 15, -1, -1, 15, -1, -1, 69, 71, -1, -1, 31, -1, -1],[14300.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 0.0, 0.0, 0.0, -4.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0004248946920283921, 0.0004819354352220031, 0.0, 0.0003500215321987598, -0.00031555685601812806, 0.0, 0.0, 0.000649381203437817, -0.00022613680256685804, 0.0, 0.001148175941871724, -0.0003427533297951969]),
  ([75, 25, 57, -1, -1, 48, -1, -1, 27, 57, -1, -1, 61, -1, -1],[1.0, 1.0, 110.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0001573234806374544, -0.00041446796242182454, 0.0, -0.00048047145157020625, 0.00041027229216940585, 0.0, 0.0, 0.0013938062885523023, 0.00038904096959731563, 0.0, 9.501404222131792e-05, -0.0011352727519918825]),
  ([47, 32, 52, -1, -1, 53, -1, -1, 45, 58, -1, -1, 52, -1, -1],[0.0, 4.0, 10.0, 0.0, 0.0, 60.0, 0.0, 0.0, -22.0, 3500.0, 0.0, 0.0, 30.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00021289863155731493, -0.0009647002952986116, 0.0, -4.277108409014463e-05, 0.0008869176196192449, 0.0, 0.0, 0.00014290939017522433, -0.004810110683864029, 0.0, 0.00034054445958213886, 0.002025014827149585]),
  ([64, 61, 60, -1, -1, 71, -1, -1, 14, 31, -1, -1, 54, -1, -1],[5.0, 3.0, 70.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 1.5517356692678253e-05, -0.0013877580705785482, 0.0, 0.00023917788012989608, -0.000260381306722143, 0.0, 0.0, -0.0036801094146268073, 5.318999535411394e-05, 0.0, 0.0005328921539774209, -0.0006871065374476768]),
  ([15, 39, 69, -1, -1, 58, -1, -1, 31, 65, -1, -1, 27, -1, -1],[130.0, 180.0, 13.0, 0.0, 0.0, 11900.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.00012556703695966215, -0.0002336430551201596, 0.0, 0.00044349148744295147, -0.0008393710434784891, 0.0, 0.0, -0.006350684527759524, -0.001389609914683354, 0.0, 0.00018138045593060902, -0.000895397854612592]),
  ([47, 59, 12, -1, -1, 65, -1, -1, 35, 25, -1, -1, 44, -1, -1],[1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 210.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00012170596243775891, 0.0001935122274837034, 0.0, 0.0028944474815259364, -0.0013583385404742341, 0.0, 0.0, 0.0009390751340296531, 0.004128646013753446, 0.0, 0.0007923414707795108, -0.003065544944775839]),
  ([52, 20, 62, -1, -1, 18, -1, -1, 20, -1, 45, -1, -1],[60.0, 0.2, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8667, 0.0, -103.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, -1, 11, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 10, -1, 12, -1, -1],[0.0, 0.0, 0.0, 0.0008284664857650894, -5.05928081377698e-05, 0.0, -0.00021799793231392595, 0.001096779293099738, 0.0, 0.004709035169492309, 0.0, -0.0008308600035303105, 0.0006582890612355764]),
  ([67, 75, 62, -1, -1, 34, -1, -1, 1, 34, -1, -1, -1],[9.0, 3.0, 9.0, 0.0, 0.0, 14.0, 0.0, 0.0, 150.0, 10.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1],[0.0, 0.0, 0.0, 1.5226860805789062e-05, 0.0003686873862792606, 0.0, 0.001762457028499609, -1.595536447834461e-05, 0.0, 0.0, 0.0008352371040573372, -0.0010240606361270421, -0.002710485848693709]),
  ([45, 35, 25, -1, -1, 20, -1, -1, 54, 5, -1, -1, 34, -1, -1],[-8.0, 360.0, 1.0, 0.0, 0.0, 1.2, 0.0, 0.0, 5.0, 1.0, 0.0, 0.0, 8.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -4.194634390122285e-05, -0.0004954310037528584, 0.0, -0.0002574097104032457, 0.0017527781017682184, 0.0, 0.0, 0.0002047671183931215, 0.0011180676777206423, 0.0, 0.0012724774453316764, -0.0004106786531474802]),
  ([47, 32, 4, -1, -1, 28, -1, -1, 66, 30, -1, -1, 30, -1, -1],[0.0, 3.0, 2.0, 0.0, 0.0, 5.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0004325097577673143, 0.0008989642587528236, 0.0, 0.0002737615222335293, -3.469485223645829e-05, 0.0, 0.0, -0.0035447549233087253, 0.0003116477761746999, 0.0, 0.003716484659228071, 0.0006642087113267177]),
  ([44, 62, 51, -1, -1, 62, -1, -1, 62, 45, -1, -1, 68, -1, -1],[3.0, 6.0, 40.0, 0.0, 0.0, 9.0, 0.0, 0.0, 4.0, -8.0, 0.0, 0.0, 200.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 8.928062139672022e-05, 0.0012099806092908923, 0.0, -0.0017607198774474485, -5.3100819219538645e-05, 0.0, 0.0, -0.0016818957065553772, 0.00018594570437805244, 0.0, 0.00036670407018247, -0.0025706260640680594]),
  ([64, 62, 63, -1, -1, 1, -1, -1, 64, 49, -1, -1, 57, -1, -1],[3.0, 6.0, 4.0, 0.0, 0.0, 140.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 140.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 4.7101474597543064e-05, -0.003148338641416581, 0.0, -0.011688671025475984, -0.001197972880107734, 0.0, 0.0, 0.0016096440622308257, 0.00043605011246152257, 0.0, -0.0005907175292188497, 0.0007557123410155332]),
  ([43, 68, 69, -1, -1, 57, -1, -1, 36, 68, -1, -1, 60, -1, -1],[107.0, 180.0, 30.0, 0.0, 0.0, 210.0, 0.0, 0.0, 1.0, 50.0, 0.0, 0.0, 180.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -5.309977885986774e-05, -0.0008685286229166138, 0.0, 0.0003511913595739375, -0.0005911867841750005, 0.0, 0.0, -0.0005508636062307554, 0.0031345684240679365, 0.0, 0.0029578167042431922, -0.0006762393063899765]),
  ([75, 33, 56, -1, -1, 56, -1, -1, 30, 40, -1, -1, 62, -1, -1],[3.0, 4.0, 7700.0, 0.0, 0.0, 40500.0, 0.0, 0.0, 2.0, 500.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -1.0348035136208955e-05, 0.00023617977966071448, 0.0, -0.0003098492135613029, 0.0010379019706163492, 0.0, 0.0, 0.0037165617013574046, 0.001232293181216544, 0.0, 0.0011200511136282012, -0.00028639468488646046]),
  ([42, 43, 29, -1, -1, 35, -1, -1, 43, 55, -1, -1, 38, -1, -1],[3.0, 4.0, 2.0, 0.0, 0.0, 140.0, 0.0, 0.0, 107.0, 380.0, 0.0, 0.0, 2.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.002041763786430363, 0.00011540368848855681, 0.0, -0.0002657993970434454, -0.0011296406429581072, 0.0, 0.0, 0.00010556422437721382, -0.000594865506935226, 0.0, 0.00011041378051945778, 0.00441793546570535]),
  ([20, 32, 58, -1, -1, 54, -1, -1, 59, 20, -1, -1, 41, -1, -1],[0.2955, 2.0, 23100.0, 0.0, 0.0, 5.0, 0.0, 0.0, 2.0, 1.2, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0010571421437664764, 0.000910761253824823, 0.0, 0.0002639404436205121, -0.00013904461428620153, 0.0, 0.0, -0.0006677356774682326, -0.00010376995856405458, 0.0, 0.006651668476690133, 0.0007016129202750372]),
  ([47, 48, 62, -1, -1, 70, -1, -1, 62, 70, -1, -1, 40, -1, -1],[0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 630.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0006702794618559804, 2.587468389243687e-05, 0.0, -0.0015156189047211886, 5.160196536094389e-05, 0.0, 0.0, -0.0014168211436203617, 0.004886564404661107, 0.0, 0.0005752095084404254, -0.0013812990448101357]),
  ([20, 62, 1, -1, -1, 51, -1, -1, 45, 62, -1, -1, 57, -1, -1],[0.3889, 3.0, 280.0, 0.0, 0.0, 60.0, 0.0, 0.0, -22.0, 4.0, 0.0, 0.0, 290.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.00032745391174027525, 0.003957177921833788, 0.0, -3.871097996975319e-06, 0.0015341748153688318, 0.0, 0.0, -0.0011822360233107148, 9.666402038448014e-05, 0.0, 0.0002383024573825002, -0.0005999647103448213]),
  ([58, 60, 26, -1, -1, 62, -1, -1, 26, 30, -1, -1, 30, -1, -1],[62100.0, 10.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 1.7319355894143127e-05, 0.0003877393846058764, 0.0, -0.000277780060987455, 0.0006508800009009731, 0.0, 0.0, 0.0037897295345222137, 0.00010477220589220295, 0.0, -0.005304881157129769, -0.00038732448567410573]),
  ([62, 37, 9, -1, -1, 58, -1, -1, 30, 37, -1, -1, 55, -1, -1],[1.0, 320.0, 0.0, 0.0, 0.0, 23100.0, 0.0, 0.0, 1.0, 320.0, 0.0, 0.0, 380.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.001717889395021033, 0.0012082593330068523, 0.0, 0.0007869661474752555, -0.001524678814841988, 0.0, 0.0, 0.002311247862034253, -0.0022969017676515427, 0.0, 2.5793563980667808e-06, -0.0005576679621963595]),
  ([32, 0, 62, -1, -1, 10, -1, -1, 44, 26, -1, -1, 26, -1, -1],[4.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00106580718504604, 2.5878612668805776e-05, 0.0, -0.0005553617464766704, 0.001753855992608796, 0.0, 0.0, 0.000703477118595276, -0.0008568446380030715, 0.0, -0.0002206026041503992, 0.0006373257539250535]),
  ([60, 62, 63, -1, -1, 30, -1, -1, 75, 49, -1, -1, 63, -1, -1],[30.0, 3.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 5.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 5.02209801997244e-05, 0.003013842416379487, 0.0, -0.0012283744259180102, -1.5944693061425007e-05, 0.0, 0.0, 0.000533918097593301, -0.00044188965412117996, 0.0, 0.0012612426678482711, -0.001600581058542119]),
  ([66, 62, 61, -1, -1, 73, -1, -1, 62, 61, -1, -1, 62, -1, -1],[3.0, 1.0, 4.0, 0.0, 0.0, 16000.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0008846044549777641, 0.0011026903620454274, 0.0, 0.00010463225823940077, -0.00018010716574821443, 0.0, 0.0, 0.004166478601393668, -0.00040359047100251493, 0.0, -0.0013324519056797626, 0.0005220974575821855]),
  ([13, 42, 30, -1, -1, 54, -1, -1, 69, 73, -1, -1, 30, -1, -1],[0.0, 3.0, 2.0, 0.0, 0.0, 16.0, 0.0, 0.0, 7.0, 22400.0, 0.0, 0.0, 1.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0008095717631611671, -0.00018249016864843723, 0.0, -7.42130726042954e-05, 0.0003803597439708455, 0.0, 0.0, 0.0017780679510900285, 0.00743683401637709, 0.0, 0.00329060662665103, 3.464357482211743e-05]),
  ([36, 72, 67, -1, -1, 36, -1, -1, 73, 34, -1, -1, 36, -1, -1],[1.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 86400.0, 10.0, 0.0, 0.0, 6.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0002996249975131354, -0.0008126877691245394, 0.0, 0.0006026241398645511, -0.0006511810348692996, 0.0, 0.0, -0.0006539200835678303, -6.0088422584327473e-05, 0.0, -0.0009937335779626663, -0.008217873286092458]),
  ([70, 33, 62, -1, -1, 62, -1, -1, 62, 31, -1, -1, 38, -1, -1],[10.0, 2.0, 1.0, 0.0, 0.0, 15.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 2.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0008509094888195883, -0.00036859514507078865, 0.0, 0.00011194424170030822, -0.0008758308100731137, 0.0, 0.0, 0.0006763914337209185, -0.000864780168118008, 0.0, 0.000652169320147981, -0.0006884234948780237]),
  ([1, 34, -1, 42, -1, -1, 38, 61, -1, -1, 44, -1, -1],[70.0, 8.0, 0.0, 3.0, 0.0, 0.0, 2.0, 6.0, 0.0, 0.0, 2.0, 0.0, 0.0],[1, 2, -1, 4, -1, -1, 7, 8, -1, -1, 11, -1, -1],[6, 3, -1, 5, -1, -1, 10, 9, -1, -1, 12, -1, -1],[0.0, 0.0, -0.0048584926327074965, 0.0, -0.0019149688300644974, -2.6742085395816485e-05, 0.0, 0.0, 0.00012464930927515099, -0.00027377433830688973, 0.0, 3.731530501959688e-05, -0.0011140441910285565]),
  ([28, 73, 44, -1, -1, 10, -1, -1, 42, 10, -1, -1, 55, -1, -1],[8.0, 5500.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 320.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.00018089210050098917, -0.0008801430056255521, 0.0, -0.0002054038757692443, 0.0003630303329079316, 0.0, 0.0, -0.0016854193798412844, 0.0005730363028057777, 0.0, 9.626164818108891e-05, 0.0005799566366531604]),
  ([61, 61, 34, -1, -1, 70, -1, -1, 49, 34, -1, -1, 70, -1, -1],[15.0, 3.0, 10.0, 0.0, 0.0, 18.0, 0.0, 0.0, 7.0, 34.0, 0.0, 0.0, 8.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0006620108561183822, 6.300637506193932e-05, 0.0, 2.790305806418697e-05, 0.0013794532047013768, 0.0, 0.0, 0.00035232171487491475, -0.0022492788874452063, 0.0, -0.00336422200648146, -6.578905341375835e-05]),
  ([75, 49, 60, -1, -1, 31, -1, -1, 68, 31, -1, -1, 36, -1, -1],[2.0, 5.0, 200.0, 0.0, 0.0, -4.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 1.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.00011952363364017049, 0.004140206829680946, 0.0, -0.004809387716783391, -0.0001382435661064517, 0.0, 0.0, -0.007863674292666847, -4.83671518536728e-05, 0.0, 0.002348906751152917, 0.0005756646142231607]),
  ([52, 52, 52, -1, -1, 67, -1, -1, 68, 35, -1, -1, 70, -1, -1],[180.0, 130.0, 60.0, 0.0, 0.0, 3.0, 0.0, 0.0, 180.0, 170.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -4.3548315485066365e-05, 0.0015221266802573194, 0.0, -0.001528012007160135, 0.00021879340003543455, 0.0, 0.0, -0.00583345402049413, 0.00036471224668845883, 0.0, 0.002379790845276981, 0.00038454110723635957]),
  ([69, 30, 69, -1, -1, 16, -1, -1, 30, 16, -1, -1, 3, -1, -1],[18.0, 5.0, 8.0, 0.0, 0.0, 70.0, 0.0, 0.0, 1.0, 210.0, 0.0, 0.0, 0.8235, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0003138748359011865, -3.8917949910617525e-05, 0.0, 0.0003692049340294264, -0.0004390443228644772, 0.0, 0.0, 0.0003878589396236771, 0.002667382905856832, 0.0, -0.0003612590432943216, 0.00016490181247249416]),
  ([65, 45, 38, -1, -1, 68, -1, -1, 32, 64, -1, -1, 39, -1, -1],[3.0, -22.0, 2.0, 0.0, 0.0, 180.0, 0.0, 0.0, 4.0, 3.0, 0.0, 0.0, 130.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.00010580857694016832, -0.001091745682478079, 0.0, 0.00010474688406767247, 0.0006020923900654048, 0.0, 0.0, -0.0016142854088962524, -0.00015757956369818714, 0.0, 0.0005723964710007948, -0.0003632059018311532]),
  ([47, 48, 20, -1, -1, 29, -1, -1, 30, -1, 31, -1, -1],[0.0, 0.0, 0.3889, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, -2.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, -1, 11, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 10, -1, 12, -1, -1],[0.0, 0.0, 0.0, 0.00016324760272526636, -6.837370237622054e-05, 0.0, 0.0011762938118917048, -0.0006094053021962896, 0.0, 0.003910607526384835, 0.0, 0.0012625326676505625, 8.98227372570185e-05]),
  ([73, 75, 73, -1, -1, 30, -1, -1, 54, 41, -1, -1, 30, -1, -1],[54600.0, 1.0, 29700.0, 0.0, 0.0, 2.0, 0.0, 0.0, 5.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 3.407253759904792e-05, -0.00037347366885616233, 0.0, 0.0014654969832766783, -2.900765615331078e-05, 0.0, 0.0, -0.0010729740013159102, 0.0007460009684526789, 0.0, -0.0013353507816471002, -6.457144309296202e-05]),
  ([32, 43, 29, -1, -1, 43, -1, -1, 43, 29, -1, -1, 31, -1, -1],[3.0, 2.0, 3.0, 0.0, 0.0, 8.0, 0.0, 0.0, 121.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0010140380610303883, 0.00029613575087744014, 0.0, -0.0010980567874087522, -0.00013303363378943455, 0.0, 0.0, 0.00036420880050349013, -1.6812388380316772e-05, 0.0, 0.0015673701316769163, -0.0002662091367773553]),
  ([64, 33, 1, -1, -1, 64, -1, -1, 35, 54, -1, -1, -1],[5.0, 3.0, 150.0, 0.0, 0.0, 4.0, 0.0, 0.0, 320.0, 7.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1],[0.0, 0.0, 0.0, 7.138938990124678e-06, 0.00045300665535938654, 0.0, -0.00019727791340917958, 0.0008457462691921523, 0.0, 0.0, -6.463815153361248e-05, -0.0007612486154114732, -0.004186384137031647]),
  ([23, 66, 31, -1, -1, 57, -1, -1, 31, 45, -1, -1, 43, -1, -1],[0.0, 0.0, -1.0, 0.0, 0.0, 210.0, 0.0, 0.0, -4.0, -4.0, 0.0, 0.0, 2.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0004643317937368598, -1.2768448974980053e-06, 0.0, -0.00019429081503562616, -0.001740172742694443, 0.0, 0.0, 0.002592603884729465, -0.0007120989958131413, 0.0, -0.00044042783462317744, -1.6960294411384852e-05]),
  ([44, 17, 35, -1, -1, 73, -1, -1, 58, 64, -1, -1, -1],[3.0, 2.0, 340.0, 0.0, 0.0, 4200.0, 0.0, 0.0, 3500.0, 2.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1],[0.0, 0.0, 0.0, 5.344549213200425e-05, -0.0005397709030901715, 0.0, 0.004328301157802957, 0.0004909858036244098, 0.0, 0.0, -0.00033064008700487275, -0.0032379560154602136, 0.002186557798127484]),
  ([67, 32, 17, -1, -1, 47, -1, -1, 61, 27, -1, -1, 40, -1, -1],[9.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 0.0, 0.0, 710.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00017928510088007676, 0.0005227286962068746, 0.0, 4.511604995055173e-05, 0.0010433750059640138, 0.0, 0.0, 0.001014372069739241, -0.0010518302756378668, 0.0, -0.0037239094583137425, -0.0010217488835046929]),
  ([64, 64, 25, -1, -1, 35, -1, -1, 62, 67, -1, -1, 39, -1, -1],[1.0, 0.0, 0.0, 0.0, 0.0, 290.0, 0.0, 0.0, 9.0, 3.0, 0.0, 0.0, 210.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0001978631454797741, -7.74431856595256e-05, 0.0, 0.003533694053583911, -0.0015168694596338931, 0.0, 0.0, -0.0035091955772470875, -0.0005847700123410042, 0.0, -2.7974198145545738e-05, 0.0030593114544914656]),
  ([51, 57, 31, -1, -1, 30, -1, -1, 31, 31, -1, -1, 57, -1, -1],[40.0, 140.0, -2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 320.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0003091881043772513, -9.662220794052447e-05, 0.0, 0.001183335842814808, -0.0007514178342538553, 0.0, 0.0, -0.0011283874635372795, 0.001591286838611437, 0.0, -0.0007249043310389859, 0.0012334469829982965]),
  ([75, 57, 57, -1, -1, 23, -1, -1, 19, 33, -1, -1, 46, -1, -1],[2.0, 110.0, 50.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -4.110371500204268e-05, 0.0005241900158664417, 0.0, -0.00034052538280792435, 0.0009877259427788808, 0.0, 0.0, 0.004443319353523181, 0.0009619690950957096, 0.0, 6.574656727784154e-05, 0.002725853392237683]),
  ([45, 20, 72, -1, -1, 53, -1, -1, 43, 36, -1, -1, 36, -1, -1],[-12.0, 0.0909, 0.0, 0.0, 0.0, 210.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 1.149483787659268e-06, 0.000517709840354803, 0.0, -0.0003977161323120076, 0.0012083256366376943, 0.0, 0.0, -1.884848164934008e-05, 0.000751189081930636, 0.0, -0.0008212247824344931, 2.2221890250975226e-05]),
  ([43, 61, 25, -1, -1, 64, -1, -1, 20, 61, -1, -1, -1],[107.0, 9.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.625, 5.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1],[0.0, 0.0, 0.0, 3.744727398624751e-05, -0.0002404847848136519, 0.0, -0.0013134111516306312, -3.872187788302182e-05, 0.0, 0.0, -0.0005988233530560806, 0.0008118472491307693, 0.005849545341163182]),
  ([75, 17, 41, -1, -1, 29, -1, -1, 30, 1, -1, -1, 52, -1, -1],[3.0, 2.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 340.0, 0.0, 0.0, 30.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -4.725625874616755e-06, -0.0002226033295814904, 0.0, -0.00040712176899524084, 0.0007082640923639186, 0.0, 0.0, 0.0017656153145536035, -0.0031444577807066237, 0.0, -0.00021502056261204152, 0.0037779642661391606]),
  ([74, 60, 52, -1, -1, 62, -1, -1, 62, 20, -1, -1, 68, -1, -1],[4.0, 200.0, 60.0, 0.0, 0.0, 6.0, 0.0, 0.0, 3.0, 0.0909, 0.0, 0.0, 200.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -3.088828969179765e-05, 0.0005117861788919665, 0.0, -0.003978014572945258, 6.313868708751136e-05, 0.0, 0.0, 0.010910241062128784, 0.0007872496984622931, 0.0, -2.0168935825247953e-05, 0.002126091340436183]),
  ([69, 60, 74, -1, -1, 66, -1, -1, 44, 69, -1, -1, 62, -1, -1],[8.0, 10.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 12.0, 0.0, 0.0, 2.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 9.806004635422998e-05, -0.0010684423058132488, 0.0, -0.0003362571327944612, -0.0020342293521213237, 0.0, 0.0, 0.0003903544559456602, 3.151586424230535e-05, 0.0, -0.0018247922170003966, -2.792056142203342e-05]),
  ([20, 62, 27, -1, -1, 10, -1, -1, 23, 25, -1, -1, 25, -1, -1],[2.4545, 1.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.001087822894253203, 0.0006957690685173265, 0.0, -0.00013842580555762342, 0.00013580442973770457, 0.0, 0.0, -2.834758714392011e-05, 0.0007893892557577274, 0.0, 0.00035674596344809805, -0.0051252645731094506]),
  ([27, 53, 53, -1, -1, 44, -1, -1, 75, 30, -1, -1, 30, -1, -1],[8.0, 180.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.00012415210916048432, 0.0009255990130257273, 0.0, -0.0015245286572219605, 0.00026704844330384983, 0.0, 0.0, -0.001420712389126128, -0.0002635742098837113, 0.0, 0.000840972233881081, -0.0007968031161424173]),
  ([62, 2, 44, -1, -1, 39, -1, -1, 44, 35, -1, -1, 10, -1, -1],[4.0, 150.0, 2.0, 0.0, 0.0, 200.0, 0.0, 0.0, 3.0, 290.0, 0.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0002459171864507303, -0.00032347797000894737, 0.0, 0.002560590987857537, -0.002033824390812239, 0.0, 0.0, -0.0002790807864207264, 4.94258015516607e-05, 0.0, -4.2756379415998086e-05, 0.0012697254822270796]),
  ([52, 52, 42, -1, -1, 24, -1, -1, 52, 25, -1, -1, 27, -1, -1],[60.0, 10.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 130.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -9.752765381715652e-05, 0.00014729991448800172, 0.0, -0.0014402089247635126, 0.0010025982257863928, 0.0, 0.0, 0.0012019191491262563, 0.003980651937136113, 0.0, -0.0010398265758228239, 0.000395851264078857]),
  ([44, 34, 44, -1, -1, 15, -1, -1, 62, 34, -1, -1, 33, -1, -1],[3.0, 10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 22.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.001124099635581433, -0.00029027694836735785, 0.0, 0.00012807486985920742, -0.00010414668263227071, 0.0, 0.0, -0.0016691723815791912, 0.00035028164031104857, 0.0, 0.0007713756233933743, -0.0002255245239799024]),
  ([29, 16, 28, -1, -1, 30, -1, -1, 23, 70, -1, -1, 16, -1, -1],[1.0, 30.0, 13.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 15.0, 0.0, 0.0, 30.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.002100443490335255, -0.0005956222708465791, 0.0, 0.0008812408482941999, -9.967729603725179e-05, 0.0, 0.0, -0.00011023346742438265, 0.00011778948066149258, 0.0, -0.0012437633612783444, -0.00019006745458353548]),
  ([57, 65, 59, -1, -1, 30, -1, -1, 42, 30, -1, -1, 67, -1, -1],[320.0, 3.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 9.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 5.256051592008752e-05, 0.0016596343447437278, 0.0, -0.0007879504687854592, 0.00019786072658100278, 0.0, 0.0, -0.003818933234993578, 0.00022692195870130402, 0.0, 0.002124362078061241, -0.002075497903766469]),
  ([73, 60, 27, -1, -1, 44, -1, -1, 10, 56, -1, -1, 73, -1, -1],[5500.0, 60.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 19800.0, 0.0, 0.0, 16800.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0009578068408524636, -0.00020655450921025294, 0.0, 0.0011231843925142123, -0.0031223255520044127, 0.0, 0.0, -3.0939574056454956e-05, 0.0002524009601924627, 0.0, 0.0005242984187631335, -0.0004945626731088803]),
  ([29, 68, 43, -1, -1, 20, -1, -1, 20, 68, -1, -1, 68, -1, -1],[1.0, 70.0, 5.0, 0.0, 0.0, 2.4545, 0.0, 0.0, 0.3889, 50.0, 0.0, 0.0, 50.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.000337970121770112, -0.0005468503905339549, 0.0, 0.0007003319735376211, 0.002279728276595157, 0.0, 0.0, 4.0790115781690897e-05, 0.0011182020384496597, 0.0, 0.0007242757820026077, -0.0003062484651194544]),
  ([37, 62, 45, -1, -1, 30, -1, -1, 62, 71, -1, -1, 71, -1, -1],[320.0, 1.0, -22.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.002454721564117628, 0.0001552171557227458, 0.0, 0.0013228835386842853, 7.778388913926781e-05, 0.0, 0.0, 0.0010138991724109592, -0.0011110970664046408, 0.0, -0.0005454958998966641, 0.0009883847795492439]),
  ([17, 70, 42, -1, -1, 61, -1, -1, 70, 37, -1, -1, 37, -1, -1],[2.0, 22.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 22.0, 320.0, 0.0, 0.0, 80.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0005831800087096479, 4.466499173269687e-05, 0.0, -0.0014482091161807926, 0.0014005900777600265, 0.0, 0.0, 0.00013103483497443687, -0.0009755113993332275, 0.0, -0.0009982917075054452, 0.0009970668868079175]),
  ([75, 75, 75, -1, -1, -1, -1],[9.0, 6.0, 1.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, -1, -1],[6, 5, 4, -1, -1, -1, -1],[0.0, 0.0, 0.0, -4.340967895752003e-05, 0.0002144045197308424, 0.001756129013996012, -0.002821921333844333]),
  ([47, 58, 64, -1, -1, 26, -1, -1, 41, 31, -1, -1, 27, -1, -1],[0.0, 62100.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, -1.0, 0.0, 0.0, 3.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -1.7003269182048308e-05, -0.0004125268424744827, 0.0, 0.0010686509455472561, -0.0016671370993012274, 0.0, 0.0, -0.0024098797984409294, 0.0004491116004698756, 0.0, -0.0017502679140541163, 0.0012490523152617406]),
  ([28, 55, 49, -1, -1, 24, -1, -1, 75, 41, -1, -1, 55, -1, -1],[3.0, 210.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 380.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.00033047075007091486, -0.0017135150962264616, 0.0, 0.0017396124125772998, 0.00660134792357702, 0.0, 0.0, 8.158486148536099e-05, -0.00014659179444226776, 0.0, 0.0010098936482916062, -0.00010524462959861192]),
  ([1, 64, 61, -1, -1, 64, -1, -1, 63, 28, -1, -1, 64, -1, -1],[340.0, 1.0, 9.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 8.0, 0.0, 0.0, 4.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 3.4188608980686996e-05, -0.003118728727444263, 0.0, -0.0009658741986399951, 0.00010365649893006943, 0.0, 0.0, 0.0021528825556172827, -0.0006631169359503529, 0.0, 0.0023731938795724774, 0.000618159561926824]),
  ([59, 72, 30, -1, -1, 30, -1, -1, 55, 25, -1, -1, -1],[2.0, 5.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 320.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1],[0.0, 0.0, 0.0, 0.000525196426849518, -0.00019477102040611234, 0.0, -0.001215425786920468, 0.00048352173212754733, 0.0, 0.0, 0.001374690715744736, -0.0013021576761129367, 0.003843390430002849]),
  ([75, 54, 58, -1, -1, 33, -1, -1, 40, 42, -1, -1, 39, -1, -1],[1.0, 5.0, 7700.0, 0.0, 0.0, 4.0, 0.0, 0.0, 800.0, 3.0, 0.0, 0.0, 200.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 5.504339470698215e-05, 0.0010149937720350612, 0.0, -9.997487532711746e-05, -0.0008735535487162757, 0.0, 0.0, -0.0006292524115314418, 0.0007775794037350949, 0.0, -5.679805986021788e-05, -0.0024534233200784717]),
  ([27, 67, 64, -1, -1, 26, -1, -1, -1],[12.0, 8.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, -1],[0.0, 0.0, 0.0, 0.00012334670636444657, -0.00031874422380991145, 0.0, -0.00015872344596500512, -0.0014389061745242017, 0.0028038937969321032]),
  ([57, 26, 62, -1, -1, 62, -1, -1, 26, 34, -1, -1, 73, -1, -1],[140.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 21.0, 0.0, 0.0, 86400.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.0011616583913389715, 7.439022173453444e-05, 0.0, 0.0021411348779849203, 0.0001625377761785446, 0.0, 0.0, 0.0006637608501612267, -0.0006274648425032665, 0.0, -0.0013810007688971806, -0.004572225711484851]),
  ([64, 64, 61, -1, -1, 1, -1, -1, 31, 33, -1, -1, 54, -1, -1],[3.0, 2.0, 12.0, 0.0, 0.0, 110.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 2.9286819152700353e-05, -0.0019661828576797223, 0.0, -0.011856675556908404, -0.0010005501887271379, 0.0, 0.0, 0.0016735738075750902, 0.00039486783943826534, 0.0, 0.0005058983159376309, -0.0005799323631810842]),
  ([52, 40, 42, -1, -1, 42, -1, -1, 41, 52, -1, -1, 40, -1, -1],[10.0, 370.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 30.0, 0.0, 0.0, 800.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0002814367456690388, 0.0009918358117525987, 0.0, -0.0013697396034438882, 6.908949120727194e-05, 0.0, 0.0, -0.0031401205790028463, -0.0007708318892504702, 0.0, -5.296147812305297e-05, 0.004208376962530294]),
  ([27, 25, 67, -1, -1, 53, -1, -1, -1],[12.0, 2.0, 6.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, -1],[0.0, 0.0, 0.0, 4.632859386657254e-05, -0.0002914477144585472, 0.0, -0.0005845863927573873, 0.0007610507224475586, -0.004874786167278119]),
  ([47, 48, 75, -1, -1, 34, -1, -1, 34, -1, 60, -1, -1],[0.0, 1.0, 7.0, 0.0, 0.0, 34.0, 0.0, 0.0, 8.0, 0.0, 30.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, -1, 11, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 10, -1, 12, -1, -1],[0.0, 0.0, 0.0, -7.98810691193625e-05, 0.0013134935949903625, 0.0, -0.00022705317823147513, -0.0015837589698978784, 0.0, -0.0052745175875129665, 0.0, 0.00046775006278464734, 0.0033460330963473437]),
  ([40, 29, 55, -1, -1, 58, -1, -1, 58, 30, -1, -1, 30, -1, -1],[530.0, 2.0, 380.0, 0.0, 0.0, 11900.0, 0.0, 0.0, 7700.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0007522387124471849, 0.0028244427237112527, 0.0, 0.00023810203653052035, -0.0008731610536052771, 0.0, 0.0, -0.0011087163873267553, -0.00016517716732395208, 0.0, 0.0007720740140077252, 2.8049352051240587e-06]),
  ([44, 12, 38, -1, -1, 25, -1, -1, 23, 61, -1, -1, -1],[2.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 6.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1],[0.0, 0.0, 0.0, 0.0002576959083795279, -9.525947562908495e-05, 0.0, 0.0007737074524521294, 0.004463149656796791, 0.0, 0.0, -6.416025837557722e-05, -0.001913548293493112, -0.0049609750962898835]),
  ([1, 49, 30, -1, -1, 30, -1, -1, 32, 37, -1, -1, 30, -1, -1],[280.0, 11.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 340.0, 0.0, 0.0, 2.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.00012530588168494712, -0.00018465675458593622, 0.0, -0.003028826687040783, 0.0005289167631036454, 0.0, 0.0, 8.044009374312287e-06, -0.0031093009691872254, 0.0, 0.002802442506495394, 0.0003087052654622302]),
  ([56, 56, 0, -1, -1, 20, -1, -1, 59, 18, -1, -1, 2, -1, -1],[55800.0, 21600.0, 2.0, 0.0, 0.0, 0.2, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 340.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -4.3225376255810613e-05, -0.0007512629732167534, 0.0, -0.002537573792901673, 0.00024571461314751834, 0.0, 0.0, -0.0004743547886147434, -0.0018607945785066749, 0.0, 0.0010190578426705844, -0.0028881185531419704]),
  ([51, 51, 66, -1, -1, 30, -1, -1, 23, 60, -1, -1, 39, -1, -1],[130.0, 60.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 210.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 7.70703074584892e-05, -0.00020177393774478214, 0.0, 0.0013748586704121668, -0.0006462787062382609, 0.0, 0.0, -0.002243547087282545, -0.0005410828397506396, 0.0, -0.0002426223772214673, 0.0015102419160929524]),
  ([43, 34, 20, -1, -1, 45, -1, -1, 73, -1, 73, -1, -1],[121.0, 32.0, 0.6364, 0.0, 0.0, -22.0, 0.0, 0.0, 20300.0, 0.0, 22400.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, -1, 11, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 10, -1, 12, -1, -1],[0.0, 0.0, 0.0, 4.2689206354582236e-05, -0.00023112412080533704, 0.0, -0.0003603560740781679, 0.00036653644628591956, 0.0, -0.0015958109021903856, 0.0, 0.0035467987656172257, 0.0006434185207330835]),
  ([36, 69, 69, -1, -1, 61, -1, -1, -1],[9.0, 6.0, 4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, -1],[0.0, 0.0, 0.0, 0.00014948680834443498, -0.0005668876289464092, 0.0, -0.0001267057460383092, 0.00016363905890110582, 0.0037262544036689546]),
  ([40, 47, 1, -1, -1, -1, 62, 17, -1, -1, 30, -1, -1],[730.0, 2.0, 150.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],[1, 2, 3, -1, -1, -1, 7, 8, -1, -1, 11, -1, -1],[6, 5, 4, -1, -1, -1, 10, 9, -1, -1, 12, -1, -1],[0.0, 0.0, 0.0, 1.4504388317041554e-05, 0.0005130431241615632, 0.004176111094531432, 0.0, 0.0, 0.0008014322726638253, -0.0032908125832155567, 0.0, 0.0017872414413113076, -0.00014594896411576183]),
  ([59, 28, 31, -1, -1, 56, -1, -1, 43, 31, -1, -1, -1],[2.0, 13.0, -3.0, 0.0, 0.0, 14300.0, 0.0, 0.0, 24.0, -3.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1],[0.0, 0.0, 0.0, 0.0003503709987911345, -2.3647665554102288e-05, 0.0, -0.000172116377348076, -0.0012641461326880058, 0.0, 0.0, -0.002151208362570868, 0.0011125646043323514, 0.008701460246299468]),
  ([32, 40, 29, -1, -1, 1, -1, -1, 69, 20, -1, -1, 16, -1, -1],[3.0, 500.0, 1.0, 0.0, 0.0, 340.0, 0.0, 0.0, 11.0, 2.4545, 0.0, 0.0, 30.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, 0.0025820463490818244, -8.471875937563541e-05, 0.0, -0.0010068088875939112, 0.0014495702619149457, 0.0, 0.0, 0.00016049377709135648, 0.001162345228642203, 0.0, -0.00030592211616637, 1.780044935946951e-05]),
  ([34, 66, 53, -1, -1, 33, -1, -1, 9, 25, -1, -1, 54, -1, -1],[25.0, 3.0, 70.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 12.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],[0.0, 0.0, 0.0, -0.00022913449025705135, 0.0005487968060202133, 0.0, 0.004523730752785702, 0.000750339956793269, 0.0, 0.0, 9.520567512100978e-05, 0.00045517176120575824, 0.0, -0.00017623546076491297, -0.00445873157201826]),
  ([47, 62, 43, -1, -1, 42, -1, -1, -1],[2.0, 3.0, 107.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],[1, 2, 3, -1, -1, 6, -1, -1, -1],[8, 5, 4, -1, -1, 7, -1, -1, -1],[0.0, 0.0, 0.0, 0.0001249811462798861, 0.002580249801761316, 0.0, -0.0003640923027106949, 2.8516915963366246e-06, 0.004413214085406092]),
]
QBASE = 0.0
QLIN = None
QLIN0 = 0.0

QW = None


_FIJOV = [(_IX[n], QFIJO[c]) for c, n in
          ((678, "c_is_mega"), (677, "c_is_riolu"), (676, "c_is_solrock"),
           (675, "c_is_lunatone"), (674, "c_is_hariyama"), (673, "c_is_makuhita"))]


def q_pred(v):
    if _Q_FIJO:
        # ABLACION FIJA: el MISMO maestro, pero el modelo solo puede usar la
        # IDENTIDAD de la carta (ventaja media aprendida), no el tablero.
        return sum(a * v[j] for j, a in _FIJOV)
    x = _expande(v)
    s = QBASE
    for F, T, L, R, V in QTREES:
        n = 0
        while F[n] >= 0:
            n = L[n] if x[F[n]] <= T[n] else R[n]
        s += V[n]
    if QLIN is not None:
        s += QLIN0
        for k in range(len(QLIN)):
            s += QLIN[k] * x[k]
    return s


# ============================================================================
# ENTRADA DE KAGGLE — DEBE SER LA ULTIMA FUNCION DEFINIDA DEL FICHERO.
# kaggle_environments/agent.py:64 hace:
#     return [v for v in env.values() if callable(v)][-1]
# o sea, coge la ULTIMA callable del modulo, no la que se llame 'agent'.
# El modelo Q se genera DESPUES de def agent(), asi que sin esto Kaggle
# llamaba a q_pred(observation) y reventaba con KeyError 7 en el paso 0.
# El arnes local no lo detecta porque importa el modulo y llama a m.agent.
# NO ANADIR NINGUNA def NI lambda DESPUES DE ESTA.
# ============================================================================
def kaggle_agent_entrypoint(obs_dict):
    return agent(obs_dict)