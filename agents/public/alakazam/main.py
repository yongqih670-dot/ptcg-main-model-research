"""Independent CPU rule policy for the Frankfurt Alakazam/Dusknoir list.

The engine owns legality. This policy chooses current option indices, using
our hand and public boards only. It plans hand growth before Powerful Hand,
and prices Cursed Blast's self-KO before selecting an opponent target.
No native search, hidden-state decoding, trained model or network is used.
"""
from collections import Counter
from copy import deepcopy
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import json
import sys

_HERE = Path(globals().get('__file__', '/kaggle_simulations/agent/main.py')).resolve().parent
for _base in (_HERE, _HERE.parent, _HERE.parent.parent):
    if (_base / 'engine' / 'cg').is_dir():
        sys.path.insert(0, str(_base / 'engine'))
        break
from cg.api import (AreaType, Card, CardType, EnergyType, LogType, OptionType, Pokemon,
                    SelectContext, SelectType, all_attack, all_card_data,
                    to_observation_class)

if '__file__' not in globals():
    _archive = Path(sys.path[-1]).resolve()
    if (_archive / 'deck.csv').is_file() and (_archive / 'cg').is_dir():
        _HERE = _archive
    else:
        _HERE = Path(sys.modules['cg.api'].__file__).resolve().parents[1]

ABRA = (109, 741)
KADABRA, ALAKAZAM = 742, 743
DUSKULL, DUSCLOPS, DUSKNOIR = 131, 132, 133
BUDEW, FEZ, SHAYMIN = 235, 140, 343
PSYCHIC, TELEPATH = 5, 19
RARE_CANDY, POFFIN, PRIME, NIGHT = 1079, 1086, 1088, 1097
ASH, TIMEPIECE, PAD = 1129, 1144, 1152
BOSS, HILDA, DAWN, RED, GWYNN = 1182, 1225, 1231, 1413, 1428
HAMMER, PATRAT, DUDUNSPARCE = 1081, 1340, 66
DUNSPARCE = (65, 305)
_MEMORY = {'round': None, 'turn': -1, 'key': None, 'attacks': {}, 'dusk_target': None, 'dusk_chain_target': None, 'enemy_prizes': None, 'last_own_ko_turn': -1, 'fez_used_turn': -1, 'devolve_serials': (), 'devolve_turn': -1, 'survival_search': None, 'resource_inventory': None, 'resource_failed': {}, 'resource_pending': None, 'public_attack_locks': {}, 'public_lock_context_valid': False}


def read_deck_csv():
    for path in (_HERE / 'deck.csv', Path('deck.csv'),
                 Path('/kaggle_simulations/agent/deck.csv')):
        if path.is_file():
            deck = [int(v) for v in path.read_text(encoding='utf-8-sig').split()]
            if len(deck) != 60:
                raise ValueError('Expected exactly 60 card IDs')
            return deck
    raise FileNotFoundError('deck.csv must accompany this candidate')


@lru_cache(maxsize=1)
def _catalog():
    return ({c.cardId: c for c in all_card_data()},
            {a.attackId: a for a in all_attack()}, Counter(read_deck_csv()))


def reset(seed=None):
    _MEMORY.update(round=None, turn=-1, key=None, attacks={}, dusk_target=None, dusk_chain_target=None, enemy_prizes=None, last_own_ko_turn=-1, fez_used_turn=-1, devolve_serials=(), devolve_turn=-1, survival_search=None, resource_inventory=None, resource_failed={}, resource_pending=None, public_attack_locks={}, public_lock_context_valid=False)


def _players(obs):
    return (obs.current.players[obs.current.yourIndex],
            obs.current.players[1 - obs.current.yourIndex])


def _board(player):
    return [p for p in player.active + player.bench if p is not None]


def _active(player):
    return player.active[0] if player.active and player.active[0] is not None else None


def _prizes(pokemon):
    card = _catalog()[0][pokemon.id]
    return 3 if card.megaEx else 2 if card.ex else 1


def _attack_prizes(obs, pokemon, attack_id):
    """Conservative Prize count for the supported attack's knockout.

    Legacy Energy reduces prizes only for attack *damage*, not Powerful Hand's
    counters or Cursed Blast. Its native once-per-game used flag is private;
    without tracking a confirmed public activation, assume a visible Legacy
    can still reduce a normal attack KO by one. Never infer that hidden flag.
    """
    if pokemon is None:
        return 0
    prizes = _prizes(pokemon)
    if attack_id != 1072 and any(c.id == 12 for c in pokemon.energyCards):
        prizes = max(0, prizes - 1)
    return prizes


def _at(obs, area, index, player_index=None):
    if area is None or index is None:
        return None
    owner = obs.current.yourIndex if player_index is None else player_index
    player = obs.current.players[owner]
    # Deck cards are visible only during the engine's search selection.
    # Neither player's facedown prize identities are accessed here.
    regions = {AreaType.HAND: player.hand if owner == obs.current.yourIndex else None,
               AreaType.ACTIVE: player.active, AreaType.BENCH: player.bench,
               AreaType.DISCARD: player.discard, AreaType.DECK: obs.select.deck,
               AreaType.LOOKING: obs.current.looking, AreaType.STADIUM: obs.current.stadium}
    collection = regions.get(area) or []
    return collection[index] if 0 <= index < len(collection) else None


def _option_card(obs, option):
    card = _at(obs, option.area, option.index, option.playerIndex)
    if card is None:
        return None
    if option.type in (OptionType.ENERGY, OptionType.ENERGY_CARD):
        index = option.energyIndex
        return card.energyCards[index] if index is not None and 0 <= index < len(card.energyCards) else None
    if option.type == OptionType.TOOL_CARD:
        index = option.toolIndex
        return card.tools[index] if index is not None and 0 <= index < len(card.tools) else None
    return card


def _known_remaining(obs):
    """Our unseen cards include unknown prizes; never treat them as deck facts."""
    own, _ = _players(obs)
    remaining = _catalog()[2].copy()
    exposed = list(own.hand or []) + list(own.discard)
    for p in _board(own):
        exposed += [p] + p.energyCards + p.tools + p.preEvolution
    for card in exposed:
        remaining[card.id] -= 1
    return +remaining


def _observe(raw):
    state = raw.get('current') or {}
    round_number, turn = state.get('round'), state.get('turn', 0)
    if (_MEMORY['round'] != round_number or turn < _MEMORY['turn']):
        reset()
    previous_turn = _MEMORY['turn']
    players = state.get('players') or []
    if len(players) == 2:
        seat = state['yourIndex']
        enemy_prizes = len(players[1 - seat].get('prize') or [])
        previous_prizes = _MEMORY['enemy_prizes']
        if previous_prizes is not None and enemy_prizes < previous_prizes:
            turn_owner = (state.get('firstPlayer', 0) + turn - 1) % 2
            if turn_owner != seat:
                _MEMORY['last_own_ko_turn'] = turn
            elif previous_turn < turn and turn > 1:
                _MEMORY['last_own_ko_turn'] = turn - 1
            # A Dusk self-KO during our current turn never unlocks Fez's draw.
        _MEMORY['enemy_prizes'] = enemy_prizes
    _MEMORY['round'], _MEMORY['turn'] = round_number, turn
    public_lock_valid_context = (
        type(round_number) is int and round_number >= 1 and type(turn) is int and turn >= 1
        and type(state.get('firstPlayer')) is int and state['firstPlayer'] in (0, 1)
        and type(state.get('yourIndex')) is int and state['yourIndex'] in (0, 1))
    pending = _MEMORY.get('survival_search')
    if pending and (pending[:2] != (round_number, turn)
                    or (raw.get('select') or {}).get('type') == int(SelectType.MAIN)):
        _MEMORY['survival_search'] = None
    logs = raw.get('logs') or []
    key = (round_number, turn, state.get('turnActionCount'),
           json.dumps(logs, sort_keys=True, separators=(',', ':')))
    _resource_observe(raw)
    if key == _MEMORY['key']:
        return
    _MEMORY['key'] = key
    _MEMORY['public_lock_context_valid'] = public_lock_valid_context
    # Native public logs can span the previous Attack and two TurnStarts.
    # Count their turn boundaries backwards from the explicitly serialized
    # current turn; re-reading a public Attack must not extend its cooldown.
    attack_turns = {}
    public_turn = turn
    if public_lock_valid_context:
        for pos in range(len(logs) - 1, -1, -1):
            log = logs[pos]
            if log.get('type') == int(LogType.TURN_START):
                owner = log.get('playerIndex')
                if (type(owner) is not int or owner not in (0, 1) or public_turn < 1
                        or (state['firstPlayer'] + public_turn - 1) % 2 != owner):
                    _MEMORY['public_lock_context_valid'] = False
                public_turn -= 1
            elif log.get('type') == int(LogType.ATTACK) and log.get('attackId') == 983:
                owner = log.get('playerIndex')
                if (type(owner) is int and owner in (0, 1) and public_turn >= 1
                        and (state['firstPlayer'] + public_turn - 1) % 2 == owner):
                    attack_turns[pos] = public_turn
                else:
                    _MEMORY['public_lock_context_valid'] = False
    for pos, log in enumerate(logs):
        if log.get('type') == int(LogType.ATTACK):
            _MEMORY['attacks'][(log.get('playerIndex'), log.get('serial'))] = (log.get('attackId'), turn)
        # Only the narrow public Mega Brave lock is new; legacy attack history stays intact.
        if log.get('type') == int(LogType.ATTACK) and log.get('attackId') == 983:
            owner, serial = log.get('playerIndex'), log.get('serial')
            if (_MEMORY['public_lock_context_valid'] and pos in attack_turns and type(owner) is int
                    and owner in (0, 1) and type(serial) is int and serial >= 0):
                _MEMORY['public_attack_locks'][(owner, serial)] = attack_turns[pos] + 2
            else:
                _MEMORY['public_lock_context_valid'] = False
        if log.get('type') == int(LogType.SWITCH):
            owner, serial = log.get('playerIndex'), log.get('serialActive')
            if type(owner) is int and owner in (0, 1) and type(serial) is int and serial >= 0:
                _MEMORY['public_attack_locks'].pop((owner, serial), None)
            else:
                _MEMORY['public_lock_context_valid'] = False



def _energy_gap(pokemon, attack):
    available = Counter(pokemon.energies)
    gap = 0
    for requirement in sorted(attack.energies, key=lambda e: e == EnergyType.COLORLESS):
        if requirement == EnergyType.COLORLESS:
            found = next((e for e, n in available.items() if n > 0), None)
        elif available[requirement]:
            found = requirement
        elif available[EnergyType.RAINBOW]:
            found = EnergyType.RAINBOW
        elif requirement in (EnergyType.PSYCHIC, EnergyType.DARKNESS) and available[EnergyType.TEAM_ROCKET]:
            found = EnergyType.TEAM_ROCKET
        else:
            found = None
        if found is None:
            gap += 1
        else:
            available[found] -= 1
    return gap


def _pokemon_location(obs, pokemon):
    """Locate a public body by serial, including projected copies of it."""
    for owner, player in enumerate(obs.current.players):
        for area, values in ((AreaType.ACTIVE, player.active), (AreaType.BENCH, player.bench)):
            if any(p is not None and p.serial == pokemon.serial for p in values):
                return owner, area
    return None, None


def _ability_enabled(obs, pokemon, owner=None, area=None):
    """Printed Ability after the engine's public, permanent suppression rules.

    For a proposed hand-to-bench play callers may supply owner and BENCH area.
    Future identities are from the shipped CardImpl.h's future() annotations;
    CardData does not expose that marker. No hidden state is consulted.
    """
    if pokemon is None:
        return False
    data = _catalog()[0][pokemon.id]
    if not data.skills or (data.cardType != CardType.POKEMON and pokemon.id != 1136):
        return False
    known_owner, known_area = _pokemon_location(obs, pokemon)
    owner = known_owner if owner is None else owner
    area = known_area if area is None else area
    if owner is None:
        owner = getattr(pokemon, 'playerIndex', obs.current.yourIndex)
    if any(c.id == 1256 for c in obs.current.stadium) and data.energyType == EnergyType.COLORLESS:
        return False  # Team Rocket's Watchtower, both sides.
    opponent_active = _active(obs.current.players[1 - owner])
    ability_effect_shield = pokemon.id in (1348, 1349, 1374, 1375, 1040)
    if (area == AreaType.ACTIVE and opponent_active is not None
            and opponent_active.id == 56 and pokemon.id != 56 and not ability_effect_shield):
        return False  # Flutter Mane exempts Midnight Fluttering itself.
    if area == AreaType.BENCH and data.stage2 and any(
            p.id == 225 for player in obs.current.players for p in player.bench if p is not None):
        return False  # Gastrodon's Sticky Bind affects both benches.
    future = pokemon.id in (27, 37, 75, 80, 87, 192, 313, 971)
    if (data.ex or data.megaEx) and not future:
        for source_owner, player in enumerate(obs.current.players):
            source = _active(player)
            other_active = _active(obs.current.players[1 - source_owner])
            if (source is not None and source.id == 37
                    and not (other_active is not None and other_active.id == 56)):
                if ability_effect_shield and source_owner != owner:
                    # NoEnemyAbility priority 3 precedes NoAbility priority 2.
                    # StaticEffect excludes shields from opposing Abilities;
                    # an allied Iron Thorns still suppresses Luminous Wing.
                    continue
                return False  # Active Iron Thorns; Flutter Mane can suppress it.
    return True


def _counter_blocked(obs, defender, ability=False):
    if defender is None:
        return True
    owner, area = _pokemon_location(obs, defender)
    if area == AreaType.BENCH and any(c.id == 1264 for c in obs.current.stadium):
        return True  # Battle Cage stops ability and attack counters on benches.
    enabled = _ability_enabled(obs, defender)
    if enabled and defender.id in (1348, 1349, 1374, 1375):
        return True  # Hide 'n' Sneak prevents both attack and Ability effects.
    if ability:
        return enabled and defender.id == 1040  # Mega Clefable's Luminous Wing.
    if not ability:
        if any(e.id == 11 for e in defender.energyCards):
            return True  # Mist prevents attack effects, not ability counters.
        if (_catalog()[0][defender.id].energyType == EnergyType.FIGHTING
                and any(e.id == 20 for e in defender.energyCards)):
            return True  # Rocky Fighting Energy.
        data = _catalog()[0][defender.id]
        if enabled and defender.id in (203, 835, 1136):
            return True  # Unaware / Emperor's Stance / Protective Cover.
        if owner is not None and data.basic and data.name.startswith("Team Rocket's ") and any(
                p.id == 414 and _ability_enabled(obs, p)
                for p in _board(obs.current.players[owner])):
            # Articuno's Repelling Veil protects its Basic Team Rocket team.
            return True
    return False


def _public_damage_base(obs, attacker, attack_id, defender, attacking_owner=None):
    """Exact variable base damage from visible board, discard and hand counts.

    Energy units are the observed effective units, including public Meganium.
    No player hand identities, hidden deck/prize cards or future draws are read.
    A projected attacker replaces its same-serial public body for attach scores.
    """
    cards, attacks, _ = _catalog()
    if attacking_owner is None:
        attacking_owner = _pokemon_location(obs, attacker)[0]
        if attacking_owner is None:
            attacking_owner = getattr(attacker, 'playerIndex', obs.current.yourIndex)
    attacking_side = obs.current.players[attacking_owner]
    if attack_id == 195:
        board = _board(attacking_side)
        grass = sum(e in (EnergyType.GRASS, EnergyType.RAINBOW)
                    for p in board for e in (attacker if p.serial == attacker.serial else p).energies)
        if not any(p.serial == attacker.serial for p in board):
            grass += sum(e in (EnergyType.GRASS, EnergyType.RAINBOW) for e in attacker.energies)
        return 30.0 + 30.0 * grass
    if attack_id == 120:
        # Defender is also the proposed Active after a public gust/promotion.
        return 30.0 + 30.0 * (len(attacker.energies) + len(defender.energies))
    if attack_id == 1708:
        hide_count = sum(any(skill.name.strip() == 'Hide ’n’ Sneak' for skill in cards[c.id].skills)
                         for c in attacking_side.discard)
        return 170.0 if hide_count >= 4 else 30.0
    if attack_id == 1240:
        return 50.0 * obs.current.players[1 - attacking_owner].handCount
    return float(attacks[attack_id].damage)


def _public_attack_locked(obs, attacker, attack_id):
    """Only the public Mega Brave next-owner-turn lock, reset by switching."""
    if attack_id != 983:
        return False
    owner = _pokemon_location(obs, attacker)[0]
    if (owner is None or not _MEMORY.get('public_lock_context_valid')
            or _MEMORY['turn'] != obs.current.turn or type(attacker.serial) is not int):
        return True  # Unknown public lock context cannot certify Mega Brave.
    turn, first = obs.current.turn, obs.current.firstPlayer
    if type(turn) is not int or turn < 1 or type(first) is not int or first not in (0, 1):
        return True
    turn_owner = (first + turn - 1) % 2
    next_turn = turn + (turn_owner != owner)
    if owner != obs.current.yourIndex and turn_owner == owner:
        # A forced promotion during the opposing turn evaluates that enemy's
        # reply after our upcoming turn, not a nonexistent owner turn T+1.
        next_turn += 2
    return _MEMORY.get('public_attack_locks', {}).get((owner, attacker.serial)) == next_turn


def _public_attack_damage(obs, attacker, attack_id, defender, attacking_owner,
                          hand_count=None, bench_target=False):
    """Supported public damage base and native damage/effect protections.

    This never reads enemy hand identities. It is not a future turn simulator:
    later modifiers, unknown gusts and hidden transient locks are not invented.
    """
    cards, attacks, _ = _catalog()
    attacking, defending = cards[attacker.id], cards[defender.id]
    attacking_side = obs.current.players[attacking_owner]
    if attack_id == 1072:
        count = attacking_side.handCount if hand_count is None else hand_count
        return 0.0 if _counter_blocked(obs, defender) else 20.0 * count
    value = _public_damage_base(obs, attacker, attack_id, defender, attacking_owner)
    if attack_id == 546:
        value = 100.0 if defending.stage1 or defending.stage2 else 50.0
    elif attack_id == 371:
        value = 20.0 + 20.0 * sum(len(p.bench) for p in obs.current.players)
    elif attack_id == 183:
        value = 100.0
    elif attack_id == 980:
        if not any(cards[p.id].name == 'Lunatone' for p in attacking_side.bench):
            return 0.0
    if (defender.id == 117 and _ability_enabled(obs, defender)
            and _ability_enabled(obs, attacker)):
        return 0.0
    if (attacking.ex or attacking.megaEx) and (
            (defender.id == 345 and _ability_enabled(obs, defender)) or
            (any(c.id == 1247 for c in obs.current.stadium)
             and not (defending.ex or defending.megaEx))):
        return 0.0
    defending_side = obs.current.players[1 - attacking_owner]
    if (bench_target and not (defending.ex or defending.megaEx)
            and any(p.id == SHAYMIN and _ability_enabled(obs, p)
                    for p in _board(defending_side))):
        return 0.0
    previous = _MEMORY['attacks'].get((1 - attacking_owner, defender.serial))
    weakness_removed = previous == (253, obs.current.turn - 1)
    weakness = defending.weakness
    if (defending.energyType == EnergyType.DRAGON
            and any(p.id == 272 and _ability_enabled(obs, p)
                    for p in _board(attacking_side))):
        weakness = EnergyType.PSYCHIC
    # Cosmic Beam and bench damage do not use weakness or resistance.
    if not bench_target and attack_id != 980:
        if not weakness_removed and weakness == attacking.energyType:
            value *= 2
        if defending.resistance == attacking.energyType:
            value = max(0.0, value - 30)
    if defending.energyType == EnergyType.METAL and any(
            c.id == 1244 for c in obs.current.stadium):
        value = max(0.0, value - 30)
    return value


def _terminal_ex_draw_ko(obs, card):
    """Fez's actual min(3, deck) hand growth may preserve a visible Alak KO.

    Only a currently legal primary attack or a printed-payable route to an
    already powered primary is used. No drawn Energy/evolution is assumed.
    """
    if card.id != FEZ or not _fez_draw_ready(obs) or not _draw_safe(obs, 3, hand_spent=1):
        return False
    own, enemy = _players(obs)
    active, target = _active(own), _active(enemy)
    if active is None or target is None or own.asleep or own.paralyzed or own.confused:
        return False
    count = own.handCount - 1 + min(3, own.deckCount)
    if active.id == ALAKAZAM and 1072 in _legal_attacks(obs):
        return _damage(obs, active, 1072, target, count) >= target.hp
    # One held P attached to an unpowered benched Alak plus an already legal,
    # payable retreat is a bounded route; no drawn Energy is credited.
    if (obs.current.turn > 1 and not obs.current.retreated and not obs.current.energyAttached
            and any(o.type == OptionType.RETREAT for o in obs.select.option)
            and _catalog()[0][active.id].retreatCost <= len(active.energies)):
        for attacker in own.bench:
            if attacker.id != ALAKAZAM or _energy_gap(attacker, _catalog()[1][1072]) != 1:
                continue
            available = any(o.type == OptionType.ATTACH and o.inPlayArea == AreaType.BENCH
                            and _at(obs, o.inPlayArea, o.inPlayIndex) is attacker
                            and _at(obs, o.area, o.index) is not None
                            and _at(obs, o.area, o.index).id in (PSYCHIC, TELEPATH)
                            for o in obs.select.option)
            if available and _damage(obs, attacker, 1072, target, count - 1) >= target.hp:
                return True
    # The existing plan validates current native retreat and known payment.
    plan = _primary_retreat_plan(obs)
    if plan is None:
        return False
    attacker = next((p for p in own.bench if p.serial == plan['attacker_serial']), None)
    return bool(attacker and _damage(obs, attacker, 1072, target,
                                    count - plan['attachment_spent']) >= target.hp)


def _terminal_ex_bench_risk(obs, card):
    """Price a possible final-Prize gust target, never claim a held gust.

    Only a new basic ex, currently charged public Active, and exact supported
    public damage establish the exposure. Preserve a real draw-created KO.
    """
    own, enemy = _players(obs)
    data = _catalog()[0][card.id]
    attacker = _active(enemy)
    if (data.cardType != CardType.POKEMON or not data.basic or not (data.ex or data.megaEx)
            or not 0 < len(enemy.prize) <= (3 if data.megaEx else 2)
            or len(_board(own)) <= 1 or attacker is None
            or _best_attack(obs) >= 100000 or _terminal_ex_draw_ko(obs, card)):
        return False
    fresh = Pokemon(card.id, card.serial, data.hp, data.hp, True, [], [], [], [])
    return _survival_public_reply_damage(obs, attacker, fresh) >= fresh.hp


def _damage(obs, attacker, attack_id, defender, hand_count=None):
    if attacker is None or defender is None:
        return 0.0
    _, enemy = _players(obs)
    return _public_attack_damage(obs, attacker, attack_id, defender,
                                 obs.current.yourIndex, hand_count,
                                 attack_id == 183 and defender in enemy.bench)


def _attack_reflection(obs, attacker, attack_id, defender, damage=None, assume_active=False):
    """Spiky Energy triggers on positive attack damage, including a knockout.

    Attack counters (Powerful Hand) do not trigger it. The source is an attached
    Energy, so Pok챕mon-only effect shields and Jamming Tower do not stop it.
    Active-target attack planning may project a currently Benched gust target.
    """
    if attacker is None or defender is None or attack_id == 1072:
        return 0
    if damage is None:
        damage = _damage(obs, attacker, attack_id, defender)
    if damage <= 0:
        return 0
    _, area = _pokemon_location(obs, defender)
    if not assume_active and area != AreaType.ACTIVE:
        return 0
    return 20 * sum(e.id == 14 for e in defender.energyCards)


def _legal_attacks(obs):
    if obs.select.type != SelectType.MAIN:
        return []
    return [o.attackId for o in obs.select.option if o.type == OptionType.ATTACK]


def _ready_attacks(obs, pokemon):
    own, _ = _players(obs)
    if pokemon is None or pokemon.hp <= 0:
        return []
    if pokemon is _active(own) and obs.select.type == SelectType.MAIN:
        return _legal_attacks(obs)  # Exact active legality beats energy guesses.
    if obs.current.turn == 1:
        return []  # Energy alone does not allow a projected first-turn attack.
    if pokemon is _active(own) and (own.asleep or own.paralyzed):
        return []
    return [i for i in _catalog()[0][pokemon.id].attacks
            if _energy_gap(pokemon, _catalog()[1][i]) == 0]


def _duskull_recovery_count(obs, reserve_slots=0):
    own, _ = _players(obs)
    return min(3, max(0, own.benchMax - len(own.bench) - reserve_slots),
               sum(c.id == DUSKULL for c in own.discard))


def _attack_score(obs, attacker, attack_id, defender, hand_count=None):
    if attacker is None or defender is None:
        return -200.0  # Both active spots can be empty during simultaneous KO.
    own, enemy = _players(obs)
    if attack_id == 169:
        recovered = _duskull_recovery_count(obs)
        return 160 + 90 * recovered if recovered else -600.0
    if attack_id == 423:
        follow = max((_attack_score(obs, p, aid, defender, hand_count)
                      for p in own.bench if p.serial != attacker.serial
                      for aid in _ready_attacks(obs, p) if aid != 423), default=-500)
        # Trading Places ends this turn. Value only a ready next attacker or
        # removing a damaging condition, not switching for its own sake.
        condition = 80 if own.poisoned or own.burned else 0
        return min(350, follow / 8) + condition if follow > 0 else -500.0
    damage = _damage(obs, attacker, attack_id, defender, hand_count)
    if damage <= 0 and attack_id != 323:
        return -500.0
    knockout = damage >= defender.hp
    prizes = min(len(own.prize), _attack_prizes(obs, defender, attack_id)) if knockout else 0
    terminal = knockout and (prizes >= len(own.prize) or len(_board(enemy)) == 1)
    reflection = _attack_reflection(obs, attacker, attack_id, defender, damage, assume_active=True)
    self_knockout = reflection >= attacker.hp and reflection > 0
    if self_knockout:
        # State::finishCheck counts prize and empty-board conditions separately.
        # A fatal reflection is not a winning terminal just because we also KO.
        own_finishes = int(knockout and prizes >= len(own.prize)) + int(knockout and len(_board(enemy)) == 1)
        enemy_finishes = int(len(_board(own)) == 1) + int(_prizes(attacker) >= len(enemy.prize))
        if enemy_finishes and own_finishes <= enemy_finishes:
            return -10000.0
    value = 100 + min(damage, defender.hp) * 0.7 + 600 * prizes + 900 * knockout
    value -= reflection * 0.8
    if self_knockout:
        value -= 600 * _prizes(attacker) + 450
    if attack_id == 323:  # Budew trades tempo rather than raw damage.
        value += 220 if obs.current.turn <= 6 else 110
    if own.confused and attacker is _active(own):
        value *= 0.5
        terminal = False
    return 100000.0 + value if terminal else value


def _best_attack(obs, pokemon=None, defender=None, hand_count=None):
    own, enemy = _players(obs)
    all_snipe_targets = defender is None
    pokemon = _active(own) if pokemon is None else pokemon
    defender = _active(enemy) if defender is None else defender
    return max((max((_fez_snipe_score(obs, pokemon, target) for target in _board(enemy)), default=-200)
                if i == 183 and all_snipe_targets else _attack_score(obs, pokemon, i, defender, hand_count)
                for i in _ready_attacks(obs, pokemon)), default=-200.0)


def _hand_win(obs, hand_count):
    own, enemy = _players(obs)
    active, target = _active(own), _active(enemy)
    if active is None or target is None or active.id != ALAKAZAM or own.confused or own.asleep or own.paralyzed:
        return False
    if obs.current.turn == 1 or (obs.select.type == SelectType.MAIN and 1072 not in _legal_attacks(obs)):
        return False
    if _energy_gap(active, _catalog()[1][1072]) != 0:
        return False
    return (_damage(obs, active, 1072, target, hand_count) >= target.hp
            and (_prizes(target) >= len(own.prize) or len(_board(enemy)) == 1))


def _mill_reserve(obs):
    # Great Tusk 58's Library Out mills 1, or 4 after an Ancient Supporter.
    # Its public presence is enough to reserve for that printed next-turn cap;
    # no assumption about a hidden Supporter identity is made.
    _, enemy = _players(obs)
    return 4 if any(p.id == 58 for p in _board(enemy)) else 0


def _draw_safe(obs, count, hand_spent=0):
    own, _ = _players(obs)
    actual = min(count, own.deckCount)
    return own.deckCount > count + _mill_reserve(obs) or (actual > 0 and _hand_win(obs, max(0, own.handCount - hand_spent + actual)))


def _has_hand(obs, card_id):
    return any(c.id == card_id for c in _players(obs)[0].hand or [])


def _counts(obs):
    own, _ = _players(obs)
    return Counter(p.id for p in _board(own)), Counter(c.id for c in own.hand or [])


def _fez_draw_ready(obs):
    return (_MEMORY['last_own_ko_turn'] >= 1
            and _MEMORY['last_own_ko_turn'] == obs.current.turn - 1
            and _MEMORY['fez_used_turn'] != obs.current.turn
            and _ability_enabled(obs, Card(FEZ, -1, obs.current.yourIndex),
                                 owner=obs.current.yourIndex, area=AreaType.BENCH))


def _primary_retreat_plan(obs):
    """One visible attach/retreat route to an already powered Alakazam.

    Only the outgoing body may receive the one attachment. No second Energy,
    evolution, draw, future hand identity or new legal menu is invented. During
    MAIN an already payable retreat must be offered by the native menu. When a
    one-Energy shortage is repaired, or during a pending Hammer selection, the
    next native menu must still confirm transient restrictions.
    """
    own, enemy = _players(obs)
    active, target = _active(own), _active(enemy)
    if (active is None or target is None or obs.current.turn <= 1
            or obs.current.retreated or own.asleep or own.paralyzed
            or _primary_ko_ready(obs) or _best_attack(obs) >= 100000):
        return None
    cost = _catalog()[0][active.id].retreatCost
    missing = max(0, cost - len(active.energies))
    if missing > 1:
        return None
    if missing:
        if obs.current.energyAttached or not any(c.id in (PSYCHIC, TELEPATH)
                                                  for c in own.hand or []):
            return None
        if obs.select.type == SelectType.MAIN and not any(
                o.type == OptionType.ATTACH and o.inPlayArea == AreaType.ACTIVE
                and o.inPlayIndex == 0 and _at(obs, o.area, o.index) is not None
                and _at(obs, o.area, o.index).id in (PSYCHIC, TELEPATH)
                for o in obs.select.option):
            return None
    elif obs.select.type == SelectType.MAIN and not any(
            o.type == OptionType.RETREAT for o in obs.select.option):
        return None
    # Paying an unknown attached card might change HP or other effects. The
    # frozen list's P/Telepathic cards have no such retreat-discard consequence.
    if any(c.id not in (PSYCHIC, TELEPATH) for c in active.energyCards):
        return None
    best = None
    before = _best_attack(obs, hand_count=own.handCount)
    for attacker in own.bench:
        if (attacker.id != ALAKAZAM or attacker.hp <= 0
                or _energy_gap(attacker, _catalog()[1][1072])):
            continue
        moved = _project_gust_board(obs, target, attacker)
        if moved is None:
            continue
        state, acting, defending = moved
        after_own, _ = _players(state)
        after_own.handCount = max(0, own.handCount - missing)
        # The old active's legal ATTACK options cannot be transferred to Alak.
        state.select.type = SelectType.CARD
        value = _attack_score(state, acting, 1072, defending, after_own.handCount)
        if value <= 0 or value <= before + 100:
            continue
        plan = {'value': value - 80 * cost - 40 * missing,
                'attack_value': value, 'attacker_serial': attacker.serial,
                'attachment_spent': missing, 'hand_after': after_own.handCount}
        if best is None or plan['value'] > best['value']:
            best = plan
    return best


def _bench_damage_threat(obs):
    """Flower Curtain prevents ordinary bench damage, including on non-ex evolutions."""
    own, enemy = _players(obs)
    cards, attacks, _ = _catalog()
    exposed = any(not (cards[p.id].ex or cards[p.id].megaEx) for p in own.bench)
    if not exposed:
        return False
    return any('Benched Pokémon' in attacks[aid].text
               and 'damage' in attacks[aid].text.lower()
               and 'damage counters' not in attacks[aid].text.lower()
               for p in _board(enemy) for aid in cards[p.id].attacks)


def _tech_wanted(obs, card_id):
    """Recognize only currently public barriers that need ordinary damage."""
    _, enemy = _players(obs)
    for pokemon in _board(enemy):
        if pokemon.id in (1348, 1349, 1374, 1375):
            return True
        if any(e.id == 11 for e in pokemon.energyCards):
            return True
        if card_id == 390 and pokemon.id == 345:
            return True
    return False


def _primary_ko_ready(obs):
    """Do not spend a hand card to pivot away from an available primary KO."""
    own, enemy = _players(obs)
    active, target = _active(own), _active(enemy)
    return (active is not None and target is not None and active.id == ALAKAZAM
            and 1072 in _legal_attacks(obs)
            and _damage(obs, active, 1072, target) >= target.hp)


def _tech_hand_plan(obs, card):
    """Bench the two-energy attacker before allocating this turn's energy."""
    own, _ = _players(obs)
    if (card.id not in (272, 390) or not _tech_wanted(obs, card.id)
            or obs.current.energyAttached or len(own.bench) >= own.benchMax
            or _primary_ko_ready(obs) or _best_attack(obs) >= 100000):
        return False
    if any(p.id in (272, 390) and _tech_wanted(obs, p.id) for p in _board(own)):
        return False
    return any(c.id in (PSYCHIC, TELEPATH) for c in own.hand or [])


def _pivot_attachment_score(obs, pokemon):
    """One available attachment must immediately pay for a useful retreat.

    Native options remain authoritative: hidden transient retreat locks and
    cost modifiers are not serialized. Only forecast the printed one-energy
    shortfall and let the next MAIN options confirm whether retreat is legal.
    """
    own, _ = _players(obs)
    active = _active(own)
    if (active is None or pokemon is not active or obs.current.retreated
            or obs.current.energyAttached or own.asleep or own.paralyzed
            or any(o.type == OptionType.RETREAT for o in obs.select.option)
            or _primary_ko_ready(obs) or _best_attack(obs) >= 100000):
        return None
    cost = _catalog()[0][active.id].retreatCost
    if cost - len(active.energies) != 1:
        return None
    primary = _primary_retreat_plan(obs)
    if primary is not None and primary['attachment_spent'] == 1:
        return _hand_cost(obs, max(950, primary['value'] + 200))
    ready = []
    for tech in own.bench:
        if tech.id not in (272, 390, FEZ) or not _tech_wanted(obs, tech.id):
            continue
        attack_id = {272: 371, 390: 546, FEZ: 183}[tech.id]
        if attack_id in _ready_attacks(obs, tech):
            ready.append(_best_attack(obs, tech))
    best = max(ready, default=-500)
    if best <= 0 or best <= _best_attack(obs) + 100:
        return None
    # The attachment is on the outgoing body, so it cannot remove the ready
    # attacker's energy. The hand-cost guard still protects a current hand KO.
    return _hand_cost(obs, max(950, best + 200))


def _tech_attach_score(obs, energy, pokemon):
    if pokemon.id not in (390, 272, FEZ):
        return None
    if not _tech_wanted(obs, pokemon.id):
        return -500.0
    own, enemy = _players(obs)
    attack_id = {390: 546, 272: 371, FEZ: 183}[pokemon.id]
    attack = _catalog()[1][attack_id]
    gap = _energy_gap(pokemon, attack)
    if gap <= 0:
        return -500.0
    # Psychic and Telepathic both pay these P/C costs. Charging is useful only
    # when at least one intended public target can take ordinary damage.
    targets = _board(enemy) if pokemon.id == FEZ else [_active(enemy)]
    if not any(p and _damage(obs, pokemon, attack_id, p) > 0 for p in targets):
        return -500.0
    hand_tech = any(_tech_hand_plan(obs, c) for c in own.hand or [])
    if pokemon.id == FEZ and (hand_tech or any(p.id in (390, 272) and _tech_wanted(obs, p.id)
                                               for p in _board(own))):
        return -500.0  # The two-energy tech gets scarce attachments first.
    target = _active(enemy)
    blocked_primary = target and _counter_blocked(obs, target)
    score = (740 if blocked_primary else 520) + (140 if gap == 1 else 0)
    if pokemon is _active(own):
        score += 80
    if pokemon.id == FEZ:
        score -= 180  # CCC is slower than either two-energy alternative.
    return _hand_cost(obs, score, essential=pokemon is _active(own) and gap == 1)


def _fez_snipe_score(obs, attacker, defender):
    """Score an actual Cruel Arrow target; bench Spiky never retaliates."""
    if attacker is None or defender is None:
        return -500.0
    own, enemy = _players(obs)
    damage = _damage(obs, attacker, 183, defender)
    if damage <= 0:
        return -500.0
    knockout = damage >= defender.hp
    prizes = min(len(own.prize), _attack_prizes(obs, defender, 183)) if knockout else 0
    terminal = knockout and (prizes >= len(own.prize) or len(_board(enemy)) == 1)
    reflection = _attack_reflection(obs, attacker, 183, defender, damage,
                                    assume_active=False)
    self_knockout = reflection > 0 and reflection >= attacker.hp
    if self_knockout:
        own_finishes = int(knockout and prizes >= len(own.prize)) + int(knockout and len(_board(enemy)) == 1)
        enemy_finishes = int(len(_board(own)) == 1) + int(_prizes(attacker) >= len(enemy.prize))
        if enemy_finishes and own_finishes <= enemy_finishes:
            return -10000.0
    value = 100 + min(damage, defender.hp) * 0.7 + 600 * prizes + 900 * knockout - reflection * 0.8
    if self_knockout:
        value -= 600 * _prizes(attacker) + 450
    if own.confused and attacker is _active(own):
        value *= 0.5
        terminal = False
    return 100000.0 + value if terminal else value


def _special_energy_units(pokemon, cid):
    """The shipped R2 energy definitions, never a card-name approximation."""
    data = _catalog()[0][pokemon.id]
    if cid == 10:
        return [EnergyType.RAINBOW] * 2 if data.stage2 else [EnergyType.COLORLESS]
    if cid == 12:
        return [EnergyType.RAINBOW]
    if cid == 15:
        return [EnergyType.TEAM_ROCKET] * 2
    if cid == 16:
        return [EnergyType.RAINBOW if data.basic else EnergyType.COLORLESS]
    if cid == 17:
        return [EnergyType.COLORLESS] * (1 if data.basic else 3)
    return {9: [EnergyType.COLORLESS], 11: [EnergyType.COLORLESS],
            13: [EnergyType.COLORLESS], 14: [EnergyType.COLORLESS],
            18: [EnergyType.GRASS], 19: [EnergyType.PSYCHIC],
            20: [EnergyType.FIGHTING]}.get(cid)


def _project_hammer(obs, target, indices, hand_spent):
    """Remove specified *card* indices on a copy; multi-unit Energy stays exact."""
    projected = deepcopy(obs)
    own, enemy = _players(projected)
    body = next((p for p in _board(enemy) if p.serial == target.serial), None)
    if body is None or len(set(indices)) != len(indices):
        return None
    body._research_removed_units_exact = True
    for index in sorted(indices, reverse=True):
        if not 0 <= index < len(body.energyCards):
            return None
        removed = body.energyCards.pop(index)
        units = _special_energy_units(body, removed.id)
        # Unknown future special Energy supplies are not guessed. In that case
        # keep the old units, understating denial while still removing its card.
        if units is not None:
            if any(Counter(body.energies)[unit] < count for unit, count in Counter(units).items()):
                body._research_removed_units_exact = False
            for unit in units:
                if unit in body.energies:
                    body.energies.remove(unit)
        else:
            body._research_removed_units_exact = False
        if removed.id == 18 and _catalog()[0][body.id].energyType == EnergyType.GRASS:
            body.maxHp -= 20
            body.hp -= 20  # Preserve damage counters when maximum HP shrinks.
    own.handCount = max(0, own.handCount - hand_spent)
    # Keep the current hand's indices for the ORIGINAL native MAIN Trainer
    # options. Only H changes in this attack-value projection; no action from
    # its copied options is executed. Removing hand entries would shift Boss.
    return projected, body


def _research_gust_available(obs, cid):
    own, _ = _players(obs)
    if not _has_hand(obs, cid) or (cid == BOSS and obs.current.supporterPlayed):
        return False
    if obs.select.type == SelectType.MAIN:
        return any(o.type == OptionType.PLAY and _at(obs, AreaType.HAND, o.index) is not None
                   and _at(obs, AreaType.HAND, o.index).id == cid
                   for o in obs.select.option if o.index is not None)
    # A pending Hammer cannot carry a fabricated MAIN option list. The next
    # native MAIN will confirm the forecasted hand Trainer is actually legal.
    return True


def _research_turn_attack(obs):
    """Current attack, one held gust, or exact-cost primary attach/retreat."""
    own, enemy = _players(obs)
    best = _best_attack(obs, hand_count=own.handCount)
    pivot = _primary_retreat_plan(obs)
    if pivot is not None:
        best = max(best, pivot['value'])
    for cid in (BOSS, PRIME):
        if not _research_gust_available(obs, cid):
            continue
        attackers = own.bench if cid == PRIME else [_active(own)]
        for target in enemy.bench:
            for attacker in attackers:
                if attacker is None or attacker.hp <= 0:
                    continue
                attacks = _ready_attacks(obs, attacker)
                moved = _project_gust_board(obs, target, attacker if cid == PRIME else None)
                if moved is None:
                    continue
                state, acting, defending = moved
                # _project_gust_board intentionally preserves the real select;
                # this helper supplies the original attack IDs and exact H-1.
                value = max((_attack_score(state, acting, aid, defending,
                                            max(0, own.handCount - 1)) for aid in attacks), default=-200)
                best = max(best, value - 86)  # One card and one scarce gust.
    return best


def _hammer_denial_score(obs, projected, before, after):
    """Tempo only when removing units disables an actually powered public attack."""
    if (before.hp <= 0 or after.hp <= 0
            or not getattr(after, '_research_removed_units_exact', False)
            or (EnergyType.RAINBOW in before.energies and EnergyType.TEAM_ROCKET in before.energies)
            or any(c.id == 1266 for c in obs.current.stadium)):
        return 0.0
    data, attacks = _catalog()[0][before.id], _catalog()[1]
    denied = []
    for aid in data.attacks:
        attack = attacks[aid]
        gap = _energy_gap(after, attack)
        if _energy_gap(before, attack) == 0 and gap > 0 and attack.damage > 0:
            denied.append(min(240, attack.damage * 0.65) + (120 if gap > 1 else 25))
    return max(denied, default=0.0)


def _hammer_target_score(obs, target, energy_index, *, already_played=False, extra_copy=0):
    """Value the actual index, including a bounded all-protection removal chain.

    A MAIN Hammer spends one hand card. At its discard selection that cost is
    already reflected in the observation. Remaining copies fund the complete
    Mist/Rock chain; one Hammer against two Mist does not claim an unlocked KO.
    """
    own, enemy = _players(obs)
    if target not in _board(enemy) or not 0 <= energy_index < len(target.energyCards):
        return -1000.0
    card = target.energyCards[energy_index]
    if _catalog()[0][card.id].cardType != CardType.SPECIAL_ENERGY:
        return -1000.0
    if target.id in (424, 994) and _ability_enabled(obs, target):
        return -1000.0  # Cetitan ex/Fraxure prevent opposing Item effects.
    held = sum(c.id == HAMMER for c in own.hand or []) + extra_copy
    copies = held + int(already_played)
    if copies < 1:
        return -400.0
    baseline = _research_turn_attack(obs)
    if baseline >= 100000:
        return -400.0  # An already available win needs no Energy removal.
    chains = [(energy_index,)]
    fighting = _catalog()[0][target.id].energyType == EnergyType.FIGHTING
    blockers = tuple(i for i, c in enumerate(target.energyCards)
                     if c.id == 11 or (fighting and c.id == 20))
    if energy_index in blockers and 1 < len(blockers) <= copies:
        chains.append(blockers)
    score = -400.0
    for indices in chains:
        spent = len(indices) - int(already_played)
        moved = _project_hammer(obs, target, indices, spent)
        if moved is None:
            continue
        state, defending = moved
        after_own, _ = _players(state)
        if defending.hp <= 0:
            prizes = min(len(own.prize), _prizes(defending))
            terminal = prizes >= len(own.prize) or len(_board(enemy)) == 1
            value = (100000 if terminal else 100 + 900 + 600 * prizes) - 16 * spent
        else:
            value = _research_turn_attack(state)
        if value >= 1000 and value > baseline + 450:
            score = max(score, value + 90 - 45 * (len(indices) - 1))
        if len(indices) == 1:
            denial = _hammer_denial_score(obs, state, target, defending)
            # Do not break a current KO for modest denial on a different body.
            active = _active(own)
            current = _active(enemy)
            if (active and current and active.id == ALAKAZAM and 1072 in _legal_attacks(obs)
                    and _damage(obs, active, 1072, current) >= current.hp
                    > _damage(state, _active(after_own), 1072, _active(_players(state)[1]))):
                denial = 0
            if denial > 0:
                score = max(score, denial - 16 * spent)
    return score


def _hammer_best_score(obs, extra_copy=0):
    _, enemy = _players(obs)
    return max((_hammer_target_score(obs, p, i, extra_copy=extra_copy)
                for p in _board(enemy) for i, c in enumerate(p.energyCards)
                if _catalog()[0][c.id].cardType == CardType.SPECIAL_ENERGY), default=-400.0)


def _patrat_utility(obs, *, proposed=True):
    """Price blocked existing-counter movement, never new Dusk/Alak placement."""
    own, enemy = _players(obs)
    if any(p.id == PATRAT and _ability_enabled(obs, p)
           for player in (own, enemy) for p in _board(player)):
        return -400.0
    if proposed and not _ability_enabled(obs, Card(PATRAT, -1, obs.current.yourIndex),
                                         owner=obs.current.yourIndex, area=AreaType.BENCH):
        return -400.0
    monks = [p for p in _board(enemy) if p.id == 112 and _ability_enabled(obs, p)
             and any(e in (EnergyType.DARKNESS, EnergyType.RAINBOW)
                     for e in p.energies)]
    if not monks:
        return -300.0
    def shielded(p):
        return p.id in (1348, 1349, 1374, 1375, 1040) and _ability_enabled(obs, p)
    damaged = sum(max(0, p.maxHp - p.hp) for p in _board(enemy) if not shielded(p))
    active, target = _active(own), _active(enemy)
    # Counters we can actually place this turn may become next turn's fuel.
    if active and target and not shielded(target) and 1072 in _legal_attacks(obs):
        hit = _damage(obs, active, 1072, target)
        if 0 < hit < target.hp:
            damaged += hit
    prevented = min(30 * len(monks), damaged)
    if prevented <= 0:
        return -300.0
    # Patrat also suppresses our movement; a later Munk variant must pay for it.
    own_monks = sum(p.id == 112 and _ability_enabled(obs, p)
                   and any(e in (EnergyType.DARKNESS, EnergyType.RAINBOW) for e in p.energies)
                   for p in _board(own))
    value = 120 + 7 * prevented - 120 * own_monks
    vulnerable = sum(0 < p.hp <= prevented for p in _board(own))
    value += min(260, vulnerable * 130)
    if len(own.bench) >= own.benchMax - 1:
        value -= 180  # Keep room for the primary attacker and sacrifice line.
    if len(enemy.prize) <= 1:
        value -= 160  # A new 70-HP gust target exposes the final Prize.
    return value


def _survival_public_reply_damage(obs, attacker, defender):
    """Current charged supported public attack; no assumed future attachment."""
    if attacker is None or defender is None:
        return 0.0
    cards, attacks, _ = _catalog()
    _, enemy = _players(obs)
    if attacker is _active(enemy) and (enemy.asleep or enemy.paralyzed or enemy.confused):
        return 0.0
    amount = 0.0
    supported = (1072, 1071, 937, 479, 1092, 980, 982, 983, 154, 371, 546, 183,
                 195, 120, 1708, 1240, 253)
    for aid in cards[attacker.id].attacks:
        if aid not in supported or _energy_gap(attacker, attacks[aid]) or _public_attack_locked(obs, attacker, aid):
            continue
        amount = max(amount, _public_attack_damage(obs, attacker, aid, defender,
                                                   1 - obs.current.yourIndex))
    return amount


def _dudun_survival_exposure(obs, source, projected=None):
    """Returning an active one-Prize body must not newly expose the final Prize.

    A benched Dudun returning while an ex is already active is a different
    situation; this guard does not suppress its potentially rescuing draw.
    """
    own, enemy = _players(obs)
    if source is not _active(own) or len(_board(own)) != 2:
        return False
    remaining = [p for p in _board(own) if p.serial != source.serial]
    if len(remaining) != 1 or not (_prizes(source) < len(enemy.prize) <= _prizes(remaining[0])):
        return False
    state = projected
    if state is None:
        moved = _dudun_project(obs, source)
        if moved is None:
            return False
        state = moved[0]
    after_own, after_enemy = _players(state)
    defender = _active(after_own)
    return bool(defender and _survival_public_reply_damage(
        state, _active(after_enemy), defender) >= defender.hp)


def _survival_backup_needed(obs):
    own, _ = _players(obs)
    if not _board(own) or len(own.bench) >= own.benchMax:
        return False
    if len(_board(own)) == 1:
        return True
    active = _active(own)
    return bool(active and active.id in (*DUNSPARCE, DUDUNSPARCE)
                and _dudun_survival_exposure(obs, active))


def _survival_pad_basic_value(obs, card, selected=None):
    """A backup search is fulfilled by a currently playable Basic, not an evo.

    Search purpose is tied to the selected own Pad's serial and current turn.
    Every candidate is still an option in the current native search menu.
    """
    effect = obs.select.effect
    if effect is None or effect.id != PAD:
        return None
    pending = _MEMORY.get('survival_search')
    requested = pending == (_MEMORY['round'], obs.current.turn, effect.serial)
    if not (_survival_backup_needed(obs) or requested):
        return None
    own, _ = _players(obs)
    data = _catalog()[0][card.id]
    if data.cardType == CardType.POKEMON and data.basic and len(own.bench) < own.benchMax:
        return 4000 + _bench_value(obs, card, selected)
    return -1000 + _card_value(obs, card, selected)


def _survival_ex_escape_targets(obs):
    """Current one-Prize pivots for a publicly lethal final-Prize ex exposure.

    This is a defensive retreat signal, not an enemy-turn survival proof. It
    never requires the replacement to attack, and does not assume a hidden
    opponent attachment, gust or ability. Native options still own legality.
    """
    own, enemy = _players(obs)
    active, attacker = _active(own), _active(enemy)
    if (active is None or attacker is None or own.asleep or own.paralyzed
            or _prizes(active) <= 1 or not (1 < len(enemy.prize) <= _prizes(active))
            or _survival_public_reply_damage(obs, attacker, active) < active.hp
            or _best_attack(obs) >= 100000):
        return []
    # A current legal KO removes this concrete charged attacker. Keep it
    # available rather than retreating merely because its pre-KO reply is high.
    if not own.confused and obs.select.type == SelectType.MAIN:
        for aid in _legal_attacks(obs):
            hit = _damage(obs, active, aid, attacker)
            if (hit >= attacker.hp and _attack_score(obs, active, aid, attacker) > 0
                    and _attack_reflection(obs, active, aid, attacker, hit,
                                           assume_active=True) < active.hp):
                return []
    targets = [p for p in own.bench if p and _prizes(p) == 1]
    if not targets:
        return []
    # Do not call moving the ex to the bench an escape if an already charged
    # supported snipe can still take its terminal Prize there.
    state = deepcopy(obs)
    projected, _ = _players(state)
    exposed = _active(projected)
    replacement = next(p for p in projected.bench if p.serial == targets[0].serial)
    projected.bench = [p for p in projected.bench if p.serial != replacement.serial] + [exposed]
    projected.active = [replacement]
    projected.asleep = projected.paralyzed = projected.confused = False
    cards, attacks, _ = _catalog()
    for aid in cards[attacker.id].attacks:
        if _energy_gap(attacker, attacks[aid]) != 0:
            continue
        if aid == 183:
            return []  # Cruel Arrow can choose the benched ex; no active-only escape.
        if aid == 154 and exposed.hp <= 60 and not _counter_blocked(state, exposed):
            return []
        if aid == 937 and exposed.hp <= 30:
            return []
    if _ability_enabled(state, exposed) and not _counter_blocked(state, exposed, ability=True):
        shrouds = sum(p.id == 104 and _ability_enabled(state, p)
                      for side in state.current.players for p in _board(side))
        if exposed.hp <= 10 * shrouds:
            return []  # The first public checkup alone would KO the benched ex.
    return targets


def _survival_ex_escape_main_score(obs, option):
    """Pay only a current legal retreat or its one-energy printed shortfall."""
    if option.type not in (OptionType.ATTACH, OptionType.RETREAT):
        return None
    if not _survival_ex_escape_targets(obs):
        return None
    if option.type == OptionType.RETREAT:
        return 2400.0  # A nonattacking Basic can still preserve the final Prize.
    own, _ = _players(obs)
    active = _active(own)
    energy = _at(obs, option.area, option.index)
    recipient = _at(obs, option.inPlayArea, option.inPlayIndex)
    if (energy is None or energy.id not in (PSYCHIC, TELEPATH) or recipient is not active
            or obs.current.retreated or obs.current.energyAttached
            or any(o.type == OptionType.RETREAT for o in obs.select.option)
            or _catalog()[0][active.id].retreatCost - len(active.energies) != 1):
        return None
    # Prefer Basic P when both actual attachment options exist; Telepathic can
    # create a search popup, so the next native menu must confirm its payment.
    return 2200.0 if energy.id == PSYCHIC else 2150.0


def _patrat_public_checkup_risk(obs):
    """Price a last-Prize Patrat that public checkups and bench damage can KO.

    Patrat prevents moving counters, not Froslass/Dragapult placing new ones.
    Keep ordinary backup survival and all Dunsparce draw lines unaffected.
    """
    own, enemy = _players(obs)
    if len(enemy.prize) != 1 or len(_board(own)) <= 1:
        return False
    proposed = Card(PATRAT, -1, obs.current.yourIndex)
    if not _ability_enabled(obs, proposed, owner=obs.current.yourIndex, area=AreaType.BENCH):
        return False
    if any(c.id == 1264 for c in obs.current.stadium):
        return False
    shrouds = sum(p.id == 104 and _ability_enabled(obs, p) for p in _board(enemy))
    checkup = 20 * shrouds  # This turn's end, then the public reply's end.
    attacker = _active(enemy)
    bench_hit = 0
    if attacker:
        for aid in _catalog()[0][attacker.id].attacks:
            if _energy_gap(attacker, _catalog()[1][aid]) != 0:
                continue
            if aid == 154:
                bench_hit = max(bench_hit, 60)
            if aid == 937 and not any(p.id == SHAYMIN and _ability_enabled(obs, p)
                                      for p in _board(own)):
                bench_hit = max(bench_hit, 30)
    return checkup + bench_hit >= _catalog()[0][PATRAT].hp


def _dudun_project(obs, source, *, evolving=False):
    projected = deepcopy(obs)
    own, _ = _players(projected)
    body = next((p for p in _board(own) if p.serial == source.serial), None)
    if body is None or len(_board(own)) <= 1 or own.deckCount <= 0:
        return None
    drawn = min(3, own.deckCount)
    returned = 1 + len(body.preEvolution) + len(body.energyCards) + len(body.tools) + int(evolving)
    own.deckCount = own.deckCount - drawn + returned
    own.handCount = max(0, own.handCount - int(evolving) + drawn)
    is_active = body is _active(own)
    own.bench = [p for p in own.bench if p.serial != body.serial]
    if is_active:
        own.active = []
        promoted = max(own.bench, key=lambda p: _promotion_score(projected, p), default=None)
        if promoted is None:
            return None
        own.bench.remove(promoted)
        own.active = [promoted]
        for condition in ('poisoned', 'burned', 'asleep', 'paralyzed', 'confused'):
            setattr(own, condition, False)
        # Original active ATTACK indices cannot be transferred to this body.
        projected.select.type = SelectType.CARD
    return projected, drawn, returned


def _dudun_draw_score(obs, source, *, evolving=False):
    own, _ = _players(obs)
    if source is None or source.id not in (*DUNSPARCE, DUDUNSPARCE):
        return -1000.0
    trial = deepcopy(source)
    trial.id = DUDUNSPARCE
    owner, area = _pokemon_location(obs, source)
    if not _ability_enabled(obs, trial, owner=owner, area=area):
        return -800.0
    moved = _dudun_project(obs, source, evolving=evolving)
    if moved is None:
        return -10000.0
    state, drawn, _ = moved
    after_own, _ = _players(state)
    before, after = _best_attack(obs), _best_attack(state, hand_count=after_own.handCount)
    if before >= 100000:
        return -400.0
    if after >= 100000:
        return after + 240 - 30 * int(evolving)
    if _dudun_survival_exposure(obs, source, state):
        if evolving:
            # Evolution does not automatically invoke Run Away Draw. Preserve
            # the HP upgrade while the later current ABILITY remains declined.
            return _hand_cost(obs, 180)
        return -600.0  # Preserve the one-Prize active until a real rescue exists.
    # The actual return replenishes the deck. Draw safety is based on that
    # final size, not D>3; retain the next draw and visible Great Tusk reserve.
    if after_own.deckCount <= _mill_reserve(state):
        return -800.0
    if after > before + 450:
        return after + 170 - 30 * int(evolving)
    value = 260 + 45 * drawn + 16 * (drawn - int(evolving))
    value -= 60 * len(source.energyCards) + 35 * len(source.tools)
    if source is _active(own):
        value += max(-250, min(300, after - before))
    else:
        value += 25  # Return opens a bench slot after drawing.
    return value - 30 * int(evolving)


def _dunsparce_setup_value(obs, selected=None):
    own, _ = _players(obs)
    selected = selected or Counter()
    if any(p.id in (*DUNSPARCE, DUDUNSPARCE) for p in _board(own)) or any(
            selected[i] for i in (*DUNSPARCE, DUDUNSPARCE)):
        return -400.0
    if not _ability_enabled(obs, Card(DUDUNSPARCE, -1, obs.current.yourIndex),
                            owner=obs.current.yourIndex, area=AreaType.BENCH):
        return -400.0
    if own.deckCount <= 1 + _mill_reserve(obs):
        return -300.0
    have_evolution = _has_hand(obs, DUDUNSPARCE)
    possible = _known_remaining(obs)[DUDUNSPARCE] > 0
    if not have_evolution and not possible:
        return -300.0
    value = 220 if have_evolution else 105  # Unseen includes prizes, not a deck fact.
    if len(own.bench) >= own.benchMax - 1:
        value -= 200
    if not any(p.id in (*ABRA, KADABRA, ALAKAZAM) for p in _board(own)):
        value -= 150
    return value


def _card_value(obs, card, selected=None):
    if card is None:
        return -200.0
    own, _ = _players(obs)
    board, hand = _counts(obs)
    selected = selected or Counter()
    cid, data = card.id, _catalog()[0][card.id]
    hand = hand + selected
    abra_count = sum(board[i] for i in ABRA)
    if cid in ABRA:
        future = abra_count + board[KADABRA] + board[ALAKAZAM]
        value = 240 if future == 0 else 160 if future == 1 else 70 if future == 2 else 20
        value -= 45 * max(0, sum(hand[i] for i in ABRA) - 1)
    elif cid == KADABRA:
        value = 210 if abra_count > hand[cid] - 1 else 80
        if not abra_count and not sum(hand[i] for i in ABRA):
            value = 45
    elif cid == ALAKAZAM:
        ready = board[KADABRA] + (abra_count if hand[RARE_CANDY] else 0)
        value = 260 if ready >= max(1, hand[cid]) else 130
        if board[ALAKAZAM] >= 2 and not ready:
            value = 55
    elif cid == DUSKULL:
        line = board[DUSKULL] + board[DUSCLOPS] + board[DUSKNOIR]
        value = 160 if line == 0 else 70 if line == 1 else 20
    elif cid in (DUSCLOPS, DUSKNOIR):
        source = DUSKULL if cid == DUSCLOPS else DUSCLOPS
        ready = board[source] + (board[DUSKULL] if cid == DUSKNOIR and hand[RARE_CANDY] else 0)
        value = (190 if cid == DUSCLOPS else 220) if ready else 85
    elif cid == FEZ:
        value = (240 if _fez_draw_ready(obs) else 90) if not board[FEZ] and own.deckCount > 5 else 15
    elif cid == BUDEW:
        value = 150 if obs.current.turn <= 4 and not board[BUDEW] else 25
    elif cid == SHAYMIN:
        value = 280 if not board[SHAYMIN] and _bench_damage_threat(obs) else 20
    elif cid in (390, 272):
        value = 300 if not board[cid] and _tech_wanted(obs, cid) else 60 if not board[cid] else 20
    elif cid == PATRAT:
        value = max(5, min(320, _patrat_utility(obs))) if not board[cid] else 5
    elif cid in DUNSPARCE:
        value = max(5, _dunsparce_setup_value(obs, selected))
    elif cid == DUDUNSPARCE:
        ready = [p for p in _board(own) if p.id in DUNSPARCE and not p.appearThisTurn]
        benefit = max((_dudun_draw_score(obs, p, evolving=True) for p in ready), default=-400)
        value = min(350, benefit) if benefit > 0 else 30 if any(hand[i] for i in DUNSPARCE) else 5
    elif cid == HAMMER:
        in_hand = any(c is card or (getattr(card, 'serial', None) is not None
                                   and c.serial == card.serial) for c in own.hand or [])
        benefit = _hammer_best_score(obs, extra_copy=0 if in_hand else 1)
        value = min(330, benefit) if benefit > 0 else 10
    elif cid in (PSYCHIC, TELEPATH):
        missing = sum(p.id in (*ABRA, KADABRA, ALAKAZAM) and not p.energies for p in _board(own))
        missing += sum(p.id in (390, 272, FEZ) and _tech_wanted(obs, p.id)
                       and _energy_gap(p, _catalog()[1][{390: 546, 272: 371, FEZ: 183}[p.id]]) > 0
                       for p in _board(own))
        value = 250 if missing and not hand[PSYCHIC] + hand[TELEPATH] > 1 else 90
        if cid == TELEPATH:
            value += 35 if len(own.bench) < own.benchMax else 5
    elif cid == RARE_CANDY:
        value = 190 if abra_count and hand[ALAKAZAM] else 90
    elif cid in (HILDA, DAWN, GWYNN):
        value = 115 if not obs.current.supporterPlayed else 55
    elif cid in (PAD, POFFIN):
        value = 150 if board[ALAKAZAM] == 0 else 70
    elif cid == NIGHT:
        value = 100 if own.discard else 50
    elif cid == PRIME:
        value = 130
    elif cid == BOSS:
        value = 80
    else:
        value = 45
    if data.cardType == CardType.POKEMON:
        value -= 35 * max(0, hand[cid] - 1)
    return max(5.0, value)


def _bench_value(obs, card, selected=None):
    own, _ = _players(obs)
    if _terminal_ex_bench_risk(obs, card):
        return -800.0
    selected = selected or Counter()
    board = Counter(p.id for p in _board(own)) + selected
    if card.id in ABRA:
        line = sum(board[i] for i in ABRA) + board[KADABRA] + board[ALAKAZAM]
        return 400 if line == 0 else 290 if line == 1 else 100 if line == 2 else -80
    if card.id == DUSKULL:
        line = board[DUSKULL] + board[DUSCLOPS] + board[DUSKNOIR]
        return 250 if line == 0 else 70 if line == 1 else -120
    if card.id == BUDEW:
        return 180 if obs.current.turn <= 3 and not board[BUDEW] else -90
    if card.id == FEZ:
        return (650 if _fez_draw_ready(obs) and _draw_safe(obs, 3, hand_spent=1) else 90) if not board[FEZ] and own.deckCount > 5 else -120
    if card.id == SHAYMIN:
        return _card_value(obs, card) - 40 if not board[SHAYMIN] else -120
    if card.id in (390, 272):
        return 330 if not board[card.id] and _tech_wanted(obs, card.id) else -500
    if card.id == PATRAT:
        if _patrat_public_checkup_risk(obs):
            return -600.0
        return _patrat_utility(obs) if not board[PATRAT] else -400
    if card.id in DUNSPARCE:
        return _dunsparce_setup_value(obs, selected)
    return -100


def _hand_cost(obs, score, spent=1, gained=0, essential=False):
    own, enemy = _players(obs)
    active, target = _active(own), _active(enemy)
    if active is not None and active.id == ALAKAZAM and 1072 in _legal_attacks(obs):
        before = _damage(obs, active, 1072, target)
        after = _damage(obs, active, 1072, target, max(0, own.handCount - spent + gained))
        if target and before >= target.hp > after and not essential:
            return -300.0  # Preserve an already legal KO rather than play a spare Basic.
        if target and after >= target.hp and not own.confused:
            current = _best_attack(obs)
            # An immediate win needs no further preparation. Otherwise grow the
            # hand or set a useful backup before committing the same safe KO.
            if current < 100000 and (gained > spent or score >= 200):
                future = _attack_score(obs, active, 1072, target,
                                       max(0, own.handCount - spent + gained))
                return max(score, future + (350 if gained > spent else 100))
    return score + 16 * (gained - spent)


def _can_switch(obs):
    return any(o.type == OptionType.RETREAT or
               (o.type == OptionType.PLAY and _at(obs, AreaType.HAND, o.index).id == PRIME)
               for o in obs.select.option)


def _promotion_hand_plan(obs, pokemon):
    """Known-hand KO route at an opposing-turn forced promotion only.

    A held Alakazam and zero/one held P attachment can ready a Kadabra;
    one held P can also ready an existing unpowered Alakazam next own turn.
    Count no draw, evolution Ability or Telepath search.
    This is a bounded plan; subsequent native EVOLVE/ATTACH/ATTACK menus
    remain authoritative. Voluntary SWITCH and current MAIN are unchanged.
    """
    own, enemy = _players(obs)
    target = _active(enemy)
    turn, first = obs.current.turn, obs.current.firstPlayer
    if (obs.select.type != SelectType.CARD
            or obs.select.context != SelectContext.TO_ACTIVE
            or first not in (0, 1) or turn <= 1
            or (first + turn - 1) % 2 == obs.current.yourIndex
            or own.deckCount <= _mill_reserve(obs)
            or pokemon not in own.bench or pokemon.id not in (KADABRA, ALAKAZAM)
            or pokemon.hp <= 0 or pokemon.appearThisTurn
            or target is None or target.hp <= 0):
        return None
    # The public current attack, not an unseen restriction, can bar evolution.
    if pokemon.id == KADABRA and any(
            owner == 1 - obs.current.yourIndex and aid == 58 and at_turn == turn
            for (owner, _), (aid, at_turn) in _MEMORY['attacks'].items()):
        return None  # Bronzong's Evolution Jammer lasts into our next turn.
    cards, attacks, _ = _catalog()
    if (pokemon.tools
            or any(c.id not in (PSYCHIC, TELEPATH) for c in pokemon.energyCards)):
        return None
    evolved = None
    if pokemon.id == KADABRA:
        evolved = next((c for c in own.hand or [] if c.id == ALAKAZAM), None)
        if (evolved is None or not cards[evolved.id].stage2
                or cards[evolved.id].evolvesFrom != cards[pokemon.id].name):
            return None
    elif _energy_gap(pokemon, attacks[1072]) != 1:
        return None  # This new route is only one held typed Energy short.
    # Keep currently ready attackers, including a ready Kadabra's terminal KO.
    if _best_attack(obs, pokemon) >= 100000 or any(
            p.serial != pokemon.serial and p.energies and _best_attack(obs, p) > 0
            for p in _board(own)):
        return None
    moved = _project_gust_board(obs, target, pokemon)
    if moved is None:
        return None
    state, acting, defending = moved
    after_own, after_enemy = _players(state)
    shrouds = sum(p.id == 104 and _ability_enabled(state, p)
                  for player in (after_own, after_enemy) for p in _board(player))
    if (_ability_enabled(state, acting) and not _counter_blocked(state, acting, ability=True)
            and acting.hp <= 10 * shrouds):
        return None  # Survive the opposing turn's checkup before we can evolve.
    spent = 0
    if evolved is not None:
        damage_counters = max(0, acting.maxHp - acting.hp)
        acting.preEvolution.append(Card(acting.id, acting.serial, obs.current.yourIndex))
        acting.id, acting.serial = evolved.id, evolved.serial
        acting.maxHp = cards[ALAKAZAM].hp
        acting.hp = acting.maxHp - damage_counters
        acting.appearThisTurn = True
        if acting.hp <= 0:
            return None
        after_own.hand = [c for c in after_own.hand if c.serial != evolved.serial]
        spent = 1
    if _energy_gap(acting, attacks[1072]):
        energy = next((c for c in after_own.hand
                       if c.id in (PSYCHIC, TELEPATH)
                       and cards[c.id].energyType == EnergyType.PSYCHIC), None)
        if energy is None:
            return None
        acting.energies.append(EnergyType.PSYCHIC)
        acting.energyCards.append(energy)
        after_own.hand = [c for c in after_own.hand if c.serial != energy.serial]
        spent += 1
    if _energy_gap(acting, attacks[1072]):
        return None  # One typed attachment must pay the entire attack cost.
    after_own.handCount = max(0, own.handCount - spent)
    # Promotion clears the outgoing active's conditions. This future attack
    # receives no old active legal menu and no invented option indices.
    state.select.type = SelectType.CARD
    hit = _damage(state, acting, 1072, defending, after_own.handCount)
    value = _attack_score(state, acting, 1072, defending, after_own.handCount)
    if hit < defending.hp or value < 1000:
        return None
    return {'value': value - 80 * spent, 'attack_value': value,
            'evolution_serial': evolved.serial if evolved is not None else None,
            'attachment_spent': spent - int(evolved is not None),
            'hand_after': after_own.handCount, 'damage': hit}


def _promotion_score(obs, pokemon):
    own, _ = _players(obs)
    value = _best_attack(obs, pokemon)
    if pokemon.id == ALAKAZAM:
        value += 150 if _energy_gap(pokemon, _catalog()[1][1072]) == 0 else 20
    if pokemon.id == BUDEW:
        value += 90
    if pokemon.id in ABRA and pokemon.id == 109 and len(_board(own)) > 1:
        value += 25
    plan = _promotion_hand_plan(obs, pokemon)
    if plan is not None:
        value = max(value, plan['value'])
    return value + pokemon.hp / 20 - 80 * (_prizes(pokemon) - 1)


def _dusk_public_reply_damage(obs, attacker, defender):
    """Public reply possibility; supported Mega Brave requires current cost and lock."""
    if attacker is None or defender is None:
        return 0.0
    cards, attacks, _ = _catalog()
    value = 0.0
    for aid in cards[attacker.id].attacks:
        gap = _energy_gap(attacker, attacks[aid])
        if gap > (0 if aid == 983 else 1) or _public_attack_locked(obs, attacker, aid):
            continue
        value = max(value, _public_attack_damage(obs, attacker, aid, defender,
                                                 1 - obs.current.yourIndex))
    return value


def _dusk_horizon_score(obs, utility, sources, target, follow, *,
                        immediate_win=False, target_ko=False,
                        active_ko=False):
    """Price every sacrifice before the engine makes it irreversible.

    Opening the enemy's last Prize is expensive even when its future hand is
    unknown. A win available this turn bypasses that price; a self-KO that
    hands the enemy its final Prize, or removes our last body, never does.
    """
    own, enemy = _players(obs)
    serials = {p.serial for p in sources}
    survivors = [p for p in _board(own) if p.serial not in serials and p.hp > 0]
    enemy_prizes = len(enemy.prize) - len(serials)
    if enemy_prizes <= 0 or not survivors:
        return -10000.0
    if immediate_win:
        return utility
    if enemy_prizes > 1:
        return utility
    # Remove only the foes that this visible plan actually knocks out. An
    # unseen replacement evolution is not included in the damage forecast.
    remaining = [p for p in _board(enemy)
                 if not (target_ko and p is target)
                 and not (active_ko and p is _active(enemy))]
    defender = follow if follow in survivors else max(
        survivors, key=lambda p: _promotion_score(obs, p), default=None)
    threatened = any(_dusk_public_reply_damage(obs, p, defender) >= defender.hp
                     for p in remaining) if defender else True
    penalty = 3000 + (1000 if threatened else 0) + (600 if len(survivors) == 1 else 0)
    # A remembered chain target must not restore an unsafe nonterminal trade
    # above ending the turn. Immediate ability/attack/chain wins stay intact.
    return min(utility - penalty, -450.0)


def _dusk_plan(obs, source):
    """Return (utility, target serial) for a public-board sacrifice and follow-up."""
    own, enemy = _players(obs)
    if source is None or source.id not in (DUSCLOPS, DUSKNOIR):
        return (-10000.0, None)
    if len(enemy.prize) <= 1 or len(_board(own)) <= 1:
        return (-10000.0, None)
    counters = 130 if source.id == DUSKNOIR else 50
    available = [p for p in _board(own) if p.serial != source.serial]
    follow = _active(own)
    if follow is source:
        follow = max(available, key=lambda p: _promotion_score(obs, p), default=None)
    elif follow and any(o.type == OptionType.RETREAT for o in obs.select.option):
        # Prime Catcher also changes the enemy active, so it cannot be assumed
        # to preserve this exact counter target in a follow-up attack.
        follow = max(available, key=lambda p: _promotion_score(obs, p), default=follow)
    follow_attacks = _ready_attacks(obs, follow)
    if follow is _active(own):
        follow_attacks = [] if own.confused else _legal_attacks(obs)
    ready = bool(follow_attacks)
    legal_others = [p for o in obs.select.option if o.type == OptionType.ABILITY
                    for p in [_at(obs, o.area, o.index)]
                    if p and p.id in (DUSCLOPS, DUSKNOIR) and p.serial != source.serial]
    candidates = []
    for target in _board(enemy):
        if _counter_blocked(obs, target, ability=True):
            continue
        lethal = counters >= target.hp
        prize_gain = min(len(own.prize), _prizes(target)) if lethal else 0
        terminal = lethal and (prize_gain >= len(own.prize) or len(_board(enemy)) == 1)
        active_enemy = _active(enemy)
        before_hits = [(aid, _damage(obs, follow, aid, active_enemy)) for aid in follow_attacks]
        before = max((damage for _, damage in before_hits), default=0)
        # Taking visible prizes adds cards to our hand; their identities are unknown.
        after_hits = [(aid, _damage(obs, follow, aid, active_enemy, own.handCount + prize_gain))
                      for aid in follow_attacks]
        after = max((damage for _, damage in after_hits), default=0)
        combo = ready and target is _active(enemy) and not lethal and before < target.hp <= after + counters
        bench_boost = (ready and target is not _active(enemy) and lethal
                       and _active(enemy) is not None and before < _active(enemy).hp <= after)
        follow_ko = (ready and target is not _active(enemy) and lethal
                     and _active(enemy) is not None and after >= _active(enemy).hp)
        utility = -500.0
        if lethal:
            utility = 180 + 550 * prize_gain - (120 if source.id == DUSKNOIR else 30)
        if combo or bench_boost:
            utility += 1900
        if follow_ko:
            utility = max(utility, _best_attack(obs, follow) + 250 + 250 * prize_gain)
        if ready and target is _active(enemy) and before >= target.hp and not lethal:
            utility = -700  # An unnecessary sacrifice before an already available KO.
        if len(enemy.prize) == 2 and not terminal and not combo and not bench_boost:
            utility -= 500  # Do not put the enemy at its last prize for marginal pressure.
        follow_prizes = max((_attack_prizes(obs, active_enemy, aid) for aid, damage in after_hits
                             if active_enemy is not None and damage >= active_enemy.hp), default=0)
        combo_prizes = max((_attack_prizes(obs, target, aid) for aid, damage in after_hits
                            if damage + counters >= target.hp), default=0)
        attack_win = (follow_ko and (prize_gain + follow_prizes >= len(own.prize)
                                    or len(_board(enemy)) == 2))
        combo_win = (combo and (combo_prizes >= len(own.prize) or len(_board(enemy)) == 1))
        if terminal:
            utility = 100000.0
        elif attack_win:
            utility = 99990.0
        elif combo_win:
            utility = 99985.0
        if _MEMORY['dusk_chain_target'] == (obs.current.turn, target.serial) and lethal:
            utility += 2500
        utility = _dusk_horizon_score(
            obs, utility, [source], target, follow,
            immediate_win=terminal or attack_win or combo_win,
            target_ko=lethal, active_ko=follow_ko or combo)
        # A wall blocking attack effects does not block Cursed Blast. Use only
        # abilities explicitly legal now; bound sacrifice count by both sides'
        # remaining prizes and our remaining bodies before combining counters.
        for n in range(1, min(3, len(legal_others)) + 1):
            if len(enemy.prize) <= n + 1 or len(_board(own)) <= n + 1:
                continue
            for others in combinations(legal_others, n):
                amount = counters + sum(130 if p.id == DUSKNOIR else 50 for p in others)
                if counters < target.hp <= amount:
                    gain = min(len(own.prize), _prizes(target))
                    chain = 250 + 550 * gain - 120 * n
                    if target.id == 58:
                        chain += 350 if target is _active(enemy) else 150
                    chain_win = gain >= len(own.prize) or len(_board(enemy)) == 1
                    chain_hits = [(aid, _damage(obs, follow, aid, active_enemy, own.handCount + gain))
                                  for aid in follow_attacks]
                    chain_follow_aids = [aid for aid, damage in chain_hits
                                         if active_enemy is not None and damage >= active_enemy.hp]
                    chain_follow = (ready and follow not in others
                                    and target is not _active(enemy)
                                    and bool(chain_follow_aids))
                    chain_follow_prizes = max((_attack_prizes(obs, active_enemy, aid)
                                               for aid in chain_follow_aids), default=0)
                    chain_attack_win = chain_follow and (
                        gain + chain_follow_prizes >= len(own.prize) or len(_board(enemy)) == 2)
                    if chain_win or chain_attack_win:
                        chain = 99980
                    if _MEMORY['dusk_chain_target'] == (obs.current.turn, target.serial):
                        chain += 2500
                    chain = _dusk_horizon_score(
                        obs, chain, [source, *others], target, follow,
                        immediate_win=chain_win or chain_attack_win,
                        target_ko=True, active_ko=chain_follow)
                    utility = max(utility, chain)
        candidates.append((utility, target.serial))
    return max(candidates, default=(-10000.0, None))


def _project_gust_board(obs, target, attacker=None):
    """Project only public switches, without mutating the real observation.

    Boss switches the opponent; Prime then switches our active as well. Bench
    counter protection and permanent Ability suppression use the new positions.
    No projected legal option list is fabricated or reused for a new attacker.
    """
    projected = deepcopy(obs)
    own, enemy = _players(projected)

    def move_active(player, serial):
        if _active(player) is not None and _active(player).serial == serial:
            return _active(player)
        index = next((i for i, p in enumerate(player.bench)
                      if p is not None and p.serial == serial and p.hp > 0), None)
        if index is None:
            return None
        previous = _active(player)
        new_active = player.bench[index]
        if previous is None:
            player.bench.pop(index)
        else:
            player.bench[index] = previous
        player.active = [new_active]
        for condition in ('poisoned', 'burned', 'asleep', 'paralyzed', 'confused'):
            setattr(player, condition, False)
        return new_active

    defender = move_active(enemy, target.serial)
    if defender is None:
        return None
    acting = _active(own) if attacker is None else move_active(own, attacker.serial)
    if acting is None:
        return None
    return projected, acting, defender


def _gust_target_score(obs, target, prime=False):
    own, _ = _players(obs)
    if target is None:
        return -200.0
    # MAIN evaluates the unplayed Trainer. During its target selection that
    # card has already left the hand, so do not charge a second card.
    hand_count = max(0, own.handCount - (obs.select.type == SelectType.MAIN))
    attackers = own.bench if prime else [_active(own)]
    scores = []
    for attacker in attackers:
        if attacker is None or attacker.hp <= 0:
            continue
        # Calculate before switching: a Prime replacement is originally on the
        # bench and cannot inherit the current active's native ATTACK options.
        attack_ids = _ready_attacks(obs, attacker)
        if not attack_ids:
            continue
        projection = _project_gust_board(obs, target, attacker if prime else None)
        if projection is None:
            continue
        projected, acting, defender = projection
        score = max((_attack_score(projected, acting, aid, defender, hand_count)
                     for aid in attack_ids), default=-200.0)
        scores.append(score)
    return max(scores, default=-200.0)


def _gwynn_count(obs, options=None):
    own, _ = _players(obs)
    cards = ([_option_card(obs, o) for o in options] if options is not None else
             [c for c in own.hand or [] if _catalog()[0][c.id].cardType == CardType.POKEMON
              and not _catalog()[0][c.id].ex])
    ranked = sorted((c for c in cards if c is not None), key=lambda c: _card_value(obs, c))
    best = (0.0, 0)
    for n in range(1, min(2, len(ranked)) + 1):
        drawn = min(3 * n, own.deckCount)
        played_cost = 1 if options is None else 0
        if own.deckCount <= drawn and not _hand_win(obs, own.handCount - played_cost - n + drawn):
            continue
        if own.deckCount <= drawn + _mill_reserve(obs) and not _hand_win(obs, own.handCount - played_cost - n + drawn):
            continue
        benefit = 100 * (drawn - n) - sum(_card_value(obs, c) * 0.45 for c in ranked[:n])
        # Do not discard the only hand Alakazam that can evolve now.
        if any(c.id == ALAKAZAM and _card_value(obs, c) >= 240 for c in ranked[:n]):
            benefit -= 400
        if len(_board(own)) <= 1:
            basics = sum(_catalog()[0][c.id].basic for c in own.hand or [])
            sacrificed_basics = sum(_catalog()[0][c.id].basic for c in ranked[:n])
            if basics and sacrificed_basics >= basics:
                benefit -= 1000
        if benefit > best[0]:
            best = (benefit, n)
    return best[1]


def _timepiece_target_score(obs, target):
    own, enemy = _players(obs)
    if target is None or target.id not in (KADABRA, ALAKAZAM, DUSCLOPS, DUSKNOIR):
        return -1000
    damage = target.maxHp - target.hp
    prior = target.preEvolution[-1] if target.preEvolution else None
    if prior is None or damage >= _catalog()[0][prior.id].hp:
        return -1000  # Devolution must not KO the body.
    if target is _active(own):
        return -900  # Do not remove this turn's attacking Alakazam.
    remaining_ready = [p for p in _board(own) if p.serial != target.serial and p.id == ALAKAZAM]
    if target.id == ALAKAZAM and not remaining_ready:
        return -600
    count = len(target.preEvolution)
    hand_after = own.handCount - 1 + count
    enables_win = _hand_win(obs, hand_after) and not _hand_win(obs, own.handCount)
    powered_attackers = sum(p.id == ALAKAZAM and bool(p.energies) for p in _board(own))
    if not enables_win and (own.deckCount <= 4 + _mill_reserve(obs)
                            or (count <= 1 and powered_attackers < 2)):
        return -400  # A one-stage devolution gains no hand cards and delays a backup.
    value = 90 if target.id in (KADABRA, ALAKAZAM) else 20
    if target.energies:
        value -= 100  # Preserve a powered backup whenever possible.
    if len(target.preEvolution) >= 2:
        value += 60
    if own.deckCount < 4:
        value -= 100
    if _active(enemy) and _active(own) and _active(own).id == ALAKAZAM:
        if (_damage(obs, _active(own), 1072, _active(enemy)) < _active(enemy).hp
                <= _damage(obs, _active(own), 1072, _active(enemy), own.handCount - 1 + count)):
            value += 2000
    return value


def _resource_card_key(card):
    return card.id, card.serial


def _resource_pokemon_key(pokemon):
    return (pokemon.id, pokemon.serial, pokemon.hp, pokemon.maxHp,
            tuple(pokemon.energies),
            tuple(sorted(_resource_card_key(c) for c in pokemon.energyCards)),
            tuple(sorted(_resource_card_key(c) for c in pokemon.tools)),
            tuple(_resource_card_key(c) for c in pokemon.preEvolution), pokemon.appearThisTurn)


def _resource_context_key(obs):
    """Strict own/public context; consuming only Pad/Poffin permits no retry.

    No unknown prizes, enemy hand identities or native search state are read.
    All actual draw, recovery, field, public reply and turn changes invalidate.
    """
    own, enemy = _players(obs)
    boards = tuple((tuple(_resource_pokemon_key(p) for p in player.active if p),
                    tuple(_resource_pokemon_key(p) for p in player.bench),
                    player.benchMax, len(player.prize),
                    tuple(getattr(player, k) for k in
                          ('asleep', 'paralyzed', 'confused', 'poisoned', 'burned')))
                   for player in (own, enemy))
    return (_MEMORY['round'], obs.current.turn, own.deckCount, boards,
            tuple(sorted(_resource_card_key(c) for c in own.hand or [] if c.id not in (PAD, POFFIN))),
            tuple(sorted(_resource_card_key(c) for c in own.discard if c.id not in (PAD, POFFIN))),
            tuple(sorted(_resource_card_key(c) for c in enemy.discard)),
            enemy.handCount, enemy.deckCount,
            tuple(_resource_card_key(c) for c in obs.current.stadium),
            obs.current.energyAttached, obs.current.supporterPlayed,
            obs.current.stadiumPlayed, obs.current.retreated, obs.current.firstPlayer)


def _resource_search_menu(obs):
    effect = obs.select.effect
    return bool(effect and effect.playerIndex == obs.current.yourIndex and (
        (effect.id == POFFIN and obs.select.context in (SelectContext.TO_BENCH, SelectContext.TO_FIELD))
        or (effect.id == PAD and obs.select.context == SelectContext.TO_HAND)))


def _resource_observe(raw):
    """Only a verified current own complete native deck may supply exact counts.

    ToJson.h supplies the whole selectPlayer deck when selectDeck is enabled.
    Matching length, unique serials and self ownership additionally establish
    that this is a complete own search list. Counts never imply prize IDs.
    """
    if not raw.get('current') or raw.get('select') is None:
        return
    obs = to_observation_class({'current': raw['current'], 'select': raw['select'], 'logs': []})
    own, _ = _players(obs)
    key = _resource_context_key(obs)
    prior = _MEMORY.get('resource_inventory')
    if prior and prior[0] != key:
        _MEMORY['resource_inventory'] = None
    _MEMORY['resource_failed'] = {cid: value for cid, value in _MEMORY['resource_failed'].items()
                                 if value == key}
    pending = _MEMORY.get('resource_pending')
    if pending and (pending[:2] != (_MEMORY['round'], obs.current.turn)):
        _MEMORY['resource_pending'] = None
        pending = None
    if obs.select.type == SelectType.MAIN and pending:
        if pending[4] == key:
            # Paid search returned with no acquired card, deck/field change.
            _MEMORY['resource_failed'][pending[2]] = key
        _MEMORY['resource_pending'] = None
    deck = obs.select.deck
    if (_resource_search_menu(obs) and deck is not None and len(deck) == own.deckCount
            and len({c.serial for c in deck}) == len(deck)
            and all(c.playerIndex == obs.current.yourIndex for c in deck)):
        _MEMORY['resource_inventory'] = key, Counter(c.id for c in deck)


def _resource_remaining(obs):
    exact = _MEMORY.get('resource_inventory')
    if exact and exact[0] == _resource_context_key(obs):
        return exact[1].copy()
    # This is an upper bound including unknown prizes, not a deck assertion.
    return _known_remaining(obs)


def _resource_pad_backup_needed(obs):
    own, _ = _players(obs)
    return (_survival_backup_needed(obs) and not any(
        _catalog()[0][c.id].cardType == CardType.POKEMON and _catalog()[0][c.id].basic
        for c in own.hand or []))


def _resource_search_value(obs, card, search_id, selected=None):
    own, _ = _players(obs)
    data = _catalog()[0][card.id]
    if data.cardType != CardType.POKEMON:
        return -1000.0
    if search_id == POFFIN:
        if not data.basic or data.hp > 70 or len(own.bench) >= own.benchMax:
            return -1000.0
        return _bench_value(obs, card, selected)
    if data.ex:
        return -1000.0  # The frozen own60 has no other Rule-Box Pokemon.
    if _resource_pad_backup_needed(obs):
        return 4000 + _bench_value(obs, card, selected) if data.basic else -1000.0
    if data.basic:
        # Pad acquires to hand; deployment bonuses must not outrank a
        # currently useful evolution. Both hand demand and legal bench use
        # bound a Basic target; Poffin's direct-to-bench utility stays separate.
        return min(_card_value(obs, card, selected), _bench_value(obs, card, selected)) if len(own.bench) < own.benchMax else -1000.0
    # A nonbasic target needs a currently visible/held predecessor and demand.
    board, hand = _counts(obs)
    selected = selected or Counter()
    predecessors = [cid for cid, n in (board + hand).items() if n > 0
                    and _catalog()[0][cid].name == data.evolvesFrom]
    demand = sum(board[cid] + hand[cid] for cid in predecessors)
    if card.id in (ALAKAZAM, DUSKNOIR) and hand[RARE_CANDY]:
        demand += sum(board[cid] for cid in (ABRA if card.id == ALAKAZAM else (DUSKULL,)))
    if demand <= hand[card.id] + selected[card.id]:
        return -1000.0
    value = _card_value(obs, card, selected)
    return value if value > 5 else -1000.0


def _resource_search_targets(obs, search_id):
    if _MEMORY['resource_failed'].get(search_id) == _resource_context_key(obs):
        return ()
    return tuple(Card(cid, -1, obs.current.yourIndex) for cid, n in sorted(_resource_remaining(obs).items())
                 if n > 0 and _resource_search_value(obs, Card(cid, -1, obs.current.yourIndex), search_id) > 0)


def _resource_search_hand_cost(obs, score, gained=0, essential=False):
    """Unconfirmed searches cannot claim a card or a current-KO prep bonus.

    An unseen target may be in unknown prizes. Keep its ordinary utility, but
    only a complete current own search can substantiate a gained-card plan.
    The existing one-body backup and current-KO loss guards stay in force.
    """
    exact = _MEMORY.get('resource_inventory')
    confirmed = bool(exact and exact[0] == _resource_context_key(obs))
    if not confirmed:
        own, enemy = _players(obs)
        active, target = _active(own), _active(enemy)
        if (not essential and active and target and active.id == ALAKAZAM
                and 1072 in _legal_attacks(obs)
                and _damage(obs, active, 1072, target) >= target.hp):
            # _hand_cost grants preparatory priority for score >= 200.
            score = min(score, 199)
        gained = 0
    return _hand_cost(obs, score, gained=gained, essential=essential)


def _resource_commit_choice(obs, action):
    """Commit only actual current selected search plays and failed menus."""
    if obs.select.type == SelectType.MAIN and action:
        option = obs.select.option[action[0]]
        card = _at(obs, AreaType.HAND, option.index) if option.type == OptionType.PLAY else None
        if card and card.id in (PAD, POFFIN):
            _MEMORY['resource_pending'] = (_MEMORY['round'], obs.current.turn, card.id,
                                           card.serial, _resource_context_key(obs))
    elif _resource_search_menu(obs):
        effect = obs.select.effect
        if not action and all(_choice_score(obs, o) <= 0 for o in obs.select.option):
            _MEMORY['resource_failed'][effect.id] = _resource_context_key(obs)
        if action:
            _MEMORY['resource_pending'] = None


def _night_recovery_value(obs, card, selected=None):
    """Value a native-recoverable card, with current visible/held demand.

    Night recovers Pokemon or Basic Energy, never Telepath Special Energy.
    Spare targets stay weakly positive in an already paid native child menu,
    so a declined MAIN play does not turn an original sequence into an empty
    recovery. MAIN requires meaningful demand, not just an eligible card.
    """
    if card is None:
        return -1000.0
    own, _ = _players(obs)
    data = _catalog()[0][card.id]
    if data.cardType == CardType.BASIC_ENERGY:
        return _card_value(obs, card, selected)
    if data.cardType != CardType.POKEMON:
        return -1000.0
    if data.basic:
        if _resource_pad_backup_needed(obs):
            return 4000 + _bench_value(obs, card, selected)
        value = min(_card_value(obs, card, selected), _bench_value(obs, card, selected)) if len(own.bench) < own.benchMax else 5.0
    else:
        value = _resource_search_value(obs, card, PAD, selected)
    return max(5.0, value)


def _telepath_bootstrap_score(obs, energy, recipient):
    """A known P body can start a missing Abra line despite a backup body.

    Remaining counts are an upper bound including unknown prizes. They give
    search eligibility only; no acquisition, new hand card or KO is promised.
    An existing primary line/held Abra or an already charged recipient leaves
    normal attachment/pivot priorities in charge.
    """
    own, _ = _players(obs)
    cards = _catalog()[0]
    if (energy.id != TELEPATH or cards[recipient.id].energyType != EnergyType.PSYCHIC
            or len(own.bench) >= own.benchMax):
        return None
    if len(_board(own)) <= 1:
        return 1600 if own.deckCount > 2 + _mill_reserve(obs) else 200
    if (recipient.energies or own.deckCount <= 2 + _mill_reserve(obs)
            or any(p.id in (*ABRA, KADABRA, ALAKAZAM) for p in _board(own))
            or any(c.id in ABRA for c in own.hand or [])):
        return None
    remaining = _resource_remaining(obs)
    if not any(remaining[cid] > 0 and cards[cid].basic
               and cards[cid].energyType == EnergyType.PSYCHIC for cid in ABRA):
        return None
    return _hand_cost(obs, 150, gained=0)


def _main_score(obs, option):
    own, enemy = _players(obs)
    active, target = _active(own), _active(enemy)
    cards = _catalog()[0]
    kind = option.type
    if kind == OptionType.ATTACK:
        if option.attackId == 183:
            return max((_fez_snipe_score(obs, active, p) for p in _board(enemy)), default=-200)
        return _attack_score(obs, active, option.attackId, target)
    if kind == OptionType.END:
        return -250
    if obs.current.turnActionCount >= 55:
        return -1000
    survival_escape = _survival_ex_escape_main_score(obs, option)
    if survival_escape is not None:
        return survival_escape
    if kind == OptionType.EVOLVE:
        card = _at(obs, option.area, option.index)
        source = _at(obs, option.inPlayArea, option.inPlayIndex)
        if card is None or source is None:
            return -1000
        if card.id == DUDUNSPARCE:
            return _dudun_draw_score(obs, source, evolving=True)
        draw = 3 if card.id == ALAKAZAM else 2 if card.id == KADABRA else 0
        if draw:
            draw = min(draw, own.deckCount) if _draw_safe(obs, draw) else 0
            score = 480 + 50 * draw + (220 if card.id == ALAKAZAM else 0)
            if source is active:
                score += 200
            return _hand_cost(obs, score, gained=draw, essential=card.id == ALAKAZAM and active.id != ALAKAZAM)
        if card.id in (DUSCLOPS, DUSKNOIR):
            score = 350 if card.id == DUSKNOIR else 220
            return _hand_cost(obs, score)
        return _hand_cost(obs, 80)
    if kind == OptionType.ATTACH:
        card = _at(obs, option.area, option.index)
        source = _at(obs, option.inPlayArea, option.inPlayIndex)
        if card is None or source is None or card.id not in (PSYCHIC, TELEPATH):
            return -200
        telepath_setup = _telepath_bootstrap_score(obs, card, source)
        if telepath_setup is not None:
            return telepath_setup
        pivot_score = _pivot_attachment_score(obs, source)
        if pivot_score is not None:
            return pivot_score
        if source.id == DUSKULL and source is active and not source.energies:
            # Telepathic may bench two basics before the attack. Charge this
            # recovery only when even that search leaves actual recovery room.
            recovered = _duskull_recovery_count(obs, 2 if card.id == TELEPATH else 0)
            if recovered:
                return _hand_cost(obs, 220 + 60 * recovered)
        tech_score = _tech_attach_score(obs, card, source)
        if tech_score is not None:
            return tech_score
        if source.energies:
            return -500  # One P is the primary attack's whole cost.
        if source.id not in (*ABRA, KADABRA, ALAKAZAM):
            return -500  # No concrete supported attack/effect pays for this card.
        score = 470 if source is active and source.id == ALAKAZAM else 240
        score += 100 if source.id == ALAKAZAM else 50 if source.id == KADABRA else 0
        if card.id == TELEPATH and len(own.bench) < own.benchMax:
            score += 100
        return _hand_cost(obs, score, essential=source is active and source.id == ALAKAZAM)
    if kind == OptionType.ABILITY:
        source = _at(obs, option.area, option.index)
        if source is None:
            return -1000
        if source.id in (DUSCLOPS, DUSKNOIR):
            return _dusk_plan(obs, source)[0]
        if source.id == DUDUNSPARCE:
            return _dudun_draw_score(obs, source)
        if source.id == FEZ:
            return _hand_cost(obs, 700, spent=0, gained=min(3, own.deckCount)) if _draw_safe(obs, 3) else -800
        if source.id == 109:
            if len(_board(own)) <= 1:
                return -10000
            best = max((_promotion_score(obs, p) for p in own.bench), default=-1000)
            return 300 + best if best > _best_attack(obs) + 100 else -300
        return -1000  # Unmodeled costly Abilities must rank below a legal END.
    if kind == OptionType.RETREAT:
        if active is None:
            return -500
        best = max((_best_attack(obs, p) for p in own.bench), default=-500)
        if best <= 0:
            return -500  # A blocked/ineffective attack is no reason to pay retreat.
        # Paid retreat reduces available hand energy indirectly; the new attacker
        # already has its own energy, so active payment cannot charge it.
        value = best - _best_attack(obs)
        cost = cards[active.id].retreatCost
        return best + 150 - 100 * cost if value > 100 else -350
    if kind != OptionType.PLAY:
        return -300
    card = _at(obs, AreaType.HAND, option.index)
    if card is None:
        return -1000
    cid, data = card.id, cards[card.id]
    if data.cardType == CardType.POKEMON:
        essential = len(_board(own)) <= 1 and data.basic
        if _terminal_ex_bench_risk(obs, card):
            return -800.0
        if _tech_hand_plan(obs, card):
            return _hand_cost(obs, 1000, essential=essential)
        if cid == FEZ and not any(p.id == FEZ for p in _board(own)) and _fez_draw_ready(obs) and _draw_safe(obs, 3, hand_spent=1):
            # Play costs one hand card; Flip the Script returns three before attacking.
            return _hand_cost(obs, 650, spent=1, gained=min(3, own.deckCount), essential=essential)
        return _hand_cost(obs, 1800 if essential else _bench_value(obs, card), essential=essential)
    if cid == POFFIN:
        if own.deckCount <= 2 + _mill_reserve(obs):
            return -500
        if not _resource_search_targets(obs, POFFIN):
            return -1000  # Never lift an unavailable/unwanted search via hand_cost.
        return _resource_search_hand_cost(obs, 390, essential=len(_board(own)) <= 1)
    if cid == PAD:
        if own.deckCount <= 1 + _mill_reserve(obs):
            return -500
        resource_targets = _resource_search_targets(obs, PAD)
        if not resource_targets:
            return -1000  # One gained card requires a concrete wanted target.
        if _survival_backup_needed(obs) and not any(
                _catalog()[0][c.id].cardType == CardType.POKEMON
                and _catalog()[0][c.id].basic for c in own.hand or []):
            return 1850.0  # Establish a real basic backup before returning Dudun.
        if len(_board(own)) <= 1:
            return 1400
        score = 350 if not any(o.type == OptionType.EVOLVE for o in obs.select.option) else 160
        # Positive targets establish usefulness, not guaranteed acquisition.
        # Retain MAIN search urgency; CARD-choice utility has another scale.
        # The separate hand-cost guard still denies unconfirmed KO prep.
        return _resource_search_hand_cost(obs, score, gained=1)
    if cid == DAWN:
        return _hand_cost(obs, 1500 if len(_board(own)) <= 1 else 490, gained=3) if own.deckCount > 3 + _mill_reserve(obs) else -500
    if cid == HILDA:
        return _hand_cost(obs, 420, gained=2) if own.deckCount > 2 + _mill_reserve(obs) else -500
    if cid == GWYNN:
        count = _gwynn_count(obs)
        return _hand_cost(obs, 530 + 40 * count, spent=count + 1,
                          gained=min(3 * count, own.deckCount)) if count else -500
    if cid == RARE_CANDY:
        if _has_hand(obs, ALAKAZAM) and any(p.id in ABRA and not p.appearThisTurn for p in _board(own)):
            # Native options enforce first-turn, devolved and other restrictions.
            return 680 + (200 if active and active.id in ABRA else 0)
        if _has_hand(obs, DUSKNOIR) and any(p.id == DUSKULL for p in _board(own)):
            return _hand_cost(obs, 370, spent=2)
        return -100
    if cid == NIGHT:
        best = max((_night_recovery_value(obs, c) for c in own.discard), default=0)
        return 340 if best > 150 else -300.0
    if cid == ASH:
        recover = sum(cards[c.id].cardType == CardType.POKEMON for c in own.discard)
        if _mill_reserve(obs):
            if own.deckCount > 8:
                return -400  # Save the list's sole recovery item for the mill phase.
            return 4000 if recover and own.deckCount <= 5 else 900 if recover else -400
        return _hand_cost(obs, 300 if own.deckCount <= 8 and recover else 100 if recover >= 3 else -100)
    if cid == TIMEPIECE:
        value = max((_timepiece_target_score(obs, p) for p in _board(own)), default=-1000)
        return value
    if cid == HAMMER:
        return _hammer_best_score(obs)
    if cid in (BOSS, PRIME):
        if cid == PRIME and not own.bench:
            return -1000
        best = max((_gust_target_score(obs, p, prime=cid == PRIME) for p in enemy.bench), default=-1000)
        current = _best_attack(obs)
        # Include a gained prize or decisive KO; do not burn a gust for damage alone.
        if best >= 100000 or best > current + 450:
            return best + 80
        return -400  # Preserve gusts rather than spend them to beat END's -250.
    if cid == RED:
        return _hand_cost(obs, 260 + 25 * (enemy.handCount - 3)) if enemy.handCount > 3 else -250
    return _hand_cost(obs, 30)


def _choice_score(obs, option, selected=None):
    own, enemy = _players(obs)
    context, effect = obs.select.context, obs.select.effect
    effect_id = effect.id if effect else None
    card = _option_card(obs, option)
    if option.type == OptionType.NUMBER:
        return float(option.number or 0)
    if option.type in (OptionType.YES, OptionType.NO):
        yes = True
        if context == SelectContext.MORE_DEVOLVE:
            subject = obs.select.contextCard
            target = next((p for p in _board(own) if
                           (subject and p.serial == subject.serial) or
                           (_MEMORY['devolve_turn'] == obs.current.turn and
                            p.serial in _MEMORY['devolve_serials'])), None)
            if target and target.preEvolution:
                prior = target.preEvolution[-1]
                yes = target.maxHp - target.hp < _catalog()[0][prior.id].hp and target is not _active(own)
            else:
                yes = False
        elif context == SelectContext.ACTIVATE and effect_id in (KADABRA, ALAKAZAM, FEZ):
            yes = _draw_safe(obs, 2 if effect_id == KADABRA else 3)
        elif context == SelectContext.ACTIVATE and obs.select.contextCard and obs.select.contextCard.id in (KADABRA, ALAKAZAM, FEZ):
            yes = _draw_safe(obs, 2 if obs.select.contextCard.id == KADABRA else 3)
        return 10 if (option.type == OptionType.YES) == yes else -10
    if option.type == OptionType.EVOLVE:
        evolved = _at(obs, option.area, option.index)
        source = _at(obs, option.inPlayArea, option.inPlayIndex)
        if evolved is None or source is None:
            return -1000
        return (900 if evolved.id == ALAKAZAM else 350) + (180 if source is _active(own) else 0)
    if option.type == OptionType.ATTACK:
        return _attack_score(obs, _active(own), option.attackId, _active(enemy))
    if card is None:
        return 0
    if effect_id == HAMMER and option.type in (OptionType.ENERGY, OptionType.ENERGY_CARD):
        body = _at(obs, option.area, option.index, option.playerIndex)
        return _hammer_target_score(obs, body, option.energyIndex, already_played=True)
    if context == SelectContext.DAMAGE and effect_id == FEZ and card in _board(enemy):
        return _fez_snipe_score(obs, _active(own), card)
    if context in (SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
                   SelectContext.TO_DECK_BOTTOM, SelectContext.TO_DECK,
                   SelectContext.DISCARD_ENERGY, SelectContext.DISCARD_ENERGY_CARD):
        if effect_id == ASH and context == SelectContext.TO_DECK:
            return _card_value(obs, card)
        return -_card_value(obs, card, selected)
    if context in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
        return _bench_value(obs, card, selected)
    if context == SelectContext.SETUP_ACTIVE_POKEMON:
        if card.id == BUDEW:
            return 500
        if card.id == 109:
            return 260  # Teleporter can release the initial active without energy.
        if card.id in ABRA:
            return 180
        if card.id == DUSKULL:
            return 150
        return -_prizes(card) * 50 + _catalog()[0][card.id].hp / 10
    if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        if option.playerIndex is not None and option.playerIndex != obs.current.yourIndex:
            return _gust_target_score(obs, card, prime=effect_id == PRIME)
        if any(p.serial == card.serial for p in _survival_ex_escape_targets(obs)):
            return 4000 + _promotion_score(obs, card)
        return _promotion_score(obs, card)
    if context in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
        if effect_id in (DUSCLOPS, DUSKNOIR):
            # At this stage self-KO has happened: source is no longer in play.
            amount = 130 if effect_id == DUSKNOIR else 50
            if card not in _board(enemy) or card.hp <= 0 or _counter_blocked(obs, card, ability=True):
                return -10000
            if _MEMORY['dusk_target'] == (obs.current.turn, card.serial):
                return 200000  # Execute the target selected in the visible plan.
            lethal = amount >= card.hp
            value = 200 + 550 * _prizes(card) if lethal else -100
            active = _active(own)
            if active and active.id == ALAKAZAM:
                damage = _damage(obs, active, 1072, _active(enemy))
                if card is _active(enemy) and damage < card.hp <= damage + amount:
                    value += 2000
            if lethal and (_prizes(card) >= len(own.prize) or len(_board(enemy)) == 1):
                value += 100000
            return value
        return 500 / max(10, card.hp) + 30 * _prizes(card)
    if context == SelectContext.DEVOLVE:
        return _timepiece_target_score(obs, card)
    if context in (SelectContext.ATTACH_FROM, SelectContext.ATTACH_TO):
        if hasattr(card, 'energies'):
            return 500 if card.id == ALAKAZAM and not card.energies else 250 if card.id in (*ABRA, KADABRA) and not card.energies else -100
    if context in (SelectContext.TO_HAND, SelectContext.LOOK, SelectContext.EFFECT_TARGET):
        if context == SelectContext.TO_HAND:
            survival = _survival_pad_basic_value(obs, card, selected)
            if survival is not None:
                return survival
            if effect_id == PAD and effect.playerIndex == obs.current.yourIndex:
                return _resource_search_value(obs, card, PAD, selected)
            if effect_id == NIGHT and effect.playerIndex == obs.current.yourIndex:
                return _night_recovery_value(obs, card, selected)
        return _card_value(obs, card, selected)
    return _card_value(obs, card, selected)


def _checked_action(select, action):
    if not select.minCount <= len(action) <= select.maxCount:
        raise ValueError('Selection count outside current legal bounds')
    if len(set(action)) != len(action) or any(type(i) is not int or not 0 <= i < len(select.option) for i in action):
        raise ValueError('Actions must be distinct current option indices')
    return action


def _choose(obs):
    selection = obs.select
    if not selection.option:
        return _checked_action(selection, [])
    if selection.type == SelectType.MAIN:
        best = max(range(len(selection.option)), key=lambda i: (_main_score(obs, selection.option[i]), -i))
        return _checked_action(selection, [best])
    ranked = sorted(range(len(selection.option)), key=lambda i: (-_choice_score(obs, selection.option[i]), i))
    count = selection.minCount
    effect_id = selection.effect.id if selection.effect else None
    if selection.type in (SelectType.YES_NO, SelectType.COUNT, SelectType.ATTACK, SelectType.EVOLVE):
        count = max(1, count)
    elif selection.type == SelectType.ENERGY and selection.remainEnergyCost > 0:
        count = max(1, count)  # The engine reissues payment with remaining units.
    elif effect_id == GWYNN and selection.context == SelectContext.DISCARD:
        count = max(count, _gwynn_count(obs, selection.option))
    elif selection.context in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH,
                               SelectContext.TO_FIELD, SelectContext.TO_HAND, SelectContext.TO_DECK):
        # Re-score after each selected card to avoid redundant search choices.
        chosen, selected = [], Counter()
        available = set(range(len(selection.option)))
        while available and len(chosen) < selection.maxCount:
            if (selection.deck is not None and selection.context in
                    (SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_FIELD)
                    and len(chosen) >= selection.minCount
                    and len(chosen) >= max(0, _players(obs)[0].deckCount - _mill_reserve(obs) - 1)):
                break
            index = max(available, key=lambda i: (_choice_score(obs, selection.option[i], selected), -i))
            if len(chosen) >= selection.minCount and _choice_score(obs, selection.option[index], selected) <= 0:
                break
            chosen.append(index)
            available.remove(index)
            card = _option_card(obs, selection.option[index])
            if card is not None:
                selected[card.id] += 1
        return _checked_action(selection, chosen)
    elif selection.minCount == 0:
        count = sum(_choice_score(obs, selection.option[i]) > 0 for i in ranked)
    count = max(selection.minCount, min(count, selection.maxCount))
    return _checked_action(selection, ranked[:count])


def _stdin_main():
    for line in sys.stdin:
        if line.strip():
            request = json.loads(line)
            if 'observation' in request:
                result = {'action': agent(request['observation']), 'userdata': request.get('userdata')}
            else:
                result = agent(request)
            print(json.dumps(result), flush=True)


# Kaggle get_last_callable: this must be the final callable in the namespace.
def agent(observation):
    if observation.get('select') is None:
        return read_deck_csv()
    _observe(observation)
    obs = to_observation_class(observation)
    action = _choose(obs)
    _resource_commit_choice(obs, action)
    if action:
        option = obs.select.option[action[0]]
        if obs.select.type == SelectType.MAIN and option.type == OptionType.PLAY:
            played = _at(obs, AreaType.HAND, option.index)
            if played and played.id == PAD and _survival_backup_needed(obs):
                _MEMORY['survival_search'] = (_MEMORY['round'], obs.current.turn, played.serial)
        if obs.select.type == SelectType.MAIN and option.type == OptionType.ABILITY:
            source = _at(obs, option.area, option.index)
            if source and source.id == FEZ:
                _MEMORY['fez_used_turn'] = obs.current.turn
            if source and source.id in (DUSCLOPS, DUSKNOIR):
                target_serial = _dusk_plan(obs, source)[1]
                _MEMORY['dusk_target'] = (obs.current.turn, target_serial)
                target = next((p for p in _board(_players(obs)[1]) if p.serial == target_serial), None)
                if target and target.hp > (130 if source.id == DUSKNOIR else 50) and _counter_blocked(obs, target):
                    _MEMORY['dusk_chain_target'] = (obs.current.turn, target_serial)
        if obs.select.context == SelectContext.DEVOLVE:
            target = _option_card(obs, option)
            if target is not None:
                _MEMORY['devolve_serials'] = tuple(c.serial for c in target.preEvolution) + (target.serial,)
                _MEMORY['devolve_turn'] = obs.current.turn
    return action


if globals().get('__name__') == '__main__':
    _stdin_main()
