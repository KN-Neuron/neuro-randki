"""
Neuro-Randki Matching & Archetype Engine.
Interpretable, fun, arcade-style biometric personality matching
that produces meaningful relationships and clear graph communities.
"""
import math
import numpy as np

import shared_stimulus as stim

ARCHETYPES = {
    'zen': {
        'code': 'zen',
        'name': 'Zen Master',
        'icon': '🧘',
        'color': '#ec4899', # Pink / Rose
        'band': 'Alpha (8–13 Hz)',
        'headline': 'Oaza spokoju i głęboka intuicja',
        'desc': 'Zrównoważony układ nerwowy, odporność na chaos, wysokie skupienie wewnętrzne.',
    },
    'focus': {
        'code': 'focus',
        'name': 'Hyper-Focus',
        'icon': '⚡',
        'color': '#06b6d4', # Cyan / Electric Blue
        'band': 'Beta (13–30 Hz)',
        'headline': 'Analityczny umysł i potężna energia',
        'desc': 'Błyskawiczne tempo myślenia, logika, determinacja i ciągły napęd do działania.',
    },
    'visionary': {
        'code': 'visionary',
        'name': 'Wizjoner',
        'icon': '🎨',
        'color': '#10b981', # Emerald
        'band': 'Theta (4–8 Hz)',
        'headline': 'Kreatywność, empatia i wyobraźnia',
        'desc': 'Głębokie odczuwanie świata, nieszablonowe idee, artystyczna wrażliwość i serce.',
    },
    'genius': {
        'code': 'genius',
        'name': 'Impuls Geniuszu',
        'icon': '💥',
        'color': '#f59e0b', # Amber / Gold
        'band': 'Gamma (30–45 Hz)',
        'headline': 'Błyskawiczna synteza i pasja',
        'desc': 'Super-połączenia neuronowe, momenty „Aha!”, intensywne przetwarzanie informacji.',
    },
    'thinker': {
        'code': 'thinker',
        'name': 'Głęboki Myśliciel',
        'icon': '🌌',
        'color': '#8b5cf6', # Violet / Indigo
        'band': 'Delta (0.5–4 Hz)',
        'headline': 'Refleksyjność i wewnętrzna mądrość',
        'desc': 'Spokój podświadomości, zakorzenienie, uważność i zdolność do wielkich przemyśleń.',
    },
}

PAIR_DIAGNOSES = {
    ('zen', 'zen'): (
        "Telepatyczny spokój. Rozumiecie się bez słów, a wasza wspólna obecność działa kojąco jak medytacja.",
        ["Idealna harmonia emocjonalna", "Brak niepotrzebnych spięć"]
    ),
    ('zen', 'focus'): (
        "Oaza spokoju spotyka wulkan energii. Zen uziemia emocje, a Focus daje impuls do działania — klasyczne Yin i Yang!",
        ["Komplementarny balans", "Wzajemne uzupełnianie słabości"]
    ),
    ('zen', 'visionary'): (
        "Głęboki rezonans empatii. Spokój spotyka artystyczną duszę — razem tworzycie bezpieczną przestrzeń dla marzeń.",
        ["Wspólna wrażliwość", "Głębokie rozmowy do późna"]
    ),
    ('zen', 'genius'): (
        "Stabilizator i rakieta. Gdy w głowie Geniusza pojawia się burza myśli, Zen pomaga wybrać tę najlepszą.",
        ["Opanowanie chaosu", "Błyskotliwe projekty"]
    ),
    ('zen', 'thinker'): (
        "Wyciszona przystań. Oboje cenicie autentyczność, spokój ducha i brak powierzchowności.",
        ["Wzajemne zaufanie", "Naturalna stabilność"]
    ),
    ('focus', 'focus'): (
        "Elektryzujący power-couple! Błyskawiczna wymiana idei, wspólne cele i nieprawdopodobna siła napędowa.",
        ["Wspólny wysoki napęd", "Nigdy nie ma nudy"]
    ),
    ('focus', 'visionary'): (
        "Wizja spotyka realizację. Wizjoner kreuje niezwykłe koncepcje, a Focus przekuwa je w rzeczywistość.",
        ["Synergia marzeń i czynów", "Kompletny zespół marzeń"]
    ),
    ('focus', 'genius'): (
        "Błyskawice myśli! Porozumienie na poziomie procesorów kwantowych — wysokie obroty i ekscytujące tempo.",
        ["Intensywna chemia intelektualna", "Szybkie tempo życia"]
    ),
    ('focus', 'thinker'): (
        "Strateg i wykonawca. Thinker patrzy na horyzont, a Focus wyznacza najszybszą ścieżkę do celu.",
        ["Mądrość i skuteczność", "Trwałe fundamenty"]
    ),
    ('visionary', 'visionary'): (
        "Świat magii i emocji. Oboje czujecie muzykę, sztukę i ludzi sercem — natychmiastowe połączenie dusz.",
        ["Niezwykła empatia", "Wspólny świat fantazji"]
    ),
    ('visionary', 'genius'): (
        "Festiwal kreatywności. Od szalonych pomysłów do genialnych odkryć — inspirujecie się nawzajem.",
        ["Niekończąca się inspiracja", "Artystyczna wolność"]
    ),
    ('visionary', 'thinker'): (
        "Poezja i filozofia. Rozmowy o sensie wszechświata i głębokie poczucie bezpieczeństwa.",
        ["Ciepło i zrozumienie", "Niezachwiana lojalność"]
    ),
    ('genius', 'genius'): (
        "Supernowa! Eksplozja energii, humoru i błyskotliwych ripost. Przy was nikt nie zaśnie.",
        ["Wysokie napięcie intelektualne", "Magnetyczne przyciąganie"]
    ),
    ('genius', 'thinker'): (
        "Iskra i głębia. Szybkie impulsy spotykają się z wielką refleksją — idealny dialog umysłów.",
        ["Równowaga myśli", "Wzajemny szacunek"]
    ),
    ('thinker', 'thinker'): (
        "Nierozerwalna kotwica. Prawdziwa głębia, mądrość i lojalność bez konieczności udowadniania czegokolwiek.",
        ["Głębokie porozumienie", "Wielka trwałość relacji"]
    ),
}

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
BAND_TO_ARCHETYPE = {'delta': 'thinker', 'theta': 'visionary', 'alpha': 'zen',
                     'beta': 'focus', 'gamma': 'genius'}

# Population reference of log relative band power (1–45 Hz, Fp1/Fp2/O1/O2),
# calibrated on the 30 recordings in data/. Profiles are z-scored against it,
# so "dominant" means "higher than typical", not "largest in absolute terms"
# (delta is always largest in absolute terms on frontal electrodes).
REF_LOG_MEAN = np.array([-0.299, -2.114, -3.535, -3.567, -4.515])
REF_LOG_STD  = np.array([ 0.185,  0.512,  0.866,  1.126,  1.275])
# Delta z-scores swing hardest (it absorbs blinks/movement) — damp it a bit
ARCHETYPE_WEIGHTS = np.array([0.8, 1.0, 1.0, 1.0, 1.0])


# Lookups use sorted pairs — normalise keys written in any order
PAIR_DIAGNOSES = {tuple(sorted(k)): v for k, v in PAIR_DIAGNOSES.items()}


def band_profile(band_powers: dict) -> np.ndarray:
    """Relative band powers → z-scored log profile vs the population reference."""
    if not band_powers:
        return np.zeros(len(BANDS))
    v = np.array([max(float(band_powers.get(k, 0.0)), 1e-6) for k in BANDS])
    v = v / v.sum()
    return (np.log(v) - REF_LOG_MEAN) / REF_LOG_STD


def get_archetype(band_powers: dict) -> dict:
    """Classify brain profile into one of the 5 archetypes."""
    if not band_powers:
        return ARCHETYPES['zen']
    z = band_profile(band_powers) * ARCHETYPE_WEIGHTS
    return ARCHETYPES[BAND_TO_ARCHETYPE[BANDS[int(np.argmax(z))]]]


def _closeness(z1: np.ndarray, z2: np.ndarray, idx: list) -> float:
    """1.0 for identical sub-profiles, decaying towards 0 as they diverge."""
    d = float(np.linalg.norm(z1[idx] - z2[idx]))
    return math.exp(-d / 1.6)


def compute_affinity(bands1: dict, arch1_code: str,
                     bands2: dict, arch2_code: str,
                     emb1: np.ndarray | None = None, emb2: np.ndarray | None = None,
                     sig1=None, sig2=None, sig_mean=None) -> dict:
    """
    Compute multi-dimensional match scores between two participants.
    Returns overall score (40..99), components, and narrative text.

    sig1/sig2 are shared-stimulus response signatures (see shared_stimulus.py).
    When both exist, 'Wspólna reakcja' dominates the overall score — it is the
    only component that reproduces between recordings.
    """
    z1, z2 = band_profile(bands1), band_profile(bands2)

    # Cosine of z-profiles — "same shape of deviation from the average brain"
    n1, n2 = np.linalg.norm(z1), np.linalg.norm(z2)
    spec_cos = float(np.dot(z1, z2) / (n1 * n2)) if n1 > 1e-6 and n2 > 1e-6 else 0.0
    raw_sim = (spec_cos + 1.0) / 2.0

    # If full feature embeddings provided, blend
    if emb1 is not None and emb2 is not None and len(emb1) == len(emb2):
        e_cos = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9))
        e_cos = max(-1.0, min(1.0, e_cos))
        raw_sim = 0.6 * raw_sim + 0.4 * ((e_cos + 1.0) / 2.0)

    # Archetype synergy bonus
    key = tuple(sorted([arch1_code, arch2_code]))
    is_same = arch1_code == arch2_code
    if is_same:
        synergy_score = 92.0
    elif key in [('focus', 'zen'), ('focus', 'visionary'), ('focus', 'thinker'), ('genius', 'zen'), ('thinker', 'visionary')]:
        synergy_score = 90.0
    else:
        synergy_score = 82.0

    # Dimension 1: Harmonia Emocjonalna (Alpha + Theta resonance)
    harmony = int(round(45 + _closeness(z1, z2, [1, 2]) * 54))

    # Dimension 2: Dynamika / Energia (Beta + Gamma stimulation)
    energy = int(round(45 + _closeness(z1, z2, [3, 4]) * 54))

    # Dimension 3: Styl Myślenia (Overall Spectral Pattern)
    cognition = int(round(40 + raw_sim * 59))

    # Overall Match Score (weighted average + mild archetype boost)
    overall = int(round(0.35 * harmony + 0.35 * cognition + 0.30 * energy))
    if is_same or synergy_score >= 90:
        overall = min(99, overall + 3)
    overall = max(42, min(99, overall))

    sync_r = stim.signature_similarity(sig1, sig2, sig_mean)
    sync = stim.sync_score(sync_r) if sync_r is not None else None
    if sync is not None:
        overall = int(round(0.7 * sync + 0.3 * overall))
        overall = max(40, min(99, overall))

    # Bond type classification
    if overall >= 88:
        bond_type = "Bratnie Dusze 💖"
    elif overall >= 78:
        bond_type = "Komplementarny Balans ⚡"
    elif overall >= 65:
        bond_type = "Wspólna Fala 🌊"
    else:
        bond_type = "Ciekawe Przeciwieństwa 🧩"

    diag_tuple = PAIR_DIAGNOSES.get(key, PAIR_DIAGNOSES[('focus', 'zen')])
    diagnosis = diag_tuple[0]
    strengths = diag_tuple[1]

    return {
        'overall': overall,
        'similarity': round(raw_sim, 4),
        'harmony': harmony,
        'energy': energy,
        'cognition': cognition,
        'sync': sync,
        'sync_r': round(sync_r, 4) if sync_r is not None else None,
        'bond_type': bond_type,
        'diagnosis': diagnosis,
        'strengths': strengths,
    }


def build_curated_graph(nodes_data: list, paired_sessions: list = None) -> dict:
    """
    Build a clean, readable network graph:
    1. Paired session partners always get an edge.
    2. Every node gets edges to its Top 2-3 highest matches (k-nearest neighbors).
    3. Additional edges added only if similarity is high (>= 75%).
    4. Limits edge density so D3 displays crisp, distinct community clusters.
    """
    paired_pairs = set()
    if paired_sessions:
        for s in paired_sessions:
            u1, u2 = s['user1_id'], s['user2_id']
            paired_pairs.add(tuple(sorted([u1, u2])))

    n = len(nodes_data)
    sig_mean = stim.population_mean([u.get('signature') for u in nodes_data])
    nodes = []
    for u in nodes_data:
        arch = ARCHETYPES.get(u.get('archetype', 'zen'), ARCHETYPES['zen'])
        nodes.append({
            'id': u['id'],
            'nickname': u['nickname'],
            'is_solo': bool(u.get('is_solo', 0)),
            'archetype': arch['code'],
            'archetype_name': arch['name'],
            'archetype_icon': arch['icon'],
            'archetype_color': arch['color'],
            'dominant_band': arch['band'],
            'has_stim': u.get('signature') is not None,
            'stim_weak': u.get('signature') is not None and stim.is_weak(u.get('stim_quality')),
            'neighbors': [],
        })

    # Compute all pairwise affinities
    sim_matrix = {}
    pair_rankings = {node['id']: [] for node in nodes}

    for i in range(n):
        for j in range(i + 1, n):
            u1 = nodes_data[i]
            u2 = nodes_data[j]
            aff = compute_affinity(
                u1.get('band_powers'), u1.get('archetype', 'zen'),
                u2.get('band_powers'), u2.get('archetype', 'zen'),
                u1.get('embedding'), u2.get('embedding'),
                u1.get('signature'), u2.get('signature'), sig_mean,
            )
            sim_matrix[(i, j)] = aff
            id1, id2 = u1['id'], u2['id']

            pair_rankings[id1].append({
                'id': id2, 'nickname': u2['nickname'],
                'score': aff['overall'], 'similarity': aff['similarity'],
                'archetype': u2.get('archetype', 'zen'),
                'bond_type': aff['bond_type'],
                'sync': aff['sync'],
            })
            pair_rankings[id2].append({
                'id': id1, 'nickname': u1['nickname'],
                'score': aff['overall'], 'similarity': aff['similarity'],
                'archetype': u1.get('archetype', 'zen'),
                'bond_type': aff['bond_type'],
                'sync': aff['sync'],
            })

    # Sort each node's neighbors descending by match score
    for node in nodes:
        node['neighbors'] = sorted(pair_rankings[node['id']], key=lambda x: -x['score'])
        if node['neighbors']:
            top = node['neighbors'][0]
            node['twin_name'] = top['nickname']
            node['twin_score'] = top['score']

    # Curate links (prevent dense mesh)
    chosen_links = set()

    # 1. Guaranteed edges: couples who played together
    for i in range(n):
        for j in range(i + 1, n):
            pair_key = tuple(sorted([nodes[i]['id'], nodes[j]['id']]))
            if pair_key in paired_pairs:
                chosen_links.add((i, j))

    # 2. Top-2 closest neighbors for each node (k-NN guarantee)
    for i in range(n):
        top_candidates = []
        for j in range(n):
            if i == j:
                continue
            pair_key = (min(i, j), max(i, j))
            aff = sim_matrix[pair_key]
            top_candidates.append((j, aff['overall']))
        top_candidates.sort(key=lambda x: -x[1])
        for target_j, _ in top_candidates[:2]:
            chosen_links.add((min(i, target_j), max(i, target_j)))

    # 3. Add exceptional connections (score >= 86) up to a max average degree
    max_total_links = max(10, int(n * 2.8))
    for (i, j), aff in sorted(sim_matrix.items(), key=lambda item: -item[1]['overall']):
        if len(chosen_links) >= max_total_links:
            break
        if aff['overall'] >= 86:
            chosen_links.add((i, j))

    # Format final links array
    links = []
    for (i, j) in chosen_links:
        aff = sim_matrix[(i, j)]
        pair_key = tuple(sorted([nodes[i]['id'], nodes[j]['id']]))
        is_game_couple = pair_key in paired_pairs
        links.append({
            'source': nodes[i]['id'],
            'target': nodes[j]['id'],
            'score': aff['overall'],
            'similarity': aff['similarity'],
            'bond_type': aff['bond_type'],
            'is_pair': is_game_couple,
            'harmony': aff['harmony'],
            'energy': aff['energy'],
            'cognition': aff['cognition'],
            'sync': aff['sync'],
        })

    return {'nodes': nodes, 'links': links, 'archetypes': ARCHETYPES}
