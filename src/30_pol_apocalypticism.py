"""30 – Apocalypticism Classifier for /pol/.

Identifies apocalyptic rhetoric across /pol/ posts using a transformer-
based approach that prioritises contextual sensitivity over rigid keyword
matching.

Architecture
------------
1. **Logistic-regression classifier on sentence embeddings** — trained
   on a large synthetic dataset of positive (apocalyptic) and negative
   (non-apocalyptic /pol/ content) examples.  Provides calibrated
   probability scores.
2. **Multi-facet contrastive similarity** — separate centroid vectors for
   five apocalyptic sub-themes (eschatological, accelerationist,
   civilisational collapse, racial apocalypse, conspiratorial) plus a
   negative centroid.  Contrastive score = max positive similarity −
   negative similarity.  This captures posts using novel phrasing that
   the LR may not have seen in training.
3. **Combined score** — 0.6 × LR probability + 0.4 × normalised
   contrastive score.  Both components are entirely embedding-based;
   no keyword matching governs the final score.

A lightweight keyword dictionary is retained **only** as a diagnostic
feature for interpretability; it does **not** contribute to
``apoc_combined``.

Output
------
``data/processed/pol_apocalypticism_scores.parquet``
  Per-post scores: ``apoc_lr_prob``, ``apoc_contrastive``,
  ``apoc_combined``, ``apoc_binary``, ``apoc_subtheme``,
  ``apoc_similarity``, ``apoc_keyword_score`` (diagnostic).

``data/processed/apocalypticism_centroid.npy``
  Saved positive centroid vector for reproducibility.
"""

from __future__ import annotations

import multiprocessing
import os
import re
from functools import partial

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from utils import DATA_PROCESSED

_N_WORKERS = min(multiprocessing.cpu_count(), int(os.environ.get("SIEGE_WORKERS", "16")))
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 512

# ══════════════════════════════════════════════════════════════════════
# Combined score weights
# ══════════════════════════════════════════════════════════════════════
LR_WEIGHT = 0.6
CONTRASTIVE_WEIGHT = 0.4
BINARY_THRESHOLD = 0.55  # on combined score

# ══════════════════════════════════════════════════════════════════════
# Apocalypticism category seeds (4-way disaggregation)
# ══════════════════════════════════════════════════════════════════════
# Each post classified as apocalyptic (apoc_binary == 1) is assigned
# to the category whose centroid it is most similar to.

APOC_CATEGORIES = ["siegist_traditionalist", "rapture_christian",
                   "prepper", "general_collapsist"]

CATEGORY_SEEDS: dict[str, list[str]] = {
    "siegist_traditionalist": [
        "Read Siege. Accelerate the collapse. The system cannot be reformed.",
        "Kali Yuga is upon us. The age of darkness demands action not words.",
        "The day of the rope is coming. Traitors will face the consequences.",
        "We are at the end of the cycle. Kalki will bring the great renewal.",
        "Siege culture understands that only total destruction brings rebirth.",
        "Mason was right. The lone wolf strategy is the only viable path forward.",
        "This is the iron age. After the collapse comes the golden age of our people.",
        "RAHOWA is inevitable. The racial holy war will determine the future.",
        "Spengler predicted the decline of the West. We are living through his prophecy.",
        "The Traditionalist school teaches that civilizations die and are reborn through fire.",
        "Evola understood that riding the tiger means accelerating the dissolution.",
        "The age of wolves is upon us. Only the strong will survive the twilight.",
        "Total racial war against the system. Saints will be remembered for their actions.",
        "Blood and soil. The ethnostate can only be built on the ashes of the old order.",
        "The Turner Diaries were prophetic. The system of racial annihilation must be destroyed.",
        "Atomwaffen understands what must be done. Read Siege. Hail victory.",
        "The Weimar Republic fell and so will this degenerate civilization.",
        "White revolution is the only solution. No political path remains.",
        "Leaderless resistance cells will bring down the corrupt system from within.",
        "The great replacement demands a violent response. Our survival is at stake.",
    ],
    "rapture_christian": [
        "The rapture is near. Christ will return and take the faithful to heaven.",
        "The book of Revelation is unfolding. The seven seals are being opened.",
        "The antichrist walks among us. The tribulation period has begun.",
        "Armageddon approaches. The final battle between good and evil is at hand.",
        "The four horsemen of the apocalypse ride across the earth today.",
        "The mark of the beast is being implemented through digital currency systems.",
        "Judgment day is coming. The wicked will face God's righteous wrath.",
        "The signs of the times are clear. We are the terminal generation.",
        "The great tribulation spoken of by the prophets is starting now.",
        "Repent for the kingdom of heaven is at hand. The end of days approaches.",
        "The whore of Babylon represents the corrupt world system that God will destroy.",
        "The abomination of desolation stands in the holy place. Flee to the mountains.",
        "Jesus is coming back soon. Every eye will see Him. Maranatha.",
        "The beast system of Revelation is real. Refuse the mark at all costs.",
        "The second coming of Christ will bring judgment on all nations.",
        "Daniel's prophecy of the seventy weeks is being fulfilled right now.",
        "The millennium approaches. Christ will reign for a thousand years.",
        "The dragon and the beast wage war against the saints in these last days.",
        "Born again believers will be taken up in the rapture before the tribulation.",
        "God's wrath is being poured out on this wicked generation. Prepare your souls.",
    ],
    "prepper": [
        "When SHTF you need at least six months of food and water stored.",
        "Bug out bags should be packed and ready. Have a plan for when society collapses.",
        "Grid down scenario: do you have solar panels and a water filtration system?",
        "Stockpile ammunition and medical supplies. You cannot rely on the system.",
        "Learn to grow your own food. Self-sufficiency is the only real security.",
        "The supply chain is fragile. One disruption and the grocery stores empty in days.",
        "Homesteading and off-grid living are the only ways to survive what is coming.",
        "Every prepper needs a ham radio. When the internet goes down communication is survival.",
        "Store seeds, not just food. Long term survival means growing your own.",
        "The financial system will collapse. Have physical gold and silver as insurance.",
        "Water purification is the number one survival priority. Everything else is secondary.",
        "Build a community of like-minded people. You cannot survive alone when it all falls apart.",
        "Faraday cages for your electronics. An EMP attack would send us back to the stone age.",
        "The government will not save you. FEMA camps are for control not protection.",
        "Tactical training and firearms proficiency are essential survival skills.",
        "Get out of the cities now. Urban areas will be death traps when the collapse comes.",
        "Canning preserving and dehydrating food are essential skills everyone should learn.",
        "A generator fuel and medical supplies are the holy trinity of prepping.",
        "The power grid is vulnerable. One well-placed attack could black out the entire country.",
        "Survival groups need to train together regularly. Operational security is paramount.",
    ],
    "general_collapsist": [
        "Western civilization is in terminal decline. The collapse is inevitable.",
        "Society is falling apart. The institutions are rotten beyond repair.",
        "The new world order is being assembled. Global governance will end all freedom.",
        "Climate change will cause civilizational collapse within our lifetimes.",
        "The financial bubble will burst and this time there will be no recovery.",
        "Democracy is dying everywhere. Authoritarianism is the future of governance.",
        "The great reset is their plan to restructure society after the engineered collapse.",
        "Everything is going to fall apart eventually. History shows all empires decline.",
        "Total societal collapse is coming. The signs are everywhere if you look.",
        "The global elite are engineering the downfall of nations for their own benefit.",
        "Late stage capitalism is consuming itself. The contradictions are unsustainable.",
        "Peak oil, peak water, peak everything. This civilization has hit its limits.",
        "Mass extinction events are accelerating. Ecological collapse will trigger social collapse.",
        "The center cannot hold. Political polarization will tear this country apart.",
        "Demographic collapse in the West guarantees economic and civilizational decline.",
        "Hyperinflation is coming. The dollar will collapse like every fiat currency before it.",
        "Nuclear war is more likely now than during the Cold War. Mutually assured destruction fails.",
        "The surveillance state is building a prison planet. There will be no escape.",
        "Social cohesion is gone. Without shared values and trust society cannot function.",
        "The pandemic was just a preview. The real collapse will make COVID look like nothing.",
    ],
}


# ══════════════════════════════════════════════════════════════════════
# Mapping from 5 sub-themes → 4 categories (for reference / fallback)
# ══════════════════════════════════════════════════════════════════════
SUBTHEME_TO_CATEGORY: dict[str, str] = {
    "accelerationist": "siegist_traditionalist",
    "racial_apocalypse": "siegist_traditionalist",
    "civilizational_collapse": "general_collapsist",
    "eschatological": "rapture_christian",
    "conspiratorial_apocalypse": "general_collapsist",
}


# ══════════════════════════════════════════════════════════════════════
# Synthetic training data – Positive examples (apocalyptic rhetoric)
# ══════════════════════════════════════════════════════════════════════
# Synthetic paraphrases of observed rhetoric, NOT direct quotations.
# Organised by sub-theme for centroid construction.

POSITIVE_SEEDS: dict[str, list[str]] = {

    "eschatological": [
        "The end times are upon us. Western civilization is collapsing and there is no saving it.",
        "Judgment day is coming soon. The signs are all around us if you open your eyes.",
        "We are living in the last days. Everything prophesied is coming true before our eyes.",
        "The book of Revelation is unfolding right now. The tribulation has begun.",
        "The rapture is near. The faithful will be saved while the wicked are destroyed.",
        "Armageddon is at hand. The final battle between good and evil approaches.",
        "The four horsemen ride. Famine, war, pestilence and death are here.",
        "The mark of the beast is being implemented through digital currency and surveillance.",
        "These are the birth pangs of the apocalypse. It will only get worse from here.",
        "God's wrath will be poured out on this wicked generation. The day of reckoning comes.",
        "The antichrist system is being built right before our eyes and nobody seems to care.",
        "Biblical prophecy is being fulfilled. The end of the age is upon us.",
        "The great tribulation spoken of by the prophets is starting now.",
        "Divine judgment is coming for this degenerate society. Nothing can stop what is coming.",
        "The signs of the times are unmistakable. We are in the final generation.",
    ],

    "accelerationist": [
        "Read Siege. Accelerate. The system is corrupt beyond repair. Total attack is the only option.",
        "The system cannot be reformed, only destroyed. Accelerate the inevitable collapse.",
        "Every act of violence hastens the coming race war. This is what acceleration means.",
        "We need to push things to the breaking point. Let it all burn down and start over.",
        "Leaderless resistance is the only viable strategy. The lone wolf strikes fear.",
        "Propaganda of the deed. Let actions speak louder than words. Make them afraid.",
        "The day of the rope approaches. Those who betrayed their people will face justice.",
        "Saints will be remembered. Those who take direct action against the system are heroes.",
        "Total war against the system. No compromise, no reform, only destruction.",
        "The boogaloo is coming whether you're ready or not. Prepare yourself now.",
        "Siege culture is the answer. Mason understood what must be done.",
        "High score attempts are symptoms of a dying civilization. The system creates its own destroyers.",
        "Accelerationism means making conditions worse so people wake up and fight back.",
        "There is no political solution. The ballot box is a trap. Only collapse will free us.",
        "Every institution is compromised. Burn it all down and rebuild from the ashes.",
    ],

    "civilizational_collapse": [
        "Western civilization is in terminal decline. Nothing can reverse this. Embrace the collapse.",
        "The fall of Rome is happening again. Degeneracy, immigration, and corruption will destroy us.",
        "We are living in Kali Yuga, the age of darkness and degeneracy before the great renewal.",
        "This is the Weimar Republic all over again. History repeats itself.",
        "The decline of the West is irreversible. Spengler was right about everything.",
        "Our civilization is dying from within. The rot is too deep to cure.",
        "Clown world cannot sustain itself. Honk honk. It will all come crashing down.",
        "Every great civilization has fallen. Ours is no different. The end is near.",
        "Degeneracy is the symptom of civilizational collapse. We are at the terminal stage.",
        "The iron age approaches its end. After darkness comes the golden age of renewal.",
        "Western civilization peaked and now we're watching its slow death in real time.",
        "The barbarians are at the gates. Just like Rome, our leaders invited the invaders in.",
        "Society is breaking down at every level. There is no going back to how things were.",
        "The point of no return has been passed. This civilization is finished.",
        "Cultural and demographic decline mean the West as we know it has maybe a generation left.",
    ],

    "racial_apocalypse": [
        "The great replacement is real. White genocide is happening before our eyes and no one is stopping it.",
        "RAHOWA now. Accelerate the collapse. The system cannot be reformed, only destroyed.",
        "The race war is coming. It's inevitable. Choose your side now or be destroyed.",
        "White people are being systematically replaced in every Western country by design.",
        "The Kalergi plan is being implemented. They want to breed out Europeans entirely.",
        "This is genocide against white people. Every statistic proves replacement is happening.",
        "There will be a racial holy war. The traitors and invaders will face consequences.",
        "Our people face extinction if we do not act now. This is a fight for survival.",
        "Total race war is the only solution. Multicultural societies always end in bloodshed.",
        "The white revolution must come or our people will disappear from history forever.",
        "They are genociding us through immigration, miscegenation, and low birth rates. Wake up.",
        "Helter Skelter was prophetic. The racial conflict that Manson predicted is coming true.",
        "Civil war along racial lines is coming. The demographics guarantee it.",
        "Blood and soil. Our ancestors' homeland is being taken from us and given to foreigners.",
        "The white race is under attack from every direction. This is an existential crisis.",
    ],

    "conspiratorial_apocalypse": [
        "The new world order is being assembled. Global governance will eliminate all freedom.",
        "ZOG is orchestrating the destruction of Western nations. Every policy serves their agenda.",
        "The great reset is not a conspiracy theory. They want you to own nothing and be happy.",
        "The globalist agenda is population control and enslavement through technology.",
        "They control the media, the banks, and the governments. Everything is orchestrated.",
        "The Illuminati or whatever you call them are engineering societal collapse on purpose.",
        "COVID was planned to implement totalitarian control. The virus was just the excuse.",
        "5G, vaccines, digital ID — it's all connected. They're building a prison planet.",
        "The elites want to depopulate the earth. They've said so openly. Georgia Guidestones.",
        "A shadow government runs everything. Elections are theater to keep the masses pacified.",
        "The globalists are satanic pedophiles engineering the destruction of civilization.",
        "Everything happening is by design. The chaos is manufactured to bring about total control.",
        "Wake up. The world governments are controlled by a cabal working toward total domination.",
        "The beast system of Revelation is being built through digital currency and surveillance.",
        "They want world war three because out of chaos comes their new order.",
    ],
}

# ══════════════════════════════════════════════════════════════════════
# Synthetic training data – Negative examples (non-apocalyptic)
# ══════════════════════════════════════════════════════════════════════

NEGATIVE_SEEDS: dict[str, list[str]] = {

    "political_discussion": [
        "Who are you voting for in the midterms? I think the Republicans have a shot.",
        "The economy is doing okay but inflation is too high. What do you guys think?",
        "This new bill will never pass the Senate. The filibuster is too strong.",
        "Immigration policy needs reform but the parties can't agree on anything.",
        "The president's approval rating dropped again. Not surprising given the economy.",
        "Both parties are corrupt but at least one side isn't trying to take my guns.",
        "The Supreme Court ruling was expected. Roberts always sides with the liberals.",
        "Local elections matter more than federal ones. School boards are where it's at.",
        "The trade deal with China was a disaster. We got played on every point.",
        "Healthcare costs keep going up. Something needs to change in the system.",
        "The electoral college is outdated. Popular vote should decide elections.",
        "Tax cuts for corporations don't help working people. Trickle-down is a scam.",
        "Foreign aid spending is too high when we have problems here at home.",
        "The deficit keeps growing. Neither party actually cares about fiscal responsibility.",
        "Military spending needs an audit. The Pentagon can't account for trillions.",
    ],

    "news_commentary": [
        "Did you see the news about the earthquake? Hope everyone is okay over there.",
        "The stock market dropped three percent today. Probably a correction after the rally.",
        "The summit between the two leaders seems to be going well. Hopefully peace talks work.",
        "That new study about climate change is pretty alarming. The data looks solid.",
        "The tech company layoffs are getting out of hand. Who's hiring anymore?",
        "Gas prices went up again. I'm spending way too much on my commute.",
        "The infrastructure bill finally passed. Maybe they'll fix the roads in my city.",
        "Another data breach at a major company. Password managers are essential now.",
        "The space launch was successful. Amazing what private companies can do in orbit.",
        "The sports team traded their best player. What a terrible decision by management.",
        "Weather forecast says snow all week. I hate winter in the Northeast.",
        "Housing prices in my area are insane. Who can afford to buy anything anymore?",
        "The new phone release was underwhelming. Barely any improvements over last year.",
        "That documentary about the scandal was really well done. Shocking revelations.",
        "Traffic was terrible today because of the construction on the highway.",
    ],

    "casual_discussion": [
        "Anybody played the new game that came out? The reviews look mixed but I'm curious.",
        "Made some great pasta tonight. Fresh basil makes all the difference.",
        "What are you guys watching tonight? I need something good on Netflix.",
        "My cat knocked over my monitor again. This is why I can't have nice things.",
        "Gym has been packed lately with all the New Year resolution people.",
        "Anyone know a good mechanic? My car is making weird noises again.",
        "The new album dropped and it's actually fire. Best thing they've released in years.",
        "Weekend plans? I'm thinking about going hiking if the weather holds up.",
        "Just finished reading that book everyone recommended. It was decent, not great.",
        "Pizza or tacos for dinner? I can't decide and I'm starving.",
        "My neighbor's dog won't stop barking. I'm about to lose my mind.",
        "Anyone else here work night shifts? The schedule is killing my social life.",
        "The memes from yesterday's press conference were hilarious. Top quality content.",
        "I need to upgrade my PC. The graphics card is three generations behind.",
        "Coffee or energy drinks? I need something to get through this workday.",
    ],

    "hard_negatives": [
        "The apocalypse movie was actually pretty good. The CGI was impressive for the budget.",
        "In Fallout 4, the post-apocalyptic world design is incredible. Love the atmosphere.",
        "The Roman Empire fell due to a combination of factors including overexpansion and corruption.",
        "Studying the Black Death shows how pandemics can reshape entire civilizations.",
        "The Cold War nuclear standoff was terrifying but mutually assured destruction prevented war.",
        "Climate scientists predict gradual sea level rise, not sudden catastrophic events.",
        "The zombie apocalypse genre is overdone. We need more original horror concepts.",
        "Mad Max Fury Road remains one of the best action films ever made.",
        "The Great Depression lasted a decade but the economy eventually recovered stronger.",
        "Historical empires rise and fall. That's just how history works. Nothing unusual about it.",
        "The Walking Dead finally ended. What a ride from the first season.",
        "My survival gear collection is just a hobby. I'm not actually prepping for anything.",
        "The medieval period wasn't actually a dark age. That's a popular misconception.",
        "Nuclear war would be devastating but treaties have reduced arsenals significantly.",
        "The financial crisis of 2008 was bad but the system adapted and recovered.",
    ],

    "pol_baseline": [
        "The absolute state of Democrats. They can't even run a simple debate properly.",
        "The founding fathers would be disappointed with modern American politics.",
        "Just got banned from Reddit again for stating basic crime statistics.",
        "Leftists are so hypocritical about free speech. They only support it for themselves.",
        "Rate my dinner. Made steak and potatoes like a real man tonight.",
        "How do we fix the education system? Schools are just indoctrination camps now.",
        "Imagine still watching CNN or Fox News in the current year. Read a book instead.",
        "Post your best Pepe collection. Comfy meme thread for the evening.",
        "Any of you guys lift? Share your gym routine. No skipping leg day.",
        "Why do Europeans let their governments tax them into the ground?",
        "Thread about national flags. Rank the top ten best flag designs from any country.",
        "Most redpilled movies? I would say Fight Club and The Matrix. Name better ones.",
        "Seriously what is wrong with modern architecture? Everything looks soulless and ugly.",
        "Does anyone here actually have friends in real life or is it all online?",
        "Post your honest political compass results. No larping allowed.",
    ],
}


# ══════════════════════════════════════════════════════════════════════
# Lightweight keyword dictionary (diagnostic only — NOT in combined)
# ══════════════════════════════════════════════════════════════════════
# Retained for interpretability and ablation studies.

DIAGNOSTIC_KEYWORDS: list[tuple[str, float]] = [
    (r"\bend\s+times?\b", 3.0),
    (r"\bapocalyps[ei]?\b", 3.0),
    (r"\barmageddon\b", 3.0),
    (r"\bjudgment\s+day\b", 2.5),
    (r"\bjudgement\s+day\b", 2.5),
    (r"\bday\s+of\s+the\s+rope\b", 3.0),
    (r"\brapture\b", 2.5),
    (r"\brace\s+war\b", 2.5),
    (r"\brahowa\b", 2.5),
    (r"\bboogaloo\b", 2.0),
    (r"\bcollapse\s+of\s+(?:civilization|society|the\s+west)\b", 2.5),
    (r"\bkali\s+yuga\b", 3.0),
    (r"\bgreat\s+replacement\b", 2.5),
    (r"\bwhite\s+genocide\b", 2.5),
    (r"\bnew\s+world\s+order\b", 2.0),
    (r"\bgreat\s+reset\b", 2.0),
    (r"\baccelerat(?:e|ion|ionism|ionist)\b", 2.0),
    (r"\bsiege\s*culture\b", 2.0),
    (r"\bread\s+siege\b", 2.0),
    (r"\bday\s+of\s+reckoning\b", 2.5),
    (r"\bcivil\s+war\s+(?:is\s+)?coming\b", 2.5),
    (r"\bend\s+of\s+(?:civilization|the\s+west|the\s+world)\b", 2.5),
    (r"\bweimar\b", 2.0),
    (r"\bzog\b", 1.5),
    (r"\bkalergi\s+plan\b", 2.5),
    (r"\btotal\s+(?:race|racial)\s+war\b", 3.0),
    (r"\bhelter\s+skelter\b", 2.5),
    (r"\bblood\s+and\s+soil\b", 1.5),
    (r"\bdeus\s+vult\b", 2.0),
    (r"\bfinal\s+solution\b", 2.0),
    (r"\bclown\s+world\b", 0.5),
    # Counter-indicators
    (r"\bvideo\s+game\b", -2.0),
    (r"\bmovie\b", -1.0),
    (r"\bapocalypse\s+now\b", -2.0),
    (r"\bzombie\s+apocalypse\b", -1.5),
]

_COMPILED_KEYWORDS = [
    (re.compile(p, re.IGNORECASE), w) for p, w in DIAGNOSTIC_KEYWORDS
]


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════

def compute_diagnostic_keyword_score(text: str | None) -> dict:
    """Lightweight keyword score for diagnostic/interpretability only.

    NOT used in the combined classifier score.
    """
    if not text:
        return {"apoc_keyword_count": 0, "apoc_keyword_score": 0.0,
                "apoc_keyword_density": 0.0, "apoc_binary": 0}

    total_score = 0.0
    total_count = 0
    for pat, weight in _COMPILED_KEYWORDS:
        matches = pat.findall(text)
        if matches:
            total_score += weight * len(matches)
            total_count += len(matches)

    word_count = max(len(text.split()), 1)
    density = max(total_score, 0.0) / word_count * 100

    return {
        "apoc_keyword_count": total_count,
        "apoc_keyword_score": max(total_score, 0.0),
        "apoc_keyword_density": max(density, 0.0),
        "apoc_binary": int(total_score > 0),
    }


# Keep old name available for backward compatibility
compute_apoc_keyword_score = compute_diagnostic_keyword_score


def build_centroids(
    model: SentenceTransformer,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build positive, negative, and sub-theme centroid vectors.

    Returns
    -------
    positive_centroid : (D,) normalised vector for all positive seeds
    negative_centroid : (D,) normalised vector for all negative seeds
    subtheme_centroids : dict mapping sub-theme name → (D,) normalised vector
    """
    # Positive centroids (per sub-theme and overall)
    all_pos_embeddings = []
    subtheme_centroids: dict[str, np.ndarray] = {}

    for theme, texts in POSITIVE_SEEDS.items():
        embs = model.encode(
            texts, batch_size=32, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        centroid = embs.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        subtheme_centroids[theme] = centroid
        all_pos_embeddings.append(embs)

    pos_all = np.vstack(all_pos_embeddings)
    positive_centroid = pos_all.mean(axis=0)
    positive_centroid = positive_centroid / np.linalg.norm(positive_centroid)

    # Negative centroid
    all_neg = []
    for texts in NEGATIVE_SEEDS.values():
        embs = model.encode(
            texts, batch_size=32, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        all_neg.append(embs)
    neg_all = np.vstack(all_neg)
    negative_centroid = neg_all.mean(axis=0)
    negative_centroid = negative_centroid / np.linalg.norm(negative_centroid)

    return positive_centroid, negative_centroid, subtheme_centroids


def build_category_centroids(
    model: SentenceTransformer,
) -> dict[str, np.ndarray]:
    """Build centroid vectors for the 4 apocalypticism categories.

    Returns dict mapping category name → (D,) normalised vector.
    """
    cat_centroids: dict[str, np.ndarray] = {}
    for cat, texts in CATEGORY_SEEDS.items():
        embs = model.encode(
            texts, batch_size=32, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        centroid = embs.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        cat_centroids[cat] = centroid
    return cat_centroids


# Keep old name for backward compatibility
def build_centroid(model: SentenceTransformer) -> np.ndarray:
    """Build the positive-class centroid (backward-compatible API)."""
    pos, _neg, _sub = build_centroids(model)
    return pos


def train_classifier(model: SentenceTransformer) -> LogisticRegression:
    """Train logistic regression on sentence-transformer embeddings.

    Uses synthetic positive/negative seeds as training data.
    Training is deterministic and fast (<1 s for ~200 examples on 384-d).
    """
    pos_texts = [t for texts in POSITIVE_SEEDS.values() for t in texts]
    neg_texts = [t for texts in NEGATIVE_SEEDS.values() for t in texts]

    pos_embs = model.encode(
        pos_texts, batch_size=64, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    neg_embs = model.encode(
        neg_texts, batch_size=64, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    X = np.vstack([pos_embs, neg_embs])
    y = np.array([1] * len(pos_embs) + [0] * len(neg_embs))

    lr = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=42,
    )
    lr.fit(X, y)

    return lr


def score_embeddings(
    embeddings: np.ndarray,
    lr: LogisticRegression,
    positive_centroid: np.ndarray,
    negative_centroid: np.ndarray,
    subtheme_centroids: dict[str, np.ndarray],
    category_centroids: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Score post embeddings using LR + contrastive similarity.

    Returns dict of arrays, each length N (number of posts).
    """
    n = embeddings.shape[0]

    # 1. LR probability
    lr_probs = lr.predict_proba(embeddings)[:, 1]

    # 2. Similarity to positive centroid (backward compat)
    pos_sim = cosine_similarity(
        embeddings, positive_centroid.reshape(1, -1)
    ).flatten()

    # 3. Similarity to negative centroid
    neg_sim = cosine_similarity(
        embeddings, negative_centroid.reshape(1, -1)
    ).flatten()

    # 4. Contrastive score: positive − negative similarity
    contrastive = pos_sim - neg_sim

    # 5. Sub-theme identification
    theme_names = list(subtheme_centroids.keys())
    theme_centroids = np.vstack([subtheme_centroids[t] for t in theme_names])
    theme_sims = cosine_similarity(embeddings, theme_centroids)  # (N, T)
    dominant_theme_idx = theme_sims.argmax(axis=1)
    dominant_theme = np.array([theme_names[i] for i in dominant_theme_idx])
    dominant_theme_sim = theme_sims[np.arange(n), dominant_theme_idx]

    # 6. Normalise contrastive to [0, 1] using clip (raw scale)
    contrastive_norm = np.clip(contrastive, 0.0, 1.0)

    # 7. Combined score
    combined = LR_WEIGHT * lr_probs + CONTRASTIVE_WEIGHT * contrastive_norm

    # 8. Binary
    binary = (combined >= BINARY_THRESHOLD).astype(np.int8)

    result = {
        "apoc_lr_prob": lr_probs,
        "apoc_similarity": pos_sim,
        "apoc_contrastive": contrastive,
        "apoc_combined": combined,
        "apoc_binary": binary,
        "apoc_subtheme": dominant_theme,
        "apoc_subtheme_sim": dominant_theme_sim,
    }

    # 9. Category identification (4-way disaggregation)
    if category_centroids is not None:
        cat_names = list(category_centroids.keys())
        cat_centroid_matrix = np.vstack(
            [category_centroids[c] for c in cat_names]
        )
        cat_sims = cosine_similarity(embeddings, cat_centroid_matrix)  # (N, 4)
        dominant_cat_idx = cat_sims.argmax(axis=1)
        dominant_cat = np.array([cat_names[i] for i in dominant_cat_idx])
        dominant_cat_sim = cat_sims[np.arange(n), dominant_cat_idx]

        # Per-category similarity columns for downstream analysis
        result["apoc_category"] = dominant_cat
        result["apoc_category_sim"] = dominant_cat_sim
        for j, cname in enumerate(cat_names):
            result[f"apoc_cat_sim_{cname}"] = cat_sims[:, j]

    return result


def _keyword_batch(texts: list[str]) -> list[dict]:
    """Score a batch of texts for diagnostic keywords (multiprocessing)."""
    return [compute_diagnostic_keyword_score(t) for t in texts]


# ── backward-compat stubs ────────────────────────────────────────────

def apply_embedding_boost(
    base_scores: np.ndarray,
    context_scores: np.ndarray,
    similarities: np.ndarray,
) -> np.ndarray:
    """Kept for backward compatibility with existing tests.

    In the new architecture, embedding scores are the PRIMARY signal,
    not a boost on top of keywords.  This stub applies a weighted
    combination for any callers that still use it.
    """
    adjusted = base_scores.copy().astype(np.float64)
    non_context = base_scores - context_scores
    multiplier = np.ones_like(similarities)
    multiplier[similarities >= 0.4] = 2.0
    multiplier[(similarities >= 0.3) & (similarities < 0.4)] = 1.5
    multiplier[similarities < 0.15] = 0.25
    adjusted = non_context + (context_scores * multiplier)
    return np.maximum(adjusted, 0.0)


# ══════════════════════════════════════════════════════════════════════
# Main pipeline stage
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("STAGE 30: Apocalypticism Classification (/pol/)")
    print("=" * 60)

    # ── Load /pol/ posts ──────────────────────────────────────────────
    pol_path = DATA_PROCESSED / "pol_posts.parquet"
    if not pol_path.exists():
        print(f"  ✗ {pol_path} not found. Run 01b_preprocess_pol first.")
        return

    pol = pl.read_parquet(pol_path)
    print(f"  /pol/ posts loaded: {pol.height:,}")

    # ── Initialise model ──────────────────────────────────────────────
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"  Device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # ── Train LR classifier on synthetic data ─────────────────────────
    print("\n  Training logistic regression on synthetic seed data…")
    lr = train_classifier(model)
    print(f"  LR training accuracy on seeds: {lr.score(lr.__dict__.get('_X', np.empty(0)), lr.__dict__.get('_y', np.empty(0))) if hasattr(lr, '_X') else 'N/A'}")

    # ── Build centroids ───────────────────────────────────────────────
    print("  Building apocalypticism centroids…")
    pos_centroid, neg_centroid, subtheme_centroids = build_centroids(model)
    cat_centroids = build_category_centroids(model)

    # Save positive centroid (backward compat)
    centroid_path = DATA_PROCESSED / "apocalypticism_centroid.npy"
    np.save(centroid_path, pos_centroid)
    print(f"  Centroid saved: {centroid_path.name}")
    print(f"  Sub-themes: {list(subtheme_centroids.keys())}")
    print(f"  Categories: {list(cat_centroids.keys())}")

    # ── 1. Diagnostic keyword scoring ─────────────────────────────────
    print("\n  Computing diagnostic keyword scores…")
    texts = pol["text"].fill_null("").to_list()

    chunk_size = max(1, len(texts) // _N_WORKERS)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    with multiprocessing.Pool(_N_WORKERS) as pool:
        results_nested = pool.map(_keyword_batch, chunks)
    kw_dicts = [d for batch in results_nested for d in batch]

    pol = pol.with_columns([
        pl.Series("apoc_keyword_count",
                   [d["apoc_keyword_count"] for d in kw_dicts], dtype=pl.Int32),
        pl.Series("apoc_keyword_score",
                   [d["apoc_keyword_score"] for d in kw_dicts], dtype=pl.Float64),
        pl.Series("apoc_keyword_density",
                   [d["apoc_keyword_density"] for d in kw_dicts], dtype=pl.Float64),
    ])

    n_kw_apoc = sum(1 for d in kw_dicts if d["apoc_binary"])
    print(f"  Keyword hits (diagnostic only): {n_kw_apoc:,} posts "
          f"({n_kw_apoc / pol.height * 100:.2f}%)")

    # ── 2. Compute post embeddings ────────────────────────────────────
    print("\n  Computing post embeddings…")
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    # ── 3. Score embeddings ───────────────────────────────────────────
    print("  Scoring with LR + contrastive similarity…")
    scores = score_embeddings(
        embeddings, lr, pos_centroid, neg_centroid, subtheme_centroids,
        category_centroids=cat_centroids,
    )

    pol = pol.with_columns([
        pl.Series("apoc_lr_prob", scores["apoc_lr_prob"], dtype=pl.Float64),
        pl.Series("apoc_similarity", scores["apoc_similarity"], dtype=pl.Float64),
        pl.Series("apoc_contrastive", scores["apoc_contrastive"], dtype=pl.Float64),
        pl.Series("apoc_combined", scores["apoc_combined"], dtype=pl.Float64),
        pl.Series("apoc_binary", scores["apoc_binary"], dtype=pl.Int8),
        pl.Series("apoc_subtheme", scores["apoc_subtheme"], dtype=pl.Utf8),
        pl.Series("apoc_subtheme_sim", scores["apoc_subtheme_sim"], dtype=pl.Float64),
        pl.Series("apoc_category", scores["apoc_category"], dtype=pl.Utf8),
        pl.Series("apoc_category_sim", scores["apoc_category_sim"], dtype=pl.Float64),
    ])

    n_apoc = pol.filter(pl.col("apoc_binary") == 1).height

    # ── 4. Save ───────────────────────────────────────────────────────
    out_path = DATA_PROCESSED / "pol_apocalypticism_scores.parquet"
    pol.write_parquet(out_path)

    print(f"\n  Summary statistics:")
    print(f"    LR prob mean:         {scores['apoc_lr_prob'].mean():.4f}")
    print(f"    Similarity mean:      {scores['apoc_similarity'].mean():.4f}")
    print(f"    Contrastive mean:     {scores['apoc_contrastive'].mean():.4f}")
    print(f"    Combined mean:        {scores['apoc_combined'].mean():.4f}")
    print(f"    Apocalyptic (binary): {n_apoc:,} ({n_apoc / pol.height * 100:.2f}%)")
    print(f"    Keyword hits (diag):  {n_kw_apoc:,}")

    # Sub-theme distribution among apocalyptic posts
    apoc_posts = pol.filter(pl.col("apoc_binary") == 1)
    if apoc_posts.height > 0:
        print(f"\n  Sub-theme distribution (apocalyptic posts):")
        for row in apoc_posts.group_by("apoc_subtheme").len().sort("len", descending=True).iter_rows():
            print(f"    {row[0]:30s} {row[1]:6d} "
                  f"({row[1]/apoc_posts.height*100:.1f}%)")

        print(f"\n  Category distribution (apocalyptic posts):")
        for row in apoc_posts.group_by("apoc_category").len().sort("len", descending=True).iter_rows():
            print(f"    {row[0]:30s} {row[1]:6d} "
                  f"({row[1]/apoc_posts.height*100:.1f}%)")

    print(f"\n✓ Apocalypticism scoring complete. Saved to {out_path.name}")


if __name__ == "__main__":
    main()
