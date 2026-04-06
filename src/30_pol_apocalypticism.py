"""30 – Apocalypticism Classifier for /pol/.

Identifies apocalyptic rhetoric across /pol/ posts using a transformer-
based approach that prioritises contextual sensitivity over rigid keyword
matching.

Architecture
------------
1. **MLP classifier on sentence embeddings** — a two-layer neural
   network trained on a large synthetic dataset of positive (apocalyptic)
   and negative (non-apocalyptic /pol/ content) examples, with Gaussian-
   noise data augmentation for robustness.  Provides calibrated
   probability scores via sigmoid.
2. **Multi-facet contrastive similarity** — separate centroid vectors for
   five apocalyptic sub-themes (eschatological, accelerationist,
   civilisational collapse, racial apocalypse, conspiratorial) plus a
   negative centroid.  Contrastive score = max positive similarity −
   negative similarity.  This captures posts using novel phrasing that
   the classifier may not have seen in training.
3. **Combined score** — 0.6 × MLP probability + 0.4 × normalised
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
import torch.nn as nn
from sentence_transformers import SentenceTransformer
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
# MLP classifier architecture
# ══════════════════════════════════════════════════════════════════════

class ApocalypticismMLP(nn.Module):
    """2-layer MLP classifier head on sentence-transformer embeddings.

    Architecture: input → Linear → BN → ReLU → Dropout →
                  Linear → BN → ReLU → Dropout → Linear → logit
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Return P(apocalyptic) for each row, sklearn-style interface."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(embeddings, dtype=torch.float32)
            logits = self.forward(x)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs


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
        "There is no political solution. The system is beyond reform. Accelerate.",
        "DOTR approaches. Every traitor and race mixer will be held accountable.",
        "The Order of Nine Angles understands the cosmic war we are engaged in.",
        "Terrorgram content shows the path forward for committed accelerationists.",
        "Saints like Tarrant showed the world what one dedicated man can accomplish.",
        "Tommasi and the NSLF pioneered the armed struggle philosophy decades ago.",
        "Tom Metzger and White Aryan Resistance understood the coming racial apocalypse.",
        "In Minecraft we will dismantle the power grid and watch civilization crumble.",
        "Infrastructure is fragile. A few well-placed actions could bring the whole thing down.",
        "The fire rises. Every act of chaos accelerates the inevitable collapse.",
        "GtkRWN is not just a slogan. The race war is an inevitability of demographics.",
        "1488. The fourteen words are the only moral imperative that matters now.",
        "Pine Tree Gang understands ecological fascism is the answer to modernity.",
        "Harassment Architecture was right about everything. Mike Ma is a prophet.",
        "The militant accelerationist handbook lays out what must be done clearly.",
        "Power grid attacks are the most effective way to bring about systemic collapse.",
        "The great cleansing fire will purify this degenerate age and usher in rebirth.",
        "Lone wolf resistance is the only strategy the system cannot defeat.",
        "The NSLF, Atomwaffen, The Base — all understood the necessity of direct action.",
        "Every fed who glows cannot stop the tide. The collapse is inevitable.",
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
        "The Babylon system will fall. Mystery Babylon will be utterly destroyed.",
        "We are the generation spoken of in Matthew 24. All these signs fulfilled.",
        "The false prophet is here. The mark is being forced on the nations.",
        "Ezekiel 38 and Gog and Magog prophecy is being fulfilled through current wars.",
        "The great falling away has begun. Churches are compromised. End times are here.",
        "Blood moons, earthquakes, wars and rumors of wars. All signs of the end.",
        "The temple will be rebuilt in Jerusalem and then the tribulation truly begins.",
        "Seal your foreheads with the blood of Christ. The day of wrath approaches.",
        "The two witnesses will appear in Jerusalem. The final countdown has started.",
        "The harvest of souls approaches. The wheat will be separated from the tares.",
        "Psalm 83 and Isaiah 17 describe the coming wars that precede the end.",
        "The dry bones of Israel are risen. Ezekiel's prophecy proves we are in the last days.",
        "America is the new Babylon. Its judgment draws nigh. Come out of her my people.",
        "The restrainer is being removed. The lawless one will be revealed soon.",
        "The seven trumpets of Revelation will sound and bring divine judgment on earth.",
        "The nephilim are returning. Days of Noah are upon us just as Jesus warned.",
        "Dreams and visions prophesied in Joel are multiplying. The spirit is being poured out.",
        "The prince of darkness rules this age but the kingdom of light approaches.",
        "Soon every knee shall bow. The day of the Lord comes like a thief in the night.",
        "The battle of Armageddon in the valley of Megiddo is imminent. Prepare your souls.",
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
        "WROL conditions are coming. Without rule of law you need to be ready to defend yourself.",
        "Innawoods is the only viable strategy. Get out of the cities before SHTF.",
        "Roof Koreans had the right idea. Community defense is essential when order breaks down.",
        "Gray man doctrine: blend in during the collapse and nobody targets you for resources.",
        "Three is two, two is one, one is none. Redundancy in all your preps.",
        "Night vision and thermal optics give you the decisive advantage in a grid down scenario.",
        "Cache supplies at multiple locations. Never keep everything in one place.",
        "The golden horde from the cities will flood the countryside when the food runs out.",
        "Antibiotics, trauma kits, surgical supplies. Medical will be the biggest gap post-collapse.",
        "Community defense perimeters and comms protocols need to be established before SHTF.",
        "Body armor and plate carriers are essential gear for any serious prepper.",
        "Solar, wind, and micro-hydro. Energy independence means survival independence.",
        "Learn primitive skills: fire starting, snares, foraging. Technology will fail eventually.",
        "The Carrington Event would destroy the entire power grid. Have an EMP plan.",
        "Two weeks of chaos is all it takes for civilization to completely unravel.",
        "NBC gear and decontamination supplies for nuclear, biological, chemical threats.",
        "Shortwave radio and mesh networks for when cell towers and internet go down.",
        "Precious metals and barter goods will replace fiat currency after the collapse.",
        "Underground bunkers and hardened shelters for the worst case scenarios.",
        "Operational security means keeping your preps quiet. Loose lips sink ships.",
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
        "It's happening. The system is coming apart at the seams and nothing can stop it.",
        "Happening thread. Everything is accelerating toward the breaking point right now.",
        "This is the big one. The happening we have been waiting for is finally here.",
        "The fire rises. Chaos breeds chaos and the spiral only goes in one direction.",
        "Clown world honk honk. The absurdity of this society guarantees its collapse.",
        "Nothing can save this system. The rot goes all the way to the foundation.",
        "The spicy times are here. Everything from here on out gets worse not better.",
        "Balkanization is inevitable. This country will fracture along racial and political lines.",
        "Fourth turning is upon us. The crisis era will transform everything we know.",
        "Rome was not built in a day but it fell in one generation. We are in that generation.",
        "The Petrodollar is dying and with it American hegemony. What replaces it will be chaos.",
        "Weimar conditions precede societal destruction. History does not repeat but it rhymes.",
        "Two weeks. Give it two weeks and everything will be different. Mark my words.",
        "Supply chains, financial systems, social trust — all are failing simultaneously.",
        "The Kali Yuga demands dissolution before renewal. We are in the darkest phase.",
        "Cities will become ungovernable. The rural areas will be the last stand of civilization.",
        "Globalization is unraveling. The interconnected system is its own greatest vulnerability.",
        "The age of abundance is ending. Scarcity and conflict will define the next century.",
        "Every empire falls. Ours is no exception. The only question is how fast.",
        "It is all so tiresome. But the collapse will cleanse the slate eventually.",
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
        "The nephilim are returning. Days of Noah repeat. The judgment comes swiftly.",
        "Blood moons and eclipses align with prophecy. The cosmic signs are undeniable.",
        "The restrainer is being removed. The man of lawlessness will be revealed imminently.",
        "Daniel's statue of iron and clay is crumbling. The stone cut without hands approaches.",
        "Gog and Magog armies are assembling. Ezekiel 38 unfolds before our eyes.",
        "The seals are being opened one by one. We are past the fourth horseman.",
        "Babylon the great has fallen. Come out of her my people before the plagues arrive.",
        "The two witnesses will appear in Jerusalem and prophecy during the final days.",
        "Trumpets are sounding in the heavens. The strange sounds people report are angelic warnings.",
        "The prince of this world rules for a season but the kingdom of light draws near.",
        "Maranatha. The Lord comes quickly. Every sign confirms the imminent return.",
        "The wheat is being separated from the tares. The harvest of souls approaches.",
        "All nations will turn against Israel and then the end will come just as prophesied.",
        "The abyss is opening. The locusts of Revelation will torment mankind for five months.",
        "The wrath of the Lamb is upon this generation. No bunker will save the wicked.",
        "Seven bowls of God's wrath will be poured out. Plagues, darkness, and fire.",
        "The final apostasy is here. Churches preach prosperity while the end approaches.",
        "The fig tree generation will not pass away. Israel's rebirth starts the clock.",
        "Wars and rumors of wars. Nation rising against nation. Birth pangs intensifying.",
        "The abomination of desolation spoken of by Daniel stands in the holy place today.",
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
        "The boog is inevitable. Big igloo time is coming. Get your Hawaiian shirt ready.",
        "Big luau energy. The boojahideen are preparing for what comes next.",
        "It's happening. The happening is finally here. This is not a drill brothers.",
        "Happening thread. Get in here. Everything is accelerating toward the breaking point.",
        "The fire rises. Every day the flames grow higher and nothing can extinguish them.",
        "DOTR is not a meme. The day of the rope is a promise not a threat.",
        "Read Siege and prepare. Mason laid out the blueprint for total system collapse.",
        "In Minecraft we will target the infrastructure. Power grid goes down first.",
        "GtkRWN. The race war will come whether you want it to or not. Choose a side.",
        "McNuke the feds. The boogaloo boys understand what the founding fathers intended.",
        "Fedposting is just truth telling. The system deserves everything that is coming.",
        "There is no political solution. Voting is cope. The only way out is through collapse.",
        "Terrorgram channels distribute the knowledge that the system fears most.",
        "The militant accelerationist handbook is required reading for serious people.",
        "Power grid attacks cause maximum disruption with minimum effort. Think strategically.",
        "Every saint who took direct action moved the timeline forward. Honor their sacrifice.",
        "The Order of Nine Angles recognized the cosmic dimension of the struggle.",
        "Atomwaffen Division understood that small cells can cause maximum systemic damage.",
        "The Base was right about organizing for post-collapse territorial control.",
        "Brenton Tarrant's manifesto section on acceleration was the most important part.",
        "Civil War 2 electric boogaloo is not a joke. The first shots have already been fired.",
        "1488 is not just numbers. It's a commitment to action when the time comes.",
        "Nothing ever happens you say? Just wait. The happening will catch you off guard.",
        "Spicy times ahead. When the balloon goes up you better be on the right side.",
        "The system is a paper tiger. One good push and the whole thing topples.",
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
        "Honkler was right. This is a clown world and clown worlds always burn themselves down.",
        "The Weimar conditions of today mirror the 1920s exactly. We know what comes next.",
        "Fourth turning crisis period has arrived. The old order must be destroyed for renewal.",
        "It is all so tiresome but the collapse will cleanse the slate eventually.",
        "Balkanization is the only possible future. There is no united anything anymore.",
        "The fire rises from the ashes of what was once a great civilization.",
        "Spengler, Toynbee, and Evola all predicted this exact stage of civilizational decay.",
        "The demographic time bomb is ticking. Mathematics guarantee civilizational transformation.",
        "Bread and circuses while Rome burns. The masses are too distracted to see the end coming.",
        "This civilization chose comfort over survival. It sealed its own fate with that choice.",
        "The social contract is broken. Without trust, cooperation, and shared purpose all falls apart.",
        "Every metric of civilizational health — fertility, social cohesion, institutional trust — is declining.",
        "Oswald Spengler The Decline of the West reads like a news report about today.",
        "Mass immigration into declining civilizations always accelerates the collapse. Every single time.",
        "The cultural immune system of the West has been deliberately compromised from within.",
        "Two more weeks until the next crisis. And the next. And the next. Accelerating collapse.",
        "It's happening in slow motion but it's happening. The trajectory is clear and irreversible.",
        "The long decline is here. Some things collapse fast but civilizations die slowly then suddenly.",
        "Atlas is shrugging. The productive class is checking out. The system cannot sustain parasitism.",
        "Anarcho-tyranny: the state punishes the law-abiding while the criminals run free. System failure.",
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
        "ZOG orchestrates the demographic destruction of every white nation systematically.",
        "1488. Fourteen words. We must secure the existence of our people. Whatever it takes.",
        "The race war is not coming. It is already here. Demographics are destiny.",
        "DOTR for every race traitor. The day of reckoning will not discriminate among the guilty.",
        "White birth rates collapse while replacement migration floods every Western nation.",
        "The Turner Diaries described a future that becomes more plausible with each passing year.",
        "Ethnostate or extinction. There is no middle ground. Choose now.",
        "South Africa shows the future of whites in a minority position. Farm murders are just the start.",
        "Diversity plus proximity equals conflict. This is a law of nature not an opinion.",
        "GtkRWN is not ironic for millions of people worldwide. The sentiment is real and growing.",
        "Every interracial crime statistic proves that multicultural society is a one-sided war.",
        "The Hart-Celler Act of 1965 was the beginning of the end for white America.",
        "The browning of America is not natural demographic change. It is engineered genocide.",
        "Fourteen eighty-eight. When you understand what those numbers mean you understand everything.",
        "Rhodesia and South Africa are the prophetic templates for what awaits white minorities everywhere.",
        "Race and IQ research proves that civilizational standards cannot be maintained through replacement.",
        "White flight is a survival instinct. But there is nowhere left to flee. Stand or perish.",
        "The race war will be fought street by street, neighborhood by neighborhood. Prepare accordingly.",
        "The Camp of the Saints is not fiction. It is documentary prophecy unfolding before our eyes.",
        "Multiculturalism is anti-white. Diversity is code for fewer white people. Wake up.",
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
        "The storm is coming. Nothing can stop what is coming. Trust the plan then watch it crumble.",
        "QAnon was right about the cabal even if the plan was compromised. The darkness is real.",
        "Central bank digital currencies are the mark of the beast implementation phase.",
        "The depopulation agenda is openly stated in their own documents. Read the Georgia Guidestones.",
        "Agenda 2030 is the blueprint for global technocratic enslavement. Every goal is a cage.",
        "Chemtrails, HAARP, weather modification — they are terraforming the planet against us.",
        "The fluoride in the water is deliberate. Mass medication without consent to dumb down the populace.",
        "Project Bluebeam will fake an alien invasion to unite the world under one government.",
        "The banking cartel created every major war for profit. World War 3 is their next harvest.",
        "MKUltra never ended. Mass shootings are programmed to advance the disarmament agenda.",
        "Bohemian Grove, Bilderberg, Davos — they plan our destruction in their secret meetings.",
        "The food supply is being deliberately poisoned to weaken and control the population.",
        "Transhumanism is the endgame. Merge with machines or be left behind in their new world.",
        "Event 201 proved they planned the pandemic. The next planned crisis will be worse.",
        "The great awakening is countered by the great deception. Most people will choose the lie.",
        "Digital ID and social credit scores are the infrastructure of the coming control grid.",
        "Klaus Schwab's fourth industrial revolution is techno-feudalism wrapped in utopian language.",
        "Fifteen minute cities are open air prisons. Restrict movement, restrict freedom, restrict life.",
        "They create the crisis, offer the solution, and consolidate power. Problem reaction solution.",
        "The controlled demolition of the economy is deliberate. Crash it and buy everything for pennies.",
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
        "Border security is a legitimate issue that both sides refuse to address honestly.",
        "The two-party system is broken. Third parties deserve a real chance.",
        "Congressional term limits would fix half the corruption in Washington overnight.",
        "The lobbying system is legalized bribery. Campaign finance reform is essential.",
        "Gerrymandering makes most House races uncompetitive. The system is rigged for incumbents.",
        "State's rights are being eroded by federal overreach on both sides of the aisle.",
        "The primary system produces extremist candidates that the general electorate doesn't want.",
        "Gun control debate is dishonest on both sides. Neither wants real compromise.",
        "Social security will be insolvent within decades. Someone needs to address it seriously.",
        "The media bias discussion is tiresome. Every outlet has an agenda. Read multiple sources.",
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
        "The wildfire coverage is heartbreaking. Those poor people losing everything.",
        "Unemployment numbers are better than expected but underemployment is still high.",
        "The chip shortage is affecting everything from cars to gaming consoles.",
        "The train derailment is getting barely any coverage. Media priorities are bizarre.",
        "Interest rates went up again. The Fed is trying to cool inflation at everyone's expense.",
        "The airline cancelled my flight again. Third time this month. Industry is broken.",
        "Crypto crashed hard today. A lot of people are going to lose their savings.",
        "The new city council voted to defund parks maintenance. Priorities are backwards.",
        "Hurricane season predictions are worse than usual this year. Better prepare.",
        "Factory shut down in my town. Three hundred people lost their jobs overnight.",
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
        "Just adopted a rescue dog. Best decision I have made in years honestly.",
        "Cooking for one is depressing. Everything is portioned for families in the store.",
        "The gym bros who never rerack their weights are a menace to society.",
        "Anyone else binge the whole season in one night? I need to go outside.",
        "Road trip playlist suggestions? Eight hours of driving ahead of me tomorrow.",
        "My internet has been dropping all week. ISP says nothing is wrong. Classic.",
        "The landlord raised rent again. Time to start looking for a new place.",
        "Who else is procrastinating right now? I should be working but here I am.",
        "The barber absolutely butchered my hair. Now I have to wear a hat for a month.",
        "Anyone recommend a good podcast? I need something for my commute.",
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
        "Nothing ever happens. People have been predicting the end for centuries and we are still here.",
        "It's happening but only in the movie I'm watching right now. Great film.",
        "The boogaloo meme used to be funny before real extremists co-opted the term.",
        "Siege is a terrible Tom Clancy game. I prefer the newer Rainbow Six entries.",
        "Happening threads on pol are always false alarms. Literally nothing ever actually happens.",
        "The collapse of that building was due to poor engineering not end times.",
        "Prepper YouTube channels are entertainment not serious survival education.",
        "Day of the Rope is just an edgy book reference. Nobody actually believes that garbage.",
        "The race war predictions from the 1960s never materialized. Society adapted and moved on.",
        "Accelerationism is a philosophy term from academia. Most people using it have no idea.",
        "Clown world is just a meme expressing frustration with absurd situations.",
        "Calling everything a happening is peak pol hyperbole. It was a minor event.",
        "The great reset is a World Economic Forum phrase for post-COVID policy suggestions.",
        "Kali Yuga references in metal music do not make someone a political extremist.",
        "Turner Diaries is poorly written propaganda fiction that only fools take seriously.",
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
        "Trad wife threads are cringe. Half of you can barely take care of yourselves.",
        "Greentext thread. Post your most embarrassing social interaction stories.",
        "Wage cage general. Another week of working for a company that doesn't care about you.",
        "Feels bar is open. Get in here and share what is bothering you tonight.",
        "Frog posting is a legitimate art form and I will die on this hill.",
        "The jannies are out of control on this board. Free speech is dead here.",
        "Post body. Oh wait you can't because you are a skinnyfat keyboard warrior.",
        "Infographic thread. Post useful charts and data visualizations on any topic.",
        "Boomer vs zoomer arguments are tiresome. Every generation has its problems.",
        "Music thread. Share what you are listening to right now. No judgment.",
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
    (r"\bdotr\b", 3.0),
    (r"\brapture\b", 2.5),
    (r"\brace\s+war\b", 2.5),
    (r"\brahowa\b", 2.5),
    (r"\bboogaloo\b", 2.5),
    (r"\bthe\s+boog\b", 2.5),
    (r"\bbig\s+igloo\b", 2.5),
    (r"\bbig\s+luau\b", 2.5),
    (r"\bboojahideen\b", 2.5),
    (r"\bcivil\s+war\s+2\b", 2.5),
    (r"\bcollapse\s+of\s+(?:civilization|society|the\s+west)\b", 2.5),
    (r"\bkali\s+yuga\b", 3.0),
    (r"\bgreat\s+replacement\b", 2.5),
    (r"\bwhite\s+genocide\b", 2.5),
    (r"\bnew\s+world\s+order\b", 2.0),
    (r"\bgreat\s+reset\b", 2.0),
    (r"\baccelerat(?:e|ion|ionism|ionist)\b", 2.0),
    (r"\bsiege\s*culture\b", 2.0),
    (r"\bread\s+siege\b", 2.5),
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
    # 4chan-specific apocalyptic/happening terminology
    (r"\bit'?s\s+happening\b", 2.0),
    (r"\bthe\s+happening\b", 2.0),
    (r"\bhappening\s+thread\b", 2.0),
    (r"\bthe\s+fire\s+rises\b", 2.0),
    (r"\bhonk\s*l?er\b", 1.0),
    (r"\bhonk\s+honk\b", 0.5),
    (r"\bnothing\s+ever\s+happens\b", -1.0),
    (r"\bshtf\b", 2.0),
    (r"\bwrol\b", 2.0),
    (r"\bgrid\s+down\b", 2.0),
    (r"\binnawoods\b", 1.5),
    (r"\bin\s+minecraft\b", 1.5),
    (r"\bfedpost(?:ing)?\b", 1.5),
    (r"\bthe\s+storm\s+is\s+coming\b", 2.0),
    (r"\bmcnuke\b", 1.5),
    (r"\bturner\s+diaries\b", 2.5),
    (r"\b1488\b", 2.0),
    (r"\b14\s*words?\b", 2.0),
    (r"\bgtkrwn\b", 3.0),
    (r"\bspicy\s+times?\b", 1.5),
    (r"\bno\s+political\s+solution\b", 2.5),
    (r"\bthere\s+is\s+no\s+political\s+solution\b", 3.0),
    (r"\bsaints?\b.*\baction\b", 1.5),
    (r"\batomwaffen\b", 3.0),
    (r"\bterrorgram\b", 3.0),
    (r"\border\s+of\s+nine\s+angles\b", 2.5),
    (r"\bthe\s+base\b.*\bcollapse\b", 2.0),
    (r"\bleaderless\s+resistance\b", 2.5),
    (r"\blone\s+wolf\b", 2.0),
    (r"\bcamp\s+of\s+the\s+saints\b", 2.5),
    (r"\bharassment\s+architecture\b", 2.0),
    (r"\bpine\s+tree\b.*\bgang\b", 1.5),
    (r"\bfourth\s+turning\b", 2.0),
    (r"\bpower\s+grid\b.*\b(?:attack|down|destroy|vulnerable)\b", 2.0),
    # Counter-indicators
    (r"\bvideo\s+game\b", -2.0),
    (r"\bmovie\b", -1.0),
    (r"\bapocalypse\s+now\b", -2.0),
    (r"\bzombie\s+apocalypse\b", -1.5),
    (r"\bfallout\s+\d\b", -1.5),
    (r"\bwalking\s+dead\b", -1.5),
    (r"\brainbow\s+six\b", -2.0),
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


def train_classifier(model: SentenceTransformer) -> ApocalypticismMLP:
    """Train 2-layer MLP classifier on sentence-transformer embeddings.

    Uses synthetic positive/negative seeds as training data, augmented
    with Gaussian noise for robustness.  Trains with Adam + BCE loss
    and early stopping.
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
    y = np.array([1.0] * len(pos_embs) + [0.0] * len(neg_embs))

    # Data augmentation: Gaussian noise copies for robustness
    rng = np.random.RandomState(42)
    X_aug = [X]
    y_aug = [y]
    for _ in range(5):  # 5× augmentation
        noise = rng.normal(0, 0.02, X.shape)
        aug = X + noise
        norms = np.linalg.norm(aug, axis=1, keepdims=True)
        aug = aug / norms  # re-normalise
        X_aug.append(aug)
        y_aug.append(y)

    X_train = np.vstack(X_aug)
    y_train = np.concatenate(y_aug)

    # Shuffle
    idx = rng.permutation(len(X_train))
    X_train = X_train[idx]
    y_train = y_train[idx]

    input_dim = X_train.shape[1]
    mlp = ApocalypticismMLP(input_dim=input_dim)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    mlp.train()
    best_loss = float("inf")
    patience = 20
    wait = 0

    for epoch in range(300):
        optimizer.zero_grad()
        logits = mlp(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    mlp.eval()

    # Report training accuracy
    with torch.no_grad():
        preds = (torch.sigmoid(mlp(X_t)) >= 0.5).float()
        acc = (preds == y_t).float().mean().item()
    print(f"  MLP training accuracy on augmented seeds: {acc:.4f} "
          f"({len(X_train)} samples, stopped at epoch {epoch + 1})")

    return mlp


def score_embeddings(
    embeddings: np.ndarray,
    classifier: ApocalypticismMLP,
    positive_centroid: np.ndarray,
    negative_centroid: np.ndarray,
    subtheme_centroids: dict[str, np.ndarray],
    category_centroids: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Score post embeddings using MLP + contrastive similarity.

    Returns dict of arrays, each length N (number of posts).
    """
    n = embeddings.shape[0]

    # 1. MLP probability
    lr_probs = classifier.predict_proba(embeddings)

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

    # ── Train MLP classifier on synthetic data ─────────────────────────
    print("\n  Training MLP classifier on synthetic seed data…")
    classifier = train_classifier(model)

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
    print("  Scoring with MLP + contrastive similarity…")
    scores = score_embeddings(
        embeddings, classifier, pos_centroid, neg_centroid, subtheme_centroids,
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
    print(f"    MLP prob mean:         {scores['apoc_lr_prob'].mean():.4f}")
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
