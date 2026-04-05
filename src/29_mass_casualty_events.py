"""29 – Mass-Casualty & Discontinuity Event Dataset Construction.

Builds a structured dataset of:
  (a) mass-casualty violent events in OECD countries since 2010, and
  (b) non-violence discontinuity events (natural disasters, economic
      shocks, political upheavals, health crises)

for use as treatment dates in interrupted time-series analyses of /pol/
apocalyptic rhetoric.

Events are drawn from public records (GTD/START, Gun Violence Archive,
Wikipedia mass-shooting and terrorist-incident lists, Europol TE-SAT
reports, ADL HEAT map, SPLC) and coded by:

  - date (UTC)
  - name / short label
  - location (country, city)
  - killed / injured
  - ideological category (for mass violence)
  - event category: mass_violence | natural_disaster | economic_shock |
    political | health_crisis
  - online nexus: whether perpetrator had known /pol//8chan/adjacent
    presence (mass violence only)
  - domestic vs. international (relative to the US)

Inclusion criteria for mass violence:
  ≥3 fatalities, OR widely covered with significant online engagement,
  OR unsuccessful but notable (e.g., foiled/low-casualty attack with
  online nexus or major media coverage).

Non-violence events are included as comparison/falsification treatments:
  if apocalyptic rhetoric responds ONLY to mass violence and NOT to
  non-violence discontinuities, the causal specificity of the violence
  effect is strengthened.

Output
------
``data/processed/mass_casualty_events.json``
``data/processed/mass_casualty_events.parquet``
``results/mass_casualty_events_summary.json``
"""

from __future__ import annotations

import json
from datetime import date, datetime

import polars as pl

from utils import DATA_PROCESSED, RESULTS_DIR

# ══════════════════════════════════════════════════════════════════════
# Valid enumerations
# ══════════════════════════════════════════════════════════════════════

VALID_CATEGORIES = {
    "mass_violence",
    "natural_disaster",
    "economic_shock",
    "political",
    "health_crisis",
}

VALID_IDEOLOGIES = {
    "far_right",
    "islamist",
    "school_shooting",
    "incel",
    "far_left",
    "other",
    "N/A",
}


# ══════════════════════════════════════════════════════════════════════
# Event catalogue
# ══════════════════════════════════════════════════════════════════════
# Sources:
#   Global Terrorism Database (START, University of Maryland)
#   Gun Violence Archive
#   Wikipedia "List of mass shootings in the United States"
#   Wikipedia "List of terrorist incidents"
#   Europol TE-SAT reports
#   Anti-Defamation League H.E.A.T. map
#   SPLC hate-crime chronology
#   NOAA Storm Events Database
#   WHO situation reports
#   OECD country membership list (38 countries as of 2023)
#
# Casualty counts reflect the most widely accepted totals.  Where
# sources disagree the lower credible estimate is used.
# ──────────────────────────────────────────────────────────────────────

EVENTS: list[dict] = [

    # ══════════════════════════════════════════════════════════════
    # 2010  —  MASS VIOLENCE
    # ══════════════════════════════════════════════════════════════
    {"date": "2010-02-18", "name": "Austin IRS plane attack",
     "location_country": "US", "location_city": "Austin, TX",
     "killed": 1, "injured": 13, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Andrew Joseph Stack III",
     "notes": "Flew plane into IRS building; anti-government manifesto"},

    {"date": "2010-06-02", "name": "Cumbria shootings",
     "location_country": "UK", "location_city": "Cumbria",
     "killed": 12, "injured": 11, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Derrick Bird",
     "notes": "Taxi driver shooting spree; personal grievances"},

    # ══════════════════════════════════════════════════════════════
    # 2010  —  NON-VIOLENCE DISCONTINUITIES
    # ══════════════════════════════════════════════════════════════
    {"date": "2010-04-20", "name": "Deepwater Horizon explosion",
     "location_country": "US", "location_city": "Gulf of Mexico",
     "killed": 11, "injured": 17, "ideology": "N/A",
     "event_category": "natural_disaster", "online_nexus": False,
     "domestic": True, "perpetrator": "Industrial accident (BP)",
     "notes": "Largest marine oil spill in history; massive environmental crisis"},

    {"date": "2010-05-06", "name": "Flash Crash",
     "location_country": "US", "location_city": "New York, NY",
     "killed": 0, "injured": 0, "ideology": "N/A",
     "event_category": "economic_shock", "online_nexus": False,
     "domestic": True, "perpetrator": "Algorithmic trading cascade",
     "notes": "DJIA dropped ~9% in minutes before recovering; systemic risk concerns"},

    # ══════════════════════════════════════════════════════════════
    # 2011  —  MASS VIOLENCE
    # ══════════════════════════════════════════════════════════════
    {"date": "2011-01-08", "name": "Tucson shooting (Giffords)",
     "location_country": "US", "location_city": "Tucson, AZ",
     "killed": 6, "injured": 13, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Jared Lee Loughner",
     "notes": "Targeted Rep. Gabby Giffords at constituent event; mixed ideology"},

    {"date": "2011-03-02", "name": "Frankfurt airport shooting",
     "location_country": "Germany", "location_city": "Frankfurt",
     "killed": 2, "injured": 2, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Arid Uka",
     "notes": "Targeted US airmen at Frankfurt airport; radicalized online"},

    {"date": "2011-07-22", "name": "Norway attacks (Breivik)",
     "location_country": "Norway", "location_city": "Oslo / Utøya",
     "killed": 77, "injured": 319, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": False, "perpetrator": "Anders Behring Breivik",
     "notes": "Bomb + mass shooting; published 1500-page manifesto online"},

    {"date": "2011-12-13", "name": "Liège attack",
     "location_country": "Belgium", "location_city": "Liège",
     "killed": 4, "injured": 125, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Nordine Amrani",
     "notes": "Grenades and gunfire at Place Saint-Lambert; attacker killed self"},

    # ══════════════════════════════════════════════════════════════
    # 2011  —  NON-VIOLENCE DISCONTINUITIES
    # ══════════════════════════════════════════════════════════════
    {"date": "2011-03-11", "name": "Tōhoku earthquake / Fukushima",
     "location_country": "Japan", "location_city": "Tōhoku region",
     "killed": 19747, "injured": 6242, "ideology": "N/A",
     "event_category": "natural_disaster", "online_nexus": False,
     "domestic": False, "perpetrator": "Earthquake / tsunami / nuclear meltdown",
     "notes": "M9.1 earthquake, tsunami, Fukushima Daiichi nuclear disaster"},

    {"date": "2011-08-05", "name": "US credit rating downgrade",
     "location_country": "US", "location_city": "Washington, DC",
     "killed": 0, "injured": 0, "ideology": "N/A",
     "event_category": "economic_shock", "online_nexus": False,
     "domestic": True, "perpetrator": "S&P downgrade from AAA to AA+",
     "notes": "First-ever US sovereign credit downgrade; markets dropped sharply"},

    {"date": "2011-09-17", "name": "Occupy Wall Street begins",
     "location_country": "US", "location_city": "New York, NY",
     "killed": 0, "injured": 0, "ideology": "N/A",
     "event_category": "political", "online_nexus": True,
     "domestic": True, "perpetrator": "Protest movement",
     "notes": "Anti-inequality protest movement; massive online mobilization"},

    # ══════════════════════════════════════════════════════════════
    # 2012  —  MASS VIOLENCE
    # ══════════════════════════════════════════════════════════════
    {"date": "2012-03-19", "name": "Toulouse and Montauban shootings",
     "location_country": "France", "location_city": "Toulouse / Montauban",
     "killed": 7, "injured": 5, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Mohammed Merah",
     "notes": "Killed French soldiers and Jewish schoolchildren; AQAP-linked"},

    {"date": "2012-07-20", "name": "Aurora theater shooting",
     "location_country": "US", "location_city": "Aurora, CO",
     "killed": 12, "injured": 70, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "James Holmes",
     "notes": "Opened fire at The Dark Knight Rises screening"},

    {"date": "2012-08-05", "name": "Wisconsin Sikh Temple shooting",
     "location_country": "US", "location_city": "Oak Creek, WI",
     "killed": 6, "injured": 4, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "Wade Michael Page",
     "notes": "White supremacist with Hammerskin Nation ties"},

    {"date": "2012-12-14", "name": "Sandy Hook school shooting",
     "location_country": "US", "location_city": "Newtown, CT",
     "killed": 26, "injured": 2, "ideology": "school_shooting",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Adam Lanza",
     "notes": "20 children + 6 staff killed; major gun control catalyst"},

    # ══════════════════════════════════════════════════════════════
    # 2012  —  NON-VIOLENCE DISCONTINUITIES
    # ══════════════════════════════════════════════════════════════
    {"date": "2012-10-29", "name": "Hurricane Sandy",
     "location_country": "US", "location_city": "US East Coast",
     "killed": 233, "injured": 0, "ideology": "N/A",
     "event_category": "natural_disaster", "online_nexus": False,
     "domestic": True, "perpetrator": "Hurricane",
     "notes": "Category 3 hurricane; $70B damage across Northeast US"},

    # ══════════════════════════════════════════════════════════════
    # 2013  —  MASS VIOLENCE
    # ══════════════════════════════════════════════════════════════
    {"date": "2013-04-15", "name": "Boston Marathon bombing",
     "location_country": "US", "location_city": "Boston, MA",
     "killed": 3, "injured": 264, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Dzhokhar & Tamerlan Tsarnaev",
     "notes": "Pressure cooker bombs; inspired by AQAP's Inspire magazine"},

    {"date": "2013-05-22", "name": "Woolwich attack (Lee Rigby)",
     "location_country": "UK", "location_city": "London",
     "killed": 1, "injured": 0, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Michael Adebolajo & Michael Adebowale",
     "notes": "British soldier hacked to death; attackers remained at scene"},

    {"date": "2013-06-07", "name": "Santa Monica shooting",
     "location_country": "US", "location_city": "Santa Monica, CA",
     "killed": 5, "injured": 4, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "John Zawahri",
     "notes": "Shooting spree across multiple locations including college campus"},

    {"date": "2013-09-16", "name": "Washington Navy Yard shooting",
     "location_country": "US", "location_city": "Washington, DC",
     "killed": 12, "injured": 3, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Aaron Alexis",
     "notes": "Military installation shooting; perpetrator had mental health issues"},

    {"date": "2013-11-01", "name": "LAX shooting",
     "location_country": "US", "location_city": "Los Angeles, CA",
     "killed": 1, "injured": 7, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Paul Ciancia",
     "notes": "Targeted TSA agents; anti-government sentiments; NWO references"},

    # ══════════════════════════════════════════════════════════════
    # 2014  —  MASS VIOLENCE
    # ══════════════════════════════════════════════════════════════
    {"date": "2014-04-02", "name": "Fort Hood shooting (2014)",
     "location_country": "US", "location_city": "Fort Hood, TX",
     "killed": 3, "injured": 16, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Ivan Lopez",
     "notes": "Second Fort Hood shooting; personal grievances + PTSD"},

    {"date": "2014-04-13", "name": "Overland Park Jewish center shooting",
     "location_country": "US", "location_city": "Overland Park, KS",
     "killed": 3, "injured": 0, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Frazier Glenn Miller Jr.",
     "notes": "Former KKK leader targeted Jewish community centers"},

    {"date": "2014-05-23", "name": "Isla Vista attack",
     "location_country": "US", "location_city": "Isla Vista, CA",
     "killed": 6, "injured": 14, "ideology": "incel",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "Elliot Rodger",
     "notes": "Misogynist manifesto; foundational incel attack"},

    {"date": "2014-05-24", "name": "Jewish Museum of Belgium shooting",
     "location_country": "Belgium", "location_city": "Brussels",
     "killed": 4, "injured": 0, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Mehdi Nemmouche",
     "notes": "Former IS fighter; first returning foreign fighter attack in Europe"},

    {"date": "2014-06-04", "name": "Moncton shootings",
     "location_country": "Canada", "location_city": "Moncton, NB",
     "killed": 3, "injured": 2, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": False, "perpetrator": "Justin Bourque",
     "notes": "Anti-police; posted gun photos and anti-government content online"},

    {"date": "2014-10-22", "name": "Ottawa Parliament Hill shooting",
     "location_country": "Canada", "location_city": "Ottawa, ON",
     "killed": 1, "injured": 3, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Michael Zehaf-Bibeau",
     "notes": "Shot soldier at National War Memorial; stormed Parliament; killed by Sergeant-at-Arms"},

    {"date": "2014-12-15", "name": "Lindt Café siege (Sydney)",
     "location_country": "Australia", "location_city": "Sydney",
     "killed": 3, "injured": 4, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Man Haron Monis",
     "notes": "Hostage situation; claimed IS affiliation"},

    # ══════════════════════════════════════════════════════════════
    # 2014  —  NON-VIOLENCE DISCONTINUITIES
    # ══════════════════════════════════════════════════════════════
    {"date": "2014-03-18", "name": "Russian annexation of Crimea",
     "location_country": "Ukraine", "location_city": "Crimea",
     "killed": 0, "injured": 0, "ideology": "N/A",
     "event_category": "political", "online_nexus": True,
     "domestic": False, "perpetrator": "Russian Federation",
     "notes": "Annexation following referendum; major /pol/ geopolitical event"},

    # ══════════════════════════════════════════════════════════════
    # 2015  —  MASS VIOLENCE
    # ══════════════════════════════════════════════════════════════
    {"date": "2015-01-07", "name": "Charlie Hebdo attack",
     "location_country": "France", "location_city": "Paris",
     "killed": 12, "injured": 11, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Saïd & Chérif Kouachi",
     "notes": "AQAP-directed attack on satirical newspaper"},

    {"date": "2015-01-09", "name": "Hypercacher kosher market siege",
     "location_country": "France", "location_city": "Paris",
     "killed": 4, "injured": 9, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Amedy Coulibaly",
     "notes": "Coordinated with Charlie Hebdo attackers; antisemitic targeting"},

    {"date": "2015-02-14", "name": "Copenhagen shootings",
     "location_country": "Denmark", "location_city": "Copenhagen",
     "killed": 2, "injured": 5, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Omar El-Hussein",
     "notes": "Targeted free speech event and synagogue; inspired by Charlie Hebdo"},

    {"date": "2015-05-03", "name": "Curtis Culwell Center attack (Garland)",
     "location_country": "US", "location_city": "Garland, TX",
     "killed": 0, "injured": 1, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "Elton Simpson & Nadir Soofi",
     "notes": "Attacked Muhammad cartoon contest; both attackers killed by security"},

    {"date": "2015-06-17", "name": "Charleston church shooting",
     "location_country": "US", "location_city": "Charleston, SC",
     "killed": 9, "injured": 1, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "Dylann Roof",
     "notes": "Manifesto posted online; later celebrated on /pol/"},

    {"date": "2015-07-16", "name": "Chattanooga recruiting center shooting",
     "location_country": "US", "location_city": "Chattanooga, TN",
     "killed": 5, "injured": 2, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Mohammad Youssuf Abdulazeez",
     "notes": "Attack on military recruiting centers; IS-inspired"},

    {"date": "2015-07-23", "name": "Lafayette Grand Theatre shooting",
     "location_country": "US", "location_city": "Lafayette, LA",
     "killed": 2, "injured": 9, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "John Russell Houser",
     "notes": "Far-right extremist; online history of anti-government posts"},

    {"date": "2015-08-21", "name": "Thalys train attack",
     "location_country": "France", "location_city": "Oignies (Thalys train)",
     "killed": 0, "injured": 4, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Ayoub El Khazzani",
     "notes": "Foiled by passengers (incl. US military); unsuccessful mass casualty attempt"},

    {"date": "2015-10-01", "name": "Umpqua Community College shooting",
     "location_country": "US", "location_city": "Roseburg, OR",
     "killed": 9, "injured": 9, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "Chris Harper-Mercer",
     "notes": "Posted on 4chan before attack; mixed ideology"},

    {"date": "2015-10-10", "name": "Ankara peace rally bombings",
     "location_country": "Turkey", "location_city": "Ankara",
     "killed": 103, "injured": 400, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "IS-attributed",
     "notes": "Deadliest terrorist attack in modern Turkish history; targeted peace rally"},

    {"date": "2015-11-13", "name": "Paris attacks (Bataclan)",
     "location_country": "France", "location_city": "Paris",
     "killed": 130, "injured": 416, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "IS cell",
     "notes": "Coordinated suicide bombings and shootings; deadliest attack in France since WWII"},

    {"date": "2015-11-27", "name": "Colorado Springs Planned Parenthood shooting",
     "location_country": "US", "location_city": "Colorado Springs, CO",
     "killed": 3, "injured": 9, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Robert Lewis Dear Jr.",
     "notes": "Anti-abortion extremist; referenced 'baby parts'"},

    {"date": "2015-12-02", "name": "San Bernardino attack",
     "location_country": "US", "location_city": "San Bernardino, CA",
     "killed": 14, "injured": 22, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Syed Rizwan Farook & Tashfeen Malik",
     "notes": "IS-inspired workplace shooting"},

    # ══════════════════════════════════════════════════════════════
    # 2015  —  NON-VIOLENCE DISCONTINUITIES
    # ══════════════════════════════════════════════════════════════
    {"date": "2015-06-16", "name": "Trump announces presidential run",
     "location_country": "US", "location_city": "New York, NY",
     "killed": 0, "injured": 0, "ideology": "N/A",
     "event_category": "political", "online_nexus": True,
     "domestic": True, "perpetrator": "Political announcement",
     "notes": "Major catalyst for online far-right mobilization; massive /pol/ event"},

    {"date": "2015-08-24", "name": "Black Monday 2015 stock market crash",
     "location_country": "US", "location_city": "New York, NY",
     "killed": 0, "injured": 0, "ideology": "N/A",
     "event_category": "economic_shock", "online_nexus": False,
     "domestic": True, "perpetrator": "Chinese market contagion",
     "notes": "DJIA dropped 1,000 pts at open; global selloff triggered by Chinese economy fears"},

    # ══════════════════════════════════════════════════════════════
    # 2016  —  MASS VIOLENCE
    # ══════════════════════════════════════════════════════════════
    {"date": "2016-03-22", "name": "Brussels bombings",
     "location_country": "Belgium", "location_city": "Brussels",
     "killed": 32, "injured": 340, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "IS cell",
     "notes": "Airport and metro station suicide bombings"},

    {"date": "2016-06-12", "name": "Orlando Pulse nightclub shooting",
     "location_country": "US", "location_city": "Orlando, FL",
     "killed": 49, "injured": 53, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Omar Mateen",
     "notes": "Pledged allegiance to IS during attack; targeted LGBTQ venue"},

    {"date": "2016-06-16", "name": "Jo Cox assassination",
     "location_country": "UK", "location_city": "Birstall, West Yorkshire",
     "killed": 1, "injured": 0, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": False, "perpetrator": "Thomas Mair",
     "notes": "Murder of Labour MP; shouted 'Britain first'; purchased manuals from National Alliance"},

    {"date": "2016-06-28", "name": "Istanbul Atatürk airport attack",
     "location_country": "Turkey", "location_city": "Istanbul",
     "killed": 45, "injured": 230, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "IS-directed cell",
     "notes": "Suicide bombing and gunfire at international airport"},

    {"date": "2016-07-07", "name": "Dallas police shooting",
     "location_country": "US", "location_city": "Dallas, TX",
     "killed": 5, "injured": 11, "ideology": "far_left",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Micah Xavier Johnson",
     "notes": "Targeted police during BLM protest; anti-white/anti-police statements"},

    {"date": "2016-07-14", "name": "Nice truck attack",
     "location_country": "France", "location_city": "Nice",
     "killed": 86, "injured": 434, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Mohamed Lahouaiej-Bouhlel",
     "notes": "Vehicle-ramming on Bastille Day promenade"},

    {"date": "2016-07-18", "name": "Würzburg train attack",
     "location_country": "Germany", "location_city": "Würzburg",
     "killed": 0, "injured": 5, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Riaz Khan Ahmadzai",
     "notes": "Axe and knife attack on train; IS-claimed; unsuccessful mass casualty"},

    {"date": "2016-07-22", "name": "Munich shooting",
     "location_country": "Germany", "location_city": "Munich",
     "killed": 9, "injured": 36, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": False, "perpetrator": "David Ali Sonboly",
     "notes": "Targeted youths at shopping center; obsessed with mass shootings; far-right links"},

    {"date": "2016-07-24", "name": "Ansbach bombing",
     "location_country": "Germany", "location_city": "Ansbach",
     "killed": 0, "injured": 15, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Mohammad Daleel",
     "notes": "Suicide bombing near music festival; IS pledge; only attacker killed"},

    {"date": "2016-07-26", "name": "Sagamihara stabbing",
     "location_country": "Japan", "location_city": "Sagamihara",
     "killed": 19, "injured": 26, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Satoshi Uematsu",
     "notes": "Attacked disabled care facility; ableist manifesto; deadliest mass killing in Japan since WWII"},

    {"date": "2016-12-19", "name": "Berlin Christmas market attack",
     "location_country": "Germany", "location_city": "Berlin",
     "killed": 12, "injured": 56, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Anis Amri",
     "notes": "Truck-ramming; IS-claimed"},

    # ══════════════════════════════════════════════════════════════
    # 2016  —  NON-VIOLENCE DISCONTINUITIES
    # ══════════════════════════════════════════════════════════════
    {"date": "2016-06-23", "name": "Brexit referendum",
     "location_country": "UK", "location_city": "Nationwide",
     "killed": 0, "injured": 0, "ideology": "N/A",
     "event_category": "political", "online_nexus": True,
     "domestic": False, "perpetrator": "Referendum vote",
     "notes": "UK voted to leave EU (51.9%); major /pol/ celebration; global market shock"},

    {"date": "2016-11-08", "name": "Trump wins US presidential election",
     "location_country": "US", "location_city": "Nationwide",
     "killed": 0, "injured": 0, "ideology": "N/A",
     "event_category": "political", "online_nexus": True,
     "domestic": True, "perpetrator": "Election result",
     "notes": "Major online far-right mobilization event; enormous /pol/ engagement"},

    # ══════════════════════════════════════════════════════════════
    # 2017  —  MASS VIOLENCE
    # ══════════════════════════════════════════════════════════════
    {"date": "2017-01-01", "name": "Istanbul Reina nightclub attack",
     "location_country": "Turkey", "location_city": "Istanbul",
     "killed": 39, "injured": 71, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Abdulkadir Masharipov",
     "notes": "New Year's Eve attack; IS-claimed"},

    {"date": "2017-01-06", "name": "Fort Lauderdale airport shooting",
     "location_country": "US", "location_city": "Fort Lauderdale, FL",
     "killed": 5, "injured": 6, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Esteban Santiago",
     "notes": "Shot travelers at baggage claim; mental health issues"},

    {"date": "2017-01-29", "name": "Quebec City mosque shooting",
     "location_country": "Canada", "location_city": "Quebec City",
     "killed": 6, "injured": 19, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": False, "perpetrator": "Alexandre Bissonnette",
     "notes": "Perpetrator frequented far-right social media; anti-Muslim motive"},

    {"date": "2017-03-22", "name": "London Westminster Bridge attack",
     "location_country": "UK", "location_city": "London",
     "killed": 5, "injured": 50, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Khalid Masood",
     "notes": "Vehicle-ramming on Westminster Bridge + stabbing of PC Keith Palmer"},

    {"date": "2017-04-07", "name": "Stockholm truck attack",
     "location_country": "Sweden", "location_city": "Stockholm",
     "killed": 5, "injured": 14, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Rakhmat Akilov",
     "notes": "Hijacked beer truck driven into pedestrians on Drottninggatan"},

    {"date": "2017-04-20", "name": "Paris Champs-Élysées shooting",
     "location_country": "France", "location_city": "Paris",
     "killed": 1, "injured": 3, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Karim Cheurfi",
     "notes": "Shot police officer on Champs-Élysées; IS-claimed"},

    {"date": "2017-05-22", "name": "Manchester Arena bombing",
     "location_country": "UK", "location_city": "Manchester",
     "killed": 22, "injured": 139, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Salman Abedi",
     "notes": "Suicide bombing at Ariana Grande concert; many child victims"},

    {"date": "2017-06-03", "name": "London Bridge attack (2017)",
     "location_country": "UK", "location_city": "London",
     "killed": 8, "injured": 48, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Khuram Butt, Rachid Redouane, Youssef Zaghba",
     "notes": "Vehicle-ramming + stabbing; IS-claimed"},

    {"date": "2017-06-14", "name": "Congressional baseball shooting",
     "location_country": "US", "location_city": "Alexandria, VA",
     "killed": 0, "injured": 6, "ideology": "far_left",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "James Hodgkinson",
     "notes": "Targeted Republican lawmakers at baseball practice"},

    {"date": "2017-06-19", "name": "Finsbury Park mosque attack",
     "location_country": "UK", "location_city": "London",
     "killed": 1, "injured": 10, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Darren Osborne",
     "notes": "Van-ramming targeting Muslims leaving mosque; radicalized online"},

    {"date": "2017-08-12", "name": "Charlottesville car attack",
     "location_country": "US", "location_city": "Charlottesville, VA",
     "killed": 1, "injured": 35, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "James Alex Fields Jr.",
     "notes": "Unite the Right rally; major online mobilization"},

    {"date": "2017-08-17", "name": "Barcelona Las Ramblas attack",
     "location_country": "Spain", "location_city": "Barcelona / Cambrils",
     "killed": 16, "injured": 152, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "IS-inspired cell",
     "notes": "Van-ramming on Las Ramblas + related attack in Cambrils"},

    {"date": "2017-09-15", "name": "Parsons Green bombing",
     "location_country": "UK", "location_city": "London",
     "killed": 0, "injured": 30, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Ahmed Hassan",
     "notes": "IED partially detonated on Tube; unsuccessful mass casualty attempt"},

    {"date": "2017-10-01", "name": "Las Vegas shooting",
     "location_country": "US", "location_city": "Las Vegas, NV",
     "killed": 60, "injured": 411, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Stephen Paddock",
     "notes": "Deadliest US mass shooting; fired from hotel into concert crowd; unclear motive"},

    {"date": "2017-10-31", "name": "NYC truck attack",
     "location_country": "US", "location_city": "New York, NY",
     "killed": 8, "injured": 11, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Sayfullo Saipov",
     "notes": "Vehicle-ramming on Manhattan bike path; IS-inspired"},

    {"date": "2017-11-05", "name": "Sutherland Springs church shooting",
     "location_country": "US", "location_city": "Sutherland Springs, TX",
     "killed": 26, "injured": 22, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Devin Patrick Kelley",
     "notes": "Shot churchgoers during Sunday service; domestic violence history"},

    # ══════════════════════════════════════════════════════════════
    # 2017  —  NON-VIOLENCE DISCONTINUITIES
    # ══════════════════════════════════════════════════════════════
    {"date": "2017-08-25", "name": "Hurricane Harvey",
     "location_country": "US", "location_city": "Houston, TX",
     "killed": 107, "injured": 0, "ideology": "N/A",
     "event_category": "natural_disaster", "online_nexus": False,
     "domestic": True, "perpetrator": "Category 4 hurricane",
     "notes": "Record-breaking rainfall and flooding in Texas; $125B damage"},

    {"date": "2017-09-20", "name": "Hurricane Maria",
     "location_country": "US", "location_city": "Puerto Rico",
     "killed": 2975, "injured": 0, "ideology": "N/A",
     "event_category": "natural_disaster", "online_nexus": False,
     "domestic": True, "perpetrator": "Category 5 hurricane",
     "notes": "Devastated Puerto Rico; estimated 2,975 excess deaths; prolonged power outage"},

    # ══════════════════════════════════════════════════════════════
    # 2018  —  MASS VIOLENCE
    # ══════════════════════════════════════════════════════════════
    {"date": "2018-02-14", "name": "Parkland school shooting",
     "location_country": "US", "location_city": "Parkland, FL",
     "killed": 17, "injured": 17, "ideology": "school_shooting",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "Nikolas Cruz",
     "notes": "Marjory Stoneman Douglas HS; YouTube comments about becoming school shooter"},

    {"date": "2018-03-23", "name": "Trèbes / Carcassonne attack",
     "location_country": "France", "location_city": "Trèbes / Carcassonne",
     "killed": 4, "injured": 15, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Radouane Lakdim",
     "notes": "Carjacking, shooting, hostage-taking; IS-pledge; officer Arnaud Beltrame died exchanging self for hostage"},

    {"date": "2018-04-23", "name": "Toronto van attack",
     "location_country": "Canada", "location_city": "Toronto",
     "killed": 10, "injured": 16, "ideology": "incel",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": False, "perpetrator": "Alek Minassian",
     "notes": "Posted on Facebook referencing Elliot Rodger; incel ideology"},

    {"date": "2018-05-18", "name": "Santa Fe High School shooting",
     "location_country": "US", "location_city": "Santa Fe, TX",
     "killed": 10, "injured": 13, "ideology": "school_shooting",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Dimitrios Pagourtzis",
     "notes": "School shooting; limited ideological signal"},

    {"date": "2018-06-28", "name": "Annapolis Capital Gazette shooting",
     "location_country": "US", "location_city": "Annapolis, MD",
     "killed": 5, "injured": 2, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Jarrod Ramos",
     "notes": "Targeted newsroom due to personal grudge from defamation case"},

    {"date": "2018-10-27", "name": "Pittsburgh synagogue shooting",
     "location_country": "US", "location_city": "Pittsburgh, PA",
     "killed": 11, "injured": 6, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "Robert Bowers",
     "notes": "Posted on Gab before attack; antisemitic motive; deadliest attack on Jews in US history"},

    {"date": "2018-11-07", "name": "Thousand Oaks Borderline Bar shooting",
     "location_country": "US", "location_city": "Thousand Oaks, CA",
     "killed": 12, "injured": 16, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Ian David Long",
     "notes": "Attacked country music bar; Marine veteran with PTSD"},

    {"date": "2018-12-11", "name": "Strasbourg Christmas market attack",
     "location_country": "France", "location_city": "Strasbourg",
     "killed": 5, "injured": 11, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Chérif Chekatt",
     "notes": "Shooting and stabbing at Christmas market; IS-claimed"},

    # ══════════════════════════════════════════════════════════════
    # 2018  —  NON-VIOLENCE DISCONTINUITIES
    # ══════════════════════════════════════════════════════════════
    {"date": "2018-11-08", "name": "Camp Fire (Paradise, CA)",
     "location_country": "US", "location_city": "Paradise, CA",
     "killed": 85, "injured": 12, "ideology": "N/A",
     "event_category": "natural_disaster", "online_nexus": False,
     "domestic": True, "perpetrator": "Wildfire",
     "notes": "Deadliest and most destructive wildfire in California history; town destroyed"},

    # ══════════════════════════════════════════════════════════════
    # 2019  —  MASS VIOLENCE
    # ══════════════════════════════════════════════════════════════
    {"date": "2019-03-15", "name": "Christchurch mosque shootings",
     "location_country": "New Zealand", "location_city": "Christchurch",
     "killed": 51, "injured": 40, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": False, "perpetrator": "Brenton Tarrant",
     "notes": "Livestreamed on Facebook; manifesto posted on 8chan; watershed online extremism event"},

    {"date": "2019-03-18", "name": "Utrecht tram shooting",
     "location_country": "Netherlands", "location_city": "Utrecht",
     "killed": 4, "injured": 3, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Gökmen Tanış",
     "notes": "Shot passengers on tram; mixed personal/extremist motive"},

    {"date": "2019-04-21", "name": "Sri Lanka Easter bombings",
     "location_country": "Sri Lanka", "location_city": "Colombo & Batticaloa",
     "killed": 269, "injured": 500, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "National Thowheed Jamath / IS",
     "notes": "Coordinated bombings at churches and hotels; included for /pol/ relevance despite non-OECD"},

    {"date": "2019-04-27", "name": "Poway synagogue shooting",
     "location_country": "US", "location_city": "Poway, CA",
     "killed": 1, "injured": 3, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "John T. Earnest",
     "notes": "Posted manifesto on 8chan citing Tarrant as inspiration"},

    {"date": "2019-05-07", "name": "STEM School Highlands Ranch shooting",
     "location_country": "US", "location_city": "Highlands Ranch, CO",
     "killed": 1, "injured": 8, "ideology": "school_shooting",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Devon Erickson, Alec McKinney",
     "notes": "School shooting; two perpetrators"},

    {"date": "2019-05-31", "name": "Virginia Beach municipal building shooting",
     "location_country": "US", "location_city": "Virginia Beach, VA",
     "killed": 12, "injured": 4, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "DeWayne Craddock",
     "notes": "Workplace shooting at municipal building; resigned hours before attack"},

    {"date": "2019-08-03", "name": "El Paso Walmart shooting",
     "location_country": "US", "location_city": "El Paso, TX",
     "killed": 23, "injured": 23, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "Patrick Crusius",
     "notes": "Anti-Hispanic manifesto posted on 8chan; Great Replacement references"},

    {"date": "2019-08-04", "name": "Dayton shooting",
     "location_country": "US", "location_city": "Dayton, OH",
     "killed": 9, "injured": 17, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Connor Betts",
     "notes": "Left-leaning social media; motive unclear; occurred day after El Paso"},

    {"date": "2019-08-10", "name": "Bærum mosque shooting",
     "location_country": "Norway", "location_city": "Bærum",
     "killed": 0, "injured": 1, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": False, "perpetrator": "Philip Manshaus",
     "notes": "Inspired by Christchurch; posted on Endchan before attack; subdued by worshipper; unsuccessful"},

    {"date": "2019-10-09", "name": "Halle synagogue shooting",
     "location_country": "Germany", "location_city": "Halle",
     "killed": 2, "injured": 2, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": False, "perpetrator": "Stephan Balliet",
     "notes": "Livestreamed on Twitch; cited Tarrant; failed to enter synagogue; killed 2 nearby"},

    {"date": "2019-11-29", "name": "London Bridge attack (2019)",
     "location_country": "UK", "location_city": "London",
     "killed": 2, "injured": 3, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Usman Khan",
     "notes": "Convicted terrorist on license; attacked at prisoner rehabilitation conference"},

    {"date": "2019-12-10", "name": "Jersey City kosher market shooting",
     "location_country": "US", "location_city": "Jersey City, NJ",
     "killed": 4, "injured": 3, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "David Anderson & Francine Graham",
     "notes": "Black Hebrew Israelite ideology; antisemitic targeting of kosher market"},

    # ══════════════════════════════════════════════════════════════
    # 2020  —  MASS VIOLENCE
    # ══════════════════════════════════════════════════════════════
    {"date": "2020-02-19", "name": "Hanau shootings",
     "location_country": "Germany", "location_city": "Hanau",
     "killed": 10, "injured": 5, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": False, "perpetrator": "Tobias Rathjen",
     "notes": "Published manifesto and YouTube videos; targeted shisha bars"},

    {"date": "2020-04-18", "name": "Nova Scotia rampage",
     "location_country": "Canada", "location_city": "Portapique, NS",
     "killed": 22, "injured": 3, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Gabriel Wortman",
     "notes": "Impersonated RCMP officer; arson and shooting over 13 hours; deadliest mass killing in Canadian history"},

    {"date": "2020-06-20", "name": "Reading stabbings",
     "location_country": "UK", "location_city": "Reading",
     "killed": 3, "injured": 3, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Khairi Saadallah",
     "notes": "Knife attack in Forbury Gardens; targeted random people"},

    {"date": "2020-10-16", "name": "Samuel Paty beheading",
     "location_country": "France", "location_city": "Conflans-Sainte-Honorine",
     "killed": 1, "injured": 0, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": False, "perpetrator": "Abdoullakh Anzorov",
     "notes": "Teacher beheaded for showing Muhammad cartoons; located via social media campaign"},

    {"date": "2020-10-29", "name": "Nice basilica stabbing",
     "location_country": "France", "location_city": "Nice",
     "killed": 3, "injured": 0, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Brahim Aouissaoui",
     "notes": "Knife attack in Notre-Dame basilica; beheaded one victim"},

    {"date": "2020-11-02", "name": "Vienna shooting",
     "location_country": "Austria", "location_city": "Vienna",
     "killed": 4, "injured": 23, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": False, "perpetrator": "Kujtim Fejzulai",
     "notes": "Multiple locations near synagogue; IS-inspired; had tried to travel to Syria"},

    # ══════════════════════════════════════════════════════════════
    # 2020  —  NON-VIOLENCE DISCONTINUITIES
    # ══════════════════════════════════════════════════════════════
    {"date": "2020-01-30", "name": "WHO declares COVID-19 PHEIC",
     "location_country": "Switzerland", "location_city": "Geneva",
     "killed": 0, "injured": 0, "ideology": "N/A",
     "event_category": "health_crisis", "online_nexus": True,
     "domestic": False, "perpetrator": "SARS-CoV-2 pandemic",
     "notes": "Public Health Emergency of International Concern; massive conspiracy discourse on /pol/"},

    {"date": "2020-03-11", "name": "WHO declares COVID-19 pandemic",
     "location_country": "Switzerland", "location_city": "Geneva",
     "killed": 0, "injured": 0, "ideology": "N/A",
     "event_category": "health_crisis", "online_nexus": True,
     "domestic": False, "perpetrator": "SARS-CoV-2 pandemic",
     "notes": "Global pandemic declaration; mass lockdowns begin; immense /pol/ apocalyptic discourse"},

    {"date": "2020-03-16", "name": "COVID-19 stock market crash (Black Monday)",
     "location_country": "US", "location_city": "New York, NY",
     "killed": 0, "injured": 0, "ideology": "N/A",
     "event_category": "economic_shock", "online_nexus": True,
     "domestic": True, "perpetrator": "Pandemic economic shock",
     "notes": "DJIA fell 2,997 pts (12.9%); worst single-day fall since 1987"},

    {"date": "2020-05-25", "name": "George Floyd killing / BLM protests",
     "location_country": "US", "location_city": "Minneapolis, MN",
     "killed": 1, "injured": 0, "ideology": "N/A",
     "event_category": "political", "online_nexus": True,
     "domestic": True, "perpetrator": "Police killing / mass protests",
     "notes": "Global protest movement; enormous /pol/ engagement on race/civil unrest"},

    # ══════════════════════════════════════════════════════════════
    # 2021  —  MASS VIOLENCE
    # ══════════════════════════════════════════════════════════════
    {"date": "2021-01-06", "name": "U.S. Capitol storming",
     "location_country": "US", "location_city": "Washington, DC",
     "killed": 5, "injured": 138, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "Multiple / crowd",
     "notes": "Insurrection; extensive /pol/ and social media coordination"},

    {"date": "2021-03-16", "name": "Atlanta spa shootings",
     "location_country": "US", "location_city": "Atlanta, GA",
     "killed": 8, "injured": 1, "ideology": "incel",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Robert Aaron Long",
     "notes": "Targeted Asian women; sexual addiction / misogynist framing"},

    {"date": "2021-03-22", "name": "Boulder King Soopers shooting",
     "location_country": "US", "location_city": "Boulder, CO",
     "killed": 10, "injured": 0, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Ahmad Al Aliwi Alissa",
     "notes": "Supermarket shooting; motive unclear"},

    {"date": "2021-05-26", "name": "San Jose VTA shooting",
     "location_country": "US", "location_city": "San Jose, CA",
     "killed": 9, "injured": 2, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Samuel Cassidy",
     "notes": "Workplace shooting at transit authority; planted bombs at home"},

    {"date": "2021-06-25", "name": "Würzburg stabbing attack",
     "location_country": "Germany", "location_city": "Würzburg",
     "killed": 3, "injured": 5, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Abdirahman Jibril A.",
     "notes": "Knife attack in city center; targeted women"},

    {"date": "2021-08-12", "name": "Plymouth shooting",
     "location_country": "UK", "location_city": "Plymouth",
     "killed": 5, "injured": 2, "ideology": "incel",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": False, "perpetrator": "Jake Davison",
     "notes": "Self-identified incel; posted on Reddit and YouTube about blackpill ideology"},

    {"date": "2021-08-26", "name": "Kabul airport bombing",
     "location_country": "Afghanistan", "location_city": "Kabul",
     "killed": 183, "injured": 150, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "IS-Khorasan",
     "notes": "13 US military killed during withdrawal; massive /pol/ reaction; included for relevance despite non-OECD"},

    {"date": "2021-10-13", "name": "Kongsberg attacks",
     "location_country": "Norway", "location_city": "Kongsberg",
     "killed": 5, "injured": 3, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Espen Andersen Bråthen",
     "notes": "Bow-and-arrow + knife attack; recent convert to Islam; mental illness"},

    {"date": "2021-10-15", "name": "David Amess assassination",
     "location_country": "UK", "location_city": "Leigh-on-Sea",
     "killed": 1, "injured": 0, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Ali Harbi Ali",
     "notes": "Conservative MP stabbed at constituency surgery; IS-motivated"},

    {"date": "2021-11-21", "name": "Waukesha parade attack",
     "location_country": "US", "location_city": "Waukesha, WI",
     "killed": 6, "injured": 62, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Darrell Brooks Jr.",
     "notes": "Drove SUV through Christmas parade; fleeing domestic incident; enormous /pol/ activity"},

    # ══════════════════════════════════════════════════════════════
    # 2021  —  NON-VIOLENCE DISCONTINUITIES
    # ══════════════════════════════════════════════════════════════
    {"date": "2021-01-27", "name": "GameStop / WallStreetBets short squeeze",
     "location_country": "US", "location_city": "New York, NY",
     "killed": 0, "injured": 0, "ideology": "N/A",
     "event_category": "economic_shock", "online_nexus": True,
     "domestic": True, "perpetrator": "Reddit retail investors vs hedge funds",
     "notes": "GME stock surged >2,500%; populist anti-establishment financial event"},

    {"date": "2021-06-24", "name": "Surfside condominium collapse",
     "location_country": "US", "location_city": "Surfside, FL",
     "killed": 98, "injured": 11, "ideology": "N/A",
     "event_category": "natural_disaster", "online_nexus": False,
     "domestic": True, "perpetrator": "Structural collapse",
     "notes": "Champlain Towers South partial collapse; prolonged search and recovery"},

    {"date": "2021-08-15", "name": "Fall of Kabul",
     "location_country": "Afghanistan", "location_city": "Kabul",
     "killed": 0, "injured": 0, "ideology": "N/A",
     "event_category": "political", "online_nexus": True,
     "domestic": False, "perpetrator": "Taliban takeover / US withdrawal",
     "notes": "Taliban captured Kabul; chaotic US evacuation; enormous /pol/ engagement"},

    # ══════════════════════════════════════════════════════════════
    # 2022  —  MASS VIOLENCE
    # ══════════════════════════════════════════════════════════════
    {"date": "2022-05-14", "name": "Buffalo supermarket shooting",
     "location_country": "US", "location_city": "Buffalo, NY",
     "killed": 10, "injured": 3, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "Payton Gendron",
     "notes": "Manifesto posted on Discord/4chan; cited Great Replacement; livestreamed on Twitch"},

    {"date": "2022-05-24", "name": "Uvalde school shooting",
     "location_country": "US", "location_city": "Uvalde, TX",
     "killed": 21, "injured": 17, "ideology": "school_shooting",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "Salvador Ramos",
     "notes": "Robb Elementary; 19 children + 2 teachers killed; social media posts before attack"},

    {"date": "2022-07-03", "name": "Copenhagen Fields mall shooting",
     "location_country": "Denmark", "location_city": "Copenhagen",
     "killed": 3, "injured": 7, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": False, "perpetrator": "Noah Esbensen",
     "notes": "Shooting at Field's shopping center; perpetrator posted YouTube videos before attack"},

    {"date": "2022-07-04", "name": "Highland Park parade shooting",
     "location_country": "US", "location_city": "Highland Park, IL",
     "killed": 7, "injured": 48, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "Robert Crimo III",
     "notes": "Shot from rooftop at Fourth of July parade; online presence with violent imagery"},

    {"date": "2022-10-12", "name": "Bratislava Tepláreň shooting",
     "location_country": "Slovakia", "location_city": "Bratislava",
     "killed": 2, "injured": 1, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": False, "perpetrator": "Juraj Krajčík",
     "notes": "Targeted LGBTQ bar; manifesto posted online; antisemitic and anti-LGBTQ content"},

    {"date": "2022-11-19", "name": "Colorado Springs Club Q shooting",
     "location_country": "US", "location_city": "Colorado Springs, CO",
     "killed": 5, "injured": 17, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Anderson Lee Aldrich",
     "notes": "Targeted LGBTQ nightclub; subdued by patrons"},

    # ══════════════════════════════════════════════════════════════
    # 2022  —  NON-VIOLENCE DISCONTINUITIES
    # ══════════════════════════════════════════════════════════════
    {"date": "2022-02-24", "name": "Russia invades Ukraine",
     "location_country": "Ukraine", "location_city": "Nationwide",
     "killed": 0, "injured": 0, "ideology": "N/A",
     "event_category": "political", "online_nexus": True,
     "domestic": False, "perpetrator": "Russian Federation full-scale invasion",
     "notes": "Largest European land war since WWII; immense /pol/ engagement; apocalyptic discourse"},

    {"date": "2022-06-24", "name": "Dobbs v. Jackson (Roe overturned)",
     "location_country": "US", "location_city": "Washington, DC",
     "killed": 0, "injured": 0, "ideology": "N/A",
     "event_category": "political", "online_nexus": True,
     "domestic": True, "perpetrator": "Supreme Court decision",
     "notes": "Overturned Roe v. Wade; massive political polarization event"},

    # ══════════════════════════════════════════════════════════════
    # 2023  —  MASS VIOLENCE
    # ══════════════════════════════════════════════════════════════
    {"date": "2023-03-09", "name": "Hamburg Jehovah's Witness hall shooting",
     "location_country": "Germany", "location_city": "Hamburg",
     "killed": 7, "injured": 8, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Philipp F.",
     "notes": "Former JW member; personal grudge against congregation"},

    {"date": "2023-03-27", "name": "Nashville Covenant School shooting",
     "location_country": "US", "location_city": "Nashville, TN",
     "killed": 6, "injured": 1, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "Audrey Hale",
     "notes": "3 children + 3 staff killed; trans perpetrator generated enormous /pol/ controversy"},

    {"date": "2023-05-06", "name": "Allen Premium Outlets shooting",
     "location_country": "US", "location_city": "Allen, TX",
     "killed": 8, "injured": 7, "ideology": "far_right",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": True, "perpetrator": "Mauricio Garcia",
     "notes": "Neo-Nazi tattoos and online activity; RWDS patch"},

    {"date": "2023-10-07", "name": "Hamas attack on Israel",
     "location_country": "Israel", "location_city": "Southern Israel",
     "killed": 1139, "injured": 3400, "ideology": "islamist",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": False, "perpetrator": "Hamas",
     "notes": "Major mass-casualty attack; enormous /pol/ apocalyptic activity"},

    {"date": "2023-10-25", "name": "Lewiston shootings",
     "location_country": "US", "location_city": "Lewiston, ME",
     "killed": 18, "injured": 13, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": False,
     "domestic": True, "perpetrator": "Robert Card",
     "notes": "Shot patrons at bowling alley and bar; Army reservist with mental health issues"},

    {"date": "2023-12-21", "name": "Prague university shooting",
     "location_country": "Czech Republic", "location_city": "Prague",
     "killed": 14, "injured": 25, "ideology": "other",
     "event_category": "mass_violence", "online_nexus": True,
     "domestic": False, "perpetrator": "David Kozák",
     "notes": "Shot from university building; had studied other mass shooters; online manifesto"},

    # ══════════════════════════════════════════════════════════════
    # 2023  —  NON-VIOLENCE DISCONTINUITIES
    # ══════════════════════════════════════════════════════════════
    {"date": "2023-02-06", "name": "Turkey–Syria earthquake",
     "location_country": "Turkey", "location_city": "Kahramanmaraş",
     "killed": 50399, "injured": 107204, "ideology": "N/A",
     "event_category": "natural_disaster", "online_nexus": False,
     "domestic": False, "perpetrator": "M7.8 earthquake",
     "notes": "Deadliest natural disaster in Turkey's modern history; OECD member; apocalyptic imagery"},

    {"date": "2023-03-10", "name": "Silicon Valley Bank collapse",
     "location_country": "US", "location_city": "Santa Clara, CA",
     "killed": 0, "injured": 0, "ideology": "N/A",
     "event_category": "economic_shock", "online_nexus": True,
     "domestic": True, "perpetrator": "Bank run / interest rate exposure",
     "notes": "Second-largest bank failure in US history; contagion to Signature Bank and Credit Suisse"},

    {"date": "2023-08-08", "name": "Maui wildfires (Lahaina)",
     "location_country": "US", "location_city": "Lahaina, HI",
     "killed": 100, "injured": 0, "ideology": "N/A",
     "event_category": "natural_disaster", "online_nexus": False,
     "domestic": True, "perpetrator": "Wildfire",
     "notes": "Historic town of Lahaina destroyed; deadliest US wildfire in over a century"},
]


# ══════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════

def validate_events(events: list[dict]) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors: list[str] = []
    seen_dates: dict[str, list[str]] = {}

    for i, ev in enumerate(events):
        prefix = f"Event {i} ({ev.get('name', 'UNNAMED')})"

        # Required fields
        for field in ("date", "name", "location_country", "killed", "injured",
                      "ideology", "event_category", "online_nexus", "domestic"):
            if field not in ev:
                errors.append(f"{prefix}: missing required field '{field}'")

        # Date format
        try:
            date.fromisoformat(ev["date"])
        except (ValueError, KeyError):
            errors.append(f"{prefix}: invalid date format '{ev.get('date')}'")

        # Event category
        if ev.get("event_category") not in VALID_CATEGORIES:
            errors.append(f"{prefix}: invalid event_category '{ev.get('event_category')}'")

        # Ideology
        if ev.get("ideology") not in VALID_IDEOLOGIES:
            errors.append(f"{prefix}: invalid ideology '{ev.get('ideology')}'")

        # Non-violence events should have ideology N/A
        if (ev.get("event_category") != "mass_violence"
                and ev.get("ideology") != "N/A"):
            errors.append(
                f"{prefix}: non-violence event should have ideology 'N/A', "
                f"got '{ev.get('ideology')}'"
            )

        # Killed/injured non-negative
        if ev.get("killed", 0) < 0:
            errors.append(f"{prefix}: killed must be non-negative")
        if ev.get("injured", 0) < 0:
            errors.append(f"{prefix}: injured must be non-negative")

        # Track duplicates
        d = ev.get("date", "")
        seen_dates.setdefault(d, []).append(ev.get("name", ""))

    # Warn about same-day events (not necessarily an error, but flag)
    for d, names in seen_dates.items():
        if len(names) > 1:
            errors.append(f"Warning: multiple events on {d}: {names}")

    return errors


def events_to_polars(events: list[dict]) -> pl.DataFrame:
    """Convert event list to a typed Polars DataFrame."""
    return pl.DataFrame([
        {
            "event_date": date.fromisoformat(ev["date"]),
            "event_name": ev["name"],
            "event_category": ev["event_category"],
            "location_country": ev["location_country"],
            "location_city": ev.get("location_city", ""),
            "killed": ev["killed"],
            "injured": ev["injured"],
            "total_casualties": ev["killed"] + ev["injured"],
            "ideology": ev["ideology"],
            "online_nexus": ev["online_nexus"],
            "domestic": ev["domestic"],
            "perpetrator": ev.get("perpetrator", ""),
            "notes": ev.get("notes", ""),
        }
        for ev in events
    ]).sort("event_date")


def events_summary(df: pl.DataFrame) -> dict:
    """Compute summary statistics for the event catalogue."""
    summary = {
        "total_events": df.height,
        "date_range": [str(df["event_date"].min()), str(df["event_date"].max())],
        "by_category": df.group_by("event_category").len().sort("len", descending=True)
            .to_dicts(),
        "by_ideology": df.group_by("ideology").len().sort("len", descending=True)
            .to_dicts(),
        "by_country": df.group_by("location_country").len().sort("len", descending=True)
            .to_dicts(),
        "online_nexus_count": int(
            df.filter(pl.col("online_nexus"))["online_nexus"].sum()
        ),
        "mean_killed": float(df["killed"].mean()),
        "median_killed": float(df["killed"].median()),
        "total_killed": int(df["killed"].sum()),
    }

    # Category-specific
    violence = df.filter(pl.col("event_category") == "mass_violence")
    summary["mass_violence"] = {
        "count": violence.height,
        "by_ideology": violence.group_by("ideology").len()
            .sort("len", descending=True).to_dicts(),
        "mean_killed": float(violence["killed"].mean()) if violence.height else 0,
        "total_killed": int(violence["killed"].sum()),
    }

    nonviolence = df.filter(pl.col("event_category") != "mass_violence")
    summary["nonviolence"] = {
        "count": nonviolence.height,
        "by_category": nonviolence.group_by("event_category").len()
            .sort("len", descending=True).to_dicts(),
    }

    return summary


def main():
    print("=" * 60)
    print("STAGE 29: Mass-Casualty & Discontinuity Event Dataset")
    print("=" * 60)

    # ── Validate ──────────────────────────────────────────────────────
    errors = validate_events(EVENTS)
    warnings = [e for e in errors if e.startswith("Warning")]
    real_errors = [e for e in errors if not e.startswith("Warning")]

    for w in warnings:
        print(f"  ⚠ {w}")
    if real_errors:
        for e in real_errors:
            print(f"  ✗ {e}")
        raise ValueError(f"{len(real_errors)} validation errors in event dataset")

    print(f"  ✓ {len(EVENTS)} events validated ({len(warnings)} warnings)")

    # ── Convert to DataFrame ──────────────────────────────────────────
    df = events_to_polars(EVENTS)

    print(f"\n  Events by category:")
    for row in df.group_by("event_category").len().sort("len", descending=True).iter_rows():
        print(f"    {row[0]:20s} {row[1]:3d}")

    violence = df.filter(pl.col("event_category") == "mass_violence")
    print(f"\n  Mass violence events by ideology:")
    for row in violence.group_by("ideology").len().sort("len", descending=True).iter_rows():
        print(f"    {row[0]:20s} {row[1]:3d}")

    print(f"\n  Events by country (top 10):")
    for row in df.group_by("location_country").len().sort("len", descending=True).head(10).iter_rows():
        print(f"    {row[0]:20s} {row[1]:3d}")

    # ── Save ──────────────────────────────────────────────────────────
    # JSON (human-readable)
    out_json = DATA_PROCESSED / "mass_casualty_events.json"
    with open(out_json, "w") as f:
        json.dump(EVENTS, f, indent=2, default=str)
    print(f"\n  Saved JSON: {out_json.name}")

    # Parquet (analysis-ready)
    out_parquet = DATA_PROCESSED / "mass_casualty_events.parquet"
    df.write_parquet(out_parquet)
    print(f"  Saved Parquet: {out_parquet.name}")

    # Summary
    summary = events_summary(df)
    out_summary = RESULTS_DIR / "mass_casualty_events_summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved summary: {out_summary.name}")

    print(f"\n✓ Event dataset complete: {df.height} events, "
          f"{df['event_date'].min()} to {df['event_date'].max()}")
    print(f"  Mass violence: {violence.height}, "
          f"Non-violence: {df.height - violence.height}")


if __name__ == "__main__":
    main()
