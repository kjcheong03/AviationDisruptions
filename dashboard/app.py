"""
Week 4 Role C: Premium Streamlit dashboard — aviation disruption risk monitor.
Run from project root: streamlit run dashboard/app.py
"""

import json
import sys
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "eda" / "cache"
MODELS_DIR = ROOT / "models"

# bq_loader lives next to this file
sys.path.insert(0, str(Path(__file__).parent))
import bq_loader
from airport_coords import AIRPORT_COORDS

st.set_page_config(
    page_title="Aviation Delay Dashboard",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Color palette ─────────────────────────────────────────────────────────────
C = {
    "bg":     "#f1f5f9",
    "card":   "#ffffff",
    "border": "#e2e8f0",
    "blue":   "#0284c7",
    "green":  "#059669",
    "amber":  "#d97706",
    "red":    "#dc2626",
    "muted":  "#64748b",
    "text":   "#0f172a",
    "subtle": "#f8fafc",
}

# ── Lookup tables ─────────────────────────────────────────────────────────────
AIRLINE_NAMES = {
    "AA": "American Airlines", "AS": "Alaska Airlines", "B6": "JetBlue Airways",
    "DL": "Delta Air Lines", "F9": "Frontier Airlines", "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines", "MQ": "Envoy Air (AA regional)", "NK": "Spirit Airlines",
    "OH": "PSA Airlines (AA regional)", "OO": "SkyWest Airlines", "QX": "Horizon Air",
    "UA": "United Airlines", "VX": "Virgin America", "WN": "Southwest Airlines",
    "YV": "Mesa Airlines", "YX": "Republic Airways", "9E": "Endeavor Air (DL regional)",
    "EV": "ExpressJet Airlines", "FL": "AirTran Airways", "US": "US Airways",
    "VX": "Virgin America", "PT": "Piedmont Airlines", "CP": "Compass Airlines",
    "ZW": "Air Wisconsin", "3M": "Silver Airways", "C5": "CommutAir",
}

AIRPORT_NAMES = {
    "ATL": "Atlanta Hartsfield-Jackson", "LAX": "Los Angeles Intl", "ORD": "Chicago O'Hare",
    "DFW": "Dallas/Fort Worth Intl", "DEN": "Denver Intl", "JFK": "New York JFK",
    "SFO": "San Francisco Intl", "SEA": "Seattle-Tacoma Intl", "LAS": "Las Vegas Harry Reid",
    "MCO": "Orlando Intl", "EWR": "Newark Liberty", "CLT": "Charlotte Douglas",
    "PHX": "Phoenix Sky Harbor", "IAH": "Houston George Bush", "MIA": "Miami Intl",
    "BOS": "Boston Logan", "MSP": "Minneapolis-St Paul", "DTW": "Detroit Metro Wayne County",
    "FLL": "Fort Lauderdale-Hollywood", "PHL": "Philadelphia Intl", "LGA": "New York LaGuardia",
    "BWI": "Baltimore/Washington Intl", "MDW": "Chicago Midway", "SLC": "Salt Lake City Intl",
    "DCA": "Washington Reagan National", "IAD": "Washington Dulles", "SAN": "San Diego Intl",
    "TPA": "Tampa Intl", "HOU": "Houston Hobby", "PDX": "Portland Intl",
    "HTS": "Huntington Tri-State (WV)", "HGR": "Hagerstown Regional (MD)",
    "SCK": "Stockton Metropolitan (CA)", "USA": "Concord Regional (NC)",
    "BRW": "Utqiagvik Wiley Post-Will Rogers (AK)", "ASE": "Aspen/Pitkin County (CO)",
    "IAG": "Niagara Falls Intl (NY)", "SWF": "New York Stewart Intl",
    "BQN": "Rafael Hernández (Puerto Rico)", "SMX": "Santa Maria Public (CA)",
    "LCK": "Rickenbacker Intl (OH)", "TOL": "Toledo Express (OH)",
    "PBG": "Plattsburgh Intl (NY)", "BLV": "MidAmerica St Louis (IL)",
    "CKB": "North Central West Virginia (WV)", "RFD": "Chicago Rockford Intl",
    "HYA": "Barnstable Municipal (MA)", "SBN": "South Bend Intl",
    "ABI": "Abilene Regional", "ACT": "Waco Regional", "ACY": "Atlantic City Intl",
    "AGS": "Augusta Regional", "ALB": "Albany Intl", "AMA": "Rick Husband Amarillo Intl",
    "ANC": "Ted Stevens Anchorage Intl", "AVL": "Asheville Regional",
    "BDL": "Bradley Intl (Hartford)", "BFL": "Meadows Field (Bakersfield)",
    "BGR": "Bangor Intl", "BHM": "Birmingham-Shuttlesworth Intl",
    "BIL": "Billings Logan Intl", "BOI": "Boise Airport",
    "BTR": "Baton Rouge Metropolitan", "BUF": "Buffalo Niagara Intl",
    "BUR": "Hollywood Burbank", "CAE": "Columbia Metropolitan",
    "CHS": "Charleston Intl", "CID": "Eastern Iowa Airport",
    "CLE": "Cleveland Hopkins Intl", "CMH": "John Glenn Columbus Intl",
    "COS": "Colorado Springs Municipal", "CRP": "Corpus Christi Intl",
    "CVG": "Cincinnati/Northern Kentucky Intl", "DAL": "Dallas Love Field",
    "DAY": "Dayton Intl", "DSM": "Des Moines Intl", "ELP": "El Paso Intl",
    "EVV": "Evansville Regional", "EYW": "Key West Intl", "FAI": "Fairbanks Intl",
    "FAT": "Fresno Yosemite Intl", "FAY": "Fayetteville Regional",
    "FNT": "Bishop Intl (Flint)", "FSD": "Sioux Falls Regional",
    "GEG": "Spokane Intl", "GPT": "Gulfport-Biloxi Intl", "GRR": "Gerald R Ford Intl",
    "GSO": "Piedmont Triad Intl", "GSP": "Greenville-Spartanburg Intl",
    "GTF": "Great Falls Intl", "GUC": "Gunnison-Crested Butte Regional",
    "HNL": "Daniel K Inouye Intl (Honolulu)", "HSV": "Huntsville Intl",
    "ICT": "Wichita Dwight D Eisenhower National", "ILM": "Wilmington Intl",
    "IND": "Indianapolis Intl", "ISP": "Long Island MacArthur",
    "JAC": "Jackson Hole Airport", "JAN": "Jackson-Medgar Wiley Evers Intl",
    "JAX": "Jacksonville Intl", "JNU": "Juneau Intl",
    "KOA": "Ellison Onizuka Kona Intl", "LBB": "Lubbock Preston Smith Intl",
    "LEX": "Blue Grass Airport (Lexington)", "LGB": "Long Beach Airport",
    "LIT": "Bill and Hillary Clinton National", "LNK": "Lincoln Airport",
    "MAF": "Midland Intl Air and Space Port", "MCI": "Kansas City Intl",
    "MEM": "Memphis Intl", "MEI": "Key Field (Meridian)",
    "MHT": "Manchester-Boston Regional", "MKE": "Milwaukee Mitchell Intl",
    "MLB": "Melbourne Orlando Intl", "MLI": "Quad City Intl",
    "MOB": "Mobile Regional", "MSN": "Dane County Regional (Madison)",
    "MSY": "Louis Armstrong New Orleans Intl", "MTJ": "Montrose Regional",
    "MYR": "Myrtle Beach Intl", "OAK": "Oakland Intl",
    "OGG": "Kahului Airport (Maui)", "OKC": "Will Rogers World Airport",
    "OMA": "Eppley Airfield (Omaha)", "ONT": "Ontario Intl",
    "ORF": "Norfolk Intl", "ORH": "Worcester Regional",
    "PBI": "Palm Beach Intl", "PIA": "General Wayne A Downing Peoria Intl",
    "PIT": "Pittsburgh Intl", "PSP": "Palm Springs Intl",
    "PVD": "T.F. Green Providence", "PWM": "Portland Intl Jetport (ME)",
    "RAP": "Rapid City Regional", "RDU": "Raleigh-Durham Intl",
    "RIC": "Richmond Intl", "RNO": "Reno-Tahoe Intl",
    "ROC": "Greater Rochester Intl", "RSW": "Southwest Florida Intl",
    "SAT": "San Antonio Intl", "SAV": "Savannah/Hilton Head Intl",
    "SBA": "Santa Barbara Municipal", "SBP": "San Luis Obispo County Regional",
    "SDF": "Louisville Muhammad Ali Intl", "SGF": "Springfield-Branson National",
    "SHV": "Shreveport Regional", "SJC": "Norman Y Mineta San Jose Intl",
    "SJU": "Luis Munoz Marin Intl (San Juan)", "SMF": "Sacramento Intl",
    "SNA": "John Wayne Airport (Orange County)", "SRQ": "Sarasota-Bradenton Intl",
    "STL": "St Louis Lambert Intl", "SYR": "Syracuse Hancock Intl",
    "TUL": "Tulsa Intl", "TUS": "Tucson Intl", "TYS": "McGhee Tyson Airport (Knoxville)",
    "VPS": "Destin-Fort Walton Beach Airport", "XNA": "Northwest Arkansas National",
}

AIRPORT_STATE = {
    "ATL":"GA","AGS":"GA","SAV":"GA","MCN":"GA",
    "LAX":"CA","SFO":"CA","SAN":"CA","OAK":"CA","SJC":"CA","SMF":"CA","FAT":"CA",
    "BUR":"CA","LGB":"CA","ONT":"CA","SBA":"CA","SBP":"CA","SCK":"CA","SNA":"CA",
    "ORD":"IL","MDW":"IL","MLI":"IL","BMI":"IL","RFD":"IL","BLV":"IL",
    "DFW":"TX","DAL":"TX","HOU":"TX","SAT":"TX","AUS":"TX","ELP":"TX",
    "ABI":"TX","ACT":"TX","AMA":"TX","CRP":"TX","GGG":"TX","HRL":"TX",
    "LBB":"TX","MAF":"TX","SHV":"TX","TYR":"TX","DEN":"CO","COS":"CO",
    "ASE":"CO","GUC":"CO","MTJ":"CO","PUB":"CO","GJT":"CO",
    "JFK":"NY","LGA":"NY","EWR":"NJ","ALB":"NY","BUF":"NY","ROC":"NY",
    "SYR":"NY","ISP":"NY","IAG":"NY","SWF":"NY","HPN":"NY","ITH":"NY",
    "SEA":"WA","GEG":"WA","BLI":"WA","PSC":"WA","YKM":"WA",
    "LAS":"NV","RNO":"NV","ELY":"NV",
    "PHX":"AZ","TUS":"AZ","FLG":"AZ","YUM":"AZ","PRC":"AZ",
    "MIA":"FL","FLL":"FL","TPA":"FL","MCO":"FL","JAX":"FL","RSW":"FL",
    "PBI":"FL","EYW":"FL","SRQ":"FL","GNV":"FL","TLH":"FL","MLB":"FL",
    "MYR":"SC","CHS":"SC","CAE":"SC","GSP":"SC",
    "CLT":"NC","RDU":"NC","ILM":"NC","AVL":"NC","FAY":"NC","OAJ":"NC",
    "IAH":"TX","MSY":"LA","BTR":"LA","MOB":"AL","BHM":"AL","HSV":"AL",
    "DSM":"IA","MLI":"IL","CID":"IA","SUX":"IA",
    "MCI":"MO","STL":"MO","SGF":"MO","JLN":"MO",
    "MSP":"MN","DLH":"MN","RST":"MN","HIB":"MN",
    "DTW":"MI","GRR":"MI","FNT":"MI","LAN":"MI","MBS":"MI","AZO":"MI",
    "CVG":"KY","LEX":"KY","SDF":"KY","OWB":"KY",
    "CMH":"OH","CLE":"OH","DAY":"OH","CAK":"OH","TOL":"OH","LCK":"OH",
    "IND":"IN","SBN":"IN","EVV":"IN","FWA":"IN",
    "PIT":"PA","PHL":"PA","ABE":"PA","AVP":"PA","MDT":"PA","ERI":"PA",
    "BWI":"MD","DCA":"VA","IAD":"VA","ORF":"VA","RIC":"VA","ROA":"VA",
    "BOS":"MA","MHT":"NH","PVD":"RI","BDL":"CT","ORH":"MA","HYA":"MA",
    "PWM":"ME","BGR":"ME","BTV":"VT","BHB":"ME","ACY":"NJ","TTN":"NJ",
    "MSN":"WI","MKE":"WI","GRB":"WI","LSE":"WI","ATW":"WI",
    "OMA":"NE","LNK":"NE","GRI":"NE","OFK":"NE",
    "MEM":"TN","BNA":"TN","TYS":"TN","CHA":"TN","MKL":"TN",
    "OKC":"OK","TUL":"OK","LAW":"OK","SWO":"OK",
    "ABQ":"NM","SAF":"NM","ROW":"NM","ALS":"CO",
    "BOI":"ID","TWF":"ID","PIH":"ID","SUN":"ID",
    "BIL":"MT","GTF":"MT","MSO":"MT","BZN":"MT","HLN":"MT","GPI":"MT",
    "ANC":"AK","FAI":"AK","JNU":"AK","KTN":"AK","SIT":"AK","BRW":"AK",
    "OME":"AK","OTZ":"AK","BET":"AK","CDV":"AK","WRG":"AK",
    "HNL":"HI","OGG":"HI","KOA":"HI","LIH":"HI","ITO":"HI",
    "PDX":"OR","EUG":"OR","MFR":"OR","RDM":"OR","SLE":"OR",
    "SLC":"UT","SGU":"UT","PVU":"UT","CDC":"UT",
    "JAC":"WY","CPR":"WY","RIW":"WY","LAR":"WY",
    "ICT":"KS","MHK":"KS","FOE":"KS","GCK":"KS",
    "FSD":"SD","RAP":"SD","ABR":"SD","PIR":"SD",
    "BIS":"ND","FAR":"ND","GFK":"ND","MOT":"ND",
    "CWA":"WI","RHI":"WI","EAU":"WI",
    "XNA":"AR","LIT":"AR","FSM":"AR","TXK":"AR",
    "JAN":"MS","GPT":"MS","MEI":"MS","GTR":"MS",
    "GSO":"NC","INT":"NC",
    "HTS":"WV","CKB":"WV","BKW":"WV","PGV":"NC",
    "HGR":"MD","SBY":"MD",
    "BQN":"PR","SJU":"PR","PSE":"PR","MAZ":"PR",
    "PBG":"NY","SLK":"NY","ELM":"NY","BGM":"NY",
    "USA":"NC",
}

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: {C["bg"]} !important;
    color: {C["text"]} !important;
}}
.stApp {{ background-color: {C["bg"]} !important; }}
.block-container {{ padding: 1.8rem 2.5rem 2.5rem 2.5rem; max-width: 1400px; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: {C["card"]} !important; border-bottom: 1px solid {C["border"]}; }}

/* Expander */
details[data-testid="stExpander"] {{
    background: {C["card"]};
    border: 1px solid {C["border"]} !important;
    border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}}
details[data-testid="stExpander"] summary {{
    font-weight: 600; font-size: .92rem; color: {C["text"]} !important;
}}

/* Metric cards */
[data-testid="metric-container"] {{
    background: {C["card"]}; border: 1px solid {C["border"]};
    border-radius: 14px; padding: 1.1rem 1.4rem;
    box-shadow: 0 1px 6px rgba(0,0,0,.07);
}}
[data-testid="metric-container"] label {{
    color: {C["muted"]} !important; font-size: 0.75rem !important;
    text-transform: uppercase; letter-spacing: .07em; font-weight: 600;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size: 1.9rem !important; font-weight: 700 !important; color: {C["text"]} !important;
}}

/* Tabs */
[data-testid="stTabs"] {{
    background: {C["card"]};
    border-radius: 12px 12px 0 0;
    border-bottom: 1px solid {C["border"]};
    padding: 0 .5rem;
}}
[data-testid="stTabs"] button {{
    font-size: .85rem; font-weight: 500; padding: .6rem 1.1rem;
    color: {C["muted"]} !important; border-radius: 0; border: none;
    background: transparent !important;
}}
[data-testid="stTabs"] button:hover {{ color: {C["text"]} !important; }}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {C["blue"]} !important;
    border-bottom: 2px solid {C["blue"]} !important;
    font-weight: 600 !important;
}}

/* Divider */
hr {{ border-color: {C["border"]} !important; margin: .6rem 0 1.4rem 0; }}

/* Dataframe */
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; border: 1px solid {C["border"]}; }}

/* Button */
.stButton > button,
[data-testid="stFormSubmitButton"] > button,
.stFormSubmitButton > button {{
    background: {C["blue"]} !important;
    color: #ffffff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important; font-size: .95rem !important;
    padding: .7rem 1.8rem !important; letter-spacing: .03em !important;
    box-shadow: 0 2px 6px rgba(2,132,199,.25) !important;
    transition: background .15s ease, box-shadow .15s ease !important;
}}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover,
.stFormSubmitButton > button:hover {{
    background: #0369a1 !important;
    box-shadow: 0 3px 10px rgba(2,132,199,.35) !important;
}}

/* Number input — premium styling with solid unified border */
[data-testid="stNumberInput"] > div {{
    border: 1.5px solid {C["border"]} !important;
    border-radius: 10px !important;
    background: {C["card"]} !important;
    overflow: hidden;
    transition: border-color .15s ease, box-shadow .15s ease;
}}
[data-testid="stNumberInput"] > div:focus-within {{
    border-color: {C["blue"]} !important;
    box-shadow: 0 0 0 3px rgba(2,132,199,.12) !important;
}}
[data-testid="stNumberInput"] input {{
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    font-weight: 500 !important;
    font-size: .95rem !important;
    color: {C["text"]} !important;
    padding: .55rem .8rem !important;
    box-shadow: none !important;
}}
[data-testid="stNumberInput"] input:focus {{
    outline: none !important;
    box-shadow: none !important;
}}
[data-testid="stNumberInput"] button {{
    background: {C["card"]} !important;
    border: none !important;
    border-left: 1.5px solid {C["border"]} !important;
    border-radius: 0 !important;
    color: {C["muted"]} !important;
}}
[data-testid="stNumberInput"] button:hover {{
    background: {C["blue"]} !important; color: #fff !important;
    border-left-color: {C["blue"]} !important;
}}

/* Selectbox — match number-input styling so Route fields align visually */
[data-testid="stSelectbox"] > div > div {{
    border: 1.5px solid {C["border"]} !important;
    border-radius: 10px !important;
    background: {C["card"]} !important;
    min-height: 42px;
}}
[data-testid="stSelectbox"] > div > div:focus-within {{
    border-color: {C["blue"]} !important;
    box-shadow: 0 0 0 3px rgba(2,132,199,.12) !important;
}}
/* Ensure selectbox + number input fill their column equally */
[data-testid="stSelectbox"],
[data-testid="stNumberInput"] {{ width: 100% !important; }}

/* Help-tooltip (? icon) — strip border/background on icon AND every ancestor wrapper */
[data-testid="stTooltipIcon"],
[data-testid="stTooltipIcon"] *,
[data-testid="stTooltipHoverTarget"],
[data-testid="stTooltipHoverTarget"] *,
[data-baseweb="tooltip"],
[data-baseweb="tooltip"] *,
[class*="tooltip"],
[class*="Tooltip"] {{
    border: none !important;
    border-left: none !important;
    border-right: none !important;
    border-top: none !important;
    border-bottom: none !important;
    background: transparent !important;
    box-shadow: none !important;
}}
[data-testid="stTooltipIcon"] svg,
[data-testid="stTooltipHoverTarget"] svg {{
    color: {C["muted"]} !important;
}}
[data-testid="stTooltipIcon"]::before,
[data-testid="stTooltipIcon"]::after,
[data-testid="stTooltipHoverTarget"]::before,
[data-testid="stTooltipHoverTarget"]::after {{
    content: none !important;
    display: none !important;
    border: none !important;
    background: transparent !important;
}}
/* Kill the vertical separator some widgets inject before the tooltip on any labeled widget */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] > div,
[data-testid="stWidgetLabel"] * ,
label[data-baseweb="checkbox"],
label[data-baseweb="checkbox"] > div,
label[data-baseweb="checkbox"] *,
div[data-baseweb="form-control-container"],
div[data-baseweb="form-control-container"] * {{
    border-left: none !important;
    border-right: none !important;
}}

/* Checkbox — align box vertically with label text, tight spacing */
[data-testid="stCheckbox"] > label,
label[data-baseweb="checkbox"] {{
    display: flex !important;
    align-items: center !important;
    gap: .25rem !important;
    min-height: 42px;
}}
[data-testid="stCheckbox"] [role="checkbox"],
label[data-baseweb="checkbox"] span[role="checkbox"],
label[data-baseweb="checkbox"] > span:first-child {{
    margin: 0 !important;
    padding: 0 !important;
    align-self: center !important;
}}
[data-testid="stCheckbox"] label p,
label[data-baseweb="checkbox"] div,
label[data-baseweb="checkbox"] > div:last-child {{
    margin: 0 !important;
    padding-left: 0 !important;
    line-height: 1.4 !important;
}}

/* Section title */
.section-title {{
    font-size: .72rem; font-weight: 700; color: {C["muted"]};
    text-transform: uppercase; letter-spacing: .1em;
    margin-bottom: .7rem; margin-top: .2rem;
}}
/* Extra breathing room between sections inside a form (e.g. Predict tab) */
[data-testid="stForm"] .section-title {{
    margin-top: 2rem !important;
    margin-bottom: 1.1rem !important;
    padding-top: .6rem;
    border-top: 1px solid {C["border"]};
}}
[data-testid="stForm"] .section-title:first-of-type {{
    margin-top: .3rem !important;
    padding-top: 0;
    border-top: none;
}}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] {{
    margin-top: 2rem;
}}

/* Stat row / cards */
.stat-row {{ display: flex; gap: 1rem; margin-bottom: 1.2rem; }}
.stat-card {{
    flex: 1; background: {C["card"]}; border: 1px solid {C["border"]};
    border-radius: 16px; padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 6px rgba(0,0,0,.06);
    transition: box-shadow .2s ease;
}}
.stat-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,.1); }}
.stat-card .sc-label {{
    font-size: .7rem; color: {C["muted"]}; font-weight: 700;
    text-transform: uppercase; letter-spacing: .09em; margin-bottom: .35rem;
}}
.stat-card .sc-value {{ font-size: 1.9rem; font-weight: 700; color: {C["text"]}; line-height: 1.1; }}
.stat-card .sc-sub   {{ font-size: .75rem; color: {C["muted"]}; margin-top: .3rem; }}

/* Accent borders */
.accent-blue  {{ border-top: 3px solid {C["blue"]}; }}
.accent-green {{ border-top: 3px solid {C["green"]}; }}
.accent-amber {{ border-top: 3px solid {C["amber"]}; }}
.accent-red   {{ border-top: 3px solid {C["red"]}; }}

/* Badges */
.bq-badge-on  {{
    background: #dcfce7; color: {C["green"]}; border: 1px solid #bbf7d0;
    border-radius: 999px; padding: .25rem .85rem; font-size: .78rem; font-weight: 600;
}}
.bq-badge-off {{
    background: #fef3c7; color: {C["amber"]}; border: 1px solid #fde68a;
    border-radius: 999px; padding: .25rem .85rem; font-size: .78rem; font-weight: 600;
}}

/* Inputs and selects */
[data-testid="stSelectbox"] > div, [data-testid="stNumberInput"] > div {{
    border-radius: 8px !important;
}}
.stSlider [data-testid="stMarkdownContainer"] {{ color: {C["muted"]}; font-size: .8rem; }}

/* Caption text */
.stCaption {{ color: {C["muted"]} !important; font-size: .78rem !important; }}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ───────────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor=C["card"],
    plot_bgcolor="#fafbfc",
    font=dict(color=C["text"], family="Inter, Segoe UI, sans-serif", size=12),
    title_font=dict(size=13, color=C["text"], family="Inter, Segoe UI, sans-serif"),
    xaxis=dict(
        gridcolor="#f1f5f9", linecolor=C["border"],
        tickcolor=C["muted"], color=C["muted"], tickfont=dict(size=11),
    ),
    yaxis=dict(
        gridcolor="#f1f5f9", linecolor=C["border"],
        tickcolor=C["muted"], color=C["muted"], tickfont=dict(size=11),
    ),
    margin=dict(l=48, r=20, t=48, b=36),
    hoverlabel=dict(bgcolor=C["card"], font_color=C["text"], bordercolor=C["border"],
                    font_size=12, font_family="Inter, Segoe UI, sans-serif"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"], font=dict(color=C["muted"])),
)
COLORS = [C["blue"], C["green"], C["amber"], C["red"], "#7c3aed", "#ea580c", "#0891b2"]

def sf(fig): fig.update_layout(**PL); return fig


# ── BQ connection check ────────────────────────────────────────────────────────
USE_BQ = bq_loader.is_available()


# ── Local parquet fallback ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_local_bts() -> pd.DataFrame:
    frames = []
    for f in ["bts_2024_1.parquet", "bts_2024_7.parquet", "bts_2024_10.parquet"]:
        p = CACHE / f
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "Cancelled" in df.columns:
        df = df[df["Cancelled"] != 1.0].copy()
    for col in ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]:
        if col in df.columns: df[col] = df[col].fillna(0)
    df["flight_month"]       = df["Month"].astype(int)
    df["scheduled_dep_hour"] = (df["CRSDepTime"].fillna(0).astype(int) // 100).clip(0, 23)
    df["airline_code"]       = df["Reporting_Airline"]
    df["DepDelay"]           = df["DepDelay"].fillna(0).clip(-720, 720)
    df["ArrDelay"]           = df["ArrDelay"].fillna(0).clip(-720, 720)
    df["is_delayed"]         = (df["DepDelay"] > 15).astype(int)
    return df

@st.cache_data(show_spinner=False)
def load_risk_scores() -> pd.DataFrame:
    p = MODELS_DIR / "risk_scores.parquet"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_parquet(p)
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    return df

@st.cache_data(show_spinner=False)
def load_centrality() -> pd.DataFrame:
    p = MODELS_DIR / "centrality_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_prediction_artifacts():
    """Load trained Gradient Boosting model + scaler + feature columns + medians.
    Cached across reruns. Returns None if any artifact is missing."""
    need = ["gradient_boosting.joblib", "feature_cols.json", "feature_medians.json"]
    if not all((MODELS_DIR / f).exists() for f in need):
        return None
    try:
        model = joblib.load(MODELS_DIR / "gradient_boosting.joblib")
        with open(MODELS_DIR / "feature_cols.json") as fh:
            feat_cols = json.load(fh)
        with open(MODELS_DIR / "feature_medians.json") as fh:
            medians = json.load(fh)
        return {"model": model, "feat_cols": feat_cols, "medians": medians}
    except Exception as e:
        print(f"[predict] artifact load failed: {e}")
        return None


@st.cache_data(show_spinner=False)
def get_airline_list() -> list:
    if USE_BQ:
        return bq_loader.load_airline_list()
    df = load_local_bts()
    return sorted(df["airline_code"].dropna().unique().tolist()) if not df.empty else []


# ── Load base data ─────────────────────────────────────────────────────────────
with st.spinner("Connecting…"):
    local_bts = load_local_bts()
    risk_df   = load_risk_scores()
    cent_df   = load_centrality()
    all_airlines = get_airline_list()

scores_ready = not risk_df.empty
TARGET_AIRPORTS = ["ATL", "DFW", "DEN", "ORD", "LAX", "JFK", "SFO", "SEA", "LAS", "MCO"]

# ── Header band (brand + connection status, above filters) ───────────────────
import datetime as _dt
DATA_MIN = _dt.date(2022, 2, 1)
DATA_MAX = _dt.date(2026, 1, 31)

_bq_badge = (
    "<span class='bq-badge-on'>⚡ BigQuery Connected</span>" if USE_BQ
    else "<span class='bq-badge-off'>📂 Local Cache · Jan / Jul / Oct 2024</span>"
)
_card   = C["card"]
_border = C["border"]
_blue   = C["blue"]
_muted  = C["muted"]
st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:space-between;
            padding:1.2rem 0 1.2rem 0;margin-bottom:1rem;
            border-bottom:1px solid {_border}'>
  <div style='display:flex;align-items:center;gap:.75rem'>
    <div style='background:{_blue};color:#ffffff;border-radius:10px;
                width:40px;height:40px;display:flex;align-items:center;
                justify-content:center;font-size:1.2rem;flex-shrink:0'>✈</div>
    <div>
      <div style='font-size:1.25rem;font-weight:700;color:{C["text"]};letter-spacing:-.02em;line-height:1.2'>
        Aviation Disruption Risk Monitor
      </div>
      <div style='font-size:.76rem;color:{_muted};margin-top:.2rem'>
        IS3107 · Data Engineering Pipeline · Feb 2022 – Jan 2026
      </div>
    </div>
  </div>
  <div style='display:flex;align-items:center;gap:1.5rem'>
    <div>{_bq_badge}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Filters removed — default to full dataset on every view ──────────────────
sel_months   = list(range(1, 13)) if USE_BQ else [1, 7, 10]
sel_airlines = []
sel_airports = []
sel_risk     = ["Low", "Medium", "High"]
start_date, end_date = DATA_MIN, DATA_MAX
date_range = None


# ── BQ filter helpers ─────────────────────────────────────────────────────────
def bq_date_range():
    """Return (start_iso, end_iso) tuple only if user narrowed from the full range."""
    if not USE_BQ:
        return None
    if start_date == DATA_MIN and end_date == DATA_MAX:
        return None
    return (start_date.isoformat(), end_date.isoformat())


def local_filter(df: pd.DataFrame) -> pd.DataFrame:
    if sel_months and "flight_month" in df.columns:
        df = df[df["flight_month"].isin(sel_months)]
    if sel_airlines and "airline_code" in df.columns:
        df = df[df["airline_code"].isin(sel_airlines)]
    if sel_airports and "Origin" in df.columns:
        df = df[df["Origin"].isin(sel_airports)]
    return df

bts_f = local_filter(local_bts) if not local_bts.empty else local_bts

risk_f = risk_df.copy()
if scores_ready:
    # Primary time filter — date range (applies to both BQ and local risk scores table)
    if "flight_date" in risk_f.columns:
        start_ts = pd.Timestamp(start_date)
        end_ts   = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        risk_f = risk_f[(risk_f["flight_date"] >= start_ts) & (risk_f["flight_date"] < end_ts)]
    if sel_months and "flight_month" in risk_f.columns:
        risk_f = risk_f[risk_f["flight_month"].isin(sel_months)]
    if sel_airlines and "airline_code" in risk_f.columns:
        risk_f = risk_f[risk_f["airline_code"].isin(sel_airlines)]
    if sel_airports and "Origin" in risk_f.columns:
        risk_f = risk_f[risk_f["Origin"].isin(sel_airports)]
    if "risk_label" in risk_f.columns:
        risk_f = risk_f[risk_f["risk_label"].isin(sel_risk)]


# ── KPI cards ─────────────────────────────────────────────────────────────────
_dr_label = (
    f"{start_date.strftime('%b %Y')} – {end_date.strftime('%b %Y')}"
    if USE_BQ else "Jan / Jul / Oct 2024"
)
_text = C["text"]

if USE_BQ:
    @st.cache_data(ttl=3600, show_spinner=False)
    def _kpis(months, airlines, airports, dr):
        return bq_loader.load_kpis(months or None, airlines or None, airports or None, dr)
    kpi = _kpis(tuple(sel_months), tuple(sel_airlines), tuple(sel_airports), bq_date_range())
    total_f    = kpi["total_flights"] if kpi else 0
    delay_rate = kpi["delay_rate"]    if kpi else 0
else:
    total_f    = len(bts_f)
    delay_rate = bts_f["is_delayed"].mean() if not bts_f.empty else 0

high_risk_n = int((risk_f["risk_score"] >= 0.6).sum()) if scores_ready else 0
n_airports  = len(cent_df) if not cent_df.empty else 0

st.markdown(f"""
<div class="stat-row">
  <div class="stat-card accent-blue">
    <div class="sc-label">Total Flights</div>
    <div class="sc-value">{total_f:,}</div>
    <div class="sc-sub">{_dr_label}</div>
  </div>
  <div class="stat-card accent-amber">
    <div class="sc-label">Delay Rate  (>15 min)</div>
    <div class="sc-value">{delay_rate:.1%}</div>
    <div class="sc-sub">Departure delays</div>
  </div>
  <div class="stat-card accent-red">
    <div class="sc-label">High-Risk Flights</div>
    <div class="sc-value">{"—" if not scores_ready else f"{high_risk_n:,}"}</div>
    <div class="sc-sub">{"Run scoring script first" if not scores_ready else f"Score ≥ 60%"}</div>
  </div>
  <div class="stat-card accent-green">
    <div class="sc-label">Airports in Network</div>
    <div class="sc-value">{n_airports if n_airports else "—"}</div>
    <div class="sc-sub">{"US domestic network" if n_airports else "Run scoring script first"}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab1, tab2, tab3, tab_weather, tab_cancel, tab4, tab_predict = st.tabs(
    ["Live Airport Status", "Historical Trends", "Network Analysis", "Risk Scores",
     "Weather Impact", "Cancellation Analysis", "Model Performance", "Predict Flight Risk"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB LIVE — LIVE AIRPORT STATUS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown(
        f"<p style='color:{C['muted']};font-size:.85rem'>"
        "Live METAR observations polled every 30 minutes via the AWC API. "
        "Data lands in <code>raw_weather.metar_raw</code> and is shown here directly from BigQuery.</p>",
        unsafe_allow_html=True,
    )

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        do_refresh = st.button("↻  Refresh now")

    @st.cache_data(ttl=300, show_spinner=False)
    def _live_metar_bq():
        return bq_loader.load_latest_metar()

    @st.cache_data(ttl=300, show_spinner=False)
    def _live_metar_api():
        try:
            from awc_extractor import fetch_metar
            return fetch_metar(hours_back=1)
        except Exception as e:
            print(f"[live] AWC direct fetch failed: {e}")
            return pd.DataFrame()

    if do_refresh:
        st.cache_data.clear()

    with st.spinner("Fetching latest METAR…"):
        if USE_BQ:
            metar_live = _live_metar_bq()
        else:
            metar_live = _live_metar_api()

    CAT_COLOR = {"VFR": C["green"], "MVFR": C["blue"], "IFR": C["amber"], "LIFR": C["red"]}
    CAT_LABEL = {
        "VFR":  "✅ VFR  — clear",
        "MVFR": "🔵 MVFR — marginal",
        "IFR":  "⚠️ IFR  — instrument",
        "LIFR": "🔴 LIFR — low instrument",
    }

    if metar_live is None or metar_live.empty:
        st.warning("No live METAR data available. BigQuery may be unreachable and the AWC API returned no data.")
    else:
        # ── Status cards (one per airport) ────────────────────────────────────
        st.markdown("<div class='section-title'>Flight Category Legend</div>", unsafe_allow_html=True)
        leg1, leg2, leg3, leg4 = st.columns(4)
        for col, cat, color, full, desc in [
            (leg1, "VFR",  "#059669", "Visual Flight Rules",      "Clear skies · Ceiling > 3,000 ft · Visibility > 5 sm · Normal operations"),
            (leg2, "MVFR", "#0284c7", "Marginal VFR",             "Some clouds · Ceiling 1,000–3,000 ft · Visibility 3–5 sm · Minor caution"),
            (leg3, "IFR",  "#d97706", "Instrument Flight Rules",  "Low cloud / fog · Ceiling 500–1,000 ft · Visibility 1–3 sm · Delays likely"),
            (leg4, "LIFR", "#dc2626", "Low Instrument Flight Rules","Dense fog / severe wx · Ceiling < 500 ft · Visibility < 1 sm · Cancellations likely"),
        ]:
            col.markdown(f"""
            <div style='background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid {color};
                        border-radius:10px;padding:.85rem 1rem;'>
              <div style='font-size:1.1rem;font-weight:700;color:{color}'>{cat}</div>
              <div style='font-size:.78rem;font-weight:600;color:#0f172a;margin:.2rem 0'>{full}</div>
              <div style='font-size:.72rem;color:#64748b;line-height:1.5'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Current Conditions · 10 Target Airports</div>",
                    unsafe_allow_html=True)

        airports_per_row = 5
        rows_data = [metar_live.iloc[i:i+airports_per_row]
                     for i in range(0, len(metar_live), airports_per_row)]

        for row_chunk in rows_data:
            cols = st.columns(len(row_chunk))
            for col, (_, r) in zip(cols, row_chunk.iterrows()):
                cat   = str(r.get("flight_category") or "N/A").strip().upper()
                color = CAT_COLOR.get(cat, C["muted"])
                wind  = r.get("wind_speed_knots")
                temp  = r.get("temp_c")
                vis   = r.get("visibility_sm")
                obs   = r.get("obs_time") or r.get("ingested_at") or ""
                obs_short = str(obs)[:16] if obs else "—"
                wind_str = f"{wind:.0f} kt"  if pd.notna(wind) else "—"
                temp_str = f"{temp:.1f} °C" if pd.notna(temp) else "—"
                vis_str  = f"{vis} sm"       if pd.notna(vis)  else "—"
                obs_str  = str(obs).replace("T", "  ").replace("Z", "")[:17] if obs else "—"
                col.markdown(f"""
                <div style="background:#ffffff;border:1px solid #e2e8f0;border-top:3px solid {color};
                            border-radius:16px;padding:1.2rem 1.4rem;min-height:170px;
                            box-shadow:0 1px 6px rgba(0,0,0,.06)">
                  <div style="font-size:.7rem;font-weight:700;color:#64748b;text-transform:uppercase;
                              letter-spacing:.09em;margin-bottom:.3rem">{r.get('iata_code','?')}</div>
                  <div style="color:{color};font-size:1.4rem;font-weight:700;margin:.1rem 0 .6rem">{cat}</div>
                  <table style="width:100%;font-size:.78rem;border-collapse:collapse">
                    <tr>
                      <td style="color:#64748b;padding:3px 8px 3px 0;font-weight:500">Wind</td>
                      <td style="color:#0f172a;font-weight:600">{wind_str}</td>
                    </tr>
                    <tr>
                      <td style="color:#64748b;padding:3px 8px 3px 0;font-weight:500">Temp</td>
                      <td style="color:#0f172a;font-weight:600">{temp_str}</td>
                    </tr>
                    <tr>
                      <td style="color:#64748b;padding:3px 8px 3px 0;font-weight:500">Visibility</td>
                      <td style="color:#0f172a;font-weight:600">{vis_str}</td>
                    </tr>
                  </table>
                  <div style="font-size:.68rem;color:#94a3b8;margin-top:.5rem">{obs_str}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Summary table ──────────────────────────────────────────────────────
        st.markdown("<div class='section-title'>Full METAR Data</div>", unsafe_allow_html=True)
        display_cols = [c for c in ["iata_code", "obs_time", "flight_category",
                                     "temp_c", "dewpoint_c", "wind_dir_deg",
                                     "wind_speed_knots", "visibility_sm", "ingested_at"]
                        if c in metar_live.columns]
        metar_display = metar_live[display_cols].copy().reset_index(drop=True)

        # Clean up timestamps
        for ts_col in ["obs_time", "ingested_at"]:
            if ts_col in metar_display.columns:
                metar_display[ts_col] = (
                    pd.to_datetime(metar_display[ts_col], utc=True, errors="coerce")
                    .dt.strftime("%Y-%m-%d  %H:%M UTC")
                )

        # Rename columns to readable headers with units
        metar_display.rename(columns={
            "iata_code":        "Airport",
            "obs_time":         "Observation Time",
            "flight_category":  "Flight Category",
            "temp_c":           "Temperature (°C)",
            "dewpoint_c":       "Dewpoint (°C)",
            "wind_dir_deg":     "Wind Direction (°)",
            "wind_speed_knots": "Wind Speed (kt)",
            "visibility_sm":    "Visibility (sm)",
            "ingested_at":      "Last Ingested",
        }, inplace=True)

        st.dataframe(metar_display, use_container_width=True, hide_index=True)

        # ── Flight category bar chart ──────────────────────────────────────────



# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HISTORICAL TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── helper: cache BQ queries keyed on filter selections ──────────────────
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_hour(m, al, ap, dr):   return bq_loader.load_delay_by_hour(m or None, al or None, ap or None, dr)
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_airline(m, ap, dr):    return bq_loader.load_delay_by_airline(m or None, ap or None, dr)
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_airport(m, al, dr):    return bq_loader.load_delay_by_airport(m or None, al or None, dr)
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_monthly(al, ap, dr, m):return bq_loader.load_monthly_trend(al or None, ap or None, dr, m or None)
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_causes(m, al, ap, dr): return bq_loader.load_delay_causes(m or None, al or None, ap or None, dr)
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_hist(m, al, ap, dr):   return bq_loader.load_delay_distribution_sample(m or None, al or None, ap or None, dr)
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_dow_hour(m, al, ap, dr): return bq_loader.load_delay_by_dow_hour(m or None, al or None, ap or None, dr)

    fk = (tuple(sel_months), tuple(sel_airlines), tuple(sel_airports), bq_date_range())

    # Row 1 — delay distribution + hourly
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Departure Delay Distribution</div>", unsafe_allow_html=True)
        if USE_BQ:
            dist_df = bq_hist(*fk)
            delay_series = dist_df["DepDelay"] if dist_df is not None else pd.Series(dtype=float)
        else:
            delay_series = bts_f["DepDelay"].clip(-60, 180).dropna()
            if len(delay_series) > 80_000: delay_series = delay_series.sample(80_000, random_state=42)

        fig = px.histogram(delay_series, nbins=80, color_discrete_sequence=[C["blue"]],
                           labels={"value": "Departure Delay (min)"})
        fig.add_vline(x=15, line_dash="dash", line_color=C["red"],
                      annotation_text="15-min threshold", annotation_font_color=C["red"])
        fig = sf(fig); fig.update_layout(showlegend=False, title="Departure Delay Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Delay Rate by Hour of Day</div>", unsafe_allow_html=True)
        if USE_BQ:
            hourly = bq_hour(*fk)
            if hourly is not None: hourly.columns = ["hour", "rate", "n"]
        else:
            hourly = bts_f.groupby("scheduled_dep_hour")["is_delayed"].mean().reset_index()
            hourly.columns = ["hour", "rate"]

        if hourly is not None and not hourly.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly["hour"], y=hourly["rate"], mode="lines+markers",
                line=dict(color=C["amber"], width=2.5), marker=dict(size=6, color=C["amber"]),
                fill="tozeroy", fillcolor="rgba(217,119,6,.12)",
            ))
            fig.update_yaxes(tickformat=".0%")
            fig.update_xaxes(tickvals=list(range(0, 24, 2)))
            fig = sf(fig); fig.update_layout(title="P(delay > 15 min) by Departure Hour")
            st.plotly_chart(fig, use_container_width=True)

    # Row 2 — airline + airport
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Delay Rate by Airline</div>", unsafe_allow_html=True)
        if USE_BQ:
            al_df = bq_airline(tuple(sel_months), tuple(sel_airports), bq_date_range())
            if al_df is not None:
                al_df = al_df.sort_values("delay_rate").tail(15)
        else:
            al_df = (bts_f.groupby("airline_code")
                     .agg(delay_rate=("is_delayed","mean"), n=("is_delayed","count"))
                     .query("n >= 500").sort_values("delay_rate").tail(15).reset_index()
                     .rename(columns={"airline_code":"airline_code"}))

        if al_df is not None and not al_df.empty:
            x_col = "delay_rate"
            y_col = "airline_code" if "airline_code" in al_df.columns else al_df.columns[0]
            al_df = al_df.copy()
            al_df["airline_name"] = al_df[y_col].map(lambda c: AIRLINE_NAMES.get(c, c))
            fig = px.bar(al_df, x=x_col, y=y_col, orientation="h",
                         color=x_col, color_continuous_scale=[C["green"], C["amber"], C["red"]],
                         labels={x_col:"Delay Rate", y_col:"Airline"},
                         custom_data=["airline_name"])
            fig.update_traces(
                hovertemplate="<b>%{customdata[0]}</b> (%{y})<br>Delay Rate: %{x:.1%}<extra></extra>"
            )
            fig.update_xaxes(tickformat=".0%")
            fig = sf(fig); fig.update_layout(coloraxis_showscale=False, title="Delay Rate by Airline")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Delay Rate by Origin Airport</div>", unsafe_allow_html=True)
        if USE_BQ:
            ap_df = bq_airport(tuple(sel_months), tuple(sel_airlines), bq_date_range())
            if ap_df is not None:
                ap_df = ap_df.sort_values("delay_rate").tail(20)
        else:
            ap_df = (bts_f.groupby("Origin")
                     .agg(delay_rate=("is_delayed","mean"), n=("is_delayed","count"))
                     .query("n >= 300").sort_values("delay_rate").tail(20).reset_index()
                     .rename(columns={"Origin":"airport"}))

        if ap_df is not None and not ap_df.empty:
            y_col = "airport" if "airport" in ap_df.columns else ap_df.columns[0]
            ap_df = ap_df.copy()
            ap_df["airport_name"] = ap_df[y_col].map(lambda c: AIRPORT_NAMES.get(c, c))
            fig = px.bar(ap_df, x="delay_rate", y=y_col, orientation="h",
                         color="delay_rate", color_continuous_scale=[C["green"], C["amber"], C["red"]],
                         labels={"delay_rate":"Delay Rate", y_col:"Airport"},
                         custom_data=["airport_name"])
            fig.update_traces(
                hovertemplate="<b>%{customdata[0]}</b> (%{y})<br>Delay Rate: %{x:.1%}<extra></extra>"
            )
            fig.update_xaxes(tickformat=".0%")
            fig = sf(fig); fig.update_layout(coloraxis_showscale=False, title="Delay Rate by Origin Airport")
            st.plotly_chart(fig, use_container_width=True)

    # Row 3 — causes + monthly trend
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Delay Cause Breakdown</div>", unsafe_allow_html=True)
        if USE_BQ:
            cause_df = bq_causes(*fk)
        else:
            cause_map = {"CarrierDelay":"Carrier","WeatherDelay":"Weather",
                         "NASDelay":"NAS","SecurityDelay":"Security","LateAircraftDelay":"Late Aircraft"}
            rows = [{"Cause":lbl,"Total Min":bts_f[col].sum()}
                    for col,lbl in cause_map.items() if col in bts_f.columns]
            cause_df = pd.DataFrame(rows) if rows else None

        if cause_df is not None and not cause_df.empty:
            fig = px.pie(cause_df, values="Total Min", names="Cause",
                         color_discrete_sequence=COLORS, hole=0.45)
            fig.update_traces(textposition="outside", textinfo="percent+label",
                              textfont=dict(color=C["text"]))
            fig = sf(fig); fig.update_layout(showlegend=False, title="Share of Total Delay Minutes")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Monthly Delay Trend</div>", unsafe_allow_html=True)
        if USE_BQ:
            mo_df = bq_monthly(tuple(sel_airlines), tuple(sel_airports), bq_date_range(), tuple(sel_months))
        else:
            month_map_label = {1:"Jan 2024", 7:"Jul 2024", 10:"Oct 2024"}
            mo_df = bts_f.groupby("flight_month")["is_delayed"].agg(delay_rate="mean", n="count").reset_index()
            mo_df["label"] = mo_df["flight_month"].map(month_map_label)

        if mo_df is not None and not mo_df.empty and "label" in mo_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=mo_df["label"], y=mo_df["delay_rate"], mode="lines+markers",
                line=dict(color=C["blue"], width=2.5), marker=dict(size=7, color=C["blue"]),
                fill="tozeroy", fillcolor="rgba(2,132,199,.12)",
                hovertemplate="%{x}: %{y:.1%}<extra></extra>",
            ))
            fig.update_yaxes(tickformat=".0%")
            fig = sf(fig); fig.update_layout(title="Monthly Delay Rate Over Time")
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 4 — Day of Week × Hour heatmap ────────────────────────────────────
    st.markdown("<div class='section-title'>Best & Worst Times to Fly</div>", unsafe_allow_html=True)
    st.caption("Each cell = delay rate for that day + departure hour. Darker red = higher chance of delay. Use this to pick the best time to book.")

    DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    if USE_BQ:
        dow_df = bq_dow_hour(*fk)
    else:
        if not bts_f.empty and {"day_of_week", "scheduled_dep_hour", "is_delayed"}.issubset(bts_f.columns):
            dow_df = (bts_f.groupby(["day_of_week", "scheduled_dep_hour"])["is_delayed"]
                      .agg(delay_rate="mean", n="count").reset_index()
                      .query("n >= 30"))
            dow_map = {2:"Mon", 3:"Tue", 4:"Wed", 5:"Thu", 6:"Fri", 7:"Sat", 1:"Sun"}
            dow_df["dow_label"] = dow_df["day_of_week"].map(dow_map)
            dow_df = dow_df.rename(columns={"scheduled_dep_hour": "hour"})
        else:
            dow_df = None

    if dow_df is not None and not dow_df.empty:
        pivot = (dow_df.pivot_table(index="dow_label", columns="hour",
                                    values="delay_rate", aggfunc="mean")
                 .reindex(DOW_ORDER))
        hour_labels = [f"{h:02d}:00" for h in pivot.columns]
        text_vals = [[f"{v:.0%}" if not pd.isna(v) else "" for v in row]
                     for row in pivot.values]

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=hour_labels,
            y=pivot.index.tolist(),
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(size=9, color="white"),
            colorscale=[[0, "#fff1f2"], [0.4, "#fca5a5"], [0.7, "#ef4444"], [1, "#7f1d1d"]],
            hovertemplate="<b>%{y} %{x}</b><br>Delay rate: %{z:.1%}<extra></extra>",
            colorbar=dict(title="Delay Rate", tickformat=".0%",
                          tickfont=dict(color=C["muted"]), thickness=12),
        ))
        fig = sf(fig)
        fig.update_layout(
            title="Delay Rate by Day of Week & Departure Hour",
            xaxis=dict(title="Scheduled Departure Hour", tickangle=-45),
            yaxis=dict(title=""),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — NETWORK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if cent_df.empty:
        st.info("Run `python scripts/generate_risk_scores.py` to generate network metrics.")
    else:
        # ── US Map — airports sized by hub importance, colored by delay rate ──
        st.markdown("<div class='section-title'>US Aviation Network Map</div>", unsafe_allow_html=True)

        map_df = cent_df.copy()
        map_df["lat"] = map_df["airport"].map(lambda a: AIRPORT_COORDS.get(a, (None, None))[0])
        map_df["lon"] = map_df["airport"].map(lambda a: AIRPORT_COORDS.get(a, (None, None))[1])
        map_df = map_df.dropna(subset=["lat", "lon"]).copy()

        # Attach per-airport delay rate (from risk_scores if available, else local bts)
        delay_lookup = pd.Series(dtype=float)
        if scores_ready and "Origin" in risk_df.columns and "is_delayed" in risk_df.columns:
            delay_lookup = (risk_df.groupby("Origin")["is_delayed"]
                              .agg(["mean", "count"]).query("count >= 100")["mean"])
        elif not bts_f.empty:
            delay_lookup = (bts_f.groupby("Origin")["is_delayed"]
                              .agg(["mean", "count"]).query("count >= 100")["mean"])
        map_df["delay_rate"] = map_df["airport"].map(delay_lookup).fillna(0.0)

        # Limit to continental US for sensible framing, keep AK/HI only if selected
        cont_us = map_df[(map_df["lat"].between(24, 50)) & (map_df["lon"].between(-125, -66))].copy()

        if not cont_us.empty:
            # ── Build route lines once, reuse across maps ──────────────────────
            route_lines = []
            route_src = risk_df if (scores_ready and {"Origin", "Dest"}.issubset(risk_df.columns)) else bts_f
            if not route_src.empty and {"Origin", "Dest"}.issubset(route_src.columns):
                top_routes = (route_src.groupby(["Origin", "Dest"]).size()
                              .nlargest(50).reset_index(name="n"))
                coord_set = set(cont_us["airport"])
                for _, r in top_routes.iterrows():
                    if r["Origin"] in coord_set and r["Dest"] in coord_set:
                        o = AIRPORT_COORDS[r["Origin"]]; d = AIRPORT_COORDS[r["Dest"]]
                        route_lines.append(dict(lon=[o[1], d[1]], lat=[o[0], d[0]], n=int(r["n"])))

            top_airports = set(cont_us.nlargest(15, "pagerank")["airport"])

            def _base_geo(title, height=400):
                f = go.Figure()
                f.update_geos(scope="usa", bgcolor=C["card"], landcolor="#e8edf2",
                               subunitcolor=C["border"], countrycolor=C["border"],
                               lakecolor=C["card"], showlakes=True)
                f.update_layout(paper_bgcolor=C["card"],
                                font=dict(color=C["text"], family="Inter, Segoe UI, sans-serif"),
                                margin=dict(l=0, r=0, t=36, b=0), height=height,
                                title=dict(text=title, font=dict(size=14, color=C["text"])))
                return f

            # ── Map 1: Which airports are most important to the network? ────────
            st.markdown("<div class='section-title'>Which airports are the most critical hubs?</div>", unsafe_allow_html=True)
            st.caption("Circle size = PageRank (network importance). Bigger = more flights depend on this airport. Labels show top 15 hubs.")

            fig1 = _base_geo("Hub Importance — sized by PageRank (bigger = more critical)")
            fig1.add_trace(go.Scattergeo(
                lon=cont_us["lon"], lat=cont_us["lat"],
                text=cont_us.apply(
                    lambda r: f"<b>{r['airport']}</b><br>PageRank: {r['pagerank']:.4f}"
                              f"<br>Betweenness: {r['betweenness']:.3f}", axis=1),
                mode="markers+text",
                textposition="top center",
                textfont=dict(size=9, color=C["text"]),
                hoverinfo="text", showlegend=False,
                marker=dict(
                    size=(cont_us["pagerank"] * 900).clip(5, 42),
                    color=C["blue"],
                    opacity=0.75,
                    line=dict(width=0.8, color="#ffffff"),
                ),
                customdata=cont_us["airport"],
            ))
            # Only show labels for top 15
            label_mask = cont_us["airport"].isin(top_airports)
            fig1.data[-1].text = cont_us["airport"].where(label_mask, "").tolist()
            st.plotly_chart(fig1, use_container_width=True)

            # ── Map 2: Delay rate heatmap ───────────────────────────────────────
            st.markdown("<div class='section-title'>Which airports have the worst delays?</div>", unsafe_allow_html=True)
            st.caption("Darker red = higher % of flights delayed >15 min. Hover any state to see the exact rate and which airports are included.")

            # Aggregate delay rate by state
            state_df = cont_us.copy()
            state_df["state"] = state_df["airport"].map(AIRPORT_STATE)
            state_agg = (state_df.dropna(subset=["state"])
                         .groupby("state")
                         .agg(delay_rate=("delay_rate", "mean"),
                              airports=("airport", lambda x: ", ".join(sorted(x))))
                         .reset_index())
            state_agg["hover"] = state_agg.apply(
                lambda r: f"<b>{r['state']}</b><br>Avg delay rate: {r['delay_rate']:.1%}<br>Airports: {r['airports']}", axis=1)

            dr_min = state_agg["delay_rate"].min()
            dr_max = state_agg["delay_rate"].max()

            fig2 = go.Figure(go.Choropleth(
                locations=state_agg["state"],
                z=state_agg["delay_rate"] * 100,
                locationmode="USA-states",
                text=state_agg["hover"],
                hoverinfo="text",
                colorscale="Reds",
                zmin=dr_min * 100,
                zmax=dr_max * 100,
                colorbar=dict(
                    title="Delay Rate (%)",
                    ticksuffix="%",
                    tickfont=dict(color=C["muted"]),
                    thickness=14, len=0.6,
                ),
                marker_line_color="#ffffff", marker_line_width=1.5,
            ))
            fig2.update_layout(
                geo=dict(scope="usa", bgcolor=C["card"], landcolor="#e5e7eb",
                         showlakes=True, lakecolor="#dbeafe",
                         subunitcolor="#ffffff"),
                paper_bgcolor=C["card"],
                font=dict(color=C["text"], family="Inter, Segoe UI, sans-serif"),
                margin=dict(l=0, r=0, t=36, b=0),
                height=450,
                title=dict(text="Average Flight Delay Rate by State", font=dict(size=14, color=C["text"])),
            )
            st.plotly_chart(fig2, use_container_width=True)

            # ── Map 3: Top 25 busiest routes (great-circle curves) ─────────────
            st.markdown("<div class='section-title'>Routes with 3,000+ Flights</div>", unsafe_allow_html=True)
            st.caption("Curved great-circle arcs — thicker & darker = more flights. Hover a route for details.")

            import numpy as np

            def _great_circle_pts(lat1, lon1, lat2, lon2, n=40):
                """Interpolate n points along a great circle arc."""
                lats = np.linspace(lat1, lat2, n)
                lons = np.linspace(lon1, lon2, n)
                # Add a slight arc by lifting the midpoint latitude
                mid = n // 2
                arc_height = abs(lat2 - lat1) * 0.18 + abs(lon2 - lon1) * 0.06
                for i in range(n):
                    t = i / (n - 1)
                    lats[i] += arc_height * 4 * t * (1 - t)
                return lats.tolist(), lons.tolist()

            route_src2 = risk_df if (scores_ready and {"Origin","Dest"}.issubset(risk_df.columns)) else bts_f
            top_route_lines = []
            if not route_src2.empty and {"Origin","Dest"}.issubset(route_src2.columns):
                top_routes_df = (route_src2.groupby(["Origin","Dest"]).size()
                                 .reset_index(name="n").query("n >= 3000")
                                 .sort_values("n", ascending=False))
                coord_set = set(cont_us["airport"])
                for _, r in top_routes_df.iterrows():
                    o_code, d_code = r["Origin"], r["Dest"]
                    if o_code in coord_set and d_code in coord_set:
                        o = AIRPORT_COORDS[o_code]; d = AIRPORT_COORDS[d_code]
                        arc_lats, arc_lons = _great_circle_pts(o[0], o[1], d[0], d[1])
                        top_route_lines.append({
                            "lats": arc_lats, "lons": arc_lons,
                            "n": int(r["n"]),
                            "origin": o_code, "dest": d_code,
                            "label": f"{o_code} → {d_code}: {int(r['n']):,} flights",
                        })

            fig3 = _base_geo("Routes with 3,000+ Flights — hover for details", height=500)
            if top_route_lines:
                max_n = max(l["n"] for l in top_route_lines)
                min_n = min(l["n"] for l in top_route_lines)
                for line in top_route_lines:
                    norm = (line["n"] - min_n) / max(max_n - min_n, 1)
                    width = 1.5 + norm * 5.0
                    alpha = 0.30 + norm * 0.60
                    fig3.add_trace(go.Scattergeo(
                        lon=line["lons"], lat=line["lats"], mode="lines",
                        line=dict(width=width, color=f"rgba(2,132,199,{alpha:.2f})"),
                        hoverinfo="text",
                        text=[line["label"]] * len(line["lats"]),
                        showlegend=False,
                    ))

            # Dots + labels at all endpoint airports
            endpoint_codes = set()
            for l in top_route_lines:
                endpoint_codes.update([l["origin"], l["dest"]])
            ep_df = cont_us[cont_us["airport"].isin(endpoint_codes)].copy()
            fig3.add_trace(go.Scattergeo(
                lon=ep_df["lon"], lat=ep_df["lat"],
                text=ep_df["airport"], mode="markers+text",
                textposition="top center",
                textfont=dict(size=9, color=C["text"]),
                hoverinfo="skip", showlegend=False,
                marker=dict(size=7, color=C["blue"], opacity=0.9,
                            line=dict(width=1, color="#ffffff")),
            ))
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            top20 = cent_df.nlargest(20, "pagerank").sort_values("pagerank")
            fig = px.bar(top20, x="pagerank", y="airport", orientation="h",
                         color="pagerank", color_continuous_scale=["#bae6fd", C["blue"]],
                         labels={"pagerank":"PageRank","airport":"Airport"},
                         title="Top 20 Hubs — PageRank (Hub Importance)")
            fig = sf(fig); fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            top20b = cent_df.nlargest(20, "betweenness").sort_values("betweenness")
            fig = px.bar(top20b, x="betweenness", y="airport", orientation="h",
                         color="betweenness", color_continuous_scale=["#ede9fe","#7c3aed"],
                         labels={"betweenness":"Betweenness","airport":"Airport"},
                         title="Top 20 Hubs — Betweenness (Relay Criticality)")
            fig = sf(fig); fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Build merged — from BTS if available, else use cent_df with dummy delay_rate
        if not bts_f.empty and "Origin" in bts_f.columns:
            ap_d = (bts_f.groupby("Origin")["is_delayed"]
                    .agg(["mean","count"]).query("count >= 300").reset_index())
            ap_d.columns = ["airport","delay_rate","count"]
            merged = ap_d.merge(cent_df, on="airport")
        elif USE_BQ:
            _dl = bq_loader.load_delay_by_airport()
            if _dl is not None and not _dl.empty:
                _dl = _dl.rename(columns={"airport":"airport","delay_rate":"delay_rate","n":"count"})
                merged = _dl.merge(cent_df, on="airport")
            else:
                merged = pd.DataFrame()
        else:
            merged = pd.DataFrame()

        if not merged.empty:
            # ── Airport Risk Matrix ────────────────────────────────────────────
            st.markdown("---")
            st.markdown("<div class='section-title'>Airport Risk Matrix</div>", unsafe_allow_html=True)
            st.caption("Each dot = one airport. Quadrant shows risk category. Bigger bubble = more flights. Hover any dot for details.")

            pr_mid = merged["pagerank"].median()
            dr_mid = merged["delay_rate"].median()

            quad_colors = {
                "High Risk (Important + Delayed)":         C["red"],
                "Unreliable (Low Importance + Delayed)":   C["amber"],
                "Safe Hub (Important + On-Time)":          "#16a34a",
                "Low Priority (Low Importance + On-Time)": "#94a3b8",
            }
            quad_bg = {
                "High Risk (Important + Delayed)":         "rgba(239,68,68,0.06)",
                "Unreliable (Low Importance + Delayed)":   "rgba(245,158,11,0.06)",
                "Safe Hub (Important + On-Time)":          "rgba(22,163,74,0.06)",
                "Low Priority (Low Importance + On-Time)": "rgba(148,163,184,0.04)",
            }

            def _quadrant(row):
                hi_pr = row["pagerank"]   >= pr_mid
                hi_dr = row["delay_rate"] >= dr_mid
                if hi_pr and hi_dr:     return "High Risk (Important + Delayed)"
                if not hi_pr and hi_dr: return "Unreliable (Low Importance + Delayed)"
                if hi_pr and not hi_dr: return "Safe Hub (Important + On-Time)"
                return "Low Priority (Low Importance + On-Time)"

            rm = merged.copy()
            rm["quadrant"] = rm.apply(_quadrant, axis=1)

            # Label only top airports by PageRank or high delay outliers
            label_threshold_pr = rm["pagerank"].quantile(0.80)
            label_threshold_dr = rm["delay_rate"].quantile(0.92)

            fig_rm = go.Figure()

            # Shaded quadrant backgrounds using shapes
            pr_min, pr_max = rm["pagerank"].min(), rm["pagerank"].max()
            dr_min_val, dr_max_val = rm["delay_rate"].min(), rm["delay_rate"].max()
            shapes = []
            quad_regions = [
                (pr_mid,  pr_max,    dr_mid,     dr_max_val, "High Risk (Important + Delayed)"),
                (pr_min,  pr_mid,    dr_mid,     dr_max_val, "Unreliable (Low Importance + Delayed)"),
                (pr_mid,  pr_max,    dr_min_val, dr_mid,     "Safe Hub (Important + On-Time)"),
                (pr_min,  pr_mid,    dr_min_val, dr_mid,     "Low Priority (Low Importance + On-Time)"),
            ]
            for x0, x1, y0, y1, q in quad_regions:
                shapes.append(dict(type="rect", xref="x", yref="y",
                                   x0=x0, x1=x1, y0=y0, y1=y1,
                                   fillcolor=quad_bg[q], line_width=0, layer="below"))

            for quad, color in quad_colors.items():
                sub = rm[rm["quadrant"] == quad]
                if sub.empty:
                    continue
                s = sub["count"].astype(float)
                s_clipped = s.clip(upper=float(s.quantile(0.95)))
                sizes = 8 + 22 * (s_clipped - s_clipped.min()) / (s_clipped.max() - s_clipped.min() + 1)
                show_label = (sub["pagerank"] >= label_threshold_pr) | (sub["delay_rate"] >= label_threshold_dr)
                fig_rm.add_trace(go.Scatter(
                    x=sub["pagerank"], y=sub["delay_rate"],
                    mode="markers+text",
                    name=quad,
                    text=sub["airport"].where(show_label, ""),
                    textposition="top center",
                    textfont=dict(size=9, color=C["text"]),
                    marker=dict(size=sizes, color=color, opacity=0.85,
                                line=dict(width=1, color="#ffffff")),
                    hovertext=sub.apply(
                        lambda r: f"<b>{r['airport']}</b><br>"
                                  f"PageRank: {r['pagerank']:.4f}<br>"
                                  f"Delay rate: {r['delay_rate']:.1%}<br>"
                                  f"Flights: {int(r['count']):,}", axis=1),
                    hoverinfo="text",
                ))

            # Quadrant divider lines
            fig_rm.add_hline(y=dr_mid, line=dict(dash="dash", color="#cbd5e1", width=1.5))
            fig_rm.add_vline(x=pr_mid, line=dict(dash="dash", color="#cbd5e1", width=1.5))

            # Quadrant labels pinned by paper fraction (immune to log scale)
            for txt, qx, qy in [
                ("HIGH RISK",    0.98, 0.98),
                ("SAFE HUB",     0.98, 0.02),
                ("UNRELIABLE",   0.02, 0.98),
                ("LOW PRIORITY", 0.02, 0.02),
            ]:
                fig_rm.add_annotation(
                    xref="paper", yref="paper", x=qx, y=qy,
                    text=txt, showarrow=False,
                    font=dict(size=11, color="#94a3b8"),
                    xanchor="right" if qx > 0.5 else "left",
                    yanchor="top"   if qy > 0.5 else "bottom",
                    opacity=0.6,
                )

            fig_rm.update_yaxes(tickformat=".0%", title="Delay Rate", gridcolor="#f1f5f9")
            fig_rm.update_xaxes(title="PageRank (Network Importance)", type="log", gridcolor="#f1f5f9")
            fig_rm = sf(fig_rm)
            fig_rm.update_layout(
                title="Airport Risk Matrix — hover a dot to see airport details",
                height=540,
                shapes=shapes,
                legend=dict(
                    orientation="h", yanchor="top", y=-0.12,
                    xanchor="center", x=0.5, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)",
                ),
            )
            st.plotly_chart(fig_rm, use_container_width=True)



# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RISK SCORES
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not scores_ready:
        st.markdown(f"""
        <div style='background:{C["card"]};border:1px solid {C["border"]};border-radius:14px;padding:2rem;text-align:center;'>
            <div style='font-size:2.5rem;margin-bottom:.8rem'>⚡</div>
            <div style='font-size:1.1rem;font-weight:600;margin-bottom:.5rem'>Risk Scores Not Generated Yet</div>
            <div style='color:{C["muted"]};margin-bottom:1rem'>Run the scoring script from the project root.</div>
            <code style='background:{C["bg"]};padding:.4rem .9rem;border-radius:6px;font-size:.88rem'>python scripts/generate_risk_scores.py</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:{C["card"]};border:1px solid {C["border"]};border-radius:12px;
                    padding:1.2rem 1.6rem;margin-bottom:1.2rem;'>
            <div style='font-size:0.8rem;font-weight:700;letter-spacing:0.08em;
                        color:{C["muted"]};margin-bottom:0.5rem;'>HOW RISK SCORE IS CALCULATED</div>
            <div style='font-size:0.95rem;color:{C["text"]};line-height:1.7;'>
                Each flight is scored <b>0 → 1</b> by an <b>XGBoost machine learning model</b>
                trained on 4 years of US domestic flight data using flight history, network position,
                schedule, and weather features. A higher score means a higher predicted probability
                of departure delay (&gt;15 min).
            </div>
            <div style='margin-top:0.8rem;font-size:0.88rem;color:{C["muted"]};'>
                Thresholds: &nbsp;
                <span style='color:{C["green"]};font-weight:600;'>● Low &lt; 0.3</span> &nbsp;
                <span style='color:{C["amber"]};font-weight:600;'>● Medium 0.3 – 0.6</span> &nbsp;
                <span style='color:{C["red"]};font-weight:600;'>● High &gt; 0.6</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        total_r = len(risk_f)
        low_n = int((risk_f["risk_score"] < 0.3).sum())
        med_n = int(((risk_f["risk_score"] >= 0.3) & (risk_f["risk_score"] < 0.6)).sum())
        hi_n  = int((risk_f["risk_score"] >= 0.6).sum())

        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card accent-green">
            <div class="sc-label">Low Risk  &lt;30%</div>
            <div class="sc-value">{low_n:,}</div>
            <div class="sc-sub">{low_n/max(total_r,1):.1%} of filtered flights</div>
          </div>
          <div class="stat-card accent-amber">
            <div class="sc-label">Medium Risk  30–60%</div>
            <div class="sc-value">{med_n:,}</div>
            <div class="sc-sub">{med_n/max(total_r,1):.1%} of filtered flights</div>
          </div>
          <div class="stat-card accent-red">
            <div class="sc-label">High Risk  &gt;60%</div>
            <div class="sc-value">{hi_n:,}</div>
            <div class="sc-sub">{hi_n/max(total_r,1):.1%} of filtered flights</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-title'>Risk Score Distribution</div>", unsafe_allow_html=True)
            fig = px.histogram(risk_f, x="risk_score", nbins=70, color_discrete_sequence=[C["blue"]],
                               labels={"risk_score":"Risk Score", "count":"Count"})
            fig.add_vline(x=0.3, line_dash="dot", line_color=C["amber"],
                          annotation_text="Low/Med", annotation_font_color=C["amber"])
            fig.add_vline(x=0.6, line_dash="dot", line_color=C["red"],
                          annotation_text="Med/High", annotation_font_color=C["red"])
            fig.add_vrect(x0=0.6, x1=1.0, fillcolor=C["red"], opacity=0.05, line_width=0)
            fig = sf(fig)
            fig.update_layout(showlegend=False, title="Risk Score Distribution",
                              yaxis_title="Count", xaxis_title="Risk Score")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "Origin" in risk_f.columns:
                st.markdown("<div class='section-title'>Avg Risk by Origin Airport</div>", unsafe_allow_html=True)
                ap_r = (risk_f.groupby("Origin")["risk_score"]
                        .agg(["mean","count"]).query("count >= 50")
                        .sort_values("mean").tail(20).reset_index())
                ap_r.columns = ["airport","avg_risk","count"]
                fig = px.bar(ap_r, x="avg_risk", y="airport", orientation="h",
                             color_discrete_sequence=[C["blue"]],
                             labels={"avg_risk":"Avg Risk Score","airport":"Airport"})
                fig = sf(fig)
                fig.update_layout(showlegend=False, title="Top 20 Airports by Avg Risk Score",
                                  xaxis_title="Avg Risk Score", yaxis_title="Airport")
                st.plotly_chart(fig, use_container_width=True)

        if "scheduled_dep_hour" in risk_f.columns:
            st.markdown("<div class='section-title'>Avg Risk by Departure Hour</div>", unsafe_allow_html=True)
            hr = risk_f.groupby("scheduled_dep_hour")["risk_score"].mean().reset_index()
            hr.columns = ["hour","risk"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hr["hour"], y=hr["risk"], mode="lines+markers",
                line=dict(color=C["red"], width=2.5), marker=dict(size=6, color=C["red"]),
                fill="tozeroy", fillcolor="rgba(220,38,38,.12)",
            ))
            fig.add_hline(y=0.3, line_dash="dot", line_color=C["amber"], opacity=0.5)
            fig.add_hline(y=0.6, line_dash="dot", line_color=C["red"], opacity=0.5)
            fig.update_xaxes(tickvals=list(range(0, 24, 2)))
            fig = sf(fig); fig.update_layout(title="Average Risk by Hour of Day", height=280)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("<div class='section-title'>Top 50 Highest-Risk Flights</div>", unsafe_allow_html=True)
        disp = ["flight_date","airline_code","Origin","Dest","scheduled_dep_hour",
                "risk_score","risk_label","rolling_6_flight_origin_delay_avg",
                "lag_1_tail_arr_delay_mins","wind_speed_knots"]
        disp = [c for c in disp if c in risk_f.columns]
        top50 = risk_f.nlargest(50, "risk_score")[disp].reset_index(drop=True)
        if "flight_date" in top50.columns:
            top50["flight_date"] = top50["flight_date"].dt.strftime("%Y-%m-%d")
        if "risk_score" in top50.columns:
            top50["risk_score"] = top50["risk_score"].round(3)
        st.dataframe(top50.style.background_gradient(subset=["risk_score"], cmap="RdYlGn_r", vmin=0, vmax=1),
                     use_container_width=True, height=420)



# ═══════════════════════════════════════════════════════════════════════════════
# TAB WEATHER — WEATHER IMPACT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_weather:
    if not scores_ready:
        st.info("Weather-delay analysis requires `models/risk_scores.parquet`. "
                "Run `python scripts/generate_risk_scores.py` first.")
    else:
        wx_cols = {"wind_speed_knots", "precipitation_mm", "temperature_c",
                   "cloud_cover_total_pct", "is_delayed"}
        if not wx_cols.issubset(risk_f.columns):
            st.warning("Weather columns missing from risk_scores.parquet — regenerate scores.")
        else:
            st.markdown("<div class='section-title'>How weather drives delay probability</div>",
                        unsafe_allow_html=True)
            st.caption("Based on Open-Meteo / METAR observations joined to flight records in "
                       "the dbt fact table. Each point is a binned mean — bin width chosen to keep "
                       "≥200 flights per bin.")

            wx_df = risk_f[list(wx_cols) + (["risk_score"] if "risk_score" in risk_f.columns else [])].copy()

            def binned_rate(df: pd.DataFrame, col: str, bins) -> pd.DataFrame:
                cut = pd.cut(df[col], bins=bins, include_lowest=True)
                g = df.groupby(cut, observed=True).agg(
                    delay_rate=("is_delayed", "mean"),
                    risk=("risk_score", "mean") if "risk_score" in df.columns else ("is_delayed", "mean"),
                    n=("is_delayed", "count"),
                ).reset_index()
                g["mid"] = g[col].apply(lambda iv: (iv.left + iv.right) / 2 if pd.notna(iv) else np.nan)
                return g[g["n"] >= 200]

            # Row 1 — wind + precip
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='section-title'>Wind Speed vs Delay Rate</div>",
                            unsafe_allow_html=True)
                bins = [0, 3, 6, 9, 12, 15, 18, 22, 26, 30, 40, 60]
                wg = binned_rate(wx_df, "wind_speed_knots", bins)
                if not wg.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=wg["mid"], y=wg["delay_rate"], mode="lines+markers",
                        line=dict(color=C["blue"], width=2.5),
                        marker=dict(size=8 + (wg["n"] / wg["n"].max()) * 14,
                                    color=C["blue"], line=dict(color="#ffffff", width=1)),
                        name="Delay rate",
                        hovertemplate="Wind %{x:.0f} kt<br>Delay %{y:.1%}<br>n=%{customdata:,}<extra></extra>",
                        customdata=wg["n"],
                    ))
                    fig.update_yaxes(tickformat=".0%")
                    fig.update_xaxes(title="Wind speed (knots)")
                    fig = sf(fig)
                    fig.update_layout(title="Delay probability rises with wind speed",
                                      showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("<div class='section-title'>Precipitation vs Delay Rate</div>",
                            unsafe_allow_html=True)
                bins = [-0.01, 0.01, 0.5, 1.5, 3, 5, 8, 12, 20, 40, 100]
                pg = binned_rate(wx_df, "precipitation_mm", bins)
                if not pg.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=pg["mid"], y=pg["delay_rate"], mode="lines+markers",
                        line=dict(color=C["red"], width=2.5),
                        marker=dict(size=8 + (pg["n"] / pg["n"].max()) * 14,
                                    color=C["red"], line=dict(color="#ffffff", width=1)),
                        hovertemplate="Precip %{x:.1f} mm<br>Delay %{y:.1%}<br>n=%{customdata:,}<extra></extra>",
                        customdata=pg["n"],
                    ))
                    fig.update_yaxes(tickformat=".0%")
                    fig.update_xaxes(title="Precipitation (mm/hr)")
                    fig = sf(fig)
                    fig.update_layout(title="Even light rain shifts the delay baseline",
                                      showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            # Row 2 — cloud cover + temperature
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='section-title'>Cloud Cover vs Delay Rate</div>",
                            unsafe_allow_html=True)
                bins = [-0.01, 10, 25, 40, 55, 70, 85, 100]
                cg = binned_rate(wx_df, "cloud_cover_total_pct", bins)
                if not cg.empty:
                    fig = px.bar(cg, x="mid", y="delay_rate",
                                 color="delay_rate",
                                 color_continuous_scale=["#bfdbfe", "#1d4ed8"],
                                 labels={"mid": "Cloud cover (%)", "delay_rate": "Delay rate"})
                    fig.update_yaxes(tickformat=".0%")
                    fig = sf(fig)
                    fig.update_layout(coloraxis_showscale=False,
                                      title="Overcast conditions correlate with higher delay")
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("<div class='section-title'>Temperature vs Delay Rate</div>",
                            unsafe_allow_html=True)
                bins = [-30, -15, -5, 5, 15, 22, 28, 34, 40, 50]
                tg = binned_rate(wx_df, "temperature_c", bins)
                if not tg.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=tg["mid"], y=tg["delay_rate"], mode="lines+markers",
                        line=dict(color=C["amber"], width=2.5),
                        marker=dict(size=10, color=C["amber"], line=dict(color="#ffffff", width=1)),
                        fill="tozeroy", fillcolor="rgba(217,119,6,.12)",
                        hovertemplate="%{x:.0f}°C<br>Delay %{y:.1%}<extra></extra>",
                    ))
                    fig.update_yaxes(tickformat=".0%")
                    fig.update_xaxes(title="Temperature (°C)")
                    fig = sf(fig)
                    fig.update_layout(title="Cold & heat extremes both lift delay rates",
                                      showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            # Hour × Wind heatmap
            if "scheduled_dep_hour" in risk_f.columns:
                st.markdown("---")
                st.markdown("<div class='section-title'>Hour × Wind-speed Heatmap</div>",
                            unsafe_allow_html=True)
                heat_df = risk_f[["scheduled_dep_hour", "wind_speed_knots", "is_delayed"]].copy()
                heat_df["wind_bin"] = pd.cut(
                    heat_df["wind_speed_knots"],
                    bins=[0, 5, 10, 15, 20, 30, 60],
                    labels=["0-5", "5-10", "10-15", "15-20", "20-30", "30+"],
                    include_lowest=True,
                )
                piv = (heat_df.groupby(["wind_bin", "scheduled_dep_hour"], observed=True)["is_delayed"]
                       .mean().unstack("scheduled_dep_hour"))
                piv = piv.reindex(columns=list(range(0, 24)))
                fig = px.imshow(
                    piv, aspect="auto",
                    color_continuous_scale=["#ffffff", "#fca5a5", "#dc2626"],
                    labels=dict(x="Departure hour", y="Wind speed (kt)", color="Delay rate"),
                    text_auto=".0%",
                )
                fig.update_xaxes(tickvals=list(range(0, 24, 2)))
                fig = sf(fig)
                fig.update_layout(title="Delay rate by hour and wind band "
                                        "— evening + high wind = worst",
                                  height=340)
                fig.update_traces(textfont=dict(size=9, color="#111111"))
                st.plotly_chart(fig, use_container_width=True)

            # Weather feature importance callout
            st.markdown("---")
            artifacts = load_prediction_artifacts()
            if artifacts is not None:
                m = artifacts["model"]
                fcols = artifacts["feat_cols"]
                fi = pd.DataFrame({"feature": fcols, "importance": m.feature_importances_})
                weather_feats = ["wind_speed_knots", "precipitation_mm", "temperature_c",
                                  "cloud_cover_total_pct", "cloud_cover_low_pct",
                                  "wind_speed_delta", "rolling_3_flight_weather_delay_flag"]
                w_fi = fi[fi["feature"].isin(weather_feats)].sort_values("importance")
                label_map = {
                    "rolling_3_flight_weather_delay_flag": "Recent weather delay (rolling 3-flight flag)",
                    "temperature_c":                       "Temperature (°C)",
                    "precipitation_mm":                    "Precipitation (mm)",
                    "cloud_cover_total_pct":               "Total cloud cover (%)",
                    "wind_speed_knots":                    "Wind speed (knots)",
                    "cloud_cover_low_pct":                 "Low cloud cover (%)",
                    "wind_speed_delta":                    "Wind speed change",
                }
                w_fi = w_fi.copy()
                w_fi["feature"] = w_fi["feature"].map(label_map).fillna(w_fi["feature"])
                total_weather = w_fi["importance"].sum()
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"""
                    <div class="stat-card accent-blue">
                      <div class="sc-label">Weather Features · Combined Importance</div>
                      <div class="sc-value">{total_weather:.1%}</div>
                      <div class="sc-sub">Share of Gradient Boosting<br>feature importance</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    fig = px.bar(w_fi, x="importance", y="feature", orientation="h",
                                 color="importance",
                                 color_continuous_scale=["#bae6fd", C["blue"]],
                                 labels={"importance": "Importance", "feature": ""})
                    fig = sf(fig)
                    fig.update_layout(coloraxis_showscale=False,
                                      title="Weather-related feature importances")
                    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB CANCEL — CANCELLATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_cancel:
    st.markdown(
        f"<p style='color:{C['muted']};font-size:.85rem'>"
        "Cancellations are more severe than delays — passengers are stranded, not just inconvenienced. "
        "This tab breaks down who cancels, where, why, and when.</p>",
        unsafe_allow_html=True,
    )

    # ── Load cancellation data ─────────────────────────────────────────────────
    @st.cache_data(ttl=3600, show_spinner=False)
    def _cancel_airline(m, ap, dr):   return bq_loader.load_cancellations_by_airline(m or None, ap or None, dr)
    @st.cache_data(ttl=3600, show_spinner=False)
    def _cancel_airport(m, al, dr):   return bq_loader.load_cancellations_by_airport(m or None, al or None, dr)
    @st.cache_data(ttl=3600, show_spinner=False)
    def _cancel_codes(m, al, ap, dr): return bq_loader.load_cancellation_codes(m or None, al or None, ap or None, dr)
    @st.cache_data(ttl=3600, show_spinner=False)
    def _cancel_trend(al, ap, dr, m): return bq_loader.load_cancellation_trend(al or None, ap or None, dr, m or None)

    fk = (tuple(sel_months), tuple(sel_airlines), tuple(sel_airports), bq_date_range())

    CODE_LABELS = {"A": "Carrier", "B": "Weather", "C": "NAS / ATC", "D": "Security"}

    if USE_BQ:
        can_airline = _cancel_airline(tuple(sel_months), tuple(sel_airports), bq_date_range())
        can_airport = _cancel_airport(tuple(sel_months), tuple(sel_airlines), bq_date_range())
        can_codes   = _cancel_codes(*fk)
        can_trend   = _cancel_trend(tuple(sel_airlines), tuple(sel_airports), bq_date_range(), tuple(sel_months))

        total_cancelled = int(can_airline["cancelled"].sum()) if can_airline is not None else 0
        total_flights_c = int(can_airline["total"].sum())    if can_airline is not None else 0
        overall_rate    = total_cancelled / total_flights_c  if total_flights_c else 0
        top_code = (can_codes.iloc[0]["code"] if can_codes is not None and not can_codes.empty else "?")
        top_code_label = CODE_LABELS.get(top_code, top_code)
    else:
        # local BTS fallback — reload raw parquet including cancelled rows
        @st.cache_data(show_spinner=False)
        def _local_bts_all():
            frames = []
            for f in ["bts_2024_1.parquet", "bts_2024_7.parquet", "bts_2024_10.parquet"]:
                p = CACHE / f
                if p.exists():
                    frames.append(pd.read_parquet(p))
            if not frames:
                return pd.DataFrame()
            df = pd.concat(frames, ignore_index=True)
            df["flight_month"] = df["Month"].astype(int)
            df["airline_code"] = df["Reporting_Airline"]
            return df

        raw_all = _local_bts_all()

        if raw_all.empty:
            can_airline = can_airport = can_codes = can_trend = None
            total_cancelled = total_flights_c = 0
            overall_rate = 0.0
            top_code_label = "?"
        else:
            can_airline = (raw_all.groupby("airline_code")
                           .apply(lambda g: pd.Series({
                               "cancelled":   (g["Cancelled"] == 1.0).sum(),
                               "total":       len(g),
                               "cancel_rate": (g["Cancelled"] == 1.0).mean(),
                           })).query("total >= 100").sort_values("cancel_rate", ascending=False)
                           .reset_index())
            can_airport = (raw_all.groupby("Origin")
                           .apply(lambda g: pd.Series({
                               "cancelled":   (g["Cancelled"] == 1.0).sum(),
                               "total":       len(g),
                               "cancel_rate": (g["Cancelled"] == 1.0).mean(),
                           })).query("total >= 100").sort_values("cancel_rate", ascending=False)
                           .head(20).reset_index().rename(columns={"Origin": "airport"}))
            if "CancellationCode" in raw_all.columns:
                can_codes = (raw_all[raw_all["Cancelled"] == 1.0]["CancellationCode"]
                             .dropna().value_counts().reset_index())
                can_codes.columns = ["code", "n"]
            else:
                can_codes = None
            can_trend = None
            total_cancelled = int((raw_all["Cancelled"] == 1.0).sum())
            total_flights_c = len(raw_all)
            overall_rate    = total_cancelled / total_flights_c if total_flights_c else 0
            top_code = (can_codes.iloc[0]["code"] if can_codes is not None and not can_codes.empty else "?")
            top_code_label = CODE_LABELS.get(top_code, top_code)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card accent-red">
        <div class="sc-label">Total Cancellations</div>
        <div class="sc-value">{total_cancelled:,}</div>
        <div class="sc-sub">{_dr_label}</div>
      </div>
      <div class="stat-card accent-amber">
        <div class="sc-label">Cancellation Rate</div>
        <div class="sc-value">{overall_rate:.2%}</div>
        <div class="sc-sub">Share of all scheduled flights</div>
      </div>
      <div class="stat-card accent-blue">
        <div class="sc-label">Most Common Cause</div>
        <div class="sc-value" style="font-size:1.3rem">{top_code_label}</div>
        <div class="sc-sub">Primary cancellation driver</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Row 1 — by airline + cancellation codes ───────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Cancellation Rate by Airline</div>", unsafe_allow_html=True)
        if can_airline is not None and not can_airline.empty:
            top_al = can_airline.sort_values("cancel_rate").tail(15)
            y_col = "airline_code" if "airline_code" in top_al.columns else top_al.columns[0]
            fig = px.bar(top_al, x="cancel_rate", y=y_col, orientation="h",
                         color_discrete_sequence=[C["blue"]],
                         labels={"cancel_rate": "Cancellation Rate", y_col: "Airline"},
                         title="Which airlines cancel most often")
            fig.update_xaxes(tickformat=".1%")
            fig = sf(fig)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Why Are Flights Cancelled?</div>", unsafe_allow_html=True)
        if can_codes is not None and not can_codes.empty:
            can_codes["label"] = can_codes["code"].map(CODE_LABELS).fillna(can_codes["code"])
            fig = px.pie(can_codes, values="n", names="label",
                         color_discrete_sequence=[C["blue"], C["red"], C["amber"], C["green"]],
                         hole=0.45,
                         title="Cancellation cause breakdown (A/B/C/D codes)")
            fig.update_traces(textposition="outside", textinfo="percent+label",
                              textfont=dict(color=C["text"]))
            fig = sf(fig)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 2 — by airport + trend ────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Cancellation Rate by Airport</div>", unsafe_allow_html=True)
        if can_airport is not None and not can_airport.empty:
            top_ap = can_airport.sort_values("cancel_rate").tail(15)
            y_col = "airport" if "airport" in top_ap.columns else top_ap.columns[0]
            fig = px.bar(top_ap, x="cancel_rate", y=y_col, orientation="h",
                         color_discrete_sequence=[C["blue"]],
                         labels={"cancel_rate": "Cancellation Rate", y_col: "Airport"},
                         title="Which origin airports cancel most often")
            fig.update_xaxes(tickformat=".1%")
            fig = sf(fig)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Cancellation Rate Over Time</div>", unsafe_allow_html=True)
        if can_trend is not None and not can_trend.empty and "label" in can_trend.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=can_trend["label"], y=can_trend["cancel_rate"],
                mode="lines+markers",
                line=dict(color=C["red"], width=2.5),
                marker=dict(size=6, color=C["red"]),
                fill="tozeroy", fillcolor="rgba(220,38,38,.12)",
                hovertemplate="%{x}: %{y:.2%}<extra></extra>",
            ))
            fig.update_yaxes(tickformat=".1%")
            fig = sf(fig)
            fig.update_layout(title="Monthly cancellation rate Feb 2022 – Jan 2026")
            st.plotly_chart(fig, use_container_width=True)
        elif not USE_BQ:
            st.caption("Trend requires BigQuery connection — only 3 months available locally.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Model Comparison</div>", unsafe_allow_html=True)
        res = pd.DataFrame({
            "Model":    ["Logistic Regression","Random Forest","Gradient Boosting"],
            "Accuracy": [0.6633, 0.6926, 0.8024],
            "F1 Score": [0.4297, 0.4608, 0.2676],
            "ROC-AUC":  [0.6954, 0.7271, 0.7278],
        })
        st.dataframe(res.style.highlight_max(subset=["Accuracy","F1 Score","ROC-AUC"],
                                              color="#bbf7d0"),
                     use_container_width=True, hide_index=True)
        fig = go.Figure()
        for i, row in res.iterrows():
            fig.add_trace(go.Bar(
                name=row["Model"],
                x=["Accuracy","F1 Score","ROC-AUC"],
                y=[row["Accuracy"],row["F1 Score"],row["ROC-AUC"]],
                text=[f"{v:.3f}" for v in [row["Accuracy"],row["F1 Score"],row["ROC-AUC"]]],
                textposition="outside", marker_color=COLORS[i], marker_line_width=0,
            ))
        fig = sf(fig)
        fig.update_layout(barmode="group", title="Metrics Side-by-Side",
                          yaxis=dict(range=[0,1.08], **PL["yaxis"]),
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Feature Importance · Gradient Boosting</div>", unsafe_allow_html=True)
        fi = pd.DataFrame({
            "feature": ["rolling_6_flight_origin_delay_avg","lag_1_tail_arr_delay_mins",
                        "scheduled_dep_hour","origin_betweenness","rolling_3_flight_weather_delay_flag",
                        "wind_speed_delta","Distance","origin_pagerank","dest_pagerank",
                        "wind_speed_knots","cloud_cover_total_pct","temperature_c",
                        "flight_month","day_of_week","precipitation_mm","cloud_cover_low_pct"],
            "importance": [0.285,0.198,0.142,0.087,0.071,0.052,0.038,
                           0.032,0.028,0.021,0.018,0.015,0.012,0.009,0.007,0.005],
        }).sort_values("importance")
        src = "Approximate values from Week 2 notebook"
        if (MODELS_DIR / "gradient_boosting.joblib").exists():
            try:
                m = joblib.load(MODELS_DIR / "gradient_boosting.joblib")
                with open(MODELS_DIR / "feature_cols.json") as fh: fcols = json.load(fh)
                fi = pd.DataFrame({"feature": fcols, "importance": m.feature_importances_}).sort_values("importance")
                src = "Live values from trained Gradient Boosting model"
            except Exception: pass
        st.caption(src)
        feat_label_map = {
            "lag_1_tail_arr_delay_mins":          "Prev. arrival delay (mins)",
            "scheduled_dep_hour":                 "Departure hour",
            "rolling_6_flight_origin_delay_avg":  "Origin delay avg (last 6 flights)",
            "origin_pagerank":                    "Origin airport PageRank",
            "dest_pagerank":                      "Destination airport PageRank",
            "Distance":                           "Flight distance",
            "origin_betweenness":                 "Origin betweenness centrality",
            "rolling_3_flight_weather_delay_flag":"Recent weather delay flag",
            "flight_month":                       "Month of year",
            "temperature_c":                      "Temperature (°C)",
            "day_of_week":                        "Day of week",
            "precipitation_mm":                   "Precipitation (mm)",
            "cloud_cover_total_pct":              "Total cloud cover (%)",
            "wind_speed_knots":                   "Wind speed (knots)",
            "cloud_cover_low_pct":                "Low cloud cover (%)",
            "wind_speed_delta":                   "Wind speed change",
        }
        fi = fi.copy()
        fi["feature"] = fi["feature"].map(feat_label_map).fillna(fi["feature"])
        fig = px.bar(fi, x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale=["#bae6fd", C["blue"]],
                     labels={"importance":"Importance","feature":""})
        fig = sf(fig)
        fig.update_layout(coloraxis_showscale=False, height=490,
                          title="Feature Importance — Gradient Boosting Model")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>Evaluation Charts — Week 2 Analysis</div>", unsafe_allow_html=True)
    eval_imgs = [("roc_curves.png","ROC Curves"),("confusion_matrices.png","Confusion Matrices"),
                 ("feature_importance.png","Feature Importance"),("model_comparison.png","Model Comparison")]
    for i in range(0, len(eval_imgs), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j < len(eval_imgs):
                fname, caption = eval_imgs[i+j]
                p = CACHE / fname
                if p.exists(): col.image(str(p), caption=caption, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB PREDICT — WHAT-IF FLIGHT RISK SCORE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    artifacts = load_prediction_artifacts()
    if artifacts is None:
        st.markdown(f"""
        <div style='background:{C["card"]};border:1px solid {C["border"]};border-radius:14px;padding:2rem;text-align:center;'>
            <div style='font-size:2.5rem;margin-bottom:.8rem'>🎯</div>
            <div style='font-size:1.1rem;font-weight:600;margin-bottom:.5rem'>Prediction model not found</div>
            <div style='color:{C["muted"]};margin-bottom:1rem'>
                Expected <code>gradient_boosting.joblib</code>, <code>feature_cols.json</code>,
                and <code>feature_medians.json</code> under <code>models/</code>.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        model     = artifacts["model"]
        feat_cols = artifacts["feat_cols"]
        medians   = artifacts["medians"]

        airport_options = (sorted(cent_df["airport"].tolist())
                           if not cent_df.empty else
                           sorted(AIRPORT_COORDS.keys()))
        cent_lookup = cent_df.set_index("airport") if not cent_df.empty else None

        st.markdown("<div class='section-title'>Enter flight details to get a disruption risk score</div>",
                    unsafe_allow_html=True)
        st.caption("Fields are pre-filled with typical values — change only what you care about. "
                   "Flight distance is computed automatically from the selected airports.")

        def _haversine_miles(a: str, b: str) -> float:
            from math import radians, sin, cos, asin, sqrt
            if a not in AIRPORT_COORDS or b not in AIRPORT_COORDS:
                return float(medians.get("Distance", 679))
            lat1, lon1 = AIRPORT_COORDS[a]
            lat2, lon2 = AIRPORT_COORDS[b]
            dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
            h = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
            return 2 * 3958.8 * asin(sqrt(h))

        with st.form("predict_form"):
            # ── Row 1 — schedule ──
            st.markdown("<div class='section-title'>Schedule</div>", unsafe_allow_html=True)
            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                in_month = st.selectbox(
                    "Month", list(range(1, 13)),
                    index=int(medians.get("flight_month", 7)) - 1,
                    format_func=lambda m: pd.Timestamp(2024, m, 1).strftime("%B"),
                )
            with r1c2:
                dow_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                in_dow = st.selectbox(
                    "Day of week", list(range(7)),
                    index=int(medians.get("day_of_week", 3)) % 7,
                    format_func=lambda d: dow_labels[d],
                )
            with r1c3:
                in_hour = st.number_input(
                    "Departure hour (0–23)", min_value=0, max_value=23, step=1,
                    value=int(medians.get("scheduled_dep_hour", 13)),
                )

            # ── Row 2 — route ──
            st.markdown("<div class='section-title'>Route</div>", unsafe_allow_html=True)
            r2c1, r2c2 = st.columns([1, 1], gap="medium")
            default_o = "ATL" if "ATL" in airport_options else airport_options[0]
            default_d = "LAX" if "LAX" in airport_options else airport_options[min(1, len(airport_options) - 1)]
            with r2c1:
                in_origin = st.selectbox("Departing from", airport_options,
                                           index=airport_options.index(default_o))
            with r2c2:
                in_dest = st.selectbox("Flying to", airport_options,
                                         index=airport_options.index(default_d))

            # ── Row 3 — weather ──
            st.markdown("<div class='section-title'>Weather conditions</div>",
                        unsafe_allow_html=True)
            r3c1, r3c2, r3c3 = st.columns(3)
            with r3c1:
                in_temp = st.number_input(
                    "Temperature (°C)", min_value=-40.0, max_value=50.0, step=0.5,
                    value=float(medians.get("temperature_c", 24.5)),
                )
                in_precip = st.number_input(
                    "Rainfall (mm per hour)", min_value=0.0, max_value=50.0, step=0.1,
                    value=float(medians.get("precipitation_mm", 0.0)),
                )
            with r3c2:
                in_wind = st.number_input(
                    "Wind speed (knots)", min_value=0.0, max_value=80.0, step=0.5,
                    value=float(medians.get("wind_speed_knots", 5.0)),
                )
                in_wind_delta = st.number_input(
                    "Wind change vs. last hour (knots)", min_value=-40.0, max_value=40.0, step=0.5,
                    value=float(medians.get("wind_speed_delta", 0.0)),
                    help="Positive = wind picking up; negative = easing off.",
                )
            with r3c3:
                in_cloud_total = st.number_input(
                    "Total cloud cover (%)", min_value=0, max_value=100, step=5,
                    value=int(medians.get("cloud_cover_total_pct", 22)),
                )
                in_cloud_low = st.number_input(
                    "Low-altitude cloud cover (%)", min_value=0, max_value=100, step=5,
                    value=int(medians.get("cloud_cover_low_pct", 0)),
                    help="Clouds below ~6,500ft — most relevant for takeoff/landing visibility.",
                )

            # ── Row 4 — operational state ──
            st.markdown("<div class='section-title'>Airport &amp; aircraft status</div>", unsafe_allow_html=True)
            r4c1, r4c2, r4c3 = st.columns(3)
            with r4c1:
                in_rolling6 = st.number_input(
                    "Avg delay of last 6 flights from this airport (min)",
                    min_value=-30.0, max_value=180.0, step=1.0,
                    value=float(medians.get("rolling_6_flight_origin_delay_avg", 4.0)),
                    help="How behind schedule the airport is running right now.",
                )
            with r4c2:
                in_lag1 = st.number_input(
                    "This aircraft's previous arrival delay (min)",
                    min_value=-60.0, max_value=240.0, step=1.0,
                    value=float(medians.get("lag_1_tail_arr_delay_mins", -6.0)),
                    help="If the plane arrived late on its previous leg, it usually departs late too.",
                )
            with r4c3:
                st.write("")  # vertical spacing
                in_wx_flag = st.checkbox(
                    "Recent flights disrupted by weather",
                    value=bool(int(medians.get("rolling_3_flight_weather_delay_flag", 0))),
                )

            submitted = st.form_submit_button("Compute Risk Score", type="primary")

        # ── Prediction ──
        if submitted:
          try:
            def _get_cent(ap: str, col: str) -> float:
                if cent_lookup is not None and ap in cent_lookup.index:
                    return float(cent_lookup.loc[ap, col])
                return 0.0

            in_dist = _haversine_miles(in_origin, in_dest)

            feat_values = {
                "flight_month":                       int(in_month),
                "day_of_week":                        int(in_dow),
                "scheduled_dep_hour":                 int(in_hour),
                "Distance":                           float(in_dist),
                "temperature_c":                      float(in_temp),
                "precipitation_mm":                   float(in_precip),
                "wind_speed_knots":                   float(in_wind),
                "cloud_cover_total_pct":              float(in_cloud_total),
                "cloud_cover_low_pct":                float(in_cloud_low),
                "rolling_6_flight_origin_delay_avg":  float(in_rolling6),
                "rolling_3_flight_weather_delay_flag": 1.0 if in_wx_flag else 0.0,
                "wind_speed_delta":                   float(in_wind_delta),
                "lag_1_tail_arr_delay_mins":          float(in_lag1),
                "origin_pagerank":                    _get_cent(in_origin, "pagerank"),
                "origin_betweenness":                 _get_cent(in_origin, "betweenness"),
                "dest_pagerank":                      _get_cent(in_dest, "pagerank"),
            }

            X_row = pd.DataFrame(
                [[float(feat_values.get(c, medians.get(c, 0))) for c in feat_cols]],
                columns=feat_cols, dtype=float,
            )
            prob = float(model.predict_proba(X_row.values)[0, 1])

            if prob < 0.3:
                band, band_color, band_icon = "Low", C["green"], "✓"
            elif prob < 0.6:
                band, band_color, band_icon = "Medium", C["amber"], "⚠"
            else:
                band, band_color, band_icon = "High", C["red"], "✕"

            st.markdown("---")
            st.markdown("<div class='section-title'>Predicted Disruption Risk</div>", unsafe_allow_html=True)

            k1, k2, k3 = st.columns([1.2, 1, 1])
            with k1:
                st.markdown(f"""
                <div class="stat-card" style="border-top:3px solid {band_color}">
                  <div class="sc-label">Risk Score · {in_origin} → {in_dest}</div>
                  <div class="sc-value" style="color:{band_color}">{prob:.1%}</div>
                  <div class="sc-sub">{band_icon} <b style="color:{band_color}">{band} Risk</b>
                    · P(delay &gt; 15 min)</div>
                </div>
                """, unsafe_allow_html=True)
            with k2:
                st.markdown(f"""
                <div class="stat-card accent-blue">
                  <div class="sc-label">Origin Hub Score</div>
                  <div class="sc-value">{feat_values['origin_pagerank']:.4f}</div>
                  <div class="sc-sub">PageRank · betweenness {feat_values['origin_betweenness']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            with k3:
                st.markdown(f"""
                <div class="stat-card accent-amber">
                  <div class="sc-label">Destination Hub Score</div>
                  <div class="sc-value">{feat_values['dest_pagerank']:.4f}</div>
                  <div class="sc-sub">PageRank (dest congestion proxy)</div>
                </div>
                """, unsafe_allow_html=True)

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"color": C["text"], "size": 34}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": C["muted"], "tickfont": {"color": C["muted"]}},
                    "bar": {"color": band_color, "thickness": 0.28},
                    "bgcolor": C["card"],
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30],  "color": "rgba(5,150,105,.18)"},
                        {"range": [30, 60], "color": "rgba(217,119,6,.18)"},
                        {"range": [60, 100], "color": "rgba(220,38,38,.18)"},
                    ],
                    "threshold": {"line": {"color": C["text"], "width": 2},
                                   "thickness": 0.75, "value": prob * 100},
                },
            ))
            fig.update_layout(paper_bgcolor=C["card"], font={"color": C["text"]},
                              height=260, margin=dict(l=30, r=30, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Driver breakdown — deviation × global importance
            st.markdown("<div class='section-title'>Top contributing factors</div>",
                        unsafe_allow_html=True)
            st.caption("Approximate attribution: normalized deviation from dataset median × "
                       "global Gradient Boosting feature importance. Red = pushes risk up, "
                       "green = pushes risk down. Not a SHAP explanation.")

            importances = model.feature_importances_
            rows = []
            for i, f in enumerate(feat_cols):
                val   = float(X_row.iloc[0, i])
                med   = float(medians.get(f, val))
                denom = max(abs(med), 1.0)
                dev   = (val - med) / denom  # signed normalized deviation
                contrib = dev * float(importances[i])
                rows.append({"feature": f, "value": val, "median": med,
                              "importance": float(importances[i]),
                              "contribution": contrib})
            drv = pd.DataFrame(rows)
            drv["abs_contrib"] = drv["contribution"].abs()
            drv = drv.nlargest(10, "abs_contrib").sort_values("contribution")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=drv["contribution"], y=drv["feature"], orientation="h",
                marker=dict(color=[C["red"] if c > 0 else C["green"] for c in drv["contribution"]]),
                customdata=drv[["value", "median", "importance"]].values,
                hovertemplate="<b>%{y}</b><br>Input: %{customdata[0]:.2f}"
                              "<br>Median: %{customdata[1]:.2f}"
                              "<br>Importance: %{customdata[2]:.3f}"
                              "<br>Contribution: %{x:+.4f}<extra></extra>",
            ))
            fig.add_vline(x=0, line_color=C["border"], line_width=1)
            fig = sf(fig)
            fig.update_layout(title="Top 10 drivers (red = increases risk · green = decreases)",
                              xaxis=dict(title="Signed contribution (dev × importance)", **PL["xaxis"]),
                              height=420)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Full input feature vector"):
                show = pd.DataFrame({
                    "feature":  feat_cols,
                    "input":    [float(X_row.iloc[0][c]) for c in feat_cols],
                    "median":   [float(medians.get(c, 0)) for c in feat_cols],
                    "importance": [float(v) for v in importances],
                })
                st.dataframe(
                    show, use_container_width=True, hide_index=True,
                    column_config={
                        "input":      st.column_config.NumberColumn(format="%.3f"),
                        "median":     st.column_config.NumberColumn(format="%.3f"),
                        "importance": st.column_config.NumberColumn(format="%.4f"),
                    },
                )
          except Exception as e:
            st.error("Prediction failed — details below.")
            st.exception(e)
