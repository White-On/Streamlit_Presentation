import streamlit as st
import numpy as np

from pages.hypothesis import write_hypothesis
from pages.reward_function import write_reward_function
from pages.scenarios import write_scenarios

img_path = "img/"

st.set_page_config(
    page_title="Learning to cross a crowd with an autonomous vehicle",
    page_icon="🤖",
    layout="wide",
)


# Sidebar
st.sidebar.page_link("pages/archive.py", label="Archive", icon="🗃️")


st.sidebar.markdown("## 🗺️ Navigation")
st.sidebar.markdown("[🔬 Hypothesis](#hypothesis)")
st.sidebar.markdown("[📦 Reward function](#reward-function)")
st.sidebar.markdown("[📝 Variables name recap](#variables-name-recap)")
st.sidebar.markdown("[🎨 Scenarios](#scenario)")
st.sidebar.markdown("[📚 References](#reference)")

st.sidebar.markdown("## 🌿 Random Seed Controls")
if st.sidebar.toggle(
    "Random Seed", value=False, help="Toggle and random seeds will be used"
):
    np.random.seed(None)
else:
    np.random.seed(42)
st.sidebar.button("Regenerate Results")


# Problem/Context
st.title("Problem/Context")
st.markdown(
    "We Consider a shared space scenario where autonomous vehicles navigate among \
    pedestrians in a simulation. Given a predefined path leading to a goal destination within this shared space, the \
    autonomous vehicle is tasked with adapting its driving behavior to ensure safety [[1]](#reference), avoid collisions with pedestrian, \
    minimize disturbance to pedestrians and social groups, maintain smooth maneuvering[[3]](#reference), and adhere to the \
    given path as closely as possible[[5]](#reference). The objective is *not merely to navigate around the crowd*, but rather \
    to follow the specified path while dynamically adjusting its trajectory based on real-time situational \
    awareness and environmental factors."
)

# Hypothesis
write_hypothesis(img_path=img_path)

# Reward function
write_reward_function(img_path=img_path)

# Senario
write_scenarios(img_path=img_path)

st.header("📚 References", anchor="reference")
st.markdown(
    "[1] Salvini, Pericle, Diego Paez-Granados, et Aude Billard. 2022. « Safety \
            Concerns Emerging from Robots Navigating in Crowded Pedestrian Areas ». International \
            Journal of Social Robotics 14 (2): 441‑62. https://doi.org/10.1007/s12369-021-00796-4."
)
st.markdown(
    "[2] Kabtoul, Maria, Anne Spalanzani, et Philippe Martinet. 2022. « Proactive And Smooth Maneuvering \
            For Navigation Around Pedestrians ». In 2022 International Conference on Robotics and Automation (ICRA), \
            4723‑29. Philadelphia, PA, USA: IEEE. https://doi.org/10.1109/ICRA46639.2022.9812255."
)
st.markdown(
    "[3] Kabtoul, Maria, Manon Prédhumeau, Anne Spalanzani, Julie Dugdale, et Philippe Martinet. 2024. \
            « How To Evaluate the Navigation of Autonomous Vehicles Around Pedestrians? » IEEE Transactions on \
            Intelligent Transportation Systems, 1‑11. https://doi.org/10.1109/TITS.2023.3323662."
)
st.markdown(
    "[4] Alia, Chebly, Tagne Gilles, Talj Reine, et Charara Ali. 2015. « Local Trajectory Planning and \
            Tracking of Autonomous Vehicles, Using Clothoid Tentacles Method ». In 2015 IEEE Intelligent Vehicles \
            Symposium (IV), 674‑79. Seoul, South Korea: IEEE. https://doi.org/10.1109/IVS.2015.7225762."
)
st.markdown(
    "[5] https://medium.com/coinmonks/how-robots-follow-routes-pid-control-2a74226c5c99"
)
st.markdown(
    "[6] Mouhagir, Hafida, Veronique Cherfaoui, Reine Talj, Francois Aioun, et Franck Guillemard. 2017. \
            « Using Evidential Occupancy Grid for Vehicle Trajectory Planning under Uncertainty with Tentacles ». \
            In 2017 IEEE 20th International Conference on Intelligent Transportation Systems (ITSC), 1‑7. Yokohama: \
            IEEE. https://doi.org/10.1109/ITSC.2017.8317808."
)
st.markdown(
    "[7] Genevois, Thomas, Anne Spalanzani, et Christian Laugier. 2023. \
            « Interaction-Aware Predicoverfittingtive Collision Detector for Human-Aware Collision Avoidance ». \
            In 2023 IEEE Intelligent Vehicles Symposium (IV), 1‑7. Anchorage, AK, USA: \
            IEEE. https://doi.org/10.1109/IV55152.2023.10186778."
)
st.markdown(
    "[8] Alia, Chebly, Tagne Gilles, Talj Reine, et Charara Ali. 2015. « Local Trajectory Planning \
            and Tracking of Autonomous Vehicles, Using Clothoid Tentacles Method ». In 2015 IEEE Intelligent \
            Vehicles Symposium (IV), 674‑79. Seoul, South Korea: IEEE. https://doi.org/10.1109/IVS.2015.7225762."
)
st.markdown(
    "[9] https://en.wikipedia.org/wiki/Preferred_walking_speed#cite_note-Browning2006-1"
)
st.markdown(
    "[10] Reynolds, T.R. (1987), Stride length and its determinants in \
            humans, early hominids, primates, and mammals. Am. J. Phys. Anthropol., 72: \
            101-115. https://doi.org/10.1002/ajpa.1330720113"
)
st.markdown(
    "[11] Renault Zoe's technical sheet https://www.renault.fr/vehicules-electriques/zoe/fiche-technique.html"
)
st.markdown(
    "[12] Deshpande, Niranjan, Dominique Vaufreydaz, et Anne Spalanzani. 2021. « Navigation in Urban Environments amongst Pedestrians Using Multi-Objective Deep Reinforcement Learning ». In 2021 IEEE International Intelligent Transportation Systems Conference (ITSC), 923‑28. Indianapolis, IN, USA: IEEE. https://doi.org/10.1109/ITSC48978.2021.9564601."
)
st.markdown(
    "[13] Deshpande, Niranjan, et Anne Spalanzani. 2019. « Deep Reinforcement Learning Based Vehicle Navigation amongst Pedestrians Using a Grid-Based State Representation ». In 2019 IEEE Intelligent Transportation Systems Conference (ITSC), 2081‑86. Auckland, New Zealand: IEEE. https://doi.org/10.1109/ITSC.2019.8917299."
)
st.markdown(
    "[14] Everett, Michael, Yu Fan Chen, et Jonathan P. How. 2021. « Collision Avoidance in Pedestrian-Rich Environments with Deep Reinforcement Learning ». IEEE Access 9: 10357‑77. https://doi.org/10.1109/ACCESS.2021.3050338."
)
