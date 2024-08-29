import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def write_scenarios(img_path) -> None:
    st.header("🎨 Scenario", anchor="scenario")
    st.markdown(
        "To conduct our experiments effectively, we are creating five scenarios to \
                observe the results of our model in different dynamic environments. Most of \
                these scenarios are inspired by the article *[Kabtoul, Maria, Manon Prédhumeau, \
                Anne Spalanzani, Julie Dugdale, et Philippe Martinet. « How To Evaluate the \
                Navigation of Autonomous Vehicles Around Pedestrians? » IEEE Transactions on \
                Intelligent Transportation Systems, 2024, 1‑11. \
                https://doi.org/10.1109/TITS.2023.3323662]* and video that follow. \n\
    1. **Opposite Direction Collision**: In this scenario, the vehicle collides with another \
                vehicle moving in the opposite direction. \n\
    2. **Same Direction Collision**: Similar to the first scenario, but both vehicles have \
                the same goal direction. However, as the car will have a faster speed goal, \
                we expect different behavior.\n\
    3. **Lateral Encounter with Crowd**: This scenario \
                is well represented in the video, involving a lateral \
                encounter between the vehicle and a crowd of pedestrians.\n\
    4. **Encounter with Two Pedestrian Crowds**: In this more complex scenario, \
                the vehicle encounters two pedestrian crowds, and the car must navigate \
                through them.\n\
    5. **Shared Space with Random Pedestrian Movements**: The last scenario represents \
        a shared space with pedestrians moving randomly around. The car needs to navigate \
            in this chaotic environment. \n\n\
    By testing our model in these diverse scenarios, we aim to gain insights into \
                its performance and behavior in various dynamic situations and avoid **overfitting**."
    )

    senario_img = [f"{img_path}scenario{i}.png" for i in range(1, 7)]
    caption = [
        "Opposite Direction Collision",
        "Same Direction Collision",
        "Lateral Encounter with Crowd",
        "Encounter with Two Pedestrian Crowds",
        "Shared Space with Random Pedestrian Movements",
        "Legend",
    ]
    st.image(senario_img, caption=caption, width=400)

    # Video
    video_file = open(f"{img_path}illustration_senario.mp4", "rb")
    video_file = video_file.read()
    st.video(video_file)
    st.write(
        "Link to the full video: [IROS21 rendered --- Anne Spalanzani](https://www.youtube.com/watch?v=DLaMMedWFn8)"
    )