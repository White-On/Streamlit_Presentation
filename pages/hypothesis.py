import streamlit as st


def write_hypothesis(img_path) -> None:
    st.header("🔬 Hypothesis", anchor="hypothesis")
    write_car_hypothesis(img_path)
    write_pedestrian_hypothesis(img_path)


def write_car_hypothesis(img_path) -> None:
    # Car Hypothesis
    st.markdown("### 🚗 Car Hypothesis")
    st.markdown(
        "- The autonomous vehicle (AV) is initially provided with a path to guide it through the scenario. This \
        path is described as a **list of waypoints** and called $W$. Depending on the precision of the path that we wish to achieve, \
        we can get those waypoints using splines such as  *Hermite splines or linear splines*. However, \
        for simplicity, we will stick to **linear splines** for now, which are just a succession of straight lines."
    )
    spline_img = ["Hermite_spline.png", "linear_spline.png"]
    st.image(
        [f"{img_path}{name}" for name in spline_img],
        width=300,
        caption=["Hermite spline", "Linear spline"],
    )

    st.markdown(
        "- The autonomous vehicle (AV) operates in a simplified 2D environment, modeled as a rectangle \
            with dimensions based on a standard Renault Zoe (used in the research team), \
                $s=[sx,sy]= 4.1 \cdot 1.8$ meters.[[11]](#reference)."
    )
    zoe_img = ["zoe1.webp", "zoe2.webp", "zoe3.webp"]
    st.image([f"{img_path}{zoe}" for zoe in zoe_img], width=300)

    st.markdown(
        "- The vehicle's movement is modeled using the \
            **[kinematic bicycle model](https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html)**. \
            Constraints are applied to ensure driving behavior remains realistic, \
            such as speed limits and steering angle restrictions."
    )

    st.image(
        [f"{img_path}bicycle_model.png", f"{img_path}vehicle_dynamic.png"],
        caption=[
            "",
            "Kabtoul, Maria, Anne Spalanzani, et Philippe Martinet. « Proactive And Smooth Maneuvering For Navigation Around Pedestrians ». In 2022 International Conference on Robotics and Automation (ICRA), 4723‑29. Philadelphia, PA, USA: IEEE, 2022. https://doi.org/10.1109/ICRA46639.2022.9812255",
        ],
        width=400,
    )

    alpha_fov = r"$\alpha_{fov}$"
    r_rov = r"$r_{rov}$"
    st.markdown(
        f"The AV perceives its environment within a radius of 30 meters ({r_rov}) and a 360° field of view ({alpha_fov})."
    )

    st.image(f"{img_path}input.gif")

    st.markdown(
        "- The vehicle represents visible pedestrians as a vector containing their current \
            position and the possible positions they may take in future time steps. The agent's \
            observation set includes:"
    )

    st.markdown(
        "    - **The state of the vehicle**, represented by a vector \
            $\eta_{AV} = [v_{AV}, \delta, W_i, W_{i+1}, \ldots, W_{i+k}]$, which includes \
                the following information: $v_{AV}$ (vehicle speed), $\delta$ (steering angle), and \
                    the positions of the remaining waypoints on the vehicle's path."
    )
    node_rep = (
        r"$\eta_{j} = \left[ p_{j}^{t}, p_{j}^{t+1}, \ldots, p_{j}^{t+l} \right]$"
    )
    st.markdown(
        f"    - **The state of the situation observable by the agent**, represented by a graph \
            where nodes represent the various pedestrians within the vehicle's field of view and \
            the vehicle itself. Each node consists of a feature vector called $\eta_j$, where $j$ \
            denotes the number of nodes. $\eta_j$ contains the pedestrian's current position and \
            an estimate of their future position calculated by the agent. The agent evaluates \
            the pedestrian's speed using their position in the previous time step and anticipates \
            the most likely positions for the next $l$ time steps. This is noted as {node_rep}."
    )

    st.image(f"{img_path}graph_representation.png")

    st.video(f"{img_path}graphe_representation.mp4", loop=True, autoplay=True, muted=True)

    delta_lim = r"$\delta \in [-\frac{\pi}{6}, \frac{\pi}{6}]$"
    v_lim = r"$v_{AV} \in [0, 4]$"
    v = r"$v_{AV}$"
    st.markdown(
        f"- The vehicle is controlled by adjusting the wheel orientation ($\delta$) in degrees and the scalar self-propulsion speed ({v}) in m/s. The agent must choose its actions in the continuous domain such that {delta_lim} and {v_lim}."
    )
    st.image(f"{img_path}input_model_illustration.gif")


def write_pedestrian_hypothesis(img_path) -> None:
    # Pedestrian Hypothesis
    st.markdown("### 🚶 Pedestrian Hypothesis")
    st.markdown(
        "- Pedestrians can move freely within the environment, and their positions are updated at each time step.\
                In the simulator, they will be represented as circles with a radius $r_{ped} = 0.3$m."
    )
    st.image(f"{img_path}pedestrian_illustration.png", width=300)
    st.markdown(
        "- Pedestrian speed varies between 0 and 2 m/s, with a preferred average speed of 1.4 m/s, \
            possibly adjusted to follow a normal distribution for added realism.\n\
        - In the simulation environment, each pedestrian is assigned a target position to reach. \
            Once the target is reached, a new target is assigned, either randomly within the environment \
                or based on specific rules to fit a particular scenario.\n\
        - Pedestrians cooperate with each other to move within the environment; however, \
            the vehicle is not always visible to them and may be considered invisible in scenarios \
                where pedestrians are non-cooperative.\n\
        - For pedestrian movements, we use ROV2 [website][GitHub], implemented in Python and \
            also used in [4] and [6]. In the case of cooperative pedestrians, the vehicle is \
                viewed as a larger pedestrian."
    )


if __name__ == "__main__":
    print("This script is not meant to be run directly.")
