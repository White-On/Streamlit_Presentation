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
        "- Although the AV can navigate in a 3D environment using Gazebo, our study will focus on a 2D representation \
            to simplify learning and task modeling. Therefore, the AV will be represented as a rectangle. To \
            maintain real-world proportions, we will use the dimensions of a standard Renault Zoe **$s = [s_x, s_y] = [4.1, 1.8]$ meters**[[11]](#reference)."
    )
    zoe_img = ["zoe1.webp", "zoe2.webp", "zoe3.webp"]
    st.image([f"{img_path}{zoe}" for zoe in zoe_img], width=300)

    st.markdown(
        "- In the reality, the AV can perceive the environment **through sensors**. Thoses sensors can then give us \
        information about the environment. We will do the Hypothesis that the AV can map pedestrians in a **range $r_{rov}$= 30 meters** and a in a \
        **field of view equal to $\\alpha_{fov}$ = 360°** around it . With this information, the Autonomous Vehicle (AV) can construct a detailed \
        map of its environment using an **Occupancy Grid Map** (here named $M$), as demonstrated in [[6]](#reference)[[7]](#reference)[[8]](#reference). \
        This map provides a bird's-eye view of the surroundings and can be augmented to include various additional information such as *velocity, \
        orientation, and the probability of occupancy in the next time step*. **To begin with simpler approach**, we will focus on tracking \
        the identification number of pedestrians, along with their velocity and orientation. A similar gird is employed in [[13]](#reference)"
    )

    st.markdown(
        "The radius of the red circle -> $r_{rov}$, the angle of the red sector -> $\\alpha_{fov}$. "
    )
    st.image(f"{img_path}input.gif")

    st.markdown(
        "- For the physical representation, we will use the **[kinematic bicycle model](https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html)**. \
                We  want to keep the model simple, easy to understand and we will disregard the dynamics of the car."
    )
    st.image(
        [f"{img_path}bicycle_model.png", f"{img_path}vehicle_dynamic.png"],
        caption=[
            "",
            "Kabtoul, Maria, Anne Spalanzani, et Philippe Martinet. « Proactive And Smooth Maneuvering For Navigation Around Pedestrians ». In 2022 International Conference on Robotics and Automation (ICRA), 4723‑29. Philadelphia, PA, USA: IEEE, 2022. https://doi.org/10.1109/ICRA46639.2022.9812255",
        ],
        width=400,
    )

    st.markdown(
        "- The output of the model will consist of two components: \
                **the self-propelled scalar speed of the autonomous vehicle's velocity** denoted as $v_{AV}$, and \
                **the steering angle of the front wheels**, represented by $\delta$. To enhance the realism of our model and guarantee safe driving behavior, so that\
                $v_{AV} \in[-1, 4]$ m/s with a prefered speed equal to $v^*_{AV} = \max(v_{AV})$ , and the acceleration is $a_{AV} \in [-1,2]$ m/s². We grant the AV the ability to \
                go **backward** to explore pentential behaviors but we may refine this behavior to only go forward.\
                If we were to model our vehicle's behavior after a real Renault Zoe, \
                we would need to constrain the steering angle ($\delta$) within the range of \
                [-13.73, 13.73] degrees (value obtained with the formula of the kinematic bicycle model), as the minimum braking range ($R$) for this car model \
                is 10.60 meters[[11]](#reference). These parameters enable us to calculate the necessary values \
                required for maneuvering within the simulator environment."
    )
    st.image(f"{img_path}input_model_illustration.gif")


def write_pedestrian_hypothesis(img_path) -> None:
    # Pedestrian Hypothesis
    st.markdown("### 🚶 Pedestrian Hypothesis")
    st.markdown(
        "- Pedestrians can move freely within the environment, and their positions are updated at each time step.\
                In the simulator, they will be represented as circles with a radius $r_{ped} = 0.3$m. The circle representation \
                provide a good approximation of the space occupied by pedestrians (In a top-down perspective, a \
                pedestrian can rotate around the center, here the head). "
    )
    st.image(f"{img_path}pedestrian_illustration.png", width=300)
    st.markdown(
        "- Pedestians have their own model to move and this model is based on the **Social Force Model** \
                which is a computational framework used to simulate the movement of individuals within a crowd by \
                modeling the interactions between them as forces. It describes how pedestrians navigate through a \
                space by considering factors such as attraction between groups and repulsion from obstacles. \
                The AV has prior knowledge of the pedestrian's model and use it to predict their movements and adapt its \
                trajectory accordingly."
    )
    st.markdown(
        "- The pedestrians will walk with a speed $v_{ped} \in [0,2]$ m/s and a prefered speed $v_{ped}^* = 1.4$ m/s [[9]](#reference)[[10]](#reference) but we may refine this values to \
                follow a normal distribution as $\mathcal{N}(1.4, 0.5)$."
    )

if __name__ == "__main__":
    print("This script is not meant to be run directly.")