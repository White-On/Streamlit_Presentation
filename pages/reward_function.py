import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def write_reward_function(img_path) -> None:
    st.header("📊 Reward Function's Components", anchor="reward-function")
    st.markdown(
        "To effectively tackle our problem, we need a comprehensive value \
                to evaluate the behavior and decisions of our agent. Primarily, \
                our focus revolves around four critical aspects for vehicle operation: ensuring safety \
                and collision avoidance, following the prescribed path and maintain prefered speed. To accomplish this, \
                we decompose our value into four distinct components that can be individually \
                fine-tuned and refined."
    )
    st.markdown("We name this value ($R$). We name each component as follows:")
    st.markdown(
        "- $r_c$: collision reward, to punish the agent for colliding with the pedestrian."
    )
    st.markdown(
        "- $r_{nc}$ : near collision reward, to punish the agent for getting too close to \
                the pedestrian. Not just a discrete reward, but a continuous one, to encourage the \
                agent to keep a safe distance from the pedestrian but not prevent it from getting close \
                to the pedestrian."
    )
    st.markdown(
        "- $r_s$ : speed reward, to reward the agent for moving fast/to the preferred speed. \
                This is to encourage the agent to move at a reasonable speed, \
                but not too fast or too slow."
    )
    st.markdown(
        "- $r_p$ : path following reward, to reward the agent for following the path. \
                This is to encourage the agent to follow the path, but not to prevent it from \
                deviating from the path if necessary."
    )
    st.markdown("The Reward function is defined as follows:")
    st.markdown(r"> $$R = r_c + r_{nc} + r_{s} + r_p$$")
    st.markdown(
        "We can **eventually** define individual component coefficients to adjust the importance of each component in the final reward. \
                Respectively, we will name them $w_c$, $w_{nc}$, $w_s$ and $w_p$."
    )
    st.markdown(
        r"> $$R = w_c \cdot r_c + w_{nc} \cdot r_{nc} + w_s \cdot r_s + w_p \cdot r_p$$"
    )

    st.markdown(
        "> 📌 **Side note**: the components $r_c,r_{nc}, r_{s}$ are based on reward function of [[12]](#reference)[[13]](#reference)[[14]](#reference).\
                The component $r_p$ is handcrafted"
    )

    number_of_points = 100
    write_path_following_component(img_path, number_of_points)
    write_speed_reward_component(img_path, number_of_points)
    write_collision_near_collision_component(img_path, number_of_points)


def write_path_following_component(img_path, n) -> None:
    st.header("🛤 Path Following component", anchor="path-following-component")

    st.markdown(
        "First, we define **the components $r_p$ responsible for the path-following reward**. The path-following \
                reward is continuous, aiming to penalize the agent for deviating too far from the path or moving in the \
                wrong direction. To achieve this, we evaluate **the angle error of the agent's trajectory concerning the \
                next waypoint on the path**, denoted as $\\epsilon_{\\theta}$. Recognizing that considering only the angle error may be \
                insufficient, we also incorporate **the distance between the agent and the path**, referred to as \
                $\\epsilon_{path}$. These two values form the basis of the path-following component. We create *two separate \
                versions*, one utilizing only $\\epsilon_{\\theta}$ to explore its individual impact on behavior, and another combining \
                both elements to assess their collective influence."
    )

    st.image(f"{img_path}Path_following_reward.png", use_column_width=True)
    st.markdown(
        "The illustration demonstrates the calculation of $\\epsilon_{\\theta}$. As our agent advances along the path, \
                we compute **the angle between the agent's current heading and the vector directed towards the next \
                waypoint**. $d_w$ represents the distance in which, **the agent is \
                to proceed to the subsequent waypoint once it falls within this threshold**."
    )

    st.image(f"{img_path}path_following_illustration.png", use_column_width=True)
    st.markdown(
        "The illustration demonstrates the calculation of $$\\epsilon_{path}$$ as the distance between the agent and \
                the path. To incentivize the agent to remain near the path, we model the reward using a **sigmoid function \
                centered around the path**, with a zero value threashold equal to $\\sigma_{path}$. $\tau$ is a parameter to \
                control the steepness of the function."
    )

    st.markdown("The path following reward with only $\\epsilon_{\\theta}$:")
    st.markdown(r"> $$r_p = e^{-\frac{\epsilon_{\theta}}{\tau}}$$")

    espison_theta = np.linspace(0, 180, n)
    m = st.slider("tau", 1, 50, 20)
    # r_p = 1/(espison_theta + 1)
    angle_component = np.exp(-espison_theta / m)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(espison_theta, angle_component)
    ax.set_xlabel("Angle difference [°]")
    ax.set_ylabel("$r_p$")
    ax.set_title("$r_p$ for the angle difference")
    ax.grid(True, linewidth=0.5, linestyle="--")
    ax.xaxis.set_ticks(np.arange(0, 181, 20))

    st.pyplot(fig)

    st.markdown(
        "The path following reward with $\\epsilon_{\\theta}$ and $\\epsilon_{path}$:"
    )
    st.markdown(
        r"> $$r_p = e^{-\frac{\epsilon_{\theta}}{\tau}}(1-\frac{2}{1+e^{-\epsilon_{path} + \sigma_{path}}})$$"
    )

    def distance_from_path(point, path):
        a = point - path[0]
        b = path[1] - path[0]
        d = np.linalg.norm(a) * np.cos(
            np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        )
        normal_point = path[0] + d * b / np.linalg.norm(b)
        distance_to_rail = np.linalg.norm(point - normal_point)
        return distance_to_rail

    def proximity_reward(distance, penalty_distance):
        return 1 - 2 / (1 + np.exp(-distance + penalty_distance))

    max_distance_display = 20
    d = np.linspace(0, max_distance_display, n)
    penalty_distance = st.slider("sigma_path [m]", 0, 20, 10)
    proximity_component = 1 - 2 / (1 + np.exp(-d + penalty_distance))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[1].plot(d, proximity_component)
    ax[1].set_xlabel("Distance between the vehicle position and the path [m]")
    ax[1].set_ylabel("Proximity component")
    # ax[1].set_title('Reward function for path following')
    ax[1].xaxis.set_ticks(np.arange(0, max_distance_display + 1, 5))
    ax[1].grid(True, linewidth=0.5, linestyle="--")
    ax[1].set_xlim(0, max_distance_display)
    ax[1].set_ylim(-1.1, 1.1)

    path = np.array([[-15, -5], [15, 10]])
    futur_point = np.random.uniform(-15, 15, 2)
    distance_to_rail = distance_from_path(futur_point, path)
    annotate_value = 1 - 2 / (1 + np.exp(-distance_to_rail + penalty_distance))
    ax[1].axvline(distance_to_rail, color="g", linestyle="--")
    ax[1].annotate(
        f"Distance to path: {distance_to_rail:.2f}\nProximity component: {annotate_value:.2f}",
        (distance_to_rail + 1, 0.5),
    )

    all_points = np.meshgrid(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100))
    # vector a is start point to futur point
    a = futur_point - path[0]
    # vector b is start point to end point
    b = path[1] - path[0]
    # d is the product of the magnitude of a and cos of the angle between a and b
    e = np.linalg.norm(a) * np.cos(
        np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    )
    normal_point = path[0] + e * b / np.linalg.norm(b)
    # path = np.array([[-15, -5], [15, 10]])
    distance = np.zeros(all_points[0].shape)
    for i in range(all_points[0].shape[0]):
        for j in range(all_points[0].shape[1]):
            distance[i, j] = distance_from_path(
                np.array([all_points[0][i, j], all_points[1][i, j]]), path
            )
    reward = proximity_reward(distance, 10)
    reward_img = ax[0].imshow(
        reward,
        cmap="plasma",
        extent=(-20, 20, -20, 20),
        origin="lower",
        interpolation="bilinear",
    )
    # reward_img = ax[0].contourf(all_points[0], all_points[1], reward, cmap='plasma')
    # reward_img = ax[0].contour(all_points[0], all_points[1], reward, cmap='plasma')
    # ax[0].clabel(reward_img, inline=True, fontsize=8)
    fig.colorbar(reward_img, ax=ax[0])

    ax[0].plot(path[:, 0], path[:, 1], "r")
    ax[0].scatter(futur_point[0], futur_point[1], c="g", label="Vehicle position")
    ax[0].scatter(path[0, 0], path[0, 1], c="purple", label="Start")
    ax[0].scatter(path[1, 0], path[1, 1], c="b", label="End")
    ax[0].scatter(normal_point[0], normal_point[1], c="r", label="Normal to the path")
    ax[0].set_xlim(-20, 20)
    ax[0].set_ylim(-20, 20)
    ax[0].legend(loc="lower right")
    fig.tight_layout(pad=3)
    fig.suptitle("Proximity component", fontsize=16)

    st.pyplot(fig)

    # r_p = 1/(espison_theta + 1)

    # TODO: figure if this is the right way to combine the two components
    # r_c = np.multiply(angle_component, proximity_component.reshape(-1, 1))
    r_c = np.add(angle_component, proximity_component.reshape(-1, 1))

    fig = plt.figure(figsize=(15, 6))
    ax = plt.subplot(121)
    cs = ax.contourf(espison_theta, d, r_c, cmap="viridis")
    # cs = ax.imshow(r_c, cmap='viridis', extent=(0, 180, 0, max_distance_display), origin='lower')
    fig.colorbar(cs)
    ax.set_xlabel("Angle difference [°]")
    ax.set_ylabel("Distance to path [m]")

    # 3D plot
    ax = fig.add_subplot(122, projection="3d")
    espison_theta_mg, d_mg = np.meshgrid(espison_theta, d)
    ax.plot_surface(espison_theta_mg, d_mg, r_c, cmap="viridis")
    ax.set_xlabel("Angle difference [°]")
    ax.set_ylabel("Distance to path [m]")
    ax.set_zlabel("$r_c$")
    ax.view_init(elev=10.0, azim=30)

    fig.tight_layout(pad=0)
    fig.suptitle(
        "$r_c$ for the angle difference and distance to path", fontsize=16, y=1.05
    )

    st.pyplot(fig)


def write_speed_reward_component(img_path, n) -> None:
    # Speed reward
    st.header("🚀 Speed component", anchor="speed-component")
    st.markdown(
        "The speed reward component is designed to encourage the agent to move at a preferred speed. \
                The reward is continuous, with higher values indicating that the agent is moving at the desired speed. \
                The speed reward is calculated using the following formula:"
    )
    st.markdown(
        r"$$r_s = \left\{ \begin{array}{rcl}2(e^{\vec{v_{AV}} - \vec{v^*_{AV}}} - 0.5) & if & \vec{v_{AV}}\leq \vec{v^*_{AV}} \\ 2(e^{-\vec{v_{AV}} + \vec{v^*_{AV}}} - 0.5) & if & \vec{v_{AV}} > \vec{v^*_{AV}}\end{array}\right.$$"
    )

    def speed_reward(current_speed, pref_speed):
        if current_speed <= pref_speed:
            return (np.exp(current_speed - pref_speed) - 0.5) * 2
        elif current_speed > pref_speed:
            return (np.exp(-current_speed + pref_speed) - 0.5) * 2

    v_ev = np.linspace(-2, 10, n)

    speed_function = np.vectorize(speed_reward)
    pref_speed = st.slider("Preferred speed [m/s]", 0.1, 10.0, 5.0)
    r_speed = speed_function(v_ev, pref_speed)

    fig = plt.figure(figsize=(10, 5))
    plt.axvline(
        pref_speed, color="r", linestyle="--", alpha=0.5, label="Preferred speed"
    )
    plt.plot(v_ev, r_speed)
    plt.xlabel("Linear velocity [m/s]")
    plt.ylabel("Reward")
    plt.title("Speed reward function")
    plt.grid(True, linewidth=0.5, linestyle="--")
    plt.legend()

    st.pyplot(fig)

    st.markdown(
        "> 📌 **Side note**: The original $r_ {s}$ component[[12]](#reference) was written like the following formula but once plotted, \
                we can see that the reward seems to reward the agent to go right above the 0 speed."
    )


def write_collision_near_collision_component(img_path, n) -> None:
    # Motion Safety score
    st.header("🦺 Collision/Near collision Component", anchor="motion-safety-score")
    st.markdown(
        " The last 2 components are the collision ($r_c$) and near collision ($r_{nc}$) components. \
                These components are designed to penalize the agent for colliding with or getting too close to the pedestrian. \
                The collision component is a discrete reward, while the near collision component is continuous. \
                The collision component need the be way more penalizing than any other component:"
    )
    st.markdown(
        r"$$r_c = \left\{ \begin{array}{rcl}-x & if & \text{collision} \\ 0 & if & \text{no collision}\end{array}\right.tq \space x >> \max(r_{nc}) + \max(r_p) + \max(r_s)$$"
    )
    st.markdown(
        "$r_{nc}$ is not just a discrete reward, but a continuous one, to encourage the agent \
                to keep a safe distance from the pedestrian but not prevent it from getting close to \
                the pedestrian. The formula is based around a safe distance calculated with the maximum \
                deceleration of the AV $d_r$:"
    )
    st.markdown(r"$$d_r = \max(\frac{\vec{v_{AV}}²}{2\min(\vec{a_{AV}})}, d_o)$$")
    st.markdown(r"$$r_{nc} =  e^{\frac{d_p - d_r}{d_r}}\vec{v_{AV}}$$")

    st.markdown(
        "Here $d_o$ is a minimum distance to keep between the AV and the pedestrian. And \
                $d_p$ is the distance between the AV and the closest pedestrian."
    )

    v_ev = np.linspace(0, 10, n)
    a_max = st.slider("Maximum deceleration [m/s²]", 0.1, 10.0, 2.0)
    min_safe_distance = st.slider("Minimum safe distance [m]", 0.1, 10.0, 5.0)
    do = np.ones(n) * min_safe_distance
    dr = np.maximum(np.power(v_ev, 2) / (2 * a_max), do)
    distance_pedesrian = np.linspace(0, 20, n).reshape(-1, 1)
    r_nc = np.exp((distance_pedesrian - dr) / dr) * v_ev

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(131)
    plt.axhline(min_safe_distance, color="r", linestyle="--", alpha=0.5)
    plt.plot(v_ev, np.power(v_ev, 2) / (2 * a_max), label="Braking distance", alpha=0.5)
    plt.plot(v_ev, dr, label="Safe distance")
    plt.xlabel("Linear velocity [m/s]")
    plt.ylabel("$d_r$")
    plt.title("Safe distance as a function of the linear velocity")
    plt.grid(True, linewidth=0.5, linestyle="--")
    plt.legend()

    # 3D plot

    ax = fig.add_subplot(132, projection="3d")
    ax.plot_surface(v_ev, distance_pedesrian, r_nc)
    ax.set_xlabel("Linear velocity [m/s]")
    ax.set_ylabel("Distance to pedestrian [m]")
    ax.set_zlabel("Reward")

    # 2D plot
    ax = fig.add_subplot(133)
    coutour = ax.contourf(v_ev.flatten(), distance_pedesrian.flatten(), r_nc)
    ax.set_xlabel("Linear velocity [m/s]")
    ax.set_ylabel("Distance to pedestrian [m]")
    ax.set_title("Reward function for the pedestrian")

    fig.colorbar(coutour)
    fig.tight_layout(pad=3)

    st.pyplot(fig)

    # Variables name recap
    st.header("📝 Variables name recap", anchor="variables-name-recap")
    st.markdown(
        "- $W$: The path of the AV, a list of waypoints. \n\
    - $M$: The occupancy grid map of the environment. \n\
    - $r_{rov}$: The range of view of the AV. \n\
    - $\\alpha_{fov}$: The field of view of the AV. \n\
    - $v_{AV}$: The linear velocity of the AV. \n\
    - $v_{AV}^*$: The prefered linear velocity of the AV. \n\
    - $a_{AV}$: The acceleration of the AV. \n\
    - $\delta$: The wheels angle of the AV. \n\
    - $s$: The dimensions of the AV. \n\
    - $r_{ped}$: The radius of the pedestrian. \n\
    - $v_{ped}$: The linear velocity of the pedestrian. \n\
    - $v_{ped}^*$: The prefered linear velocity of the pedestrian. \n\
    - $d_o$: The minimum distance to keep between the AV and the pedestrian. \n\
    - $d_r$: The safe distance between the AV and the pedestrian. \n\
    - $d_p$: The distance between the AV and the closest pedestrian. \n\
    - $r_c$: The collision reward. \n\
    - $r_{nc}$: The near collision reward. \n\
    - $\\epsilon_{\\theta}$: The angle error of the agent's trajectory concerning the next waypoint on the path. \n\
    - $\\epsilon_{path}$: The distance between the agent and the path. \n\
    - $\\sigma_{path}$: The zero value threashold of the sigmoid function. \n\
    - $\\tau$: The parameter to control the steepness of the path following component.\n\
    - $r_s$: The speed reward. \n\
    - $r_p$: The path following reward. \n\
    - $w_c$: The collision reward coefficient. \n\
    - $w_{nc}$: The near collision reward coefficient. \n\
    - $w_s$: The speed reward coefficient. \n\
    - $w_p$: The path following reward coefficient. "
    )


def write_progression_component(img_path, n) -> None:
    st.header("📈 Progression component", anchor="progression-component")
    st.markdown("**WIP**")
