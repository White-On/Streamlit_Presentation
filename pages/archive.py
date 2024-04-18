import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

img_path = "img/"

st.sidebar.page_link("main_presentation.py", label="Main Presentation", icon="👉")
st.sidebar.markdown("## 🗺️ Navigation")
st.sidebar.markdown("[🛤 Path following score](#path-following-score)")
st.sidebar.markdown("[🦺 Motion Safety score](#motion-safety-score)")
st.sidebar.markdown("[🎯 Trajectory Quality](#trajectory-quality)")
st.sidebar.markdown("[🚶 Pedestrian Confort Score](#pedestrian-confort-score)")

st.markdown("### 🚗 Car Hypothesis")
st.markdown(
    "- The autonomous vehicle (AV) is initially provided with a path to guide it through the scenario. This \
            path is described as a **list of waypoints**. Depending on the precision of the path that we wish to achieve, \
            we can get those waypoints using splines such as *Bezier curves, Hermite splines, or B-Splines*. However, \
            for simplicity, we will stick to **linear splines** for now, which are just a succession of straight lines."
)
spline_img = [
    "Bezier_forth_anim.gif",
    "Hermite_spline.png",
    "B_spline.png",
    "linear_spline.png",
]
st.image(
    [f"{img_path}{name}" for name in spline_img],
    width=300,
    caption=["Bezier curve", "Hermite spline", "B-Spline", "Linear spline"],
)

st.markdown(
    "- In the reality, the AV can perceive the environment **through sensors**. As we are working in a simulator, we choose \
            to represent the environment percieved by the AV as a **2D Occupancy Grid Map** like in [[6]](#reference)[[7]](#reference)[[8]](#reference)\
            and we capture it though a camera or a Lidar. This map will be updated \
            We will do the Hypothesis that the AV can map pedestrians in a **range $r_{rov}$= 30 meters** and a in a \
            **field of view equal to $\\alpha_{fov}$ = 360°** around it . This "
)

st.markdown(
    "The radius of the red circle -> $r_{rov}$, the angle of the red sector -> $\\alpha_{fov}$. "
)
st.image(f"{img_path}AV_perception.gif")

# Fitness function
st.header("Old Fitness function [Deprecated]", anchor="fitness-function")
st.markdown(
    "To address our problem effectively, we require a comprehensive function to assess the behavior \
            and decisions made by our agent. In essence, our focus lies on four core aspects crucial for vehicle \
            operation: **safety, collision avoidance with pedestrians, minimizing disruption to pedestrians and social \
            groups, ensuring smooth maneuvering, and adhering closely to the prescribed path**. To achieve this, we \
            decompose our function into four distinct components that can be tuned and refined independently."
)
st.markdown(
    "We name this function ($f$). Each criterion is weighted by a corresponding coefficient ($c_{fr}$, $c_{s}$, $c_{pc}$, \
            $c_{t}$) to reflect its importance in the overall evaluation process. The fitness function is defined as \
            follows:"
)
st.markdown(
    r"> $$f = c_{fr} \cdot R_c + c_{s} \cdot S_c + c_{pc} \cdot P_c + c_{t} \cdot T_c$$"
)
st.markdown(
    "Here, $R_c$, $R_s$, $R_{pc}$, and $R_t$ represent the scores obtained for Path following, \
            safety, pedestrian comfort, and trajectory quality, respectively. These scores are computed based \
            on the corresponding evaluation criteria.**Each component falls within the range $[-1, 1]$** to ensure clarity and prevent any single score from taking priority \
         in the overall representation. To achieve this, all scores are derived from a modified sigmoid function \
         tailored to effectively reward or penalize specific behaviors.The resulting fitness score ($f$) falls \
            within the range $[-4, 4]$, with higher values indicating better overall performance of the navigation system."
)

# Sigmoid
st.markdown("### 📈 Sigmoid Function")
m = st.slider("m", -10.0, 10.0, 2.0, step=1.0)
n = st.slider("n", 0.0, 10.0, 0.2, step=0.1)
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
x = np.linspace(-10, 10, 100)
y = 1 / (1 + np.exp(-x))
ax[0, 0].plot(x, y)
ax[0, 0].set_title(r"$\frac{1}{1 + e^{x}}$")
ax[0, 0].grid(True, linewidth=0.5, linestyle="--")
ax[0, 0].yaxis.set_ticks(np.arange(0, 1.1, 0.25))

y = 2 / (1 + np.exp(-x))
ax[0, 1].plot(x, y)
ax[0, 1].set_title(r"$\frac{2}{1 + e^{x}}$")
ax[0, 1].grid(True, linewidth=0.5, linestyle="--")
ax[0, 1].yaxis.set_ticks(np.arange(0, 2.1, 0.5))

y = 2 / (1 + np.exp(-x)) - 1
ax[0, 2].plot(x, y)
ax[0, 2].set_title(r"$\frac{2}{1 + e^{x}} - 1$")
ax[0, 2].grid(True, linewidth=0.5, linestyle="--")
ax[0, 2].yaxis.set_ticks(np.arange(-1, 1.1, 0.5))

y = -2 / (1 + np.exp(-x)) + 1
ax[1, 0].plot(x, y)
ax[1, 0].set_title(r"$-\frac{2}{1 + e^{x}} + 1$")
ax[1, 0].grid(True, linewidth=0.5, linestyle="--")
ax[1, 0].yaxis.set_ticks(np.arange(-1, 1.1, 0.5))


y = 2 / (1 + np.exp((-x + m))) - 1
ax[1, 1].plot(x, y)
ax[1, 1].set_title(r"$\frac{2}{1 + e^{x + m}} - 1$")
ax[1, 1].grid(True, linewidth=0.5, linestyle="--")
ax[1, 1].yaxis.set_ticks(np.arange(-1, 1.1, 0.5))

y = 2 / (1 + np.exp(-x * n)) - 1
ax[1, 2].plot(x, y)
ax[1, 2].set_title(r"$\frac{2}{1 + e^{nx}} - 1$")
ax[1, 2].grid(True, linewidth=0.5, linestyle="--")
ax[1, 2].yaxis.set_ticks(np.arange(-1, 1.1, 0.5))
ax[1, 2].set_ylim(-1.1, 1.1)

# Fitness function
st.header("📊 Fitness Function's Components", anchor="fitness-function")
st.markdown(
    "To effectively tackle our problem, we need a comprehensive function \
            to evaluate the behavior and decisions of our agent. Primarily, \
            our focus revolves around two critical aspects for vehicle operation: ensuring safety \
            and collision avoidance, and closely following the prescribed path. To accomplish this, \
            we decompose our function into two distinct components that can be individually \
            fine-tuned and refined."
)
st.markdown("We name this function ($f$). We name each component as follows:")
st.markdown(
    "- **$R_p$**: Path following score with the corresponding coefficient $c_{fp}$"
)
st.markdown(
    "- **$R_s$**: Motion Safety score with the corresponding coefficient $c_{s}$"
)
st.markdown("The fitness function is defined as follows:")
st.markdown(r"> $$f = c_{fp} \cdot R_p + c_{s} \cdot R_s$$")
st.markdown(
    "These scores are computed based on the corresponding evaluation criteria.**Each \
            component falls within the range $[-1, 1]$** to ensure clarity and prevent any single \
            score from taking priority in the overall representation. To achieve this, all scores \
            are derived from a modified sigmoid function \
            tailored to effectively reward or penalize specific behaviors. The resulting fitness score ($f$) falls \
            within the range $[-2, 2]$, with higher values indicating better overall performance of the navigation system."
)

# Path following score
st.header("🛤 Path following score", anchor="path-following-score")

st.markdown(
    "> 📌 **Side note**: This metric is inspired by path following in the context of autonomous agents."
)

st.markdown(
    "The path following score represents the distance between the initially given path to the autonomous \
            vehicle (AV) and its future position.\
             In this context, the variables $d_p$ and $d$ represent the penalty distance and the distance between \
            the AV and the path, respectively. The formula used to calculate the path following score ($R_c$) is \
            given by:"
)
st.markdown(r"> $$R_c = 1-\frac{2}{1+e^{-d + d_p}}$$")
st.markdown(
    "This formula provides a measure of how closely the AV is following the path, with higher values \
            of $R_c$ indicating better adherence to the track."
)

st.image(f"{img_path}path_following_illustration.png", use_column_width=True)

# Plot
max_distance_display = 20
d = np.linspace(0, max_distance_display, 100)
penalty_distance = st.slider("Penalty distance", 0, 20, 10)
Rc = 1 - 2 / (1 + np.exp(-d + penalty_distance))
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[1].plot(d, Rc)
ax[1].set_xlabel("distance between the future position and the path")
ax[1].set_ylabel("Reward")
ax[1].set_title("Reward function for path following")
ax[1].xaxis.set_ticks(np.arange(0, max_distance_display + 1, 5))
ax[1].grid(True, linewidth=0.5, linestyle="--")
ax[1].set_xlim(0, max_distance_display)
ax[1].set_ylim(-1.1, 1.1)

path = np.array([[-15, -5], [15, 10]])
futur_point = np.random.uniform(-15, 15, 2)
# vector a is start point to futur point
a = futur_point - path[0]
# vector b is start point to end point
b = path[1] - path[0]
# d is the product of the magnitude of a and cos of the angle between a and b
d = np.linalg.norm(a) * np.cos(
    np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
)
normal_point = path[0] + d * b / np.linalg.norm(b)
distance_to_rail = np.linalg.norm(futur_point - normal_point)
Rc = 1 - 2 / (1 + np.exp(-distance_to_rail + penalty_distance))
ax[1].axvline(distance_to_rail, color="g", linestyle="--")
ax[1].annotate(
    f"Distance to path: {distance_to_rail:.2f}\n path following reward: {Rc:.2f}",
    (distance_to_rail + 1, 0.5),
)

ax[0].plot(path[:, 0], path[:, 1], "r")
ax[0].scatter(futur_point[0], futur_point[1], c="g", label="Future point")
ax[0].scatter(path[0, 0], path[0, 1], c="y", label="Start")
ax[0].scatter(path[1, 0], path[1, 1], c="b", label="End")
ax[0].scatter(normal_point[0], normal_point[1], c="r", label="Normal point")
ax[0].set_xlim(-20, 20)
ax[0].set_ylim(-20, 20)
ax[0].legend(loc="lower right")

st.pyplot(fig)

st.pyplot(fig)

# Motion Safety score
st.header("🦺 Motion Safety score", anchor="motion-safety-score")

st.markdown(
    "> 📌 **Side note**: This metric is inspired by the paper \
            [Kabtoul, Maria, Manon Prédhumeau, Anne Spalanzani, Julie Dugdale, et \
            Philippe Martinet. « How To Evaluate the Navigation of Autonomous Vehicles \
            Around Pedestrians? » IEEE Transactions on Intelligent Transportation Systems, \
            2024, 1‑11. https://doi.org/10.1109/TITS.2023.3323662]. In the paper, this \
            metric was sugested but **not actually used**"
)
st.image(f"{img_path}motion_safety_notation.png")
st.markdown(
    "Motion Safety refers to the assessment of potential risks and hazards associated with vehicle \
            movement, particularly in the presence of pedestrians. Key parameters involved in evaluating \
            motion safety include the evaluation radius ($eval_r$), the number of pedestrians within \
            this radius ($nb_{p}$), the distance between the vehicle and each pedestrian ($d_i$), and the penalty \
            distance ($d_p$). The safety score ($S_c$) is computed using the following formula:"
)
st.markdown(
    r"> $$S_c = -1 + \frac{2}{1+e^{-\frac{\sum_{i=1}^{nb_{p}}d_i}{d_i} + d_p}}$$"
)
st.write("which is equivalent to:")
st.markdown(r"> $$S_c = -1 + \frac{2}{1+e^{-\bar{d_i} + d_p}}$$")
st.markdown(
    "This formula quantifies the level of safety during vehicle motion, with higher values of $S_c$ indicating \
            a greater degree of safety. We also add the same score but only evaluated with the closest pedestrian \
            to compare the two scores."
)


def safety_reward(distance, penalty_distance):
    return 2 / (1 + np.exp(-distance + penalty_distance)) - 1


eval_radius = st.slider("Evaluation radius", 10, 30, 20)
nb_pedestrian = st.slider("Number of pedestrian", 1, 100, 10)
mu = 2.5  # in meter
# radom direction from the vehicle
random_vector = np.random.uniform(-np.pi, np.pi, (2, nb_pedestrian))
random_vector = np.array([np.cos(random_vector[0]), np.sin(random_vector[1])])
# normalize the vector
random_vector /= np.linalg.norm(random_vector, axis=0)
# random distance
d = np.random.randn(nb_pedestrian) * mu + eval_radius / 2
# random position
pedestrian_pos = d * random_vector

penalty_distance = st.slider("Penalty distance", 0, 20, 6)
mean_distance = np.mean(d)
lowess_distance = d[np.argmin(np.abs(d))]
linspace_distance = np.linspace(0, eval_radius, 100)
Sc = safety_reward(linspace_distance, penalty_distance)

fig, axe = plt.subplots(1, 2, figsize=(10, 5))

axe[0].scatter(pedestrian_pos[0], pedestrian_pos[1], label="Pedestrian")
# the vehicle is a rectangle
axe[0].scatter(0, 0, label="Vehicle", c="r", marker="s", s=100)
axe[0].set_title("Pedestrian position")
axe[0].set_xlabel("x")
axe[0].set_ylabel("y")
axe[0].grid(True, linewidth=0.5, linestyle="--")
axe[0].legend()
axe[0].set_xlim(-eval_radius, eval_radius)
axe[0].set_ylim(-eval_radius, eval_radius)

axe[1].plot(linspace_distance, Sc)
axe[1].set_title("Safety reward")
axe[1].set_xlabel("Mean distance to the pedestrian")
axe[1].set_ylabel("Reward")
axe[1].grid(True, linewidth=0.5, linestyle="--")
axe[1].axvline(mean_distance, color="r", linestyle="--", label="Mean distance")
axe[1].annotate(
    f"Mean distance: {mean_distance:.2f}\nSafety reward: {safety_reward(mean_distance,penalty_distance):.2f}",
    (mean_distance + 1, 0.5),
)
axe[1].axvline(lowess_distance, color="r", linestyle="--", label="Lowess distance")
axe[1].annotate(
    f"Lowess Distance: {lowess_distance:.2f}\nSafety reward: {safety_reward(lowess_distance,penalty_distance):.2f}",
    (lowess_distance + 1, 0.0),
)


st.write(
    r"We created a scenario where our vehicle is positioned in the middle of a shared space with pedestrians around. In this representation, the pedestrians are positioned according to a normal distribution ($mu = \frac{eval_r}{2}, \sigma = 2.5$) shaped like a donut around the car."
)
st.pyplot(fig)

### Trajectory Quality
st.header("🎯 Trajectory Quality", anchor="trajectory-quality")

st.markdown(
    "> 📌 **Side note**: A similar metric was used in [Kabtoul, Maria, Manon Prédhumeau, Anne Spalanzani, \
            Julie Dugdale, et Philippe Martinet. « How To Evaluate the Navigation of Autonomous Vehicles Around \
            Pedestrians? » IEEE Transactions on Intelligent Transportation Systems, \
            2024, 1‑11. https://doi.org/10.1109/TITS.2023.3323662] but wasn't adapted to our case"
)

st.markdown(
    "Trajectory quality refers to the assessment of the smoothness and stability of a vehicle's trajectory \
            during motion. It is influenced by parameters such as the magnitude of acceleration ($|a|$) and the \
            magnitude zero limit ($m_0$). The magnitude of acceleration is computed using the formula:"
)
st.markdown(r"> $$|a| = \sqrt{\frac{dx^2}{dt^2} + \frac{dy^2}{dt^2}}$$")
st.markdown(
    "where $dx$ and $dy$ represent the changes in position with respect to time."
)
st.markdown(
    "The quality score ($T_c$) for the trajectory is then determined using the following formula:"
)
st.markdown(r"> $$T_c = 1 - \frac{2}{1+e^{-|a| + m_0}}$$")
st.markdown(
    "This formula quantifies the quality of the trajectory, \
            with higher values of $T_c$ indicating a smoother and more stable trajectory."
)

st.write(
    "In this scenario, we generated a mock trajectory for the autonomous vehicle (AV) represented \
         by a random quadratic Bezier curve, with little black arrows indicating the velocity at each \
         specific point along the trajectory."
)
# Plot


def lerp(a, b, t):
    return a * (1 - t) + b * t


P1 = np.random.randn(2)
P2 = np.random.randn(2)
P3 = np.random.randn(2)
P4 = np.random.randn(2)

n = 100

l = np.linspace(0, 1, n).reshape(-1, 1)

P5 = lerp(P1, P2, l)
P6 = lerp(P2, P3, l)
P7 = lerp(P3, P4, l)
P8 = lerp(P5, P6, l)
P9 = lerp(P6, P7, l)
P10 = lerp(P8, P9, l)

# the tangent vector at each point
tangent = np.diff(P10, axis=0)
# normalize the tangent
tangent = tangent / np.linalg.norm(tangent, axis=1).reshape(-1, 1)
tangent *= 0.1

fig = plt.figure(figsize=(5, 5))
plt.scatter(P10[:, 0], P10[:, 1], c=l.flatten(), cmap="viridis_r", label="v", s=100)
for i in range(len(P10) - 1):
    plt.arrow(P10[i, 0], P10[i, 1], tangent[i, 0], tangent[i, 1], alpha=0.5)
plt.axis("off")
clor = plt.colorbar(shrink=0.5)
clor.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
clor.set_ticklabels(["start", 0.2, 0.4, 0.6, 0.8, "end"])

st.pyplot(fig)

dxdl = np.gradient(tangent[:, 0], l[:-1].flatten())
dydl = np.gradient(tangent[:, 1], l[:-1].flatten())


magnitude_roc = np.sqrt(dxdl**2 + dydl**2)
esp = 0.0001
# reward  = 1 / (magnitude_roc + esp)
magniture_zero_limit = st.slider("Magnitude zero limit", 0.0, 5.0, 1.0, step=0.5)
reward = -2 / (1 + np.exp(-magnitude_roc + magniture_zero_limit)) + 1

fig, axe = plt.subplots(1, 4, figsize=(15, 5))
axe[0].plot(l[:-1], dxdl, label="x")
axe[0].set_title(
    "rate of change of the x component\n of the tangent vector", fontsize=10
)
axe[0].grid(True, linewidth=0.5, linestyle="--")
axe[0].legend()

axe[1].plot(l[:-1], dydl, label="y")
axe[1].set_title(
    "rate of change of the y component\n of the tangent vector", fontsize=10
)
axe[1].grid(True, linewidth=0.5, linestyle="--")
axe[1].legend()

axe[2].plot(l[:-1], magnitude_roc, label="magnitude")
axe[2].set_title("magnitude of the rate of change\n of the tangent vector", fontsize=10)
axe[2].grid(True, linewidth=0.5, linestyle="--")
axe[2].legend()

axe[3].plot(l[:-1], reward, label="reward")
axe[3].set_title("reward", fontsize=10)
axe[3].grid(True, linewidth=0.5, linestyle="--")
axe[3].legend()

fig.supxlabel("normalized time")

st.pyplot(fig)

# Pedestrian Confort Score
st.header("🚶 Pedestrian Confort Score", anchor="pedestrian-confort-score")

st.markdown(
    r"> 📌 **Side note**: $\bar{I}_{ucf}$ is a metric used in [D. Helbing, P. Moln ́ ar, I. J. Farkas, and K. Bolay, “Self-organizing pedestrian movement,” Environment and Planning B: Planning and Design, vol. 28, no. 3, pp. 361–383, 2001.]"
)

st.markdown(
    r"Pedestrian comfort is evaluated based on parameters such as the frequency of linear velocity changes experienced by pedestrians ($\bar{I}_{ucf}$) during their navigation and the frequency zero penalty ($f_0$).The comfort score ($T_c$) for pedestrians is computed using the following formula:"
)
st.markdown(r"> $$T_c = 1 - \frac{2}{1+e^{10(-\bar{I}_{ucf} + m_0)}}$$")
st.markdown(
    "This formula provides a measure of pedestrian comfort, with higher values of $T_c$ indicating \
            a more comfortable and less disruptive interaction with the vehicle."
)

# Plot
freq_domain = np.linspace(0, 1, 1000)
freq_penalty = st.slider("Frequency zero penalty magnitude", 0.0, 1.0, 0.4, step=0.1)

Pc = -2 / (1 + np.exp(10 * (-freq_domain + freq_penalty))) + 1

fig = plt.figure(figsize=(5, 5))
plt.plot(freq_domain, Pc)
plt.xlabel("Frequency of linear velocity change")
plt.ylabel("Reward")
plt.title("Pedestrian reward function")
plt.grid(True, linewidth=0.5, linestyle="--")
st.pyplot(fig)

st.markdown("### 🚨 Warning 🚨")
st.markdown(
    "This score can't really be used to learn a policy as it is a post simulation evaluation metric. \
            We need a metric that can evaluate the pedestrian comfort in a given time/state.\n\n"
)
st.markdown(
    "Following some advice, the best option to evaluate the comfort level of pedestrians in our environment \
            would be to perform a similar operation as the trajectory quality but from the perspective of pedestrians. \
            We would evaluate the rate of change of velocity of all pedestrians and use it as a metric to assess how \
            our AV constrains pedestrian movements. However, this metric may be too computationally expensive. An \
            alternative approach for evaluating comfort in pedestrian areas would be to assess how limited \
            pedestrians are in their movements. Another idea would be to use the number of points added in \
            Alexis' physical model (the one with springs to constrain pedestrians)."
)
