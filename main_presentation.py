import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

img_path = 'img/'

st.set_page_config(page_title="Learning to cross a crowd with an autonomous vehicle", page_icon="🤖", layout="wide")



#Sidebar
st.sidebar.page_link("pages/archive.py", label="Archive", icon="🗃️")
st.sidebar.markdown("## 🌿 Random Seed Controls")
if st.sidebar.toggle("Random Seed", value=False, help="Toggle and random seeds will be used"):
    np.random.seed(None)
else:
    np.random.seed(42)
st.sidebar.button("Regenerate Results")

st.sidebar.markdown("## 🗺️ Navigation")
st.sidebar.markdown("[🔬 Hypothesis](#hypothesis)")
st.sidebar.markdown("[📦 Reward function](#reward-function)")
st.sidebar.markdown("[🎨 Scenarios](#scenario)")
st.sidebar.markdown("[📚 References](#reference)")


#Problem/Context
st.title("Problem/Context")
st.markdown('We Consider a shared space scenario where autonomous vehicles navigate among \
            pedestrians in a simulation. Given a predefined path leading to a goal destination within this shared space, the \
            autonomous vehicle is tasked with adapting its driving behavior to ensure safety [[1]](#reference), avoid collisions with pedestrian, \
            minimize disturbance to pedestrians and social groups, maintain smooth maneuvering[[3]](#reference), and adhere to the \
            given path as closely as possible[[5]](#reference). The objective is *not merely to navigate around the crowd*, but rather \
            to follow the specified path while dynamically adjusting its trajectory based on real-time situational \
            awareness and environmental factors.')

# Hypothesis
st.header("🔬 Hypothesis", anchor="hypothesis")
# Car Hypothesis
st.markdown("### 🚗 Car Hypothesis")
st.markdown("- The autonomous vehicle (AV) is initially provided with a path to guide it through the scenario. This \
            path is described as a **list of waypoints**. Depending on the precision of the path that we wish to achieve, \
            we can get those waypoints using splines such as  *Hermite splines or linear splines*. However, \
            for simplicity, we will stick to **linear splines** for now, which are just a succession of straight lines.")
spline_img = ['Hermite_spline.png', 'linear_spline.png']
st.image([f"{img_path}{name}" for name in spline_img], width=300, caption=['Hermite spline', 'Linear spline'])

st.markdown("- Although the AV can navigate in a 3D environment using Gazebo, our study will focus on a 2D representation \
         to simplify learning and task modeling. Therefore, the AV will be represented as a rectangle. To \
         maintain real-world proportions, we will use the dimensions of a standard Renault Zoe **$s = [s_x, s_y] = [4.1, 1.8]$m**[[11]](#reference).")
zoe_img = ['zoe1.webp', 'zoe2.webp', 'zoe3.webp']
st.image([f"{img_path}{zoe}" for zoe in zoe_img], width=300) 

st.markdown("- In the reality, the AV can perceive the environment **through sensors**. As we are working in a simulator, we choose \
            to represent the environment percieved by the AV as a **2D Occupancy Grid Map**[[6]](#reference)[[7]](#reference)[[8]](#reference)\
             like we capture it though a camera or a Lidar. This map will be updated \
            We will do the Hypothesis that the AV can map pedestrians in a **range $r_{rov}$= 20m** and a in a \
            **field of view equal to $\\alpha_{fov}$ = 360°** around it .")
st.markdown("The radius of the red circle -> $r_{rov}$, the angle of the red sector -> $\\alpha_{fov}$. ")
st.image(f"{img_path}AV_perception.gif")

st.markdown("- For the physical representation, we will use the **[kinematic bicycle model](https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html)**. \
            We  want to keep the model simple, easy to understand and we will disregard the dynamics of the car.")
st.image([f"{img_path}bicycle_model.png", f"{img_path}vehicle_dynamic.png"], caption=["", "Kabtoul, Maria, Anne Spalanzani, et Philippe Martinet. « Proactive And Smooth Maneuvering For Navigation Around Pedestrians ». In 2022 International Conference on Robotics and Automation (ICRA), 4723‑29. Philadelphia, PA, USA: IEEE, 2022. https://doi.org/10.1109/ICRA46639.2022.9812255"], width=400)

st.markdown("- To enhance the realism of our model and guarantee safe driving behavior, the speed for the AV is\
            $\\vec{v_{AV}} \in[-1, 4]$ m/s with a prefered speed equal to $\\vec{v^*_{AV}} = \max(\\vec{v_{AV}})$ , and the acceleration is $\\vec{a_{AV}} \in [-0.5,2]$ m/s². We grant the AV the ability to \
            go **backward** to explore pentential behaviors but we may refine this behavior to only go forward.")

st.markdown("- The output of the model will consist of two components: \
            **the self-propelled scalar speed of the autonomous vehicle's velocity** denoted as $v_{AV}$, and \
            **the steering angle of the front wheels**, represented by $\delta$. \
            If we were to model our vehicle's behavior after a real Renault Zoe, \
            we would need to constrain the steering angle ($\delta$) within the range of \
            [-13.73, 13.73] degrees (value obtained with the formula of the kinematic bicycle model), as the minimum braking range ($R$) for this car model \
            is 10.60 meters[[11]](#reference). These parameters enable us to calculate the necessary values \
            required for maneuvering within the simulator environment.")
st.image(f"{img_path}input_model_illustration.gif")


# Pedestrian Hypothesis
st.markdown("### 🚶 Pedestrian Hypothesis")
st.markdown("- Pedestrians can move freely within the environment, and their positions are updated at each time step.\
            In the simulator, they will be represented as circles with a radius $r_{ped} = 0.3$m. The circle representation \
            provide a good approximation of the space occupied by pedestrians (In a top-down perspective, a \
            pedestrian can rotate around the center, here the head). ")
st.image(f"{img_path}pedestrian_illustration.png", width=300)
st.markdown("- Pedestians have their own model to move and this model is based on the **Social Force Model** \
            which is a computational framework used to simulate the movement of individuals within a crowd by \
            modeling the interactions between them as forces. It describes how pedestrians navigate through a \
            space by considering factors such as attraction between groups and repulsion from obstacles. \
            The AV has prior knowledge of the pedestrian's model and use it to predict their movements and adapt its \
            trajectory accordingly.")
st.markdown("- The pedestrians will walk with a speed $v_{ped} \in [0,2]$ m/s and a prefered speed $v_{ped}^* = 1.4$ m/s [[9]](#reference)[[10]](#reference) but we may refine this values to \
            follow a normal distribution as $\mathcal{N}(1.4, 0.5)$.")

# Reward function
st.header("📊 Reward Function's Components", anchor="reward-function")
st.markdown("To effectively tackle our problem, we need a comprehensive value \
            to evaluate the behavior and decisions of our agent. Primarily, \
            our focus revolves around four critical aspects for vehicle operation: ensuring safety \
            and collision avoidance, following the prescribed path and maintain prefered speed. To accomplish this, \
            we decompose our value into four distinct components that can be individually \
            fine-tuned and refined.")
st.markdown("We name this value ($R$). We name each component as follows:")
st.markdown("- $r_c$: collision reward, to punish the agent for colliding with the pedestrian.")
st.markdown("- $r_{nc}$ : near collision reward, to punish the agent for getting too close to \
            the pedestrian. Not just a discrete reward, but a continuous one, to encourage the \
            agent to keep a safe distance from the pedestrian but not prevent it from getting close \
            to the pedestrian.")
st.markdown("- $r_s$ : speed reward, to reward the agent for moving fast/to the preferred speed. \
            This is to encourage the agent to move at a reasonable speed, \
            but not too fast or too slow.")
st.markdown("- $r_p$ : path following reward, to reward the agent for following the path. \
            This is to encourage the agent to follow the path, but not to prevent it from \
            deviating from the path if necessary.")
st.markdown("The Reward function is defined as follows:")
st.markdown(r"> $$R = r_c + r_{nc} + r_{s} + r_p$$")
st.markdown("We can **eventually** define individual component coefficients to adjust the importance of each component in the final reward. \
            Respectively, we will name them $w_c$, $w_{nc}$, $w_s$ and $w_p$.")
st.markdown(r"> $$R = w_c \cdot r_c + w_{nc} \cdot r_{nc} + w_s \cdot r_s + w_p \cdot r_p$$")
 
st.markdown("> 📌 **Side note**: the components $r_c,r_{nc}, r_{s}$ are based on reward function of **ADD_REF**.\
            The component $r_p$ is handcrafted")

# st.markdown("The path following score represents the distance between the initially given path to the autonomous \
#             vehicle (AV) and its future position.\
#              In this context, the variables $d_p$ and $d$ represent the penalty distance and the distance between \
#             the AV and the path, respectively. The formula used to calculate the path following score ($R_c$) is \
#             given by:")
# st.markdown(r"> $$R_c = 1-\frac{2}{1+e^{-d + d_p}}$$")
# st.markdown("This formula provides a measure of how closely the AV is following the path, with higher values \
#             of $R_c$ indicating better adherence to the track.")

st.markdown("First, we define the components $r_p$ responsible for the path-following reward. The path-following \
            reward is continuous, aiming to penalize the agent for deviating too far from the path or moving in the \
            wrong direction. To achieve this, we evaluate the angle error of the agent's trajectory concerning the \
            next waypoint on the path, denoted as $\\epsilon_{\\theta}$. Recognizing that considering only the angle error may be \
            insufficient, we also incorporate the distance between the agent and the path, referred to as \
            $\\epsilon_{path}$. These two values form the basis of the path-following component. We create two separate \
            versions—one utilizing only $\\epsilon_{\\theta}$ to explore its individual impact on behavior, and another combining \
            both elements to assess their collective influence.") 

st.image(f"{img_path}Path_following_reward.png", use_column_width=True)
st.markdown("The illustration demonstrates the calculation of $\\epsilon_{\\theta}$. As our agent advances along the path, \
            we compute the angle between the agent's current heading and the vector directed towards the next \
            waypoint. $d_w$ represents the distance between the agent and the next waypoint, triggering the agent \
            to proceed to the subsequent waypoint once it falls within a certain threshold.")

st.image(f"{img_path}path_following_illustration.png", use_column_width=True)
st.markdown("The illustration demonstrates the calculation of $$\\epsilon_{path}$$ as the distance between the agent and \
            the path. To incentivize the agent to remain near the path, we model the reward using a sigmoid function \
            centered around the path, with a variance equal to $\\sigma_{path}$.")

st.markdown("The path following reward with only $\\epsilon_{\\theta}$:")
st.markdown(r"> $$r_p = e^{-\frac{\epsilon_{\theta}}{20}}$$")
st.markdown("The path following reward with $\\epsilon_{\\theta}$ and $\\epsilon_{path}$:")
st.markdown(r"> $$r_p = 1-\frac{2}{1+e^{-d + d_p}}$$")#TODO: add the formula



# Plot
max_distance_display = 20
d = np.linspace(0, max_distance_display, 100)
penalty_distance = st.slider("Penalty distance", 0, 20, 10)
Rc = 1-2 / (1 + np.exp(-d + penalty_distance))
fig , ax = plt.subplots(1, 2 , figsize=(10, 5))

ax[1].plot(d, Rc)
ax[1].set_xlabel('distance between the future position and the path')
ax[1].set_ylabel('Reward')
ax[1].set_title('Reward function for path following')
ax[1].xaxis.set_ticks(np.arange(0, max_distance_display+1, 5))
ax[1].grid(True, linewidth=0.5, linestyle='--')
ax[1].set_xlim(0, max_distance_display)
ax[1].set_ylim(-1.1, 1.1)

path = np.array([[-15, -5], [15, 10]])
futur_point = np.random.uniform(-15, 15, 2)
# vector a is start point to futur point
a = futur_point - path[0]
# vector b is start point to end point
b = path[1] - path[0]
# d is the product of the magnitude of a and cos of the angle between a and b
d = np.linalg.norm(a) * np.cos(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
normal_point = path[0] + d * b / np.linalg.norm(b)
distance_to_rail = np.linalg.norm(futur_point - normal_point)
Rc = 1 - 2 / (1 + np.exp(-distance_to_rail + penalty_distance))
ax[1].axvline(distance_to_rail, color='g', linestyle='--')
ax[1].annotate(f'Distance to path: {distance_to_rail:.2f}\n path following reward: {Rc:.2f}',
                (distance_to_rail + 1, 0.5))

ax[0].plot(path[:, 0], path[:, 1], 'r')
ax[0].scatter(futur_point[0], futur_point[1], c='g', label='Future point')
ax[0].scatter(path[0, 0], path[0, 1], c='y', label='Start') 
ax[0].scatter(path[1, 0], path[1, 1], c='b', label='End')
ax[0].scatter(normal_point[0], normal_point[1], c='r', label='Normal point')
ax[0].set_xlim(-20, 20)
ax[0].set_ylim(-20, 20)
ax[0].legend(loc='lower right')

st.pyplot(fig)

# Motion Safety score
st.header("🦺 Motion Safety score", anchor="motion-safety-score")

st.markdown("> 📌 **Side note**: This metric is inspired by the paper \
            [Kabtoul, Maria, Manon Prédhumeau, Anne Spalanzani, Julie Dugdale, et \
            Philippe Martinet. « How To Evaluate the Navigation of Autonomous Vehicles \
            Around Pedestrians? » IEEE Transactions on Intelligent Transportation Systems, \
            2024, 1‑11. https://doi.org/10.1109/TITS.2023.3323662]. In the paper, this \
            metric was sugested but **not actually used**")
st.image(f'{img_path}motion_safety_notation.png')
st.markdown("Motion Safety refers to the assessment of potential risks and hazards associated with vehicle \
            movement, particularly in the presence of pedestrians. Key parameters involved in evaluating \
            motion safety include the evaluation radius ($eval_r$), the number of pedestrians within \
            this radius ($nb_{p}$), the distance between the vehicle and each pedestrian ($d_i$), and the penalty \
            distance ($d_p$). The safety score ($S_c$) is computed using the following formula:")
st.markdown(r"> $$S_c = -1 + \frac{2}{1+e^{-\frac{\sum_{i=1}^{nb_{p}}d_i}{d_i} + d_p}}$$")
st.write("which is equivalent to:")
st.markdown(r"> $$S_c = -1 + \frac{2}{1+e^{-\bar{d_i} + d_p}}$$")
st.markdown("This formula quantifies the level of safety during vehicle motion, with higher values of $S_c$ indicating \
            a greater degree of safety. We also add the same score but only evaluated with the closest pedestrian \
            to compare the two scores.")

def safety_reward(distance, penalty_distance):
    return 2 / (1 + np.exp(-distance + penalty_distance)) - 1

eval_radius = st.slider("Evaluation radius", 10, 30, 20)
nb_pedestrian = st.slider("Number of pedestrian", 1, 100, 10)
mu = 2.5 # in meter
# radom direction from the vehicle
random_vector = np.random.uniform(-np.pi, np.pi, (2, nb_pedestrian))
random_vector = np.array([np.cos(random_vector[0]), np.sin(random_vector[1])])
# normalize the vector
random_vector /= np.linalg.norm(random_vector, axis=0)
# random distance
d = np.random.randn(nb_pedestrian) * mu + eval_radius/2
# random position
pedestrian_pos = d * random_vector

penalty_distance = st.slider("Penalty distance", 0, 20, 6)
mean_distance = np.mean(d)
lowess_distance = d[np.argmin(np.abs(d))]
linspace_distance = np.linspace(0, eval_radius, 100)
Sc = safety_reward(linspace_distance, penalty_distance)

fig, axe = plt.subplots(1, 2, figsize=(10, 5))

axe[0].scatter(pedestrian_pos[0], pedestrian_pos[1], label='Pedestrian')
# the vehicle is a rectangle
axe[0].scatter(0, 0, label='Vehicle', c='r', marker='s', s=100)
axe[0].set_title('Pedestrian position')
axe[0].set_xlabel('x')
axe[0].set_ylabel('y')
axe[0].grid(True, linewidth=0.5, linestyle='--')
axe[0].legend()
axe[0].set_xlim(-eval_radius, eval_radius)
axe[0].set_ylim(-eval_radius, eval_radius)

axe[1].plot(linspace_distance, Sc)
axe[1].set_title('Safety reward')
axe[1].set_xlabel('Mean distance to the pedestrian')
axe[1].set_ylabel('Reward')
axe[1].grid(True, linewidth=0.5, linestyle='--')
axe[1].axvline(mean_distance, color='r', linestyle='--', label='Mean distance')
axe[1].annotate(f'Mean distance: {mean_distance:.2f}\nSafety reward: {safety_reward(mean_distance,penalty_distance):.2f}',
                (mean_distance + 1, 0.5))
axe[1].axvline(lowess_distance, color='r', linestyle='--', label='Lowess distance')
axe[1].annotate(f'Lowess Distance: {lowess_distance:.2f}\nSafety reward: {safety_reward(lowess_distance,penalty_distance):.2f}',
                (lowess_distance + 1, 0.0))


st.write(r"We created a scenario where our vehicle is positioned in the middle of a shared space with pedestrians around. In this representation, the pedestrians are positioned according to a normal distribution ($mu = \frac{eval_r}{2}, \sigma = 2.5$) shaped like a donut around the car.")
st.pyplot(fig)



# Senario
st.header("🎨 Scenario", anchor="scenario")
st.markdown("To conduct our experiments effectively, we are creating five scenarios to \
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
            its performance and behavior in various dynamic situations and avoid **overfitting**.")

senario_img = [f'{img_path}scenario{i}.png' for i in range(1,7)]
caption = ['Opposite Direction Collision',
            'Same Direction Collision',
            'Lateral Encounter with Crowd',
            'Encounter with Two Pedestrian Crowds', 
            'Shared Space with Random Pedestrian Movements', 
            'Legend']
st.image(senario_img, caption=caption, width=400) 

# Video
video_file = open(f'{img_path}illustration_senario.mp4', 'rb')
video_file = video_file.read()
st.video(video_file)
st.write('Link to the full video: [IROS21 rendered --- Anne Spalanzani](https://www.youtube.com/watch?v=DLaMMedWFn8)')

st.header("📚 References", anchor="reference")
st.markdown("[1] Salvini, Pericle, Diego Paez-Granados, et Aude Billard. 2022. « Safety \
            Concerns Emerging from Robots Navigating in Crowded Pedestrian Areas ». International \
            Journal of Social Robotics 14 (2): 441‑62. https://doi.org/10.1007/s12369-021-00796-4.")
st.markdown("[2] Kabtoul, Maria, Anne Spalanzani, et Philippe Martinet. 2022. « Proactive And Smooth Maneuvering \
            For Navigation Around Pedestrians ». In 2022 International Conference on Robotics and Automation (ICRA), \
            4723‑29. Philadelphia, PA, USA: IEEE. https://doi.org/10.1109/ICRA46639.2022.9812255.")
st.markdown("[3] Kabtoul, Maria, Manon Prédhumeau, Anne Spalanzani, Julie Dugdale, et Philippe Martinet. 2024. \
            « How To Evaluate the Navigation of Autonomous Vehicles Around Pedestrians? » IEEE Transactions on \
            Intelligent Transportation Systems, 1‑11. https://doi.org/10.1109/TITS.2023.3323662.")
st.markdown("[4] Alia, Chebly, Tagne Gilles, Talj Reine, et Charara Ali. 2015. « Local Trajectory Planning and \
            Tracking of Autonomous Vehicles, Using Clothoid Tentacles Method ». In 2015 IEEE Intelligent Vehicles \
            Symposium (IV), 674‑79. Seoul, South Korea: IEEE. https://doi.org/10.1109/IVS.2015.7225762.")
st.markdown("[5] https://medium.com/coinmonks/how-robots-follow-routes-pid-control-2a74226c5c99")
st.markdown("[6] Mouhagir, Hafida, Veronique Cherfaoui, Reine Talj, Francois Aioun, et Franck Guillemard. 2017. \
            « Using Evidential Occupancy Grid for Vehicle Trajectory Planning under Uncertainty with Tentacles ». \
            In 2017 IEEE 20th International Conference on Intelligent Transportation Systems (ITSC), 1‑7. Yokohama: \
            IEEE. https://doi.org/10.1109/ITSC.2017.8317808.")
st.markdown("[7] Genevois, Thomas, Anne Spalanzani, et Christian Laugier. 2023. \
            « Interaction-Aware Predicoverfittingtive Collision Detector for Human-Aware Collision Avoidance ». \
            In 2023 IEEE Intelligent Vehicles Symposium (IV), 1‑7. Anchorage, AK, USA: \
            IEEE. https://doi.org/10.1109/IV55152.2023.10186778.")
st.markdown("[8] Alia, Chebly, Tagne Gilles, Talj Reine, et Charara Ali. 2015. « Local Trajectory Planning \
            and Tracking of Autonomous Vehicles, Using Clothoid Tentacles Method ». In 2015 IEEE Intelligent \
            Vehicles Symposium (IV), 674‑79. Seoul, South Korea: IEEE. https://doi.org/10.1109/IVS.2015.7225762.")
st.markdown("[9] https://en.wikipedia.org/wiki/Preferred_walking_speed#cite_note-Browning2006-1")
st.markdown("[10] Reynolds, T.R. (1987), Stride length and its determinants in \
            humans, early hominids, primates, and mammals. Am. J. Phys. Anthropol., 72: \
            101-115. https://doi.org/10.1002/ajpa.1330720113")
st.markdown("[11] Renault Zoe's technical sheet https://www.renault.fr/vehicules-electriques/zoe/fiche-technique.html")