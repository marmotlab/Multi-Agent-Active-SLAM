Welcome to the Multi-Agent SLAM and Active SLAM code repository!

This README will guide you through the structure of the code, setting up your environment, running experiments, and checking results.

#Code Structure
The codebase consists of the following files:
    - param.py: contains the parameters for the experiments, such as Environment, Agents, Episodes, Planning mode, Rendering, etc.
    - imposed_world.py: contains predefined worlds for the experiments, including indoor, semi-open, and open environments
    - driver.py: the root file to be executed to start the experiment, it gathers performance metrics for the entire experiment
    - runner.py: contains the Runner class, used to run an episode and gather performance metrics.
    - env.py: defines the Environment class
    - agent.py: defines the Agent class and includes correction abilities
    - components.py: contains the components of the agent, including Move base, Sensor, Map, and Planner
    - planner.py: defines the Cost-maps and Viewpoint planner classes
    - tree.py: implements the RRT tree for the multiverse approach
    - virtual_agent.py (and virtual_agent2.py): used for the RRT multiverse
    - objects.py: defines Observation, Track, and Trace objects, as well as Memory, History, and other data recorded during the episode
    - tools.py: contains utility functions to write files after an episode

Additional functions files include:
    - a_star_v2.py
    - clustering.py
    - connected_points.py
    - diffuse_array.py
    - distance_to_line.py
    - entropy_functions.py
    - generate_path.py
    - get_loop_pos_distrib.py
    - get_meet_pos_distrib.py (and get_pos_distrib_plans.py and get_pos_distrib_mc.py and get_pos_distrib_mc_separate.py)
    - get_pos_distrib.py (and get_pos_distrib_mc.py)
    - intersection.py
    - meeting.py
    - merge_segments.py
    - multi_obj_tsp.py (and multi_obj_tsp_2.py)
    - path_exists.py
    - render.py
    - union_intervals.py

#Setting Up Your Environment
    - Create a new environment: you can use either Linux or Conda.
        - Linux (or WSL) :
        - Conda : conda create --name env_name python=3.9

    - Install Packages:
        - python : version = 3.9 (terminal : sudo apt-get install python3-pip)
        - numpy : version = 1.24.1 (terminal : pip3 install numpy)
        - matplotlib : version = 3.6.3
        - imageio : version = 2.25.0
        - shapely : version = last
        - tqdm : version = last
        - wandb : version = last
        - gym : version = 0.21.0
            - terminal : python3 -m install gym=0.21.0
            - linux : pip3 install gym==0.21.0 or pip3 install git+https://github.com/openai/gym.git@9180d12e1b66e7e2a1a622614f787a6ec147ac40
            - alternatively, install from the GitHub repository : https://github.com/openai/gym/releases/tag/v0.21.0
            - install gym and add the rendering file from from gym==0.21 envs.classic_control
        - pyglet==1.5.27 (terminal : "python3 -m pip install pyglet==1.5.27")

    - Source the environment :
        - Linux : $source env_folder/bin/activate
        - Conda : $conda activate env_name

#Running an Experiment
    - Edit Parameters:
        - open param.py in a code editor like VS Code.
        - adjust parameters as needed (for multiple runs, parallelize the planner by setting the Boolean to True in param.py)

    - Execute driver : $python3 driver.py or $python driver.py

    - Checking results:
        For a single episode :
            - navigate to the gif/test***1r*** folder
            - inside each agent's folder, you will find:
                - images
                - gif
                - history.txt
                - execution time files (txt)
                - records (observations, meetings, tracks)
                - metrics (cost maps, rrt, viewpoint planner, path planner)
            - review sequences (graphs), episode performance files (data frame and text files), and global execution time (txt).
        
        For multiple runs :
            - go to the gif/test***100r*** folder (example for 100 runs)
            - check mission stats (mean performance values, sequences, and median progressions - txt)
            - check median curves (graphs)


Enjoy exploring and utilizing the Multi-Agent SLAM and Active SLAM code! If you have any issues or questions, feel free to reach out.
Maxime de Montlebert
maxime.montleb@hotmail.com
