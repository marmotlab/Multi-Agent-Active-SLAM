from imposed_world import Env_1515, Env_3030

class EnvParameters:
    #Env param
    N_STEPS = 250
    N_AGENTS = 4

    #Map param
    MAP_HEIGHT = 30
    MAP_WIDTH = MAP_HEIGHT

    OCC_RATE = 0.15

    random_map = False
    '''
    #Select map and initial poses
    #Generate map
    #IMPOSED_MAP = np.random.choice([0,1], [MAP_HEIGHT, MAP_WIDTH], p=[1-OCC_RATE, OCC_RATE])
    #Generate initial poses
    #INIT_POSES = []
    #for _ in range(N_AGENTS):
    #    INIT_POSES.append((rd.randrange(0, MAP_HEIGHT), rd.randrange(0, MAP_WIDTH)))
    #Preselect initial poses
    #INIT_POSES = [(6,1), (2,5), (8,4), (15,9), (4,19), (14,27), (25,3), (12,32), (39,22)]
    #Comment above to set a random initial pose (for each run)
    #'''
    #Random map
    if random_map: 
        IMPOSED_MAP = [None for _ in range(10)]
        INIT_POSES = [None for _ in range(10)]
    #Predefined map
    elif MAP_HEIGHT == 15:
        IMPOSED_MAP = Env_1515.OPEN_MAP_1515
        INIT_POSES = Env_1515.OPEN_POSES_1515
    elif MAP_HEIGHT == 30:
        # #open map
        # IMPOSED_MAP = Env_3030.OPEN_MAP_3030    
        # INIT_POSES = Env_3030.OPEN_POSES_3030

        # #semi open map
        # IMPOSED_MAP = Env_3030.SEMI_OPEN_MAP_3030    
        # INIT_POSES = Env_3030.SEMI_OPEN_POSES_3030

        # #indoor map
        # IMPOSED_MAP = Env_3030.INDOOR_MAP_3030    
        # INIT_POSES = Env_3030.INDOOR_POSES_3030

        #open same start
        IMPOSED_MAP = Env_3030.OPEN_MAP_3030    
        INIT_POSES = Env_3030.OPEN_SAME_POSES_3030
    #else random map

class AgentParameters:
    #Scanner
    RANGE = 2

    #Odometry param
    ODOM_ERROR_RATE = 0.15

    #Submap param
    OFF_SET = (3, 3)
    SUBMAP_MAX_HEIGHT = int(EnvParameters.MAP_HEIGHT + 2* OFF_SET[0])
    SUBMAP_MAX_WIDTH = int(EnvParameters.MAP_WIDTH+ 2* OFF_SET[1])
    FRONTIER_DEPTH = RANGE

    #Sharing
    SHARING_DATA = True
    NEIGHBOURHOOD_PERIOD = 4

    #Correcting
    SELF_LOOP_COR = True
    MA_LOOP_COR = True
    MEETING_LOOP_COR = True

    #blind
    INIT_BLIND = 1e3

    #Entropy
    MH_UNKNOWN_PENALTY = 1/3 #so that mh_uncertain ~= mh_unknown at start
    MH_THRES_1 = None
    MH_THRES_2 = 50
    PHW = 1
    MHW = 1

    #Metrics param
    RECENTLY_SEEN_THRESHOLD = 30 #threshold above which an agent is considered to have been seen recently (to measure the proximity of an agent to the others)

    #Path Planner param
    PLANNER_MODE = 'fast cm' #in ['random', 'rd explore', 'max cm' ,'fast cm', 'long term cm', 'frontier', 'rrt']
    
    if PLANNER_MODE == 'random':
        PLANNER_PARAM = {
            'COSTMAPS' : False,
            'TREE_MODE' : False,
            'VPP_MODE' : False,
            'MULTI_GOALS' : False,
            'PLANNER_RANGE' : 8, #-> short term
        }
    elif PLANNER_MODE == 'rd explore':
        PLANNER_PARAM = {
            'COSTMAPS' : False,
            'TREE_MODE' : False,
            'VPP_MODE' : False,
            'MULTI_GOALS' : False,
            'PLANNER_RANGE' : 8, #-> short term
            'EXPLO_RATE' : 0.7, #Random Explore planner
            'RANDOM_RATE' : 0,
        }
    elif PLANNER_MODE == 'frontier':
        PLANNER_PARAM = {
            'COSTMAPS' : False,
            'TREE_MODE' : False,
            'VPP_MODE' : False,
            'MULTI_GOALS' : False,

            'PLANNER_RANGE' : 70, #-> short term
            'RD_FRONTIER_RATE' : 0,
        }
    elif PLANNER_MODE == 'max cm':
        PLANNER_PARAM = {
            'COSTMAPS' : True,
            'TREE_MODE' : False,
            'VPP_MODE' : True,
            'MULTI_GOALS' : False,
            'PLANNER_RANGE' : 11, #-> short term
            'RANDOM_RATE' : 0, #Max Costmap planner and Long Term planner -> set to zero by default
            'ST_PLANNING_RANGE' : 8, #-> random
            'INTEREST_THRES' : 5,
        }
    elif PLANNER_MODE == 'fast cm':
        PLANNER_PARAM = {
            'COSTMAPS' : True,
            'TREE_MODE' : False,
            'VPP_MODE' : True,
            'MULTI_GOALS' : False,
            'PLANNER_RANGE' : 50, #-> all
            'MIN_RANGE' : 3, #-> all
            'DISCOUNT_RATE' : 0.99,
            'RANDOM_RATE' : 0, #Short Term planner and Long Term planner -> set to zero by default
            'ST_PLANNING_RANGE' : 8, #-> random
            'INTEREST_THRES' : 5,
        }
    elif PLANNER_MODE == 'long term cm':
        PLANNER_PARAM = {
            'COSTMAPS' : True,
            'TREE_MODE' : False,
            'VPP_MODE' : True,
            'MULTI_GOALS' : True,

            'PLANNER_RANGE' : 50,
            'MIN_RANGE' : 3,

            'RANDOM_RATE' : 0, #Short Term planner and Long Term planner -> set to zero by default
            'ST_PLANNING_RANGE' : 8, #-> if random

            'MAX_N_GOALS' : 20,
            'MIN_N_GOALS' : 8,
            'MAX_GOALS_RATIO' : 15,
            'MAX_VALUE_TOL_IN' : 0.5,
            'MAX_VALUE_TOL_OUT' : 0.2,
            'MAX_HORIZON_PLANNING' : 70, #max_distance
            'MAX_DIST_BTW_TARGETS' : 70,

            'CLUSTERING_METHOD' : 'v2', #optimization per step or with sum
            'PER_STEP' : True, #optimization per step or with sum

            'DISCOUNT_RATE' : 0.97,
            'DISCOUNT_RATIO' : 0.8,

            'INTEREST_THRES' : 5,
        }
    elif PLANNER_MODE == 'rrt':
        PLANNER_PARAM = {
            'COSTMAPS' : True, #turn to True for a bias tree sampling #can be False if no bias
            'TREE_MODE' : True,
            'VPP_MODE' : False,
            'MULTI_GOALS' : True,
            'PLANNER_RANGE' : 50, #-> long term
            'ACTION_METHOD' : 'max_step_IG', #in [rd, rd_leaf, max_IG, max_step_IG]
        }
    else :
        print("Planner Error")

    PLANNER_MAX_TRY = 5

    #ma penalty
    MA_PENALTY_MODE = True
    MA_PENALTY_RANGE = 5
    MA_PENALTY_MAX_TIME = 10

    #automatic replanning
    MAX_TS_NON_REPLANNING = 10

    #thresholds
    COMPLETENESS_THRESHOLD = 95
    CORRECTNESS_THRESHOLD = 90
    MEAN_ERROR_THRESHOLD = 1

    COMPLETENESS_DONE = 99
    CORRECTNESS_DONE = 99
    MEAN_ERR_DONE = 0

class CostmapsParameters:
    #team distrib param
    DISTRIB_METHOD = 'mc' #'mc' or 'anything' => reg model 
    LOST_STEPS = 30 #same
    IMPACT_STEPS = 20 #get other pos distrib, linear factor for no presence impact (meeting cost map)
    TRUST_PLAN_FACTOR = 0.9

    #global costmap balance
    EXPLORE_WEIGHT = 1
    LOOP_WEIGHT = 1
    MEET_WEIGHT = 1
    MEET_OFFSET = 0

class RRTParameters:
    #tree
    MAX_N_NODES = 25
    MIN_N_NODES = 10
    NODES_RATIO = 15
    MIN_NEW = 5
    IDEAL_NEARBY_NODES = 7
    SAMPLING_MAX_RANGE = None
    SAMPLING_METHOD = 'bias' #in [explore, known, bias, bias]
    BIAS_RATE = 0.8 #else random

    RRT_METHOD = 'rrt_star_rewire' #in [rrt, rrt_star, rrt_star_rewire]
    MIN_DIST_B_NODES = 3
    MAX_EDGE_LENGHT = 8
    MIN_EDGE_LENGHT = 4
    NEIGHBOURHOOD_DISTANCE = MAX_EDGE_LENGHT

    #info
    DISTRIB_METHOD = 'reg' #'anything' => reg model or 'mc' (too expensive)
    MC_MEMO = False #lighten the calculation for mc distrib method
    LOST_STEPS = CostmapsParameters.LOST_STEPS
    IMPACT_STEPS = CostmapsParameters.IMPACT_STEPS
    TRUST_PLAN_FACTOR = CostmapsParameters.TRUST_PLAN_FACTOR

    CALCULATE_LOOP = True
    CALCULATE_MEET = False

class RewardParameters:
    #agent related events : meet/loop and correct
    TRAVELLED_DIST = 0
    N_MEETINGS = 1
    N_NEW_NEIGHBOURS = 1
    N_NEW_DATA = 1
    N_SELF_LOOPS = 1
    N_MA_META_LOOPS = 1
    N_MEETING_META_LOOPS = 1
    N_MEETING_CORR = 0
    N_CORRECTIONS = 1

    #submap
    N_SQUARE_KNOWN = 1
    N_AGENTS = 1

    #tracks
    N_TRACKS = 0
    N_SELF_OBS = 0
    N_OBS = 0
    N_VISITED_POS = 1

class RunningParameters:
    N_ENVS = 1
    N_RUNS = 1

    PAR_POOL = False
    N_PROCESSES = 6

class RenderingParameters:
    SCREEN_DIM = (500, 500)

    #Rendering param
    RENDER_GT = True

    RENDER_TRACE = True

    RENDER_LOOPS = True
    RENDER_TRACE_LENGTH = 100
    
    RENDER_VISITS = True
    FULLY_SCANNED = 30
    INIT_BLIND = AgentParameters.INIT_BLIND
    MAX_BLIND = 100

class MetaParameters:
    exp_id = 642
    special_test = 'gif'

    if AgentParameters.PLANNER_MODE == 'random':
        planner_carcteristics = 'rd'
    elif AgentParameters.PLANNER_MODE == 'rd explore':
        planner_carcteristics = 'rdex' + str(int(AgentParameters.PLANNER_PARAM['EXPLO_RATE']*10)) + str(int(AgentParameters.PLANNER_PARAM['PLANNER_RANGE']))
    elif AgentParameters.PLANNER_MODE == 'frontier':
        planner_carcteristics = 'fr' + str(int(AgentParameters.PLANNER_PARAM['PLANNER_RANGE']))
    elif AgentParameters.PLANNER_MODE == 'max cm':
        planner_type = 'maxcm' + str(AgentParameters.PLANNER_PARAM['PLANNER_RANGE'])
        cm_balance = str(CostmapsParameters.EXPLORE_WEIGHT) + str(CostmapsParameters.LOOP_WEIGHT) + str(CostmapsParameters.MEET_WEIGHT)
        planner_carcteristics = planner_type + '_' + cm_balance + 'b'
    elif AgentParameters.PLANNER_MODE == 'fast cm':
        planner_type = 'fastcm' + str(int(AgentParameters.PLANNER_PARAM['DISCOUNT_RATE']*100))
        cm_balance = str(CostmapsParameters.EXPLORE_WEIGHT) + str(CostmapsParameters.LOOP_WEIGHT) + str(CostmapsParameters.MEET_WEIGHT)
        planner_carcteristics = planner_type + '_' + cm_balance + 'b'
    elif AgentParameters.PLANNER_MODE == 'long term cm':
        planner_type = str(AgentParameters.PLANNER_PARAM['PER_STEP']*'fast') + 'ltcm' + str(AgentParameters.PLANNER_PARAM['PLANNER_RANGE'])
        lt_param = str(int(AgentParameters.PLANNER_PARAM['DISCOUNT_RATE']*100)) + str(int(AgentParameters.PLANNER_PARAM['DISCOUNT_RATIO']*10))
        cm_balance = str(CostmapsParameters.EXPLORE_WEIGHT) + str(CostmapsParameters.LOOP_WEIGHT) + str(CostmapsParameters.MEET_WEIGHT)
        planner_carcteristics = planner_type + '_' + lt_param + 'd_' + cm_balance + 'b'
    elif AgentParameters.PLANNER_MODE == 'rrt':
        planner_type = 'rrt' + str(AgentParameters.PLANNER_PARAM['PLANNER_RANGE'])
        method = str(RRTParameters.SAMPLING_METHOD) + '_' + str(RRTParameters.SAMPLING_MAX_RANGE) + '_'
        rrt_param = str(RRTParameters.MAX_EDGE_LENGHT) + '_' + str(RRTParameters.NODES_RATIO)
        planner_carcteristics = planner_type + method + rrt_param

    penalty_mode = 'mp' + str(AgentParameters.MA_PENALTY_RANGE) + str(AgentParameters.MA_PENALTY_MAX_TIME) if AgentParameters.MA_PENALTY_MODE else 'np'
    name = str(exp_id) + '_' + str(EnvParameters.N_AGENTS) + 'a_' + str(RunningParameters.N_ENVS) + 'e' + str(RunningParameters.N_RUNS) + 'r_'  + planner_carcteristics + '_' + penalty_mode + '_'  + special_test

    #display
    DISPLAY = False
    #DISP_STEP = True

    #render
    rend = True
    RENDER_AG_MAP = True
    RENDER_BLIND = True
    RENDER_POG = True
    RENDER_PL = True
    RENDER_COSTMAPS = (AgentParameters.PLANNER_PARAM['VPP_MODE'] == True or (AgentParameters.PLANNER_PARAM['TREE_MODE'] == True and AgentParameters.PLANNER_PARAM['COSTMAPS'] == True))
    RENDER_VPP = True * AgentParameters.PLANNER_PARAM['VPP_MODE'] == True
    RENDER_RRT = True * AgentParameters.PLANNER_PARAM['TREE_MODE'] == True
    RENDER_PDP = True * AgentParameters.PLANNER_PARAM['COSTMAPS'] == True

    RENDER = (RENDER_AG_MAP or RENDER_PL or RENDER_VPP or RENDER_RRT or RENDER_BLIND or RENDER_POG) * rend
    SPLIT_SCREEN = True
    SAVE_GIF = True
    FPS = 4

    #record
    #RECORD = EnvParameters.N_STEPS >= 150
    RECORD = True
    GIFS_PATH = 'gifs/test' + name
    MEASURE_TIME = True

    #draw
    DRAW = True

    #wandb
    #WANDB = (RunningParameters.N_RUNS > 1)
    WANDB = False
    WANDB_PROJECT_NAME = 'project_name'
    WANDB_EXPERIMENT_NAME = 'test' + name
    WANDB_ENTITY = 'entity'
    WANDB_NOTES = 'planner_caracteristics : ' + planner_carcteristics