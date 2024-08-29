#import ray
#import tensorflow as tf
#import torch
import wandb
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import numpy

#from model import Model
from runner import Runner

from param import*
from tools import*

def get_episode_performances_dataframe(game_perf):
    perf_dict = {'ag_id': range(1, len(game_perf['final']['agents_success_step_final'])+1)}
    
    for type_m in game_perf:
        for metric in game_perf[type_m]:
            perf_dict[metric] = game_perf[type_m][metric]

    df = pd.DataFrame(perf_dict)
    return  df

def get_episode_sequences_dataframe(game_seq):
    seq_dict = {'ag_id': [ag_id for ag_id in game_seq]}
    
    for type_r in game_seq[1]:
        for type_a in game_seq[1][type_r]:
            for seq in game_seq[1][type_r][type_a]:
                seq_dict[str(type_r)+'_'+str(type_a)+'_'+seq] = [game_seq[ag_id][type_r][type_a][seq] for ag_id in game_seq] #this is a sequence

    df = pd.DataFrame(seq_dict)
    return  df

def record_episode(runner, dir_path):    
    #set paths
    env = runner.env
    world_dir_path = dir_path + "/world"
    agents_dir_paths = {"Agent"+str(id) : (dir_path + "/agent_" + str(id)) for id in env.team}
    
    #create folders
    if not os.path.exists(world_dir_path) : os.makedirs(world_dir_path)
    for agent in agents_dir_paths:
        if not os.path.exists(agents_dir_paths[agent]) : os.makedirs(agents_dir_paths[agent])
    
    #save agents' history
    for _id, agent in env.team.items():
        ag_dir_path = agents_dir_paths["Agent"+str(agent.id)]
        write_history('{}/history.txt'.format(ag_dir_path), agent.history)
        for oth_id in agent.memory.tracks: #save other agents' observations
            write_observations('{}/obervations_from_agent_{}.txt'.format(ag_dir_path, str(oth_id)), agent.memory.tracks[oth_id].obs_list, oth_id, agent.id)
        write_tracks_records('{}/tracks_records_from_agent_{}.txt'.format(ag_dir_path, str(agent.id)), agent.tracks_records)
        write_meeting_records('{}/meeting_records_from_agent_{}.txt'.format(ag_dir_path, str(agent.id)), agent.meetings_records)
        write_cm_metrics_records('{}/costmaps_metrics_of_agent_{}.txt'.format(ag_dir_path, str(env.team[_id].id)), env.team[_id].episode_cm_metrics)
        write_vpp_metrics_records('{}/viewpoint_planner_metrics_of_agent_{}.txt'.format(ag_dir_path, str(env.team[_id].id)), env.team[_id].episode_vpp_metrics)
        write_rrt_metrics_records('{}/rrt_metrics_of_agent_{}.txt'.format(ag_dir_path, str(env.team[_id].id)), env.team[_id].episode_rrt_metrics)
        write_pp_history('{}/path_planner_history_of_agent_{}.txt'.format(ag_dir_path, str(env.team[_id].id)), env.team[_id].episode_path_planner_history)

        write_idv_sequences('{}/sequences_of_agent_{}.txt'.format(ag_dir_path, str(agent.id)), agent.sequences)


    #save overall episode performance
    file_name = '{}/episode_performance.txt'.format(dir_path)
    write_episode_perf(file_name, runner.game_perf)
    
    #save overall episode performance
    df_perf = get_episode_performances_dataframe(runner.game_perf)
    df_perf.to_csv('{}/episode_perf_df.csv'.format(dir_path))
    df_perf.to_pickle('{}/episode_perf_df.pkl'.format(dir_path))

    #save overall episode performance
    df_seq = get_episode_sequences_dataframe(runner.game_seq)
    df_seq.to_csv('{}/episode_seq_df.csv'.format(dir_path))
    df_seq.to_pickle('{}/episode_seq_df.pkl'.format(dir_path))

    if MetaParameters.DRAW:
        #draw agent sequences
        pre_name = '{}/agents_sequence_'.format(dir_path)
        draw_team_sequence(pre_name, runner.game_seq, env.team)

    if MetaParameters.MEASURE_TIME:
        #record execution time
        file_name = '{}/global_exec_times.txt'.format(dir_path)
        write_exec_times(file_name, runner.run_exec_dic, runner.run_exec_stats)
        for _id, agent in env.team.items():
            ag_dir_path = agents_dir_paths["Agent"+str(agent.id)]
            write_exec_times('{}/exec_times_step.txt'.format(ag_dir_path), agent.step_exec_time, agent.step_exec_stats)
            write_exec_times('{}/exec_times_costmaps.txt'.format(ag_dir_path), agent.costmaps.cm_exec_time, agent.costmaps.cm_exec_stats)
            write_exec_times('{}/exec_times_viewpoint_planner.txt'.format(ag_dir_path), agent.viewpoint_planner.vpp_exec_time, agent.viewpoint_planner.vpp_exec_stats)
            write_exec_times('{}/exec_times_rrt.txt'.format(ag_dir_path), agent.RRTtree.rrt_exec_time, agent.RRTtree.rrt_exec_stats)

    #save GIF and images
    if MetaParameters.RENDER:

        #save Images
        for _id, agent in env.team.items():
            ag_dir_path = agents_dir_paths["Agent"+str(agent.id)]

            n_frames = len(agent.episode_agents_frames)

            if not MetaParameters.SPLIT_SCREEN:
                write_images('{}/image_'.format(ag_dir_path), agent.episode_agents_frames)
                write_images('{}/planner_'.format(ag_dir_path), agent.episode_planner_frames)
                write_images('{}/blind_'.format(ag_dir_path), agent.episode_blind_frames)
                if MetaParameters.RENDER_PDP:
                    write_images('{}/pdp_'.format(ag_dir_path), agent.episode_distrib_frames)
                else:
                    write_images('{}/pog_'.format(ag_dir_path), agent.episode_pog_frames)

                if MetaParameters.SAVE_GIF and n_frames > 3:
                    write_gif('{}/episode.gif'.format(ag_dir_path), agent.episode_agents_frames, fps = MetaParameters.FPS, true_image=True, salience=False)
                    write_gif('{}/planner.gif'.format(ag_dir_path), agent.episode_planner_frames, fps = MetaParameters.FPS, true_image=True, salience=False)
                    write_gif('{}/blind_episode.gif'.format(ag_dir_path), agent.episode_blind_frames, fps = MetaParameters.FPS, true_image=True, salience=False)
                    if MetaParameters.RENDER_PDP:
                        write_gif('{}/pdp_episode.gif'.format(ag_dir_path), agent.episode_distrib_frames, fps = MetaParameters.FPS, true_image=True, salience=False)
                    else:
                        write_gif('{}/pog_episode.gif'.format(ag_dir_path), agent.episode_pog_frames, fps = MetaParameters.FPS, true_image=True, salience=False)

            else: #combine frames                    
                combined_frames = []
                for i in range(0, n_frames):
                    if type(agent.episode_agents_frames[i]) == np.ndarray:
                        line_1 = np.concatenate((agent.episode_agents_frames[i], agent.episode_planner_frames[i]), axis=1)
                        if MetaParameters.RENDER_PDP:
                            line_2 = np.concatenate((agent.episode_blind_frames[i], agent.episode_distrib_frames[i]), axis=1)
                        else:
                            line_2 = np.concatenate((agent.episode_blind_frames[i], agent.episode_pog_frames[i]), axis=1)
                        array = np.concatenate((line_1, line_2), axis=0)
                        combined_frames.append(array)
                    else:
                        combined_frames.append(None)
                write_images('{}/image_'.format(ag_dir_path), combined_frames)
                if MetaParameters.SAVE_GIF and n_frames > 3:
                    write_gif('{}/episode.gif'.format(ag_dir_path), combined_frames, fps = MetaParameters.FPS, true_image=True, salience=False)
        
        #world frames
        if True:
            write_images('{}/image_'.format(world_dir_path), env.episode_world_frames)
            if MetaParameters.SAVE_GIF and len(env.episode_world_frames) > 3:
                write_gif('{}/episode.gif'.format(world_dir_path), env.episode_world_frames, fps = MetaParameters.FPS, true_image=True, salience=False)

def get_mission_dataframe(games_perf_saved):
    mission_dict = {
        'run_id': [],
        'ag_id': [],
        }
    
    for type_r in games_perf_saved[0]:
        for metric in games_perf_saved[0][type_r]:
            mission_dict[metric] = []

    for i_run in range(len(games_perf_saved)):
        for i_ag in range(len(games_perf_saved[i_run]['final']['exploring_completeness_final'])):
            mission_dict['run_id'].append(i_run+1)
            mission_dict['ag_id'].append(i_ag+1)
            for type_r in games_perf_saved[i_run]:
                for metric in games_perf_saved[i_run][type_r]:
                    mission_dict[metric].append(games_perf_saved[i_run][type_r][metric][i_ag])

    df = pd.DataFrame(mission_dict)
    return  df

def get_mission_stats(games_perf_saved):
    #import pandas as pd
    # stats = pd.DataFrame(stats)
    # stats['key'].mean()
    # stats.to_csv()
    metrics_lists = {
        # exemple : 'agents_success_step_list' : [games_perf_saved[i_run]['final']['agents_success_step_final'][i_ag] for i_run in range(len(games_perf_saved)) for i_ag in range(len(games_perf_saved[i_run]['final']['agents_success_step_final']))],
        # exemple : 'exploring_completeness_final_list' : [games_perf_saved[i_run]['final']['exploring_completeness_final'][i_ag] for i_run in range(len(games_perf_saved)) for i_ag in range(len(games_perf_saved[i_run]['final']['exploring_completeness_final']))], #exploration
        # exemple : 'team_known_perc_are_list' : [games_perf_saved[i_run]['are']['team_known_perc_are'][i_ag] for i_run in range(len(games_perf_saved)) for i_ag in range(len(games_perf_saved[i_run]['are']['team_known_perc_are']))],       
    }

    for type_m in games_perf_saved[0]:
        for metric in games_perf_saved[0][type_m]:
            name = str(metric)+'_list'
            metrics_lists[name] = [games_perf_saved[i_run][type_m][metric][i_ag] for i_run in range(len(games_perf_saved)) for i_ag in range(len(games_perf_saved[i_run][type_m][metric]))]

    #calculate overall mission_stats
    mission_stats = {}
    for m_list in metrics_lists:

        if 'bool' in m_list:
            avg_name = str(m_list).replace('list','ratio')
            mission_stats[avg_name] = round(statistics.mean(metrics_lists[m_list]), 2)
            continue

        avg_name = str(m_list).replace('list','overall_avg')

        if type(metrics_lists[m_list][0]) is tuple:
            values = [elem[0] for elem in metrics_lists[m_list]]
            weights =  [elem[1] for elem in metrics_lists[m_list]]
            mission_stats[avg_name] = round(numpy.average(values, weights=weights), 2) if max(weights)>0 else False
            continue

        nn_false_list = [elem for elem in metrics_lists[m_list] if elem is not False]
        mission_stats[avg_name] = round(statistics.mean(nn_false_list), 2) if nn_false_list != [] else False

        if len(games_perf_saved) >= 3 and nn_false_list != []:
            std_name = str(m_list).replace('list', 'overall_std')
            mission_stats[std_name] = round(statistics.stdev(metrics_lists[m_list]), 1)

    # exemples:
    # mission_stats['agents_succes_step_overall_avg'] = int(statistics.mean([elem for elem in metrics_lists['agents_success_step_list'] if elem !=0]))
    # if len(games_perf_saved) >= 3 and [elem for elem in metrics_lists['agents_success_step_list'] if elem] != [] : mission_stats['agents_succes_step_overall_stdev'] = int(statistics.stdev([elem for elem in metrics_lists['agents_success_step_list'] if elem !=0]))

    # mission_stats['exploring_completeness_final_overall_avg'] = round(statistics.mean(metrics_lists['exploring_completeness_final_list']), 1)
    # if len(games_perf_saved) >= 3 : mission_stats['exploring_completeness_final_overall_stdev'] = round(statistics.stdev(metrics_lists['exploring_completeness_final_list']), 1)

    # if [elem for elem in metrics_lists['team_known_perc_are_list'] if elem] != [] : mission_stats['team_known_perc_are_overall_avg'] = round(statistics.mean(metrics_lists['team_known_perc_are_list']), 1)
    # if len(games_perf_saved) >= 3 and [elem for elem in metrics_lists['team_known_perc_are_list'] if elem] != [] : mission_stats['team_known_perc_are_overall_stdev'] = round(statistics.stdev(metrics_lists['team_known_perc_are_list']), 1)

    return metrics_lists, mission_stats

def get_median_progression(games_seq_saved):
    
    median_prog = {}
    for type_m in games_seq_saved[0][1]:
        median_prog[type_m] = {}
        for type_a in games_seq_saved[0][1][type_m]:
            median_prog[type_m][type_a] = {}
            for seq in games_seq_saved[0][1][type_m][type_a]:

                if 'inst' in seq:
                    continue
                
                #init
                median_prog[type_m][type_a][seq] = []

                #fill
                for i_step in range(EnvParameters.N_STEPS):
                    pre_list = []
                    for i_run in range(len(games_seq_saved)):
                        for i_agent in games_seq_saved[i_run]:
                            if len(games_seq_saved[i_run][i_agent][type_m][type_a][seq]) > i_step:
                                if games_seq_saved[i_run][i_agent][type_m][type_a][seq][i_step]:
                                    pre_list.append(games_seq_saved[i_run][i_agent][type_m][type_a][seq][i_step])
                            else:
                                if games_seq_saved[i_run][i_agent][type_m][type_a][seq][-1]:
                                    pre_list.append(games_seq_saved[i_run][i_agent][type_m][type_a][seq][-1])
                    
                    if [elem for elem in pre_list if elem] != [] :
                        median_prog[type_m][type_a][seq].append(statistics.median(pre_list)) 
                    else : median_prog[type_m][type_a][seq].append(None)

    return median_prog

def init_wandb():
    # start a new wandb run to track this script
    wandb_id = wandb.util.generate_id()

    wandb.init(
        # set the wandb project where this run will be logged
        project = MetaParameters.WANDB_PROJECT_NAME,
        name = MetaParameters.WANDB_EXPERIMENT_NAME,
        entity = MetaParameters.WANDB_ENTITY,
        notes = MetaParameters.WANDB_NOTES,
        id = wandb_id,
        resume = 'allow',
        
        # track hyperparameters and run metadata
        config={
            "n_episodes": RunningParameters.N_RUNS,
        }
    )
    print('id is:{}'.format(wandb_id))
    print('Launching wandb ...\n')
    #setproctitle.setproctitile(MetaParameters.WANDB_PROJECT_NAME+MetaParameters.WANDB_EXPERIMENT_NAME+'@'+MetaParameters.WANDB_ENTITY)

def write_to_wandb(game_stats):
        
    dic = {
        #exemple : "agents_success_ratio": game_stats['agents_success_ratio'],
    }
    for stat in game_stats:
        if 'avg' in stat:
            dic[stat] = game_stats[stat]
    wandb.log(dic)
    

#---main function---
def single_episode(i_run, runner, render = False, measure_time = False, record = False, dir_path = None, wand = False):
    print("\n-----\ni_run :", i_run+1, "\nRunning episode ...")
    runner.run_episode(render)
    print("\n-----\nEpisode is over")

    #evaluate performance of the episode and record it into the dic
    runner.save_episode_data()
    game_perf = {k : v for k, v in runner.game_perf.items()} #a way to copy the dictionary
    game_stats = {k : v for k, v in runner.game_stats.items()} #a way to copy the dictionary
    game_seq = {k : v for k, v in runner.game_seq.items()} #a way to copy the dictionary
    if wand : write_to_wandb(game_stats)

    #measure time, record episode and sent to wandb
    if measure_time : record_exec_time_stats(runner)
    if record : record_episode(runner, dir_path)

    #reset episode, world (map, agents including closing viewers) and meta variables
    runner.env.reset()

    return game_perf, game_stats, game_seq

def mission(i_env):
    print('Mission starts!')

    #ENV AND RUNNER
    print("\n-----\n-----\ni_env :", i_env+1, "\nCreating Environment ...")
    imposed_map = EnvParameters.IMPOSED_MAP[i_env]
    init_poses = EnvParameters.INIT_POSES[i_env]

    runner = Runner(i_env, imposed_map, init_poses)
    n_runs = RunningParameters.N_RUNS

    #
    if MetaParameters.WANDB : init_wandb()

    #RUN IN SERIAL OR PARALLEL
    if RunningParameters.PAR_POOL:
        p = Pool(RunningParameters.N_PROCESSES)
        render=False
        measure_time=False
        record=False
        dir_path=None
        wand = MetaParameters.WANDB
        results = p.starmap(single_episode, [[i_run, runner, render, measure_time, record, dir_path, wand] for i_run in range(n_runs)])
        p.close()
        p.join()
        games_perf_saved = [game_res[0] for game_res in results]
        games_stats_saved = [game_res[1] for game_res in results]
        games_seq_saved = [game_res[2] for game_res in results]

    else:
        games_perf_saved = []
        games_stats_saved = []
        games_seq_saved = []
        for i_run in tqdm(range(n_runs), position = 0):
            render = MetaParameters.RENDER and (i_run == n_runs -1)
            measure_time = MetaParameters and (i_run == n_runs -1)
            record = MetaParameters.RECORD and (i_run == n_runs -1)
            if n_runs == 1 and RunningParameters.N_ENVS == 1 : dir_path = MetaParameters.GIFS_PATH
            else: dir_path = MetaParameters.GIFS_PATH + '/env_'+str(i_env+1)+'/episode_'+str(i_run+1)
            wand =  MetaParameters.WANDB

            game_perf, game_stats, game_seq = single_episode(i_run, runner, render, measure_time, record, dir_path, wand)
            games_perf_saved.append(game_perf)
            games_stats_saved.append(game_stats)
            games_seq_saved.append(game_seq)
    
    #MISSION STATS AND GLOBAL PROGRESSION
    mission_df = get_mission_dataframe(games_perf_saved)
    metrics_lists, mission_stats = get_mission_stats(games_perf_saved)
    median_prog = get_median_progression(games_seq_saved)

    #record stats
    if MetaParameters.RECORD and (n_runs > 1 or RunningParameters.N_ENVS > 1):
        if n_runs == 1 and RunningParameters.N_ENVS == 1 : dir_path = MetaParameters.GIFS_PATH
        else : dir_path = MetaParameters.GIFS_PATH + '/env_'+str(i_env+1)
        if not os.path.exists(dir_path) : os.makedirs(dir_path)

        #write parameters
        file_name = '{}/parameters.txt'.format(dir_path)
        shutil.copyfile('param.py', file_name)
        print("Parameters file has been saved!")

        #write mission data and stats
        mission_df.to_csv('{}/mission_df.csv'.format(dir_path))
        mission_df.to_pickle('{}/mission_df.pkl'.format(dir_path))
        write_mission_stats('{}/mission_stats.txt'.format(dir_path), metrics_lists, mission_stats, median_prog, games_perf_saved, games_stats_saved)
        
        if MetaParameters.DRAW:
            draw_median_progression('{}/median_'.format(dir_path), median_prog)

    if MetaParameters.WANDB :
        wandb.finish()
    
    if MetaParameters.RENDER:
        print("Closing viewers ...")
        runner.env.close_viewers()


#------ MAIN ------
if __name__ == '__main__':
    print("Welcome to MA_SLAM")
    
    for i_env in range(RunningParameters.N_ENVS):
        mission(i_env)

    print("Good bye world!")






