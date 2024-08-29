import os
import shutil
import imageio
import statistics
import numpy as np
import matplotlib.pyplot as plt

from agent import History, Observation

from param import MetaParameters, AgentParameters

#---execution time statistics---
def gen_function_statistics(list):
    function_stats = {}
    function_stats['n time called'] = len(list)
    function_stats['max (s)'] = max(list)
    list_w_index = enumerate(list)
    function_stats['higher values'] = sorted(list_w_index, key = lambda item : item[1], reverse = True)[:10]
    function_stats['min (s)'] = min(list)
    if len(list) > 3 :
        function_stats['mean (s)'] = round(statistics.mean(list), 3)
        function_stats['std (s)'] = round(statistics.stdev(list), 3)
    function_stats['total time (s)'] = round(sum(list), 2)
    return function_stats

def get_statistics(times_dic):
    exec_stats = {}
    for function_name in times_dic:
        if times_dic[function_name] != []:
            if type(times_dic[function_name][0]) not in [tuple, list]:
                exec_stats[function_name] = gen_function_statistics(times_dic[function_name])
    return exec_stats

def record_exec_time_stats(runner):
    #for the episode
    runner.run_exec_stats = get_statistics(runner.run_exec_dic)

    #for each agent
    for _id, agent in runner.env.team.items():
        agent.step_exec_stats = get_statistics(agent.step_exec_time)
        agent.costmaps.cm_exec_stats = get_statistics(agent.costmaps.cm_exec_time)
        agent.viewpoint_planner.vpp_exec_stats = get_statistics(agent.viewpoint_planner.vpp_exec_time)
        agent.RRTtree.rrt_exec_stats = get_statistics(agent.RRTtree.rrt_exec_time)

#---writing functions---
def write_gif(file_name, images, fps, true_image = False, salience = False, salIMGS = None):
    kargs = {'fps'  : fps, 'subrectangles' : True}
    imageio.mimwrite(file_name, [im for im in images if (type(im) == np.ndarray)], 'GIF', **kargs)
    print("GIF has been created!")

def write_images(folder_name, images):
    for i in range(len(images)):
        image = images[i]
        if type(image) == np.ndarray : imageio.imwrite(str(folder_name)+str(i+1)+".png", image)
    print("Images have been saved!")

def write_history(file_name, history):
    if type(history) is History:
        with open(file_name,"w") as f:
            f.write("History of Agent "+str(history.self_id))
            f.write("\n---------------")
            f.write("\n---------------")

            for i in range(len(history.state_history)):
                state = history.state_history[i]
                f.write("\n\n\nstep : "+str(i+1))
                f.write("\n---------------")
                f.write("\n done : "+str(state.done))
                f.write("\n has planned : "+str(state.has_planned))
                f.write("\n goal : "+str(state.goal))
                f.write("\n path : "+str(state.path))
                f.write("\n action : "+str(state.action))
                f.write("\n---------------")
                f.write("\n has moved ? : "+str(state.has_moved))
                f.write("\n odom : "+str(state.odom))
                f.write("\n blind_v : "+str(state.blind_v))
                f.write("\n---------------")
                f.write("\n real pos : "+str(state.pos))
                f.write("\n local map : \n"+str(state.local_map))
                f.write("\n borders : "+str(state.borders))
                f.write("\n team in range : "+str(state.team_in_range))
                f.write("\n---------------")
                f.write("\n neighbours : "+str(state.neighbours))
                f.write("\n new neighbours : "+str(state.new_neighbours))
                f.write("\n data : "+str(state.data))
                f.write("\n new data : "+str(state.new_data))
                f.write("\n last data : "+str(state.last_data))
                f.write("\n self closed loop gt : "+str(state.self_loop_gt))
                if state.self_loop_gt != None:

                    f.write("\n self closed loop ag: "+str(state.self_loop_ag))
                    f.write("\n self last observation step : "+str(state.self_loop_ref_observation_step))
                    f.write("\n self correction : "+str(state.self_loop_correction))
                
                if state.ma_gt_loops_dic == {}:
                    f.write("\n ma closed loops : None")
                else:
                    f.write("\n ma meta loops : "+str(state.ma_meta_loops))
                    f.write("\n ma semi loops : "+str(state.ma_semi_loops))
                    f.write("\n ma corrections : "+str(state.ma_corrections))

                    f.write("\n ma gt closed loops : "+str(state.ma_gt_loops_dic))
                    f.write("\n ma ag closed loops : "+str(state.ma_ag_loops_dic))

                
                if state.meeting_gt_loops_dic == {}:
                    f.write("\n meeting closed loops : None")
                else:
                    f.write("\n meeting rpr : "+str(state.meeting_rpr))
                    f.write("\n meeting batches : "+str(state.meeting_batches))
                    
                    f.write("\n meeting meta loops : "+str(state.meeting_meta_loops))
                    f.write("\n meeting semi loops : "+str(state.meeting_semi_loops))
                    f.write("\n meeting corrections : "+str(state.meeting_corrections))

                    f.write("\n meeting gt closed loops : "+str(state.meeting_gt_loops_dic))
                    f.write("\n meeting ag closed loops : "+str(state.meeting_ag_loops_dic))
                
                f.write("\n has corrected map ? : "+str(state.has_corrected_map))
                f.write("\n has treated holes ? : "+str(state.has_treated_holes))
                f.write("\n n_corr_inst ? : "+str(state.n_corr_inst))
                f.write("\n---------------")
                f.write("\n agent pos (believed pos): "+str(state.ag_pos))
                f.write("\n agent localisation error : "+str(state.ag_loc_error))
                #f.write("\n agent map : \n"+str(state.ag_map))
                f.write("\n agent team pos : "+str(state.ag_team_pos))
                #f.write("\n blind v map : \n"+str(state.blind_v_map))
                #f.write("\n blind d map : \n"+str(state.blind_d_map))
                #f.write("\n POG : \n"+str(state.pog))
                f.write("\n---------------")
                f.write("\n team plans : "+str(state.team_plans))
                f.write("\n---------------")
                f.write("\n robot_entropy : "+str(state.robot_entropy))
                f.write("\n path_entropy : "+str(state.path_entropy))
                f.write("\n map_entropy_uncertain : "+str(state.map_entropy_uncertain))
                f.write("\n map_entropy_unknown : "+str(state.map_entropy_unknown))
                f.write("\n global_entropy : "+str(state.global_entropy))

                
        print("History has been saved!")

def write_observations(file_name, obs_list, ag_id, self_id):
    with open(file_name,"w") as f:
        f.write("Observations of Agent "+str(ag_id)+" - collected by Agent "+str(self_id))
        f.write("\n---------------")
        f.write("\n---------------")

        for observation in obs_list:
            if type(observation) is Observation:
                f.write("\n\nagent id : "+str(observation.ag_id))
                f.write("\nstep id : "+str(observation.time_step))
                f.write("\n---------------")
                f.write("\n real pos : "+str(observation.real_pos))                
                f.write("\n local map : \n"+str(observation.local_map))
                f.write("\n scan agents : \n"+str(observation.agents_scan))
                f.write("\n neighbours : \n"+str(observation.neighbours))
                f.write("\n---------------")
                f.write("\n pos belief : \n"+str(observation.pos_belief))

    print("Observations files have been saved!")

def write_meeting_records(file_name, meeting_records):
    if meeting_records == []:
        with open(file_name,"w") as f:
            f.write("No meeting records from this agent")

    else:
        with open(file_name,"w") as f:
            f.write("Meeting recorded by Agent " + str(meeting_records[0].self_id))
            f.write("\n---------------")
            f.write("\n---------------")

            for meeting in meeting_records:
                f.write("\n\n\n\nmeeting step : " + str(meeting.meeting_time_step))
                f.write("\nneighbours : " + str(meeting.neighbours))
                
                if meeting.tracks != None:
                    for oth_id in meeting.tracks:
                        f.write("\n---------------")
                        f.write("\nagent track id : "+str(meeting.tracks[oth_id].ag_id))
                        f.write("\n---------------")
                        f.write("\nmeeting dic : "+str(meeting.tracks[oth_id].meeting_dic))
                        f.write("\nlen obs list : "+str(len(meeting.tracks[oth_id].obs_list)))
                        f.write("\ngt trace : "+str(len(meeting.tracks[oth_id].trace.gt_pos)))
                        f.write("\nag trace : "+str(len(meeting.tracks[oth_id].trace.ag_pos)))
                        f.write("\nstep trace : "+str(len(meeting.tracks[oth_id].trace.time_steps)))                    
                        f.write("\nn_corr trace : "+str(len(meeting.tracks[oth_id].trace.n_corr)))                    
                        #f.write("\nlen ext obs list : "+str(len(meeting.tracks[oth_id].ext)))
                        f.write("\nmeta ext obs list : "+str(meeting.tracks[oth_id].meta_ext))
                        f.write("\nowners : "+str(meeting.tracks[oth_id].owners))

                        if oth_id in meeting.meeting_batches:
                            f.write("\n---------------")
                            f.write("\nmeeting rpr : "+str(meeting.meeting_rpr[oth_id]))
                            f.write("\nmeeting batches : "+str(meeting.meeting_batches[oth_id]))
                
                f.write("\n---------------")
                f.write("\n---------------")
                f.write("\nmeeting rpr : "+str(meeting.meeting_rpr))
                f.write("\nmeeting batches : "+str(meeting.meeting_batches))
                f.write("\nmeeting meta loops : "+str(meeting.meeting_meta_loops))
                f.write("\nmeeting semi loops : "+str(meeting.meeting_semi_loops))
                f.write("\nmeeting corrections : "+str(meeting.meeting_corrections))
                f.write("\nmeeting gt loops dic : "+str(meeting.meeting_gt_loops_dic))
                f.write("\nmeeting ag loops dic : "+str(meeting.meeting_ag_loops_dic))

    print("Meeting records have been saved!")


def write_tracks_records(file_name, tracks_records):
    with open(file_name,"w") as f:
        f.write("Tracks recorded by Agent")
        f.write("\n---------------")
        f.write("\n---------------")

        for step in range(len(tracks_records)):
            f.write("\n\n\n\nstep : " + str(step +1))
            tracks = tracks_records[step]
            for oth_id in tracks:
                track = tracks_records[step][oth_id]
                f.write("\n---------------")
                f.write("\nagent track id : "+str(track.ag_id))
                f.write("\n---------------")
                f.write("\nlast update : "+str(track.last_update))
                f.write("\nlen obs list : "+str(len(track.obs_list)))
                f.write("\nlen gt trace : "+str(len(track.trace.gt_pos)))
                f.write("\nag trace : "+str(track.trace.ag_pos))
                f.write("\nstep trace : "+str(track.trace.time_steps))           
                f.write("\nblind_v trace : "+str(track.trace.blind_v))
                f.write("\nn_corr trace : "+str(track.trace.n_corr))                   
                #f.write("\nlen ext obs list : "+str(len(track.ext)))
                f.write("\nmeeting dic : "+str(track.meeting_dic))
                f.write("\nmeta ext obs list : "+str(track.meta_ext))
                f.write("\nowners : "+str(track.owners))

    print("Tracks records have been saved!")


def write_metrics_records(file_name, metrics_records):
    with open(file_name,"w") as f:
        f.write("Metrics records from Agent " + str(metrics_records[0].self_id))
        f.write("\n---------------")
        f.write("\n---------------")

        for metrics_r in metrics_records:
            f.write("\n\n\n\nstep : " + str(metrics_r.time_step))

            f.write("\n---------------historical metrics")
            for metric in metrics_r.agent_metrics:
                f.write("\n" + str(metric) + " : " + str(metrics_r.agent_metrics[metric]))

            f.write("\n---------------submap metrics")
            for metric in metrics_r.submaps_metrics:
                f.write("\n" + str(metric) + " : " + str(metrics_r.submaps_metrics[metric]))
                
            f.write("\n---------------tracks metrics")
            for metric in metrics_r.tracks_metrics:
                f.write("\n" + str(metric) + " : " + str(metrics_r.tracks_metrics[metric]))

            f.write("\n---------------")
            f.write("\nscore : "+str(metrics_r.score))
            f.write("\nutility : "+str(metrics_r.utility))

    print("Metrics records have been saved!")

def write_cm_metrics_records(file_name, vpp_metrics_records):
    with open(file_name,"w") as f:
        f.write("CostMaps metrics records")
        f.write("\n---------------")
        f.write("\n---------------")

        for _ in range(len(vpp_metrics_records)):
            metric_dic = vpp_metrics_records[_]
            for metric in metric_dic:
                if metric == 'step':
                    f.write("\n\n\n\nstep : " + str(metric_dic[metric]))
                    f.write("\n---------------")
                    continue
                if type(metric_dic[metric]) is dict:
                    f.write("\n" + str(metric) + " : ")
                    for sub_metric in metric_dic[metric]:
                        if type(metric_dic[metric][sub_metric]) is np.ndarray:
                            if np.max(metric_dic[metric][sub_metric]) < 1:
                                f.write("\n " + str(sub_metric) + " :\n" + str((metric_dic[metric][sub_metric]*1000).astype(int)))
                            else:
                                f.write("\n " + str(sub_metric) + " :\n" + str(metric_dic[metric][sub_metric]))
                        else:
                            f.write("\n " + str(sub_metric) + " : "+ str(metric_dic[metric][sub_metric]))

                elif type(metric_dic[metric]) is np.ndarray:
                    if np.max(metric_dic[metric]) < 1:
                        f.write("\n" + str(metric) + " :\n " + str((metric_dic[metric]*1000).astype(int)))
                    else:
                        f.write("\n" + str(metric) + " :\n " + str(metric_dic[metric]))
                else:
                    f.write("\n" + str(metric) + " : "+ str(metric_dic[metric]))

    print("CostMaps metrics file has been saved!")

def write_vpp_metrics_records(file_name, vpp_metrics_records):
    with open(file_name,"w") as f:
        f.write("Viewpoint Planner metrics records")
        f.write("\n---------------")
        f.write("\n---------------")

        for _ in range(len(vpp_metrics_records)):
            metric_dic = vpp_metrics_records[_]
            for metric in metric_dic:
                #print(metric)
                #print(metric_dic[metric])
                if metric == 'step':
                    f.write("\n\n\n\nstep : " + str(metric_dic[metric]))
                    f.write("\n---------------")
                    continue
                if type(metric_dic[metric]) != np.ndarray:
                    if metric_dic[metric] in [None, {}]:
                        continue
                if type(metric_dic[metric]) is not dict:
                    f.write("\n" + str(metric) + " : "+ ("\n" * (type(metric_dic[metric]) == np.ndarray)) + str(metric_dic[metric]))
                else:
                    f.write("\n" + str(metric) + " : ")
                    for sub_metric in metric_dic[metric]:
                        f.write("\n " + str(sub_metric) + " : "+ ("\n" * (type(metric_dic[metric][sub_metric]) is np.ndarray)) + str(metric_dic[metric][sub_metric]))

    print("VPP metrics file has been saved!")

def write_rrt_metrics_records(file_name, rrt_metrics_records):
    with open(file_name,"w") as f:
        f.write("RRT metrics records")
        f.write("\n---------------")
        f.write("\n---------------")

        for _ in range(len(rrt_metrics_records)):
            rrt_metric_dic = rrt_metrics_records[_]
            for metric in rrt_metric_dic:
                contend = rrt_metric_dic[metric]
                if metric == 'step':
                    f.write("\n\n\n\nstep : " + str(contend))
                    f.write("\nindex in list : "+ str(_))
                    f.write("\n---------------")
                    f.write("\n---------------")
                elif type(contend) != np.ndarray and contend in [None, {}]:
                    continue
                elif type(contend) is not dict:
                    f.write("\n" + str(metric) + " : "+ ("\n" * (type(contend) == np.ndarray)) + str(contend))
                elif metric == 'situation':
                    f.write("\n" + str(metric) + " : ")
                    for node in contend:
                        f.write("\n    " + str(node) + " : ")
                        for sub_metric in contend[node]:
                            if sub_metric not in ['tracks', 'submaps']:
                                f.write("\n        " + str(sub_metric) + " : "+ str(contend[node][sub_metric]))
                elif metric == 'w_sit':
                    f.write("\n" + str(metric) + " : ")
                    for node in contend:
                        f.write("\n " + str(node) + " : ")
                        if type(contend[node]) == dict:
                            for event in contend[node]:
                                f.write("\n  " + str(event) + " : " + str(contend[node][event]))
                elif metric == 'multiverse':
                    f.write("\n" + str(metric) + " : ")
                    for node in contend:
                        f.write("\n " + str(node) + " : ")
                        for event in contend[node]:
                            f.write("\n     " + str(event) + " : ")
                            for sub_metric in contend[node][event]:
                                if sub_metric not in ['VA', 'init_VA']:
                                    f.write("\n         " + str(sub_metric) + " : "+ str(contend[node][event][sub_metric]))
                                
                                #optional:
                                # elif sub_metric in ['VA', 'init_VA']:
                                #     f.write("\n         " + str(sub_metric) + " v_step : "+ str(contend[node][event][sub_metric].v_step))
                                #     f.write("\n         " + str(sub_metric) + " bv : "+ str(contend[node][event][sub_metric].bv))
                                #     f.write("\n         " + str(sub_metric) + " h : "+ str(contend[node][event][sub_metric].h))

                                #     if 'root' and 'added' in contend[node][event][sub_metric].tracks:
                                #         for tr_id in contend[node][event][sub_metric].tracks['root']:
                                #             f.write("\n         " + str(sub_metric) + " root_track" + str(tr_id) + " : ag_pos : " + str(contend[node][event][sub_metric].tracks['root'][tr_id].trace.ag_pos))
                                #             f.write("\n         " + str(sub_metric) + " root_track" + str(tr_id) + " : ts : " + str(contend[node][event][sub_metric].tracks['root'][tr_id].trace.time_steps))
                                #             f.write("\n         " + str(sub_metric) + " root_track" + str(tr_id) + " : blind_v : " + str(contend[node][event][sub_metric].tracks['root'][tr_id].trace.blind_v))
                                #         for tr_id in contend[node][event][sub_metric].tracks['added']:
                                #             f.write("\n         " + str(sub_metric) + " added_track" + str(tr_id) + " : ag_pos : " + str(contend[node][event][sub_metric].tracks['added'][tr_id].trace.ag_pos))
                                #             f.write("\n         " + str(sub_metric) + " added_track" + str(tr_id) + " : ts : " + str(contend[node][event][sub_metric].tracks['added'][tr_id].trace.time_steps))
                                #             f.write("\n         " + str(sub_metric) + " added_track" + str(tr_id) + " : blind_v : " + str(contend[node][event][sub_metric].tracks['added'][tr_id].trace.blind_v))
                                #         for tr_id in contend[node][event][sub_metric].tracks['updates']:
                                #             f.write("\n         " + str(sub_metric) + " updates_track : " + str(tr_id) + " : " + str(contend[node][event][sub_metric].tracks['updates'][tr_id]))
                                #     else:
                                #         for tr_id in contend[node][event][sub_metric].tracks:
                                #             f.write("\n         " + str(sub_metric) + " trace" + str(tr_id) + " : ag_pos : " + str(contend[node][event][sub_metric].tracks[tr_id].trace.ag_pos))
                                #             f.write("\n         " + str(sub_metric) + " trace" + str(tr_id) + " : ts : " + str(contend[node][event][sub_metric].tracks[tr_id].trace.time_steps))
                                #             f.write("\n         " + str(sub_metric) + " trace" + str(tr_id) + " : blind_v : " + str(contend[node][event][sub_metric].tracks[tr_id].trace.blind_v))
                              
                                #     if 'traces' and 'meta_ext' in contend[node][event][sub_metric].tracks:
                                #         for tr_id in contend[node][event][sub_metric].tracks['traces']:
                                #             f.write("\n         " + str(sub_metric) + " tracks_trace" + str(tr_id) + " : ag_pos : " + str(contend[node][event][sub_metric].tracks['traces'][tr_id].ag_pos))
                                #             f.write("\n         " + str(sub_metric) + " tracks_trace" + str(tr_id) + " : ts : " + str(contend[node][event][sub_metric].tracks['traces'][tr_id].time_steps))
                                #             f.write("\n         " + str(sub_metric) + " tracks_trace" + str(tr_id) + " : blind_v : " + str(contend[node][event][sub_metric].tracks['traces'][tr_id].blind_v))
                                #         for tr_id in contend[node][event][sub_metric].tracks['meta_ext']:
                                #             f.write("\n         " + str(sub_metric) + " tracks_meta_ext" + str(tr_id) + " : " + str(contend[node][event][sub_metric].tracks['meta_ext'][tr_id]))


                else:
                    f.write("\n" + str(metric) + " : ")
                    for sub_metric in contend:
                        f.write("\n " + str(sub_metric) + " : "+ ("\n" * (type(contend[sub_metric]) is np.ndarray)) + str(contend[sub_metric]))

    print("RRT metrics have been saved!")

def write_pp_history(file_name, pp_history_records):
    with open(file_name,"w") as f:
        f.write("Path Planner History")
        f.write("\n---------------")
        f.write("\n---------------")
    
        for step in range(len(pp_history_records)):
            f.write("\n\n\n\nstep : " + str(step + 1))
            f.write("\n---------------")

            for i_try in pp_history_records[step]:
                f.write("\n---------------")
                f.write("\ntry " + str(i_try))

                for metric in pp_history_records[step][i_try]:
                    f.write("\n" + str(metric) + " : " + str(pp_history_records[step][i_try][metric]))

    print("Path Planner History has been saved!")

def write_idv_sequences(file_name, sequences):
    with open(file_name,"w") as f:
        f.write("Sequences")
        f.write("\n---------------")
        f.write("\n---------------")

        for type_s in sequences:
            for type_a in sequences[type_s]:
                f.write("\n\n" + str(type_s) + " - " + str(type_a) + ' : ')
                for seq in sequences[type_s][type_a]:
                    f.write("\n" + str(seq) + " : " + str(sequences[type_s][type_a][seq]))

    print("Individual sequences have been saved!")

def write_episode_perf(file_name, game_perf):
    with open(file_name,"w") as f:
        f.write("Global Performance of the Episode")
        f.write("\n---------------")
        f.write("\n---------------\n")
        for i_type in game_perf:
            if i_type != 'prog':
                f.write("\n\n" + str(i_type) + " : ")
                for metric in game_perf[i_type]:
                    f.write("\n" + str(metric) + " : " + str(game_perf[i_type][metric]))
                
    print("Episode Performances have been saved!")



def draw_team_sequence(pre_name, game_seq, team):
    #matplotlib ...
    for type_r in game_seq[1]:
        if type_r not in ['perf']:
            for type_a in game_seq[1][type_r]:
                for seq in game_seq[1][type_r][type_a]:
                    if 'per_step' in seq:
                        continue

                    #init figure
                    plt.figure()

                    for ag_id in game_seq:
                        y = game_seq[ag_id][type_r][type_a][seq]
                        x = range(1,len(y)+1)
                        color = team[ag_id].color
                        label = 'agent_' + str(ag_id)

                        if 'submaps' in seq:
                            linestyle = '-'
                        elif 'obs' in seq or 'tracks' in seq:
                            linestyle = '--'
                        else:
                            linestyle = ':'

                        plt.plot(x, y, color = color, marker = '', linestyle = linestyle, linewidth = 1, label = label)
                    
                    plt.xlabel('step')
                    plt.title(str(type_r)+'_'+str(type_a)+'_'+str(seq))
                    plt.grid()
                    plt.legend()

                    fig_name = pre_name + str(type_r)+'_'+str(type_a)+'_'+str(seq) + '.png'
                    plt.savefig(fig_name)
                    plt.close()

    print("Team Sequences have been drawn!")



def write_mission_stats(file_name, metrics_lists, mission_stats, median_prog, games_perf_saved, games_stats_saved):
    with open(file_name,"w") as f:
        f.write("Global Statistics over the Mission")
        f.write("\n---------------")
        f.write("\n---------------")
        f.write("\nNumber of Runs : " + str(len(games_perf_saved)))
        f.write("\n---------------")
        f.write("\nCOMPLETENESS_THRESHOLD : " + str(AgentParameters.COMPLETENESS_THRESHOLD))
        f.write("\nCORRECTNESS_THRESHOLD : " + str(AgentParameters.CORRECTNESS_THRESHOLD))
        f.write("\nMEAN_ERROR_THRESHOLD : " + str(AgentParameters.MEAN_ERROR_THRESHOLD))
        f.write("\n---------------")

        f.write("\n\n\nMission Stats")
        for metric in mission_stats:
            f.write("\n" + str(metric) + " : " + str(mission_stats[metric]))

        f.write("\n\n\nMetrics Lists")
        for metric in metrics_lists:
            f.write("\n" + str(metric) + " : " + str(metrics_lists[metric]))
        
        f.write("\n\n\nMedian Progression")
        for type_m in median_prog:
            for type_a in median_prog[type_m]:
                for seq in median_prog[type_m][type_a]:
                    f.write("\n" + str(type_m) + " - " + str(type_a) +  " - " + str(seq) + " : " + str(median_prog[type_m][type_a][seq]))
        
        for i_run in range(len(games_perf_saved)):
            f.write("\n\n\nRun " + str(i_run+1) + " recap")
            f.write("\n---------------")
            f.write("\n---------------")

            for metric in games_stats_saved[i_run]:
                f.write("\n" + str(metric) + " : " + str(games_stats_saved[i_run][metric]))
        
    print("Mission stats have been saved!")

def draw_median_progression(pre_name, median_prog):
    #matplotlib ...
    
    for type_r in median_prog:
        if type_r not in ['perf']:
            for type_a in median_prog[type_r]:
                for seq in median_prog[type_r][type_a]:
                    if 'per_step' in seq:
                        continue
                    y = median_prog[type_r][type_a][seq]
                    if y == []:
                        continue

                    x = range(1,len(y)+1)

                    if 'submaps' in type_a:
                        linestyle = ':'
                    elif 'obs' in type_a or 'tracks' in type_a:
                        linestyle = '--'
                    else:
                        linestyle = '-'

                    if 'metrics' in type_r :
                        color = 'blue'
                    elif 'eval' in type_r :
                        color = 'red'
                    elif 'perf' in type_r :
                        color = 'purple'
                    else:
                        color = 'grey'

                    plt.figure()
                    plt.plot(x, y, color = color, marker = '', linestyle = linestyle, linewidth = 1)
                    plt.xlabel('step')
                    plt.title(str(type_r)+'_'+str(type_a)+'_'+str(seq) + ' - median_prog')
                    plt.grid()

                    fig_name = pre_name + str(type_r)+'_'+str(type_a)+'_'+str(seq) + '.png'
                    plt.savefig(fig_name)
                    plt.close()
        
    print("Median performance progression has been drawn!")

def write_exec_times(file_name, exec_dic, stats):
    with open(file_name,"w") as f:
        f.write("Execution time to record and get statistics")
        f.write("\n---------------")
        f.write("\n---------------")

        for function_name in exec_dic:
            f.write("\n\n"+str(function_name)+" exec times : \n" + str(exec_dic[str(function_name)]))
            if function_name in stats:
                for stat in stats[function_name]:
                    f.write("\n "+str(stat)+" : " + str(stats[function_name][stat]))

    print("Execution time record file has been saved!")