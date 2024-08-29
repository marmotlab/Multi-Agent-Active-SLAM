from env import MA_SLAM_Env

from param import *

import statistics
import time
import numpy

class Runner:
    def __init__(self, env_id, imposed_map = None, init_poses = None):
        #init env
        self.env_id = env_id
        self.n_agents = EnvParameters.N_AGENTS
        self.n_steps = EnvParameters.N_STEPS
        self.env = MA_SLAM_Env(self.n_agents, self.n_steps, imposed_map, init_poses)

        self.run_exec_dic = {'run' : []}
        self.run_exec_stats = {}

    def run_episode(self, render = MetaParameters.RENDER):
        #init running variables
        self.env.running = True
        for _id, agent in self.env.team.items():
            agent.running = True

        if render : self.env.render()

        #run the episode
        try:
            while self.env.running and self.env.time_step < self.n_steps and not self.env.done:
                #increment time step
                self.env.time_step += 1
                print("\nstep :", self.env.time_step)

                #run each agent one by one
                for _id, agent in self.env.team.items():

                    if agent.running:
                        tic_run = time.time()
                        agent.set_time(self.env)

                        agent.pre_update(self.env)
                        agent.policy() #update the planner and choose the next action to do
                        agent.step(self.env) #agent moves, process SLAM and calculate a new entropy                   
                        agent.save() #save variables

                        if agent.done:
                            print("Mission done for Agent")

                        agent.eval(self.env) #agent compare its mapping with the ground truth

                        agent.update_sequences() #get sequences to analyse the episode later

                        if render : agent.render(self.env)
                        self.run_exec_dic['run'].append(round(time.time() - tic_run, 3))

                #final
                self.env.done = self.env.is_task_done()
                if render : self.env.render()
                
        except KeyboardInterrupt:
            print("CTRL-C pressed. Process interrupted!")

        if render : self.env.close_viewers()

    
    
    #record agents individual performance and measure collective performance (of the whole team) at the end of the episode
    def save_episode_data(self):
        game_perf_final = {
            #speed
            'agents_success_bool_final' : [self.env.team[ag_id].perf['success'] for ag_id in self.env.team],
            'agents_success_step_final' : [self.env.team[ag_id].perf['success_step'] for ag_id in self.env.team],
            
            #exploring
            'exploring_completeness_final' : [self.env.team[ag_id].sequences['eval']['submaps']['squares_known_perc'][-1] for ag_id in self.env.team], #individual exploration final state
            'exploring_redundancy_final' : [self.env.team[ag_id].sequences['metrics']['submaps']['n_scans_med'][-1] for ag_id in self.env.team],
            
            #precision
            'mapping_correctness_final' : [self.env.team[ag_id].sequences['eval']['submaps']['obstacles_corr_perc'][-1] for ag_id in self.env.team], #individual obstacles positionning correctness final state
            'obs_mean_error_team_final' : [self.env.team[ag_id].sequences['eval']['obs']['obs_mean_err'][-1] for ag_id in self.env.team], #team mean_error final state
            
            #certainty
            'obs_mean_bv_team_final' : [self.env.team[ag_id].sequences['metrics']['tracks']['obs_blind_v_mean'][-1] for ag_id in self.env.team], #team mean_bv final state            
            'mapping_entropy_uncertain_final' : [self.env.team[ag_id].sequences['state']['agent']['mapping_entropy_uncertain'][-1] for ag_id in self.env.team], #map entropy final state
            'mapping_entropy_unknown_final' : [self.env.team[ag_id].sequences['state']['agent']['mapping_entropy_unknown'][-1] for ag_id in self.env.team], #map entropy final state

            #connectivity
            'n_new_meetings_final' : [self.env.team[ag_id].sequences['metrics']['agent']['n_new_neighbours'][-1] for ag_id in self.env.team],
            'n_new_data_final' : [self.env.team[ag_id].sequences['metrics']['agent']['n_new_data'][-1] for ag_id in self.env.team],

            #correction cost
            'meeting_avg_cost_final' : [self.env.team[ag_id].sequences['metrics']['agent']['n_meeting_corrections_per_new_meeting'][-1] for ag_id in self.env.team],
            'meeting_loops_avg_cost_final' : [self.env.team[ag_id].sequences['metrics']['agent']['n_meeting_corrections_per_meeting_loop'][-1] for ag_id in self.env.team],
            'loops_avg_cost_final' : [self.env.team[ag_id].sequences['metrics']['agent']['n_corrections_per_loop'][-1] for ag_id in self.env.team],
            'loops_correcting_avg_cost_final' : [self.env.team[ag_id].sequences['metrics']['agent']['n_corrections_per_loop_correcting'][-1] for ag_id in self.env.team],
        
            'score_final' : [self.env.team[ag_id].sequences['state']['agent']['score'][-1] for ag_id in self.env.team], #individual mapping global utility final state
        }

        n_missing_time_steps = self.n_steps - self.env.time_step
        game_perf_are = {
            #exploring
            'exploring_completeness_are' : [round(statistics.mean(self.env.team[ag_id].sequences['eval']['submaps']['squares_known_perc']+[self.env.team[ag_id].sequences['eval']['submaps']['squares_known_perc'][-1] for _ in range(n_missing_time_steps)]), 1) for ag_id in self.env.team], #individual exploration
            'exploring_redundancy_are' : [round(statistics.mean(self.env.team[ag_id].sequences['metrics']['submaps']['n_scans_med']+[self.env.team[ag_id].sequences['metrics']['submaps']['n_scans_med'][-1] for _ in range(n_missing_time_steps)]), 1) for ag_id in self.env.team], #individual obstacles positionning correctness
            
            #precision
            'mapping_correctness_are' : [round(statistics.mean(self.env.team[ag_id].sequences['eval']['submaps']['obstacles_corr_perc']+[self.env.team[ag_id].sequences['eval']['submaps']['obstacles_corr_perc'][-1] for _ in range(n_missing_time_steps)]), 1) for ag_id in self.env.team], #individual obstacles positionning correctness
            'robot_localization_error_are' : [round(statistics.mean(self.env.team[ag_id].sequences['state']['agent']['robot_localization_error']+[self.env.team[ag_id].sequences['state']['agent']['robot_localization_error'][-1] for _ in range(n_missing_time_steps)]), 1) for ag_id in self.env.team],
            'obs_mean_error_team_are' : [round(statistics.mean(self.env.team[ag_id].sequences['eval']['obs']['obs_mean_err']+[self.env.team[ag_id].sequences['eval']['obs']['obs_mean_err'][-1] for _ in range(n_missing_time_steps)]), 1) for ag_id in self.env.team],
            'obs_mean_error_self_are' : [round(statistics.mean(self.env.team[ag_id].sequences['eval']['obs']['obs_self_mean_err']+[self.env.team[ag_id].sequences['eval']['obs']['obs_self_mean_err'][-1] for _ in range(n_missing_time_steps)]), 1) for ag_id in self.env.team],
            
            #certainty
            'robot_entropy_are' : [round(statistics.mean(self.env.team[ag_id].sequences['state']['agent']['robot_entropy']+[self.env.team[ag_id].sequences['state']['agent']['robot_entropy'][-1] for _ in range(n_missing_time_steps)]), 1) for ag_id in self.env.team], 
            'obs_mean_bv_team_are' : [round(statistics.mean([elem for elem in self.env.team[ag_id].sequences['metrics']['tracks']['obs_blind_v_mean'] if elem != None]+[self.env.team[ag_id].sequences['metrics']['tracks']['obs_blind_v_mean'][-1] for _ in range(n_missing_time_steps)]), 1) for ag_id in self.env.team], #individual exploration mean state
            'mapping_entropy_uncertain_are' : [round(statistics.mean(self.env.team[ag_id].sequences['state']['agent']['mapping_entropy_uncertain']+[self.env.team[ag_id].sequences['state']['agent']['mapping_entropy_uncertain'][-1] for _ in range(n_missing_time_steps)]), 1) for ag_id in self.env.team], #individual mapping entropy
            'mapping_entropy_unknown_are' : [round(statistics.mean(self.env.team[ag_id].sequences['state']['agent']['mapping_entropy_unknown']+[self.env.team[ag_id].sequences['state']['agent']['mapping_entropy_unknown'][-1] for _ in range(n_missing_time_steps)]), 1) for ag_id in self.env.team], #individual mapping entropy
            'global_entropy_are' : [round(statistics.mean(self.env.team[ag_id].sequences['state']['agent']['global_entropy']+[self.env.team[ag_id].sequences['state']['agent']['global_entropy'][-1] for _ in range(n_missing_time_steps)]), 1) for ag_id in self.env.team],
            
            #connectivity
            'team_lastly_seen_mean_ts_are' : [],
            'team_lastly_seen_mean_ts_corr_are' : [round(statistics.mean(self.env.team[ag_id].sequences['eval']['submaps']['team_lastly_seen_mean_ts_corr']+[self.env.team[ag_id].sequences['eval']['submaps']['team_lastly_seen_mean_ts_corr'][-1] for _ in range(n_missing_time_steps)]), 1) for ag_id in self.env.team],
            'team_agents_known_perc_are' : [round(statistics.mean(self.env.team[ag_id].sequences['eval']['submaps']['team_agents_known_perc']+[self.env.team[ag_id].sequences['eval']['submaps']['team_agents_known_perc'][-1] for _ in range(n_missing_time_steps)]), 1) for ag_id in self.env.team],
            'team_known_perc_are' : [round(statistics.mean(self.env.team[ag_id].sequences['eval']['submaps']['team_known_perc']+[self.env.team[ag_id].sequences['eval']['submaps']['team_known_perc'][-1] for _ in range(n_missing_time_steps)]), 1) for ag_id in self.env.team],
        }

        #calculate conn
        list_mlts = []
        for ag_id in self.env.team:
            completed_list = self.env.team[ag_id].sequences['metrics']['submaps']['team_lastly_seen_mean_ts']+[self.env.team[ag_id].sequences['metrics']['submaps']['team_lastly_seen_mean_ts'][-1] for _ in range(n_missing_time_steps)]
            nn_list = [elem for elem in completed_list if elem != None]
            if nn_list != []:
                lts_tuple = (round(statistics.mean(nn_list)), len(nn_list))
                list_mlts.append(lts_tuple)
            else:
                list_mlts.append((0,0))
        game_perf_are['team_lastly_seen_mean_ts_are'] = list_mlts


        game_perf_oth = {
            #connectivity
            'new_meetings_frequency' : [],
            'new_meetings_itv_median' : [],
            'disconnection_max' : [],

            #corrections charge
            'correcting_max_cost' : [max(self.env.team[ag_id].sequences['metrics']['agent']['n_corrections_inst']) for ag_id in self.env.team]
        }

        for ag_id in self.env.team:
            sequence = self.env.team[ag_id].sequences['metrics']['agent']['n_new_neighbours']
            if max(sequence) < 2:
                game_perf_oth['new_meetings_frequency'].append(False)
                game_perf_oth['new_meetings_itv_median'].append(False)
                game_perf_oth['disconnection_max'].append(False)
                continue
            
            new_meetings_ts = []
            if sequence[0] > 0:
                new_meetings_ts.append(0)
            for i in range(len(sequence) -1):
                if sequence[i+1] > sequence[i]:
                    new_meetings_ts.append(i+1)
            
            new_meetings_itv = []
            for i_nm in range(len(new_meetings_ts) -1):
                itv = new_meetings_ts[i_nm+1]-new_meetings_ts[i_nm]
                new_meetings_itv.append(itv)
            
            if len(new_meetings_itv) >= 3:
                game_perf_oth['new_meetings_frequency'].append(round(statistics.mean(new_meetings_itv)))
            else:
                game_perf_oth['new_meetings_frequency'].append(False)
            if len(new_meetings_itv) >= 3:
                game_perf_oth['new_meetings_itv_median'].append(round(statistics.median(new_meetings_itv)))
            else:
                game_perf_oth['new_meetings_itv_median'].append(False)

            new_meetings_itv.append(len(sequence)-new_meetings_ts[-1])
            game_perf_oth['disconnection_max'].append(max(new_meetings_itv))



                    



        #gathering
        self.game_perf = {
            'final' : game_perf_final,
            'are' : game_perf_are,
            'oth' : game_perf_oth,
        }

        #calculate mean (avg) and stdev
        game_stats = {}

        #speed
        game_stats['agents_success_ratio'] = round(sum([1 * self.env.team[ag_id].perf['success'] for ag_id in self.env.team])/ self.n_agents ,2)
        game_stats['agents_success_step_avg'] = int(statistics.mean([self.env.team[ag_id].perf['success_step'] + ((self.env.team[ag_id].perf['success_step'] == False) * self.n_steps) for ag_id in self.env.team]))

        #exploration
        game_stats['exploring_completeness_final_avg'] = round(statistics.mean(game_perf_final['exploring_completeness_final']), 1)
        game_stats['exploring_completeness_are_avg'] = round(statistics.mean(game_perf_are['exploring_completeness_are']), 1) 
        if [elem for elem in game_perf_final['exploring_redundancy_final'] if elem != None] != [] : game_stats['exploring_redundancy_final_avg'] = round(statistics.mean(game_perf_final['exploring_redundancy_final']), 1)
        if [elem for elem in game_perf_are['exploring_redundancy_are'] if elem != None] != [] : game_stats['exploring_redundancy_are_avg'] = round(statistics.mean(game_perf_are['exploring_redundancy_are']), 1)
        
        #precision
        if [elem for elem in game_perf_final['mapping_correctness_final'] if elem != None] != [] : game_stats['mapping_correctness_final_avg'] = round(statistics.mean(game_perf_final['mapping_correctness_final']), 1)
        if [elem for elem in game_perf_are['mapping_correctness_are'] if elem != None] != [] : game_stats['mapping_correctness_are_avg'] = round(statistics.mean(game_perf_are['mapping_correctness_are']), 1)
        game_stats['obs_mean_error_team_final_avg'] = round(statistics.mean(game_perf_final['obs_mean_error_team_final']), 2)
        game_stats['obs_mean_error_team_are_avg'] = round(statistics.mean(game_perf_are['obs_mean_error_team_are']), 2)
        game_stats['robot_localization_error_are_avg'] = round(statistics.mean(game_perf_are['robot_localization_error_are']), 2)

        #certainty
        game_stats['robot_entropy_are_avg'] = round(statistics.mean(game_perf_are['robot_entropy_are']), 2)
        game_stats['obs_mean_bv_team_final_avg'] = round(statistics.mean(game_perf_final['obs_mean_bv_team_final']), 1)
        game_stats['obs_mean_bv_team_are_avg'] = round(statistics.mean(game_perf_are['obs_mean_bv_team_are']), 1)
        game_stats['mapping_entropy_uncertain_final_avg'] = round(statistics.mean(game_perf_final['mapping_entropy_uncertain_final']), 1)
        game_stats['mapping_entropy_uncertain_are_avg'] = round(statistics.mean(game_perf_are['mapping_entropy_uncertain_are']), 1)
        game_stats['mapping_entropy_unknown_final_avg'] = round(statistics.mean(game_perf_final['mapping_entropy_unknown_final']), 1)
        game_stats['mapping_entropy_unknown_are_avg'] = round(statistics.mean(game_perf_are['mapping_entropy_unknown_are']), 1)
        game_stats['global_entropy_are_avg'] = round(statistics.mean(game_perf_are['global_entropy_are']), 2)

        #connectivity
        game_stats['n_new_meetings_avg'] = round(statistics.mean(game_perf_final['n_new_meetings_final']), 2)
        game_stats['n_new_data_avg'] = round(statistics.mean(game_perf_final['n_new_data_final']), 2)

        team_lastly_seen_mean_ts_are = [elem[0] for elem in game_perf_are['team_lastly_seen_mean_ts_are']]
        weights =  [elem[1] for elem in game_perf_are['team_lastly_seen_mean_ts_are']]
        game_stats['team_lastly_seen_mean_ts_are_avg'] = round(numpy.average(team_lastly_seen_mean_ts_are, weights=weights), 1) if max(weights)>0 else None

        game_stats['team_lastly_seen_mean_ts_corr_are_avg'] = round(statistics.mean(game_perf_are['team_lastly_seen_mean_ts_corr_are']), 1)
        game_stats['team_agents_known_perc_are_avg'] = round(statistics.mean(game_perf_are['team_agents_known_perc_are']), 1)
        game_stats['team_known_perc_are_avg'] = round(statistics.mean(game_perf_are['team_known_perc_are']), 1)
        
        #corrections charge
        game_stats['meeting_avg_cost_avg'] = round(statistics.mean([elem for elem in game_perf_final['meeting_avg_cost_final'] if type(elem) is not bool]), 2) if [elem for elem in game_perf_final['meeting_avg_cost_final'] if type(elem) is not bool] != [] else None
        game_stats['meeting_loops_avg_cost_avg'] = round(statistics.mean([elem for elem in game_perf_final['meeting_loops_avg_cost_final'] if type(elem) is not bool]), 2) if [elem for elem in game_perf_final['meeting_loops_avg_cost_final'] if type(elem) is not bool] != [] else None
        game_stats['loops_avg_cost_avg'] = round(statistics.mean([elem for elem in game_perf_final['loops_avg_cost_final'] if type(elem) is not bool]), 2) if [elem for elem in game_perf_final['loops_avg_cost_final'] if type(elem) is not bool] != [] else None
        game_stats['loops_correcting_avg_cost_avg'] = round(statistics.mean([elem for elem in game_perf_final['loops_correcting_avg_cost_final'] if type(elem) is not bool]), 2) if [elem for elem in game_perf_final['loops_correcting_avg_cost_final'] if type(elem) is not bool] != [] else None

        game_stats['score_final_avg'] = int(statistics.mean(game_perf_final['score_final']))
        if self.n_agents > 1:
            game_stats['score_final_std'] = int(statistics.stdev(game_perf_final['score_final']))

        self.game_stats = game_stats

        #sequences
        self.game_seq = {ag_id : self.env.team[ag_id].sequences for ag_id in self.env.team}