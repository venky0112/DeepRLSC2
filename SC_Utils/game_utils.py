from pysc2.env import sc2_env
from pysc2.lib import actions 
import numpy as np
import torch

def init_game(game_params, map_name='MoveToBeacon', max_steps=256, step_multiplier=8, **kwargs):

    race = sc2_env.Race(1) # 1 = terran
    agent = sc2_env.Agent(race, "_") # NamedTuple [race, agent_name]
    agent_interface_format = sc2_env.parse_agent_interface_format(**game_params) #AgentInterfaceFormat instance

    game_params = dict(map_name=map_name,
                       players=[agent],
                       game_steps_per_episode = max_steps*step_multiplier,
                       agent_interface_format=[agent_interface_format]
                       )  
    env = sc2_env.SC2Env(**game_params, **kwargs)

    return env

def get_action_dict(action_names):
    action_ids = [actions.FUNCTIONS[a_name].id for a_name in action_names]
    action_dict = {i:action_ids[i] for i in range(len(action_ids))}
    return action_dict

class ObsProcesser():
    def __init__(self, screen_names=[], minimap_names=[], select_all=False):
        
        self.screen_var = { 'height_map': (0,'unknown'), 
                            'visibility_map': (1,'ohe', 2), 
                            'creep': (2,'unknown'), 
                            'power': (3,'unknown'), 
                            'player_id': (4,'ohe', 3), # 1,2,16
                            'player_relative': (5,'ohe', 3), # friendly (1), neutral (3), enemy (4)
                            #'unit_type': (6,'ohe', 9), # dependent on the number of minigames considered and unit in them 
                            'unit_type': (6,'ohe', 11), # updated version
                            'selected': (7,'ohe', 1),
                            'unit_hit_points': (8,'log'), 
                            'unit_hit_points_ratio': (9,'log'), 
                            'unit_energy': (10,'unknown'), 
                            'unit_energy_ratio': (11,'unknown'), 
                            'unit_shields': (12,'unknown'),
                            'unit_shields_ratio': (13, 'unknown'),
                            'unit_density': (14, 'float'), 
                            'unit_density_aa': (15, 'float'),
                            'effects': (16, 'unknown'),
                            'hallucinations': (17, 'unknown'),
                            'cloaked': (18, 'unknown'),
                            'blip': (19, 'unknown'),
                            'buffs': (20, 'unknown'), 
                            'buff_duration': (21, 'unknown'),  
                            'active': (22, 'unknown'), 
                            'build_progress': (23, 'unknown'), 
                            'pathable': (24, 'ohe', 1),  
                            'buildable': (25, 'ohe', 1),  
                            'placeholder': (26, 'unknown')}
        
        self.minimap_var = {'height_map': (0,'unknown'),
                            'visibility_map': (1, 'ohe', 2),
                            'creep': (2, 'unknown'), 
                            'camera': (3, 'ohe', 1), 
                            'player_id': (4, 'ohe', 3),
                            'player_relative': (5, 'ohe', 3),
                            'selected': (6,'ohe', 1),  
                            'unit_type': (7,'unknown'),
                            'alerts': (8, 'unknown'),
                            'pathable': (9, 'ohe', 1),  
                            'buildable': (10,'ohe', 1)}
        
        # Contains specifics both for screen and minimap
        self.categorical_specs = {
                                  'visibility_map':np.array([1,2]),
                                  'player_id':np.array([1,2,16]),
                                  'player_relative':np.array([1,3,4]),
                                  'unit_type':np.array([9, 18, 20, 45, 48, 105, 110, 317, 341, 342, 1680]), # updated version
                                  #'unit_type':np.array([9, 18, 48, 105, 110, 317, 341, 342, 1680]),
                                  'selected':np.array([1]),
                                  'pathable':np.array([1]),
                                  'buildable':np.array([1]),
                                  'camera':np.array([1])
                                 }
        if select_all:
            # overrides layer_names selecting all known layers
            self.screen_names = [k for k in self.screen_var.keys() if self.screen_var[k][1] != 'unknown']
            self.minimap_names = [k for k in self.minimap_var.keys() if self.minimap_var[k][1] != 'unknown']
        else:
            # names of the layers to be selected
            self.screen_names = screen_names 
            self.minimap_names = minimap_names 
       
        self.screen_indexes =[self.screen_var[n][0] for n in self.screen_names]
        self.minimap_indexes =[self.minimap_var[n][0] for n in self.minimap_names]
        
    def get_state(self, obs):
        feature_screen = obs[0].observation['feature_screen']
        feature_minimap = obs[0].observation['feature_minimap']
        
        screen_layers, screen_names = self._process_screen_features(feature_screen)
        minimap_layers, minimap_names = self._process_minimap_features(feature_minimap)
        state = {'screen_layers':screen_layers, 'minimap_layers':minimap_layers}
        names = {'screen_names':screen_names, 'minimap_names':minimap_names}
        return state, names
    
    def get_n_channels(self):
        screen_channels, minimap_channels = 0, 0
        
        for name in self.screen_names:
            if self.screen_var[name][1] == 'ohe':
                screen_channels +=  self.screen_var[name][2]
            elif (self.screen_var[name][1] == 'log') or (self.screen_var[name][1]=='float'):
                screen_channels += 1
            else:
                raise Exception("Unknown number of channels of screen feature "+name)
                
        for name in self.minimap_names:
            if self.minimap_var[name][1] == 'ohe':
                minimap_channels +=  self.minimap_var[name][2]
            elif (self.minimap_var[name][1] == 'log') or (self.minimap_var[name][1]=='float'):
                minimap_channels += 1
            else:
                raise Exception("Unknown number of channels of minimap feature "+name)
                
        return screen_channels, minimap_channels
        
    def _process_screen_features(self, feature_screen):
        names = list(feature_screen._index_names[0].keys())
        processed_layers = []
        processed_names = []

        for i, idx in enumerate(self.screen_indexes):
            try:
                assert names[idx] == self.screen_names[i], 'Mismatched state indexes'
            except:
                print('Mismatched state indexes')
                print('names[%d]'%idx, names[idx])
                print('self.screen_names[%d]'%i, self.screen_names[i])
                
            layer = np.array(feature_screen[idx])
            if self.screen_var[names[idx]][1] == 'ohe':
                layer, name = self._process_ohe_layer(layer, names[idx])
            elif self.screen_var[names[idx]][1] == 'float':
                layer, name = self._process_float_layer(layer, names[idx])
            elif self.screen_var[names[idx]][1] == 'log':
                layer, name = self._process_log_layer(layer, names[idx])
            else:
                raise Exception('Type of layer '+names[idx]+' not understood')
                
            processed_layers.append(layer)
            processed_names.append(name)
        if len(processed_layers) > 0:
            processed_layers = np.concatenate(processed_layers).astype(float)
            processed_names = np.concatenate(processed_names)
        return processed_layers, processed_names
    
    def _process_minimap_features(self, feature_minimap):
        names = list(feature_minimap._index_names[0].keys())
        processed_layers = []
        processed_names = []

        for i, idx in enumerate(self.minimap_indexes):
            try:
                assert names[idx] == self.minimap_names[i], 'Mismatched state indexes'
            except:
                print('Mismatched state indexes')
                print('names[%d]'%idx, names[idx])
                print('self.minimap_names[%d]'%i, self.minimap_names[i])
                
            layer = np.array(feature_minimap[idx])
            if self.minimap_var[names[idx]][1] == 'ohe':
                layer, name = self._process_ohe_layer(layer, names[idx])
            elif self.minimap_var[names[idx]][1] == 'float':
                layer, name = self._process_float_layer(layer, names[idx])
            elif self.minimap_var[names[idx]][1] == 'log':
                layer, name = self._process_log_layer(layer, names[idx])
            else:
                raise Exception('Type of layer '+names[idx]+' not understood')
                
            processed_layers.append(layer)
            processed_names.append(name)
        if len(processed_layers) > 0:
            processed_layers = np.concatenate(processed_layers).astype(float)
            processed_names = np.concatenate(processed_names)
        return processed_layers, processed_names
    
    def _process_ohe_layer(self, layer, name):
        u = np.unique(layer)
        unique = u[u!=0] # 0 is always the background value, it doesn't get a dedicated layer
        possible_values = self.categorical_specs[name]
        try:
            assert np.all(np.isin(unique, possible_values))
        except:
            print("Found unexpected value in "+name+" layer")
            print("Expected possible values: \n", possible_values)
            print("Actual unique values (0 excl.): \n", unique)
        ohe_layer = (layer == possible_values.reshape(-1,1,1)).astype(float) # mask broadcasting + casting to float
        names = np.array([name+'_'+str(x) for x in possible_values]) 
        return ohe_layer, names

    def _process_float_layer(self, layer, name):
        return layer.reshape((1,)+layer.shape[-2:]).astype(float), [name]
    
    def _process_log_layer(self, layer, name):
        """Apply log2 to all nonzero elements"""
        mask = (layer!=0) 
        layer[mask] = np.log2(layer[mask])
        return layer.reshape((1,)+layer.shape[-2:]).astype(float), [name]

class FullObsProcesser(ObsProcesser):
    def __init__(self, screen_names=[], minimap_names=[], select_all=False):
        super().__init__(screen_names, minimap_names, select_all)
        self.useful_indexes = np.arange(1,9)
        
    def get_state(self, obs):
        feature_screen = obs[0].observation['feature_screen']
        feature_minimap = obs[0].observation['feature_minimap']
        player_info = obs[0].observation['player'].astype(float)
        
        screen_layers, screen_names = self._process_screen_features(feature_screen)
        minimap_layers, minimap_names = self._process_minimap_features(feature_minimap)
        player_features, player_names = self._process_player_features(player_info)
        state = {'screen_layers':screen_layers, 'minimap_layers':minimap_layers, 'player_features':player_features}
        names = {'screen_names':screen_names, 'minimap_names':minimap_names, 'player_names':player_names}
        return state, names
    
    def _process_player_features(self, player):
        x = player[self.useful_indexes]
        f =  lambda x: np.log2(x) if x != 0 else x
        x['minerals'] = f(x['minerals'])
        x['vespene'] = f(x['vespene'])
        return np.array(x), np.array(list(x._index_names[0].keys()))
    
    def get_n_channels(self):
        screen_channels, minimap_channels = super().get_n_channels()
        player_channels = len(self.useful_indexes)
        return screen_channels, minimap_channels, player_channels

class IMPALA_ObsProcesser(FullObsProcesser):
    def __init__(self, action_table, screen_names=[], minimap_names=[], select_all=False):
        super().__init__(screen_names, minimap_names, select_all)
        self.action_table = action_table
        
    def get_action_mask(self, available_actions):
        """
        Creates a mask of length action_table with zeros (negated True casted to float) 
        in the positions of available actions and ones (negated False casted to float) 
        in the other positions. 
        """
        action_mask = ~torch.tensor([a in available_actions for a in self.action_table], dtype=torch.bool)
        return action_mask

class IMPALA_ObsProcesser_v2(IMPALA_ObsProcesser):
    def __init__(self, env, action_table, screen_names=[], minimap_names=[], select_all=False):
        super().__init__(action_table, screen_names, minimap_names, select_all)
        # Used to tile binary masks to screen and minimap layers depending on where the last action acted
        self.screen_mask = list(map(lambda x: self.check_if_screen(env, x, True), self.action_table))
        self.minimap_mask = list(map(lambda x: self.check_if_screen(env, x, False), self.action_table))
        
    def get_state(self, obs):
        feature_screen = obs[0].observation['feature_screen']
        feature_minimap = obs[0].observation['feature_minimap']
        player_info = obs[0].observation['player'].astype(float)
        
        last_action = obs[0].observation['last_actions']
        last_action_idx = self.get_last_action_idx(last_action)
        
        screen_layers, screen_names = self._process_screen_features(feature_screen)
        screen_layers = self.tile_binary_mask(screen_layers, last_action_idx, screen=True)
        
        minimap_layers, minimap_names = self._process_minimap_features(feature_minimap)
        minimap_layers = self.tile_binary_mask(minimap_layers, last_action_idx, screen=False)
        
        player_features, player_names = self._process_player_features(player_info)
        state = {'screen_layers':screen_layers, 
                 'minimap_layers':minimap_layers, 
                 'player_features':player_features,
                 'last_action':last_action_idx
                }
        names = {'screen_names':screen_names, 'minimap_names':minimap_names, 'player_names':player_names}
        return state, names
    
    def get_last_action_idx(self, last_action):
        if len(last_action) == 0:
            # explicitly code no_op action
            last_action = 0
        else:
            last_action = last_action[0]
        last_action_idx = np.where(self.action_table == last_action)[0] # (batch,)
        if len(last_action_idx) == 0:
            # sometimes the environment translates an action like select_army in select_unit,
            # which is not in the action_table, so last_action_idx will be an empty list,
            # throwing an error in tile_binary_mask. I decided to map it to no_op if there is no
            # correspondence
            last_action_idx = [0]
        
        return last_action_idx 
    
    def tile_binary_mask(self, spatial_layers, last_action_idx, screen=True):
        if screen:
            binary_mask = np.array([self.screen_mask[last_action_idx[0]]])
        else:
            binary_mask = np.array([self.minimap_mask[last_action_idx[0]]])
        binary_mask2D = np.tile(binary_mask, [1,*spatial_layers.shape[-2:]])
        spatial_layers = np.concatenate([spatial_layers, binary_mask2D])
        return spatial_layers
    
    def get_n_channels(self):
        """
        Add 1 to the screen and minimap channels because of default binary mask tiling
        """
        screen_channels, minimap_channels = super().get_n_channels()
        player_channels = len(self.useful_indexes)
        return screen_channels+1, minimap_channels+1, player_channels
    
    @staticmethod
    def check_if_screen(env, sc_env_action, screen=True):
        """
        If screen=True checks if sc_env_action acts on the screen.
        If screen=False checks if sc_env_action acts on the minimap.
        Note: some actions without spatial parameters are considered 
              not to act on any of the two.
        """
        all_actions = env.action_spec()[0][1]
        all_arguments = env.action_spec()[0][0]
        args = all_actions[sc_env_action].args
        names = [all_arguments[arg.id].name for arg in args]
        if screen:
            return np.any(['screen' in n for n in names])
        else:
            return np.any(['minimap' in n for n in names])

    

