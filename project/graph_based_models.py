import numpy as np
import networkx as nx
import pandas as pd

# Prepare the data
def convert_dataset_to_relevant_interactions_list(df):
    df_convert = df[df.rating >= 4]
    df_convert = df_convert[["item_id","user_id"]]
    df_convert_arr = df_convert.values
    return df_convert_arr

''' Building Graph '''
class InteractionGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        
    def add_nodes_from_edge_array(self, edge_array, type_1, type_2):
        nodes = [(x[0], {'type': type_1}) for x in edge_array] \
        + [(x[1], {'type': type_2}) for x in edge_array]
        self.graph.add_nodes_from(nodes)

    def add_edges_from_array(self, array, weight_front=1.0, weight_back=1.0):
        forward_edges = [(x[0], x[1], weight_front) for x in array]
        back_edges = [(x[1], x[0], weight_back) for x in array]
        self.graph.add_weighted_edges_from(forward_edges)
        self.graph.add_weighted_edges_from(back_edges)

    @staticmethod
    def build_graph(user_item_array):
        multigraph = InteractionGraph()
        multigraph.add_nodes_from_edge_array(user_item_array, 'item', 'user')
        multigraph.add_edges_from_array(user_item_array)
        return multigraph


class PersonalizedPageRankRecommendation:
    def __init__(self, multigraph, damping_factor = 0.3):
        self.graph = nx.DiGraph()
        
        # if we have multple edges with the same source and destination, then create a single edge with the cummulative sum of those edges' weight
        for u,v,d in multigraph.graph.edges(data=True):
            w = d['weight']
            if self.graph.has_edge(u,v):
                self.graph[u][v]['weight'] += w
            else:
                self.graph.add_edge(u,v,weight=w)
        self.nodes = list(self.graph.nodes)
        self.damping_factor = damping_factor
        
        #this part keeps track of items that have been rated by each user in the training set
        self.user_item_dict = {}
        for n in multigraph.graph.nodes.data():
            if n[1]['type'] == 'user':
                self.user_item_dict[n[0]] = set()
        for e in multigraph.graph.edges:
            if e[0] in self.user_item_dict:
                self.user_item_dict[e[0]].add(e[1])

    def generate_pr(self, user, damping_factor):
        # Searching for the node corresponding to the user in the graph
        pers = [1 if n==user else 0 for n in self.nodes]
        pers_dict = None
        
        # If the user isn't in our graph, then pagerank won't have a dedicated starting point, hence any node can be its starting point
        if sum(pers) != 0:
            pers_dict = dict(zip(self.nodes, pers))       
        pr = nx.pagerank(self.graph, damping_factor, personalization=pers_dict)
        pr_sorted = dict(
            #sort pr by descending probability values
            sorted(pr.items(), key=lambda x: x[1], reverse=True)
            )
        pr_list = [(k, v) for k, v in pr_sorted.items()]
        return pr_list
    
    def generate_recommendations(self, user, k=10):
        pr_list = self.generate_pr(user,self.damping_factor)
        if user not in self.user_item_dict.keys():
            return [item for (item, _) in pr_list if item not in self.user_item_dict.keys()]
        
        # Given the user, remove items in their recommendation list that they have rated in the training set
        # The pr_list contains both user and item nodes that were regularly visited during the traversals
        result = [item for (item, _) in pr_list if item not in self.user_item_dict.keys() and item not in self.user_item_dict[user]][:k]
        return result

class KatzSimilarityRecommender: 
    def __init__(self, relevant_interaction_list):
        
        self.user_list = []
        self.item_list = []
        
        self.rated_items_per_user = {}
        
        for item, user in relevant_interaction_list:
            if user not in self.user_list:
                self.rated_items_per_user[user] = set()
            
            self.user_list.append(user)
            self.item_list.append(item)
            self.rated_items_per_user[user].add(item)

        self.user_list = list(set(self.user_list))
        self.item_list = list(set(self.item_list))
        
        self.num_nodes = len(self.user_list) + len(self.item_list)
        
        self.katz_similarities = self._calculate_katz_similarity(relevant_interaction_list)
    
    def generate_recommendation(self, user, k=10):
        adjacency_matrix_row = 0
        try:
            adjacency_matrix_row = self.user_list.index(user)
        except ValueError:
            raise ValueError(f"No information about user {user}.") 
        
        top_k_node_indeces = np.argsort(-self.katz_similarities[adjacency_matrix_row,len(self.user_list):])
        return [
            self.item_list[idx] for idx in top_k_node_indeces if self.item_list[idx] not in self.rated_items_per_user[user]
        ][:k]

    def _calculate_katz_similarity(self, relevant_interaction_list):
        self.adjacency_matrix = self._build_adjacency_matrix(relevant_interaction_list)
        
        # discount factor must me less than the max absolute value of the eigenvalues of the adjacency matrix
        eigenvalues = np.linalg.eigvals(self.adjacency_matrix)
        discount_factor = 1/(np.max(np.abs(eigenvalues)) + 10**(-6))

        I = np.identity(self.num_nodes)
        katz_similarities = np.linalg.inv(I - discount_factor* self.adjacency_matrix) - I
        return katz_similarities

    def _build_adjacency_matrix(self, relevant_interaction_list):
        node_name_list = self.user_list + self.item_list
               
        df = pd.DataFrame(np.zeros((self.num_nodes, self.num_nodes)), columns=node_name_list, index=node_name_list)

        for item_id, user_id in relevant_interaction_list:
            df.loc[item_id, user_id] = 1
            df.loc[user_id, item_id] = 1
            
        return df.to_numpy()