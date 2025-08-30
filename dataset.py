# import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_networkx, subgraph, degree, to_undirected, is_undirected
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.datasets import Planetoid, DBLP, IMDB, HGBDataset, LastFM, MovieLens1M, AMiner, OGB_MAG
import networkx as nx
import numpy as np
import random
import torch_geometric.transforms as T
from sklearn.metrics.pairwise import cosine_similarity
import os
import csv
import pandas as pd
import torch_geometric as geo

def split_metapaths(full_data, metapaths1, metapaths2):
	data1 = HeteroData()
	data2 = HeteroData()

	def collect_node_types(metapaths):
		node_set = set()
		for src, rel, dst in metapaths:
			node_set.add(src)
			node_set.add(dst)
		return node_set

	node_types1 = collect_node_types(metapaths1)
	node_types2 = collect_node_types(metapaths2)

	# Copy only required node features
	for node_type in node_types1:
		for key, val in full_data[node_type].items():
			data1[node_type][key] = val

	for node_type in node_types2:
		for key, val in full_data[node_type].items():
			data2[node_type][key] = val

	# Assign only the specified edge types
	for edge_type in metapaths1:
		if edge_type in full_data.edge_types:
			data1[edge_type].edge_index = full_data[edge_type].edge_index

	for edge_type in metapaths2:
		if edge_type in full_data.edge_types:
			data2[edge_type].edge_index = full_data[edge_type].edge_index

	return data1, data2
def load_DBLP(seed=42):

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	metapaths = [[('author', 'paper'), ('paper', 'author')],
			 [('author', 'paper'),('paper','term'),('term','paper'), ('paper', 'author')],
			 [('author', 'paper'),('paper','conference'),('conference','paper'), ('paper', 'author')]]
	transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
						   drop_unconnected_node_types=True)

	dataset = DBLP(root='tmp/DBLP',transform=transform)[0]
	dblp_graph = DBLP(root='tmp/DBLP2')
	dataset2 = dblp_graph[0]
	#print(dataset)
	out = group_hetero_graph(dataset2.edge_index_dict, dataset2.num_nodes_dict)
	edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out
	homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
				 node_type=node_type, local_node_idx=local_node_idx,
				 num_nodes=node_type.size(0))
	feat_list = []
	for key, x in dataset2.x_dict.items():
		x_pad = x
		if x.dim() == 1:
			x_pad = x.unsqueeze(1)
		feat_list.append(x_pad)

	# Step 2: Create global node feature tensor
	x_dict = {key2int[key]: x for key, x in dataset2.x_dict.items()}
	node_feats = torch.zeros((node_type.size(0), max(x.shape[1] for x in x_dict.values())))

	for t, x in x_dict.items():
		idx = (node_type == t)
		node_feats[idx, :x.shape[1]] = x

	homo_data.x = node_feats
	target_ntype = 'author'  # Or 'paper', if you're predicting for papers

	target_nid = local2global[target_ntype]
	homo_data.y = torch.full((node_type.size(0),), -1, dtype=torch.long)
	homo_data.y[target_nid] = dataset2[target_ntype].y

	# Train/Val/Test masks
	homo_data.train_mask = torch.zeros((node_type.size(0),), dtype=torch.bool)
	homo_data.val_mask = torch.zeros((node_type.size(0),), dtype=torch.bool)
	homo_data.test_mask = torch.zeros((node_type.size(0),), dtype=torch.bool)

	homo_data.train_mask[target_nid[dataset2[target_ntype].train_mask]] = True
	homo_data.val_mask[target_nid[dataset2[target_ntype].val_mask]] = True
	homo_data.test_mask[target_nid[dataset2[target_ntype].test_mask]] = True

	x = dataset['author'].x
	y = dataset['author'].y

	APA = dataset['author', 'metapath_0', 'author'].edge_index
	APTPA = dataset['author', 'metapath_1', 'author'].edge_index
	APCPA = dataset['author', 'metapath_2', 'author'].edge_index

	data_APA = Data(x=x, edge_index=APA, y=y)
	data_APTPA = Data(x=x, edge_index=APTPA, y=y)
	data_APCPA = Data(x=x, edge_index=APCPA, y=y)

	num_con = 20
	num_paper = 14328
	num_term = 7724
	train_mask = dataset['author'].train_mask
	val_mask = dataset['author'].val_mask
	test_mask = dataset['author'].test_mask

	dataset2['conference'].x = torch.Tensor(num_con, 128).uniform_(-0.5, 0.5)
	#dataset2['paper'].x = torch.Tensor(num_paper, 128).uniform_(-0.5, 0.5)
	#dataset2['term'].x = torch.Tensor(num_term, 128).uniform_(-0.5, 0.5)
	metapaths1 = [
		('author','to', 'paper'),
		('paper', 'to','author'),
		('paper', 'to','term'),
		('term', 'to','paper')
		

	]

	metapaths2 = [
		('author','to', 'paper'),
		('paper', 'to','author'),
		('paper', 'to','conference'),
		('conference', 'to','paper')

	]

	data1, data2 = split_metapaths(dataset2, metapaths1, metapaths2)
	dblp_graph = [data1,data2]

	return homo_data,dataset2, dblp_graph,train_mask, val_mask, test_mask

def load_IMDB(seed=42):

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	metapaths = [[('movie', 'actor'), ('actor', 'movie')],
			 [('movie', 'director'), ('director', 'movie')],
			 [('movie', 'actor'), ('actor', 'movie'),('movie', 'director'), ('director', 'movie')],
			 [('movie', 'director'), ('director', 'movie'),('movie', 'actor'), ('actor', 'movie')]]
	transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
						   drop_unconnected_node_types=True)

	dataset = IMDB(root='tmp/IMDB',transform=transform)[0]
	dataset2 = IMDB(root='tmp/IMDB2')[0]
	out = group_hetero_graph(dataset2.edge_index_dict, dataset2.num_nodes_dict)
	edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out
	homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
				 node_type=node_type, local_node_idx=local_node_idx,
				 num_nodes=node_type.size(0))
	feat_list = []
	for key, x in dataset2.x_dict.items():
		x_pad = x
		if x.dim() == 1:
			x_pad = x.unsqueeze(1)
		feat_list.append(x_pad)

	# Step 2: Create global node feature tensor
	x_dict = {key2int[key]: x for key, x in dataset2.x_dict.items()}
	node_feats = torch.zeros((node_type.size(0), max(x.shape[1] for x in x_dict.values())))

	for t, x in x_dict.items():
		idx = (node_type == t)
		node_feats[idx, :x.shape[1]] = x

	homo_data.x = node_feats
	target_ntype = 'movie'  # Or 'paper', if you're predicting for papers

	target_nid = local2global[target_ntype]
	homo_data.y = torch.full((node_type.size(0),), -1, dtype=torch.long)
	homo_data.y[target_nid] = dataset2[target_ntype].y

	# Train/Val/Test masks
	homo_data.train_mask = torch.zeros((node_type.size(0),), dtype=torch.bool)
	homo_data.val_mask = torch.zeros((node_type.size(0),), dtype=torch.bool)
	homo_data.test_mask = torch.zeros((node_type.size(0),), dtype=torch.bool)

	homo_data.train_mask[target_nid[dataset2[target_ntype].train_mask]] = True
	homo_data.val_mask[target_nid[dataset2[target_ntype].val_mask]] = True
	homo_data.test_mask[target_nid[dataset2[target_ntype].test_mask]] = True
	x = dataset2['movie'].x
	y = dataset2['movie'].y

	num_movies = 4278
	num_directors = 2081
	num_actors = 5257

	# MD_edge = dataset2['movie','to','director'].edge_index
	# DM_edge = dataset2['director','to','movie'].edge_index
	# movie_to_director = torch.zeros((num_movies, num_directors))
	# movie_to_director[MD_edge[0], MD_edge[1]] = 1
	# director_to_movie = torch.zeros((num_directors, num_movies))
	# director_to_movie[DM_edge[0], DM_edge[1]] = 1
	# MDM = torch.matmul(movie_to_director, director_to_movie)
	# MDM = (MDM > 0).int()

	# MA_edge = dataset2['movie','to','actor'].edge_index
	# AM_edge = dataset2['actor','to','movie'].edge_index
	# movie_to_actor = torch.zeros((num_movies, num_actors))
	# movie_to_actor[MA_edge[0], MA_edge[1]] = 1
	# actor_to_movie = torch.zeros((num_actors, num_movies))
	# actor_to_movie[AM_edge[0], AM_edge[1]] = 1
	# MAM = torch.matmul(movie_to_actor, actor_to_movie)
	# MAM = (MAM > 0).int()

	# MAMDM = torch.matmul(MAM, MDM)
	# MAMDM = (MAMDM > 0).int()

	# MDMAM = torch.matmul(MDM, MAM)
	# MDMAM = (MDMAM > 0).int()

	# torch.save({'feature':x,'graph_actor':MAM,'graph_director': MDM, 'actor': MAMDM, 'director':MDMAM, 'labels':y}, 'combined_movie.pt')

	MAM = torch.load('combined_movie.pt')['graph_actor']
	MDM = torch.load('combined_movie.pt')['graph_director']	
	AMA = torch.load('combined_movie.pt')['actor']
	DMD = torch.load('combined_movie.pt')['director']

	data_MAM = Data(x=x, edge_index=dense_to_sparse(MAM.fill_diagonal_(1))[0], y=y)
	data_MDM = Data(x=x, edge_index=dense_to_sparse(MDM.fill_diagonal_(1))[0], y=y)
	data_AMA = Data(x=x, edge_index=dense_to_sparse(AMA.fill_diagonal_(1))[0], y=y)
	data_DMD = Data(x=x, edge_index=dense_to_sparse(DMD.fill_diagonal_(1))[0], y=y)

	train_mask = dataset['movie'].train_mask
	val_mask = dataset['movie'].val_mask
	test_mask = dataset['movie'].test_mask
	metapaths1 = [
		('movie','to','director'),
		('director','to','movie'),
		#('movie','to','movie')

	]

	metapaths2 = [
		('movie','to','actor'),
		('actor','to','movie'),
		#('movie','to','movie')

	]


	data1, data2 = split_metapaths(dataset2, metapaths1, metapaths2)
	data1['movie','movie'].edge_index = data_MAM.edge_index
	data2['movie','movie'].edge_index = data_MDM.edge_index
	graph = [data1,data2]
	
	return homo_data,dataset2, graph,train_mask, val_mask, test_mask


def load_ACM(seed=42):

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	metapaths = [[('paper', 'author'), ('author', 'paper')],
			 [('paper', 'subject'), ('subject', 'paper')],[('paper', 'term'), ('term', 'paper')]]
	transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
						   drop_unconnected_node_types=True)

	dataset = HGBDataset(root='tmp/ACM',name='ACM',transform=transform)[0]
	#print(dataset)
	dataset2 = HGBDataset(root='tmp/ACM2',name='ACM')[0]
	#print(dataset2)
	x = dataset2['paper'].x
	#print("x",x.shape)
	y = dataset2['paper'].y
	num_papers = 3025
	num_authors = 5959
	num_subjects = 56
	num_term = 1902


	#print(dataset2)
	# PA_edge = dataset2['paper','to','author'].edge_index
	# AP_edge = dataset2['author','to','paper'].edge_index
	# paper_to_author = torch.zeros((num_papers, num_authors))
	# paper_to_author[PA_edge[0], PA_edge[1]] = 1
	# author_to_paper = torch.zeros((num_authors, num_papers))
	# author_to_paper[AP_edge[0], AP_edge[1]] = 1
	# PAP = torch.matmul(paper_to_author, author_to_paper)
	# PAP = (PAP > 0).int()

	# PS_edge = dataset2['paper','to','subject'].edge_index
	# SP_edge = dataset2['subject','to','paper'].edge_index
	# paper_to_subject = torch.zeros((num_papers, num_subjects))
	# paper_to_subject[PS_edge[0], PS_edge[1]] = 1
	# subject_to_paper = torch.zeros((num_subjects, num_papers))
	# subject_to_paper[SP_edge[0], SP_edge[1]] = 1
	# PSP = torch.matmul(paper_to_subject, subject_to_paper)
	# PSP = (PSP > 0).int()

	# torch.save({'feature':x,'graph_author':PAP,'graph_subject': PSP, 'labels':y}, 'combined_paper.pt')

	PAP = torch.load('combined_paper.pt')['graph_author']
	PSP = torch.load('combined_paper.pt')['graph_subject']

	data_PAP = Data(x=x, edge_index=dense_to_sparse(PAP.fill_diagonal_(1))[0], y=y)
	data_PSP = Data(x=x, edge_index=dense_to_sparse(PSP.fill_diagonal_(1))[0], y=y)

	train_mask = dataset['paper'].train_mask
	test_mask = dataset['paper'].test_mask

	train_indices = train_mask.nonzero(as_tuple=True)[0]
	val_size = int(0.5 * len(train_indices))
	val_indices = train_indices[:val_size]
	new_train_indices = train_indices[val_size:]

	val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
	val_mask[val_indices] = True

	new_train_mask = torch.zeros_like(train_mask, dtype=torch.bool)
	new_train_mask[new_train_indices] = True
	dataset2['paper'].train_mask = new_train_mask
	dataset2['paper'].val_mask = val_mask
	out = group_hetero_graph(dataset2.edge_index_dict, dataset2.num_nodes_dict)
	edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out
	homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
				 node_type=node_type, local_node_idx=local_node_idx,
				 num_nodes=node_type.size(0))
	feat_list = []
	for key, x in dataset2.x_dict.items():
		x_pad = x
		if x.dim() == 1:
			x_pad = x.unsqueeze(1)
		feat_list.append(x_pad)

	# Step 2: Create global node feature tensor
	x_dict = {key2int[key]: x for key, x in dataset2.x_dict.items()}
	node_feats = torch.zeros((node_type.size(0), max(x.shape[1] for x in x_dict.values())))

	for t, x in x_dict.items():
		idx = (node_type == t)
		node_feats[idx, :x.shape[1]] = x

	homo_data.x = node_feats
	target_ntype = 'paper'  # Or 'paper', if you're predicting for papers

	target_nid = local2global[target_ntype]
	homo_data.y = torch.full((node_type.size(0),), -1, dtype=torch.long)
	homo_data.y[target_nid] = dataset2[target_ntype].y

	# Train/Val/Test masks
	homo_data.train_mask = torch.zeros((node_type.size(0),), dtype=torch.bool)
	homo_data.val_mask = torch.zeros((node_type.size(0),), dtype=torch.bool)
	homo_data.test_mask = torch.zeros((node_type.size(0),), dtype=torch.bool)

	homo_data.train_mask[target_nid[dataset2[target_ntype].train_mask]] = True
	homo_data.val_mask[target_nid[dataset2[target_ntype].val_mask]] = True
	homo_data.test_mask[target_nid[dataset2[target_ntype].test_mask]] = True

	dataset2['term'].x = torch.Tensor(num_term, 128).uniform_(-0.5, 0.5)
	metapaths1 = [
		('paper','cite','paper'),
		('paper','ref','paper'),
		('author','to', 'paper'),
		('paper', 'to','author')
		

	]

	metapaths2 = [
		('subject','to', 'paper'),
		('paper', 'to','subject'),
		('term','to', 'paper'),
		('paper', 'to','term'),


	]

	data1, data2 = split_metapaths(dataset2, metapaths1, metapaths2)
	data1['paper','to','paper'].edge_index = data_PSP.edge_index
	data2['paper','paper'].edge_index = data_PAP.edge_index
	graph = [data1,data2]
	#torch.save({'feature':x,'graph_author':data_PAP,'graph_subject': data_PSP, 'labels':y,'train':new_train_mask,'val':val_mask,'test':test_mask}, 'paper.pt')
	return homo_data,dataset2,graph, new_train_mask, val_mask, test_mask

def drop_edges_random(edge_index, keep_ratio=0.1):
	num_edges = edge_index.size(1)
	keep_num = int(keep_ratio * num_edges)
	perm = torch.randperm(num_edges)
	keep_idx = perm[:keep_num]
	edge_index_new = edge_index[:, keep_idx]
	return edge_index_new
def load_OgbMag(seed=42,feature=''):

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)

	dataset = OGB_MAG(root='tmp/Ogb',preprocess='metapath2vec',transform=T.ToUndirected())[0]
	dataset2 = OGB_MAG(root='tmp/Ogb',preprocess='metapath2vec')[0]
	#print(dataset2)
	x = dataset['paper'].x
	y = dataset['paper'].y
	train_mask = dataset['paper'].train_mask
	val_mask = dataset['paper'].val_mask
	test_mask = dataset['paper'].test_mask

	for edge_type in dataset2.edge_types:
		original_edge_index = dataset2[edge_type].edge_index
		dataset2[edge_type].edge_index = drop_edges_random(original_edge_index,0.3)
	dataset2['institution','author'].edge_index = dataset2['author','institution'].edge_index[[1, 0]]
	dataset2['field_of_study','paper'].edge_index = dataset2['paper','field_of_study'].edge_index[[1,0]]
	dataset2['paper', 'author'].edge_index = dataset2['author','paper'].edge_index[[1, 0]]
	out = group_hetero_graph(dataset2.edge_index_dict, dataset2.num_nodes_dict)
	edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out
	homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
				 node_type=node_type, local_node_idx=local_node_idx,
				 num_nodes=node_type.size(0))
	feat_list = []
	for key, x in dataset2.x_dict.items():
		x_pad = x
		if x.dim() == 1:
			x_pad = x.unsqueeze(1)
		feat_list.append(x_pad)

	# Step 2: Create global node feature tensor
	x_dict = {key2int[key]: x for key, x in dataset2.x_dict.items()}
	node_feats = torch.zeros((node_type.size(0), max(x.shape[1] for x in x_dict.values())))

	for t, x in x_dict.items():
		idx = (node_type == t)
		node_feats[idx, :x.shape[1]] = x

	homo_data.x = node_feats
	target_ntype = 'paper'  # Or 'paper', if you're predicting for papers

	target_nid = local2global[target_ntype]
	homo_data.y = torch.full((node_type.size(0),), -1, dtype=torch.long)
	homo_data.y[target_nid] = dataset2[target_ntype].y

	# Train/Val/Test masks
	homo_data.train_mask = torch.zeros((node_type.size(0),), dtype=torch.bool)
	homo_data.val_mask = torch.zeros((node_type.size(0),), dtype=torch.bool)
	homo_data.test_mask = torch.zeros((node_type.size(0),), dtype=torch.bool)

	homo_data.train_mask[target_nid[dataset2[target_ntype].train_mask]] = True
	homo_data.val_mask[target_nid[dataset2[target_ntype].val_mask]] = True
	homo_data.test_mask[target_nid[dataset2[target_ntype].test_mask]] = True
	
	metapaths = [#[('paper', 'paper2'), ('paper2','paper')],
		 [('paper', 'author'), ('author', 'paper')],
	   [('paper', 'field_of_study'), ('field_of_study', 'paper')]]

	transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,drop_unconnected_node_types=True)

	data = HeteroData()
	data['paper'].x = x
	data['paper'].y = y

	data['paper', 'paper2'].edge_index = dataset['paper','paper'].edge_index
	data['paper2', 'paper'].edge_index = dataset['paper','paper'].edge_index[[1, 0]]

	data['paper', 'author'].edge_index = dataset['author','paper'].edge_index[[1, 0]]
	data['author', 'paper'].edge_index = dataset['author','paper'].edge_index

	data['paper', 'field_of_study'].edge_index = dataset['paper','field_of_study'].edge_index
	data['field_of_study','paper'].edge_index = dataset['paper','field_of_study'].edge_index[[1,0]]

	data['author','institution'].edge_index = dataset['author','institution'].edge_index
	data['institution','author'].edge_index = dataset['author','institution'].edge_index[[1, 0]]

	# Get number of nodes in each type
	num_papers = data['paper'].num_nodes
	num_author = 1134649
	num_inst = 8740
	num_fields=59965
	num_institution = 8740

	author_emb = dataset['author'].x
	author_x = torch.cat([x,author_emb], dim=0)
	author_idx, paper_idx = dataset['author', 'paper'].edge_index
	author_edge = torch.stack([paper_idx, author_idx + num_papers], dim=0)

	topic_emb = dataset['field_of_study'].x
	topic_x = torch.cat([x,topic_emb], dim=0)
	paper_idx, field_idx = dataset['paper', 'field_of_study'].edge_index
	field_edge = torch.stack([paper_idx, field_idx + num_papers], dim=0)
	institution_emb = torch.Tensor(num_institution, 256).uniform_(-0.5, 0.5)
	
	if feature == 'uniform':
		dataset['author'].x = torch.Tensor(num_author, 128).uniform_(-0.5, 0.5)
		dataset['field_of_study'].x = torch.Tensor(num_fields, 128).uniform_(-0.5, 0.5)
		dataset['institution'].x = torch.Tensor(num_institution, 128).uniform_(-0.5, 0.5)
	metapaths1 = [
		('paper', 'cites', 'paper'),
		('paper', 'has_topic', 'field_of_study'),
		('field_of_study', 'rev_has_topic', 'paper')

	]

	metapaths2 = [
		('author', 'affiliated_with', 'institution'),
		('institution', 'rev_affiliated_with', 'author'),
		('author', 'writes', 'paper'),
		('paper', 'rev_writes', 'author')

	]
	
	data1, data2 = split_metapaths(dataset, metapaths1, metapaths2)
	ogb_graph = [data1,data2]
	return homo_data,dataset,ogb_graph ,train_mask, val_mask, test_mask

def load_Freebase(seed=42):

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)

	df = pd.read_csv("Freebase/label.dat.test", sep='\t', header=None)
	#print(df.shape)
	metapaths = [#[('book','and', 'book'), ('book','and', 'book')],

				[('book','about','organization'),('organization','to','music'),('music','in','book')],
			 [('book','about','organization'),('organization','for','business'),('business','about','book')]]
	transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
						   drop_unconnected_node_types=True)
	#dataset = HGBDataset(root='tmp/Freebase',name='Freebase')[0]
	dataset = HGBDataset(root='tmp/Freebase',name='Freebase',transform=transform)[0]
	dataset2 = HGBDataset(root='tmp/Freebase2',name='Freebase',transform=T.ToUndirected())[0]
	#print(dataset2)
	num_books=dataset['book'].y.shape[0]

	num_books = 40402
	free_type = ['film','music','sports','people','location','organization','business']
	free_node = [19427,82351,1025,17641,9368,2731,7153]
	for index,type in enumerate(free_type):
		dataset2[type].x = torch.Tensor(free_node[index], 128).uniform_(-0.5, 0.5)
	dataset2['book'].x = torch.Tensor(num_books, 128).uniform_(-0.5, 0.5)
	dataset['book'].x = torch.Tensor(num_books, 128).uniform_(-0.5, 0.5)
	metapaths1 = [
		('book','and', 'book'),
		('book','about','organization'),
		('organization','to','music'),
		('music','in','book'),
		('music','and','music')

	]

	metapaths2 = [
		('book','and','book'),
		('book','about','organization'),
		('organization','for','business'),
		('business','about','book')

	]

	data1, data2 = split_metapaths(dataset2, metapaths1, metapaths2)
	graph = [data1,data2]
	train_mask = torch.load('freebase.pt')['train']
	val_mask = torch.load('freebase.pt')['val']
	test_mask = torch.load('freebase.pt')['test']
	dataset2['book'].train_mask = train_mask
	dataset2['book'].val_mask = val_mask
	out = group_hetero_graph(dataset2.edge_index_dict, dataset2.num_nodes_dict)
	edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out
	homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
				 node_type=node_type, local_node_idx=local_node_idx,
				 num_nodes=node_type.size(0))
	feat_list = []
	for key, x in dataset2.x_dict.items():
		x_pad = x
		if x.dim() == 1:
			x_pad = x.unsqueeze(1)
		feat_list.append(x_pad)

	# Step 2: Create global node feature tensor
	x_dict = {key2int[key]: x for key, x in dataset2.x_dict.items()}
	node_feats = torch.zeros((node_type.size(0), max(x.shape[1] for x in x_dict.values())))

	for t, x in x_dict.items():
		idx = (node_type == t)
		node_feats[idx, :x.shape[1]] = x

	homo_data.x = node_feats
	target_ntype = 'book'  # Or 'paper', if you're predicting for papers

	target_nid = local2global[target_ntype]
	homo_data.y = torch.full((node_type.size(0),), -1, dtype=torch.long)
	homo_data.y[target_nid] = dataset2[target_ntype].y

	# Train/Val/Test masks
	homo_data.train_mask = torch.zeros((node_type.size(0),), dtype=torch.bool)
	homo_data.val_mask = torch.zeros((node_type.size(0),), dtype=torch.bool)
	homo_data.test_mask = torch.zeros((node_type.size(0),), dtype=torch.bool)

	homo_data.train_mask[target_nid[dataset2[target_ntype].train_mask]] = True
	homo_data.val_mask[target_nid[dataset2[target_ntype].val_mask]] = True
	homo_data.test_mask[target_nid[dataset2[target_ntype].test_mask]] = True

	return homo_data,dataset2,graph,train_mask, val_mask, test_mask


#load_ACM(seed=42)
#load_Freebase(seed=42)
#load_DBLP(seed=42)
#load_OgbMag(seed=42)
#load_IMDB(seed=42)
