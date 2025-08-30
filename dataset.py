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

def add_reverse_edges(data):
	new_data = HeteroData()

	for node_type in data.node_types:
		new_data[node_type].num_nodes = data[node_type].num_nodes

	for edge_type in data.edge_types:
		src, rel, dst = edge_type
		edge_index = data[edge_type].edge_index

		# Add original edge
		new_data[(src, rel, dst)].edge_index = edge_index

		# Add reversed edge
		reverse_rel = rel + '_rev'  # Give a new name like 'writes_rev'
		new_data[(dst, reverse_rel, src)].edge_index = edge_index.flip(0)  # reverse source and target

	return new_data
def get_binary_mask(total_size, indices):
	mask = torch.zeros(total_size)
	mask[indices] = 1
	return mask.byte().to(torch.bool)
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
	#print(homo_data)
	#print(dataset['author', 'metapath_0', 'author'].edge_index)
	#print(dataset2.edge_index_dict)
	#print(dataset2)
	x = dataset['author'].x
	#print("x",x.shape)
	y = dataset['author'].y
	# APA = torch.load('combined_author.pt')['APA']
	# APTPA = torch.load('combined_author.pt')['APTPA']
	# APCPA = torch.load('combined_author.pt')['APCPA']
	APA = dataset['author', 'metapath_0', 'author'].edge_index
	APTPA = dataset['author', 'metapath_1', 'author'].edge_index
	APCPA = dataset['author', 'metapath_2', 'author'].edge_index
	# print("APA",APA.shape)
	# print("APTPA",APTPA.shape)
	# print("APCPA",APCPA.shape)
	data_APA = Data(x=x, edge_index=APA, y=y)
	data_APTPA = Data(x=x, edge_index=APTPA, y=y)
	data_APCPA = Data(x=x, edge_index=APCPA, y=y)
	# data_APA = Data(x=x, edge_index=dense_to_sparse(APA.fill_diagonal_(1))[0], y=y)
	# data_APTPA = Data(x=x, edge_index=dense_to_sparse(APTPA.fill_diagonal_(1))[0], y=y)
	# data_APCPA = Data(x=x, edge_index=dense_to_sparse(APCPA.fill_diagonal_(1))[0], y=y)
	# print(f'APA:{len(data_APA.edge_index[0])}')
	# print(f'APTPA:{len(data_APTPA.edge_index[0])}')
	# print(f'APCPA:{len(data_APCPA.edge_index[0])}')

	# num_nodes = x.shape[0]
	# float_mask = np.random.permutation(np.linspace(0, 1, num_nodes))

	# train_idx = np.where(float_mask <= 0.2)[0]
	# val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
	# test_idx = np.where(float_mask > 0.3)[0]
	# train_mask = get_binary_mask(num_nodes, train_idx)
	# val_mask = get_binary_mask(num_nodes, val_idx)
	# test_mask = get_binary_mask(num_nodes, test_idx)

	# np.save(f'train_mask_{seed}', train_mask.numpy())
	# np.save(f'test_mask_{seed}', test_mask.numpy())
	# np.save(f'val_mask_{seed}', val_mask.numpy())

	# train_mask = torch.from_numpy(np.load('train_mask_42.npy', allow_pickle=True))
	# val_mask = torch.from_numpy(np.load('val_mask_42.npy', allow_pickle=True))
	# test_mask = torch.from_numpy(np.load('test_mask_42.npy', allow_pickle=True))
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
	#torch.save({'feature':x,'graph_paper':data_APA,'graph_term': data_APTPA, 'graph_c': data_APCPA, 'labels':y, 'train':train_mask,'val':val_mask,'test':test_mask}, 'combined_author.pt')
	# print(torch.sum(train_mask==True))
	# print(torch.sum(val_mask==True))
	# print(torch.sum(test_mask==True))

	return homo_data,dataset2, dblp_graph,train_mask, val_mask, test_mask
	#return dataset,homo_data,dataset2, dblp_graph,data_APA,data_APCPA,data_APTPA, train_mask, val_mask, test_mask
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
	#print(dataset)
	dataset2 = IMDB(root='tmp/IMDB2')[0]
	#print(dataset2)
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
	#print(homo_data)
	x = dataset2['movie'].x
	#print("x",x.shape)
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
	#degrees = degree(data_MAM.edge_index[0], num_nodes=num_movies)
	# print(f'node{num_movies}')
	# print(f'degree:{(degrees <= 1).sum().item()}')

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
	#dataset2['movie','to','movie'] = dense_to_sparse(MDM.fill_diagonal_(1))[0]
	#print(dataset2)

	data1, data2 = split_metapaths(dataset2, metapaths1, metapaths2)
	#print(data1)
	#print(data2)
	data1['movie','movie'].edge_index = data_MAM.edge_index
	data2['movie','movie'].edge_index = data_MDM.edge_index
	graph = [data1,data2]
	# print(torch.sum(train_mask==True))
	# print(torch.sum(val_mask==True))
	# print(torch.sum(test_mask==True))

	return homo_data,dataset2, graph,train_mask, val_mask, test_mask
	#return dataset,data_MAM, data_MDM,data_AMA, data_DMD, train_mask, val_mask, test_mask

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
	#print(homo_data)
	# print(torch.sum(train_mask))
	# print(torch.sum(new_train_mask==True))
	# print(torch.sum(val_mask==True))
	# print(torch.sum(test_mask==True))
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
	#return dataset,data_PAP, data_PSP, new_train_mask, val_mask, test_mask

def load_Ogb(seed=42,path=''):
	if path != '':
		os.chdir(path)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	# metapaths = [[('paper', 'paper'), ('paper','paper')],
	#          [('paper', 'author'), ('author', 'paper')],
	#          [('paper', 'author'),('author','institution'),('institution','author'), ('author', 'paper')]]
	#transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,drop_unconnected_node_types=True)

	#dataset = OGB_MAG(root='tmp/Ogb',transform=transform)[0]
	#print(dataset)
	dataset2 = OGB_MAG(root='tmp/Ogb2')[0]
	#print(dataset2)

	x = dataset2['paper'].x
	#print("x",x.shape)
	y = dataset2['paper'].y
	# print(y)
	# label_counts = torch.bincount(y)
	# sorted_counts, sorted_indices = torch.sort(label_counts)
	# for label, count in enumerate(sorted_counts):
	#     print(f"Label {label}: {count.item()} occurrences")
	#print(f'class:{y.unique().shape[0]}')
	# num_papers = 1000
	# num_authors = 1000
	# num_institutions = 1000

	# selected_nodes = []
	# index_mask = []
	# for i in range(x.shape[0]):
	# 	if y[i] <=120 and y[i]>=65:
	# 		index_mask.append(i)
	# index_mask = torch.tensor(index_mask)
	# torch.save(index_mask,"index_mask_OGB_MAG.pt")

	index_mask =torch.load("index_mask_OGB_MAG.pt")
	x = x[index_mask]
	y = y[index_mask]
	unique_labels = torch.unique(y)
	label_mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
	y_consecutive = torch.tensor([label_mapping[label.item()] for label in y])
	y = y_consecutive
	#print(y)
	num_papers = index_mask.shape[0]
	print(f'num of papers selected{num_papers}')
	#print(f'class:{y.unique().shape[0]}')


	# Step 3: Create the subgraph with the selected nodes
	#subgraph_edge_index, subgraph_edge_attr = subgraph(index_mask, PP_edge, relabel_nodes=True)
	#paper_nodes = torch.arange(num_papers)
	#-------------------------------------------------------------------------------------------------------
	# PP_edge = dataset2['paper', 'cites', 'paper'].edge_index
	# paper_to_paper1, _ = subgraph(index_mask, PP_edge, relabel_nodes=True)
	# paper_to_paper = torch.zeros((num_papers, num_papers))
	# paper_to_paper[paper_to_paper1[0], paper_to_paper1[1]] = 1
	# PP = torch.matmul(paper_to_paper, paper_to_paper.T)
	# PP = (PP > 0).int()

	# PA_edge = dataset2['author', 'writes', 'paper'].edge_index
	# PA_edge[0] = dataset2['author', 'writes', 'paper'].edge_index[1]
	# PA_edge[1] = dataset2['author', 'writes', 'paper'].edge_index[0]
	# paper_sub, _ = subgraph(index_mask, PA_edge, relabel_nodes=True)
	# unique_authors = torch.unique(paper_sub[1])
	# num_authors = len(unique_authors)
	# label_mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_authors)}
	# paper_sub[1] = torch.tensor([label_mapping[label.item()] for label in paper_sub[1]])
	# paper_to_author = torch.zeros((num_papers,num_authors))
	# paper_to_author[paper_sub[0], paper_sub[1]] = 1
	# #PAP_edge = torch.matmul(author_to_paper.T, author_to_paper)
	# PAP_edge = torch.matmul(paper_to_author,paper_to_author.T)
	# PAP = (PAP_edge > 0).int()
	#----------------------------------------------------------------------------------------------------------
	# AI_edge = dataset2['author', 'affiliated_with', 'institution'].edge_index
	# author_nodes = torch.arange(num_authors)
	# institution_nodes = torch.arange(num_institutions)
	# institution_sub, _ = subgraph(author_nodes, AI_edge, relabel_nodes=True)
	# author_to_institution = torch.zeros((num_authors, num_papers))
	# author_to_institution[institution_sub[0], institution_sub[1]] = 1
	# PAIAP_edge = torch.matmul(torch.matmul(torch.matmul(author_to_paper.T, author_to_institution), author_to_institution.T),author_to_paper)
	# PAIAP = (PAIAP_edge > 0).int()

	# torch.save({'feature':x,'graph_paper':PP,'graph_author': PAP, 'graph_institution': PAIAP, 'labels':y}, 'combined_ogb.pt')
	#torch.save({'feature':x,'graph_paper':PP,'graph_author': PAP, 'labels':y}, 'combined_ogb_selected_by_65_120.pt')
	PP = torch.load('combined_ogb_selected_by_65_120.pt')['graph_paper']

	#print(f'PP_edges{torch.sum(PP)}')
	PAP = torch.load('combined_ogb_selected_by_65_120.pt')['graph_author']
	#print(f'PAP_edges{torch.sum(PAP)}')
	# PP = torch.load('combined_ogb.pt')['graph_paper']
	# print(f'PP_edges{torch.sum(PP)}')
	# PAP = torch.load('combined_ogb.pt')['graph_author']
	# print(f'PAP_edges{torch.sum(PAP)}')
	#PAIAP = torch.load('combined_ogb.pt')['graph_institution']
	data_PP = Data(x=x, edge_index=dense_to_sparse(PP.fill_diagonal_(1))[0], y=y)
	data_PAP = Data(x=x, edge_index=dense_to_sparse(PAP.fill_diagonal_(1))[0], y=y)
	#degrees = degree(data_PP.edge_index[0], num_nodes=num_papers)
	#print(torch.sum)
	# print(f'node{num_papers}')
	# print(f'degree:{(degrees <= 1).sum().item()}')
	#data_PAIAP = Data(x=x, edge_index=dense_to_sparse(PAIAP.fill_diagonal_(1))[0], y=y)
	metapaths = [[('paper', 'paper2'), ('paper2', 'paper')],
			 [('paper', 'author'),('author','paper')]]
	transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
						   drop_unconnected_node_types=True)


	# data = HeteroData()
	# data['paper'].x = x
	# data['paper'].y = y

	# data['paper', 'paper2'].edge_index = dense_to_sparse(paper_to_paper.fill_diagonal_(1))[0]
	# data['paper2', 'paper'].edge_index = data['paper', 'paper2'].edge_index[[1, 0]]

	# data['paper', 'author'].edge_index = dense_to_sparse(paper_to_author.fill_diagonal_(1))[0]
	# data['author', 'paper'].edge_index = data['paper', 'author'].edge_index[[1, 0]]

	# dataset = transform(data)

	# torch.save(dataset, 'ogb_10w.pt')
	dataset = torch.load('ogb_10w.pt')
	# num_nodes = num_papers
	# float_mask = np.random.permutation(np.linspace(0, 1, num_nodes))

	# train_idx = np.where(float_mask <= 0.2)[0]
	# val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
	# test_idx = np.where(float_mask > 0.3)[0]
	# train_mask = get_binary_mask(num_nodes, train_idx)
	# val_mask = get_binary_mask(num_nodes, val_idx)
	# test_mask = get_binary_mask(num_nodes, test_idx)

	# np.save("train_mask_ogb10w.npy", train_mask)
	# np.save("val_mask_ogb10w.npy", val_mask)
	# np.save("test_mask_ogb10w.npy", test_mask)
	train_mask = torch.from_numpy(np.load('train_mask_ogb10w.npy', allow_pickle=True))
	val_mask = torch.from_numpy(np.load('val_mask_ogb10w.npy', allow_pickle=True))
	test_mask = torch.from_numpy(np.load('test_mask_ogb10w.npy', allow_pickle=True))

	return dataset, data_PP, data_PAP, train_mask, val_mask, test_mask

	# return dataset2, data_PP, data_PAP,data_PAIAP, train_mask, val_mask, test_mask
def edge_index_to_adj(edge_index, src_size, dst_size, device):
	values = torch.ones(edge_index.size(1), device=device)
	return torch.sparse_coo_tensor(edge_index, values, (src_size, dst_size)).coalesce()

def drop_edges_random(edge_index, keep_ratio=0.1):
	# sparse_tensor = sparse_tensor.coalesce()
	# indices = sparse_tensor.indices()
	# values = sparse_tensor.values()
	# num_edges = values.size(0)

	# perm = torch.randperm(num_edges)
	# keep_num = int(keep_ratio * num_edges)
	# keep_idx = perm[:keep_num]

	# filtered_indices = indices[:, keep_idx]
	# #print(filtered_indices.size(1))
	# filtered_values = values[keep_idx]

	# return torch.sparse_coo_tensor(
	# 	filtered_indices,
	# 	filtered_values,
	# 	size=sparse_tensor.size()
	# ).coalesce()
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
	
	#print(homo_data)
	# edge_index_dict = dataset.edge_index_dict
	# r, c = edge_index_dict[('author', 'affiliated_with', 'institution')]
	# edge_index_dict[('institution', 'to', 'author')] = torch.stack([c, r])

	# r, c = edge_index_dict[('author', 'writes', 'paper')]
	# edge_index_dict[('paper', 'to', 'author')] = torch.stack([c, r])

	# r, c = edge_index_dict[('paper', 'has_topic', 'field_of_study')]
	# edge_index_dict[('field_of_study', 'to', 'paper')] = torch.stack([c, r])

	# # Convert to undirected paper <-> paper relation.
	# edge_index = to_undirected(edge_index_dict[('paper', 'cites', 'paper')])
	# edge_index_dict[('paper', 'cites', 'paper')] = edge_index


	# out = group_hetero_graph(dataset.edge_index_dict, dataset.num_nodes_dict)
	# edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out
	# data_BB4 = Data(x=x, edge_index=edge_index, y=y)
	# torch.save(data_BB4,'one_ogb.pt')

	#dataset = OGB_MAG(root='tmp/Ogb',transform=transform)[0]
	# dataset = add_reverse_edges(dataset)
	# metapaths = [[('paper','cites', 'paper'), ('paper','cites','paper')],
	# 		 [('paper','writes_rev', 'author'), ('author','writes', 'paper')],
	# 		 [('paper', 'has_topic', 'field_of_study'), ('field_of_study', 'has_topic_rev', 'paper')]]
	metapaths = [#[('paper', 'paper2'), ('paper2','paper')],
		 [('paper', 'author'), ('author', 'paper')],
	   [('paper', 'field_of_study'), ('field_of_study', 'paper')]]
	#metapaths=[[('paper', 'author'),('author','institution'),('institution','author'), ('author', 'paper')],]
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

	#print(data)
	#HAN_dataset = transform(data)
	#print(HAN_dataset)
	#torch.save(HAN_dataset, 'ogb_HAN.pt')

	#HAN_dataset=torch.load('ogb_HAN.pt')
	#HAN_dataset['paper', 'metapath_0', 'paper'].edge_index = drop_edges_random(HAN_dataset['paper', 'metapath_0', 'paper'].edge_index)
	#HAN_dataset['paper', 'metapath_1', 'paper'].edge_index = drop_edges_random(HAN_dataset['paper', 'metapath_1', 'paper'].edge_index)
	#print(HAN_dataset)


	# bb1 = to_undirected(dataset['paper', 'cites', 'paper'].edge_index)
	# #print(dataset)
	# #print(dataset)
	# bb2 = data_HAN['paper','metapath_0','paper'].edge_index
	# #torch.save(bb2,'PFP.pt')
	# bb3 = data_HAN['paper','metapath_1','paper'].edge_index


	# #bb4 = dataset['paper','metapath_2','paper'].edge_index

	# data_BB1 = Data(x=x, edge_index=bb1, y=y)
	# data_BB2 = Data(x=x, edge_index=bb2, y=y)
	# data_BB3 = Data(x=x, edge_index=bb3, y=y)
	#data_BB4 = Data(x=x, edge_index=bb4, y=y)
	#torch.save({'graph_bb1':data_BB1,'graph_bb2':data_BB2,'graph_bb3':data_BB3}, 'ogbmag_new.pt')

	# Get number of nodes in each type
	num_papers = data['paper'].num_nodes
	#print(num_paper)
	num_author = 1134649
	#print(num_author)
	num_inst = 8740
	num_fields=59965
	num_institution = 8740

	# A_pa = drop_edges_random(torch.sparse_coo_tensor(
	# 	indices=data['paper', 'author'].edge_index,
	# 	values=torch.ones(data['paper', 'author'].edge_index.size(1)),
	# 	size=(num_papers, num_author)))

	# A_ap = drop_edges_random(torch.sparse_coo_tensor(
	# 	indices=data['author', 'paper'].edge_index,
	# 	values=torch.ones(data['author', 'paper'].edge_index.size(1)),
	# 	size=(num_author, num_papers)))

	# A_ai = drop_edges_random(torch.sparse_coo_tensor(
	# 	indices=data['author', 'institution'].edge_index,
	# 	values=torch.ones(data['author', 'institution'].edge_index.size(1)),
	# 	size=(num_author, num_institution)))

	# A_ia = drop_edges_random(torch.sparse_coo_tensor(
	# 	indices=data['institution', 'author'].edge_index,
	# 	values=torch.ones(data['institution', 'author'].edge_index.size(1)),
	# 	size=(num_institution, num_author)))
	# A_PAIAP = torch.sparse.mm(A_pa, A_ai)
	# print(A_PAIAP._nnz())
	# A_PAIAP = torch.sparse.mm(A_PAIAP, A_ia)
	# print(A_PAIAP._nnz())
	# A_PAIAP = torch.sparse.mm(A_PAIAP, A_ap)
	# print(A_PAIAP._nnz())
	# A_PAIAP = A_PAIAP.coalesce()
	# A_PAIAP = A_PAIAP.indices()
	# data_PAIAP = Data(x=x, edge_index=A_PAIAP, y=y)
	# torch.save(data_PAIAP,'ogb_PAIAP_0.9.pt')

	# A_pf = drop_edges_random(torch.sparse_coo_tensor(
	# indices=data['paper', 'field_of_study'].edge_index,
	# values=torch.ones(data['paper', 'field_of_study'].edge_index.size(1)),
	# size=(num_papers, num_fields)))

	# A_fp = drop_edges_random(torch.sparse_coo_tensor(
	# 	indices=data['field_of_study', 'paper'].edge_index,
	# 	values=torch.ones(data['field_of_study', 'paper'].edge_index.size(1)),
	# 	size=(num_fields, num_papers)
	# ))

	# # Compute P→F→P metapath
	# A_pfp = torch.sparse.mm(A_pf, A_fp)

	# # Get non-zero indices (i.e., valid paper-paper connections via field)
	# A_pfp = A_pfp.coalesce()
	# PFP = A_pfp.indices()
	# torch.save(PFP, 'ogbmag_PFP_0.9.pt')
	#data_BB0 = Data(x=x,edge_index=dataset['paper','paper'].edge_index,y=y) #5416271
	#data_BB1 = torch.load('ogbmag_new.pt')['graph_bb1'] #PP 10792672
	#print(data_BB1.edge_index.shape)
	#data_BB2 = torch.load('ogbmag_new.pt')['graph_bb2'] #PPP 253157532
	#print(data_BB2.edge_index.shape)
	#data_BB3 = torch.load('ogbmag_new.pt')['graph_bb3'] #PAP 65933339
	#print(data_BB3.edge_index.shape)
	#data_BB4 = torch.load('one_ogb.pt')
	#data_PFP = Data(x=x,edge_index=torch.load('ogbmag_PFP_0.9.pt'),y=y) #PFP 6673031450
	#print(data_PFP.edge_index.shape)
	#print(data_PFP)
	#data_PAIAP = torch.load('ogb_PAIAP_0.9.pt') #PAIAP 28560050

	author_emb = dataset['author'].x
	author_x = torch.cat([x,author_emb], dim=0)
	author_idx, paper_idx = dataset['author', 'paper'].edge_index
	author_edge = torch.stack([paper_idx, author_idx + num_papers], dim=0)

	topic_emb = dataset['field_of_study'].x
	topic_x = torch.cat([x,topic_emb], dim=0)
	paper_idx, field_idx = dataset['paper', 'field_of_study'].edge_index
	field_edge = torch.stack([paper_idx, field_idx + num_papers], dim=0)
	institution_emb = torch.Tensor(num_institution, 256).uniform_(-0.5, 0.5)
	# data_PP1 = Data(x=x,edge_index=dataset['paper','paper'].edge_index,y=y)
	# data_PP2 = torch.load('ogbmag_new.pt')['graph_bb1'] #PP 10792672
	# data_PA = Data(x=author_x,edge_index=author_edge,y=y)
	# data_PF = Data(x=topic_x,edge_index=field_edge,y=y)
	# ogb_graph = [data_PP1, data_PA, data_PF]
	#print(data_PAIAP.edge_index.shape)

	#print(data_BB3)
	#train_mask = torch.load('ogbmag.pt')['train']
	#val_mask = torch.load('ogbmag.pt')['val']
	#test_mask = torch.load('ogbmag.pt')['test']
	# print(torch.sum(train_mask==True))
	# print(torch.sum(val_mask==True))
	# print(torch.sum(test_mask==True))
	#dataset = torch.load('ogb_HAN.pt')
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
	# print(torch.sum(train_mask==True))
	# print(torch.sum(val_mask==True))
	# print(torch.sum(test_mask==True))
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
	#print(dataset['book'].y.shape)
	#----------------------------------------------------------------
	# book_feature = torch.randn(dataset['book'].y.shape[0], 20)
	# dataset['book'].x = book_feature
	# # #print(book_feature.shape)
	# # #print(dataset['book'].y.shape)
	# bb1 = dataset['book','metapath_0','book'].edge_index
	# bb2 = dataset['book','metapath_1','book'].edge_index
	# bb3 = dataset['book','metapath_2','book'].edge_index

	# data_BB1 = Data(x=book_feature, edge_index=bb1, y=dataset['book'].y)
	# data_BB2 = Data(x=book_feature, edge_index=bb2, y=dataset['book'].y)
	# data_BB3 = Data(x=book_feature, edge_index=bb3, y=dataset['book'].y)
	# train_mask = dataset['book'].train_mask
	# test_mask = dataset['book'].test_mask

	# #torch.save({'graph_bb1':data_BB1,'graph_bb2': data_BB2,'train':train_mask,'test':test_mask},'freebase.pt')
	# # train_mask = torch.load('freebase.pt')['train']
	# # test_mask = torch.load('freebase.pt')['test']

	# train_indices = train_mask.nonzero(as_tuple=True)[0]
	# val_size = int(0.3 * len(train_indices))
	# val_indices = train_indices[:val_size]
	# new_train_indices = train_indices[val_size:]

	# val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
	# val_mask[val_indices] = True

	# new_train_mask = torch.zeros_like(train_mask, dtype=torch.bool)
	# new_train_mask[new_train_indices] = True
	# dataset['book'].train_mask = new_train_mask
	# dataset['book'].val_mask = val_mask
	# torch.save({'graph_bb1':data_BB1,'graph_bb2': data_BB2,'graph_bb3': data_BB3,'train':new_train_mask,'val':val_mask,'test':test_mask}, 'freebase.pt')
	#------------------------------------------------------------------
	#data_BB1 = torch.load('freebase.pt')['graph_bb1']
	#data_BB2 = torch.load('freebase.pt')['graph_bb2']
	#data_BB3 = torch.load('freebase.pt')['graph_bb3']


	#data_BB1.x=eye
	#data_BB2.x=eye
	#data_BB3.x=eye
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
	#print(is_undirected(homo_data.edge_index))
	#print(homo_data)
	# print(data_BB1.y[new_train_mask].unique())
	# print(data_BB1.y[val_mask].unique())
	# print(data_BB1.y[test_mask].unique())

	# print(torch.sum(train_mask==True))
	# print(torch.sum(val_mask==True))
	# print(torch.sum(test_mask==True))

	return homo_data,dataset2,graph,train_mask, val_mask, test_mask
	#return dataset,train_mask, val_mask, test_mask

def reindex(values):
	unique_values = torch.unique(values)
	#print(unique_values)
	# unique_values = unique_values)  # Get unique values and sort them
	mapping = {old.item(): new for new, old in enumerate(unique_values, start=1)}  # Create a mapping
	#print("mapping",mapping)
	return torch.tensor([mapping[v.item()] for v in values],dtype=torch.long)


def load_company(seed=42,dataname='a',path=''):
	if path != '':
		os.chdir(path)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	metapaths = [[('company', 'organization'), ('organization', 'company')],
			 [('company', 'person'),('person','company')]]
	transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
						   drop_unconnected_node_types=True)

	X = torch.from_numpy(np.load('company_data_new.npy',allow_pickle=True).item()['feature'])
	CPC = torch.from_numpy(np.load('company_data_new.npy',allow_pickle=True).item()['graph_people'])
	COC = torch.from_numpy(np.load('company_data_new.npy',allow_pickle=True).item()['graph_org'])
	CO = torch.from_numpy(np.load('CO.npy'))
	CP = torch.from_numpy(np.load('CP.npy'))
	Y = torch.from_numpy(np.load('company_data_new.npy',allow_pickle=True).item()['labels'])
	data = HeteroData()
	data['company'].x = X
	data['company'].y = Y

	data['company', 'organization'].edge_index = dense_to_sparse(CO.fill_diagonal_(1))[0]
	data['company', 'person'].edge_index = dense_to_sparse(CP.fill_diagonal_(1))[0]
	data['organization', 'company'].edge_index = data['company', 'organization'].edge_index[[1, 0]]
	data['person', 'company'].edge_index = data['company', 'person'].edge_index[[1, 0]]

	dataset = transform(data)
	#print(dataset)
	#print(data)

	data_CPC = Data(x=X, edge_index=dense_to_sparse(CPC.fill_diagonal_(1))[0], y=Y)
	data_COC = Data(x=X, edge_index=dense_to_sparse(COC.fill_diagonal_(1))[0], y=Y)

	train_mask = torch.from_numpy(np.load('train_mask_42.npy', allow_pickle=True))
	val_mask = torch.from_numpy(np.load('val_mask_42.npy', allow_pickle=True))
	test_mask = torch.from_numpy(np.load('test_mask_42.npy', allow_pickle=True))

	# print(torch.sum(train_mask==True))
	# print(torch.sum(val_mask==True))
	# print(torch.sum(test_mask==True))


	return dataset, data_CPC, data_COC, train_mask, val_mask, test_mask

#load_ACM(seed=42)
#load_Freebase(seed=42)
#load_LastFM(seed=42,path='')
#load_DBLP(seed=42)
#load_AMiner(seed=42,path='')
#load_Urban(seed=42,path='')
#load_OgbMag(seed=42)
#load_IMDB(seed=42)
#load_Protein(seed=42,path='')
#load_company(seed=42,dataname='',path='')
# print(torch.load('cov_mean3_test.pt'))
# print(torch.load('cov_mean5_test.pt'))
# print(torch.load('cov_mean7_test.pt'))
