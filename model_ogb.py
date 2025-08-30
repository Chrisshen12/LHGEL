import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, HANConv, GATv2Conv,HeteroConv, to_hetero, RGCNConv
from typing import Dict, List, Union

from torch_geometric.utils import to_dense_adj,dense_to_sparse, softmax,subgraph
from torch_geometric.utils.dropout import dropout_edge, dropout_node
import random
from collections import defaultdict
#from layers import GraphConvolution
import warnings
warnings.filterwarnings("ignore")


class HeteroSAGE(nn.Module):
	def __init__(self, hidden, out_dim, dropout, layer_s):
		super().__init__()
		self.layers = nn.ModuleList()
		self.dropout = dropout

		# First layer
		self.enc = (SAGEConv((-1, -1), hidden))
		#self.enc = (GCNConv((-1, -1), hidden))

		# Hidden layers
		for _ in range(layer_s):
			self.layers.append(SAGEConv((-1, -1), hidden))
			#self.layers.append(GCNConv((-1, -1), hidden))

		# Output layer
		self.dec =(SAGEConv((-1, -1), out_dim))
		#self.dec =(GCNConv((-1, -1), out_dim))

	def forward(self, x, edge_index):
		x = F.dropout(x,self.dropout,training=self.training)
		x = F.leaky_relu(self.enc(x,edge_index),0.1)
		for i, conv in enumerate(self.layers):
			x = F.leaky_relu(conv(x,edge_index),0.1)
		x = self.dec(x,edge_index)

		return F.log_softmax(x,dim=1)





class AttentionH(nn.Module):
	def __init__(self, feat_dim, atten_dim,num_gcn):
		super(AttentionH, self).__init__()
		self.attention_list = nn.ModuleList()
		self.num_gcn = num_gcn
		self.att = torch.nn.Linear(feat_dim,atten_dim)
		for i in range(num_gcn):
			self.attention_list.append(torch.nn.Linear(feat_dim,atten_dim))
		self.agg = torch.nn.Linear(num_gcn*atten_dim,num_gcn)
		# self.dec_list = nn.ModuleList()
		# for i in range(num_gcn):
		# 	self.dec_list.append(nn.Linear(feat_dim,out_dim))
	def forward(self,embed):
		gcn_attention = []
		#print(embed[0].shape)
		#print(len(embed))
		#print(self.num_gcn)
		#mean_embed = torch.mean(embed,dim=0)
		for i in range(self.num_gcn):
			#gcn_attention.append(self.attention_list[i].forward(embed[i]))
			#print(f'mean:{torch.mean(embed[i],dim=0).unsqueeze(0).shape}')
			#print(f"embed[{i}].shape = {embed[i].shape}") #[64]
			gcn_attention.append(self.attention_list[i].forward(embed[i]))
			#gcn_attention.append(self.attention_list[i].forward(torch.mean(embed[i],dim=0).unsqueeze(0)))
		#print(f'gcn:{gcn_attention[0].shape}')
		#attention = torch.softmax(self.agg(torch.cat(tuple(gcn_attention),dim=1)),dim=1)

		attention = self.agg(torch.cat(tuple(gcn_attention),dim=1)) #[node,num_gcn]
		# print(torch.min(attention,dim=1).values.unsqueeze(1))
		# print(attention.shape)
		# row_max_values = torch.max(attention, dim=1).values  # Get max of each row
		# row_with_max = torch.argmax(row_max_values)  # Get row index with highest max
		# print(attention[row_with_max,:])  # e.g., tensor(1)
		# print("------------------")
		#attention = torch.sigmoid(attention)
		attention = (attention-torch.mean(attention,dim=1).unsqueeze(1))
		# print(attention[row_with_max,:])  # e.g., tensor(1)
		# print("------------------")
		# ------ min max way ---------------

		atten_min = torch.min(attention,dim=1).values.unsqueeze(1)
		atten_max = torch.max(attention,dim=1).values.unsqueeze(1)
		# print(atten_min[0,:])
		# print(atten_max[0,:])
		attention = (attention-atten_min)/(atten_max-atten_min)
		# print(attention[row_with_max,:])  # e.g., tensor(1)
		# print("------------------")
		final_embed = (attention[:,0].unsqueeze(1)+(1/self.num_gcn))*embed[0]
		for i in range(1,self.num_gcn):
			final_embed += (attention[:,i].unsqueeze(1)+(1/self.num_gcn))*embed[i] #no_MLP
		#---------------------------------------------------------------------------softmax
		# attention = torch.softmax(attention,dim=1)
		# final_embed = (attention[:,0].unsqueeze(1)+(1/self.num_gcn))*embed[0]
		# for i in range(1,self.num_gcn):
		# 	final_embed += (attention[:,i].unsqueeze(1)+(1/self.num_gcn))*embed[i] #no_MLP

		#attention = attention-torch.mean(attention,dim=1).unsqueeze(1)
		#print("attention shape",attention.shape)
		#attention = attention-torch.mean(attention,dim=1).unsqueeze(1)
		#attention = attention/self.num_gcn
		#print('att',torch.max(attention))
		#print('attmin',torch.min(torch.abs(attention)))
		#print(torch.min(attention))
		#print(f'final_embed: {final_embed.shape}')
		#print("attention",attention[0,:])
		#---------------------------------------------------------------------------minmax
		# final_embed = (attention[:,0].unsqueeze(1)+(1/self.num_gcn))*embed[0]
		# for i in range(1,self.num_gcn):
		# 	final_embed += (attention[:,i].unsqueeze(1)+(1/self.num_gcn))*embed[i] #no_MLP
		#---------------------------------------------------------------------------softmax
		# final_embed = (attention[:,0].unsqueeze(1))*embed[0]
		# for i in range(1,self.num_gcn):
		# 	final_embed += (attention[:,i].unsqueeze(1))*embed[i] #no_MLP
		return final_embed

class HeteroBatch(nn.Module):
	def __init__(self, hidden, out_dim, dropout, layer_s):
		super().__init__()
		self.layers = nn.ModuleList()
		self.dropout = dropout

		# First layer
		self.enc = (SAGEConv((-1, -1), hidden))
		#self.enc = (GCNConv((-1, -1), hidden))

		# Hidden layers
		for _ in range(layer_s):
			self.layers.append(SAGEConv((-1, -1), hidden))
			#self.layers.append(GCNConv((-1, -1), hidden))

		# Output layer
		self.dec =(SAGEConv((-1, -1), out_dim))
		#self.dec =(GCNConv((-1, -1), out_dim))

	def forward(self, x, edge_index):
		x = F.dropout(x,self.dropout,training=self.training)
		x = F.leaky_relu(self.enc(x,edge_index),0.1)
		for i, conv in enumerate(self.layers):
			x = F.leaky_relu(conv(x,edge_index),0.1)
		x = self.dec(x,edge_index)

		return F.log_softmax(x,dim=1)

class HeteroBatchv2(nn.Module):
	def __init__(self, hidden, out_dim, dropout, layer_s):
		super().__init__()
		self.layers = nn.ModuleList()
		self.dropout = dropout

		# First layer
		self.enc = (SAGEConv((-1, -1), hidden))
		#self.enc = (GCNConv((-1, -1), hidden))

		# Hidden layers
		for _ in range(layer_s):
			self.layers.append(SAGEConv((-1, -1), hidden))
			#self.layers.append(GCNConv((-1, -1), hidden))

		# Output layer
		self.dec =(SAGEConv((-1, -1), out_dim))
		#self.dec =(GCNConv((-1, -1), out_dim))

	def forward(self, x, edge_index):
		x = F.dropout(x,self.dropout,training=self.training)
		x = F.leaky_relu(self.enc(x,edge_index),0.1)
		for i, conv in enumerate(self.layers):
			x = F.leaky_relu(conv(x,edge_index),0.1)
		x = self.dec(x,edge_index)

		return x
#several rgcn models
class Multi_Hetero(nn.Module):
	def __init__(self, hidden, out_dim, attention_dim,num_gnn, dropout, layer_s):
		super(Multi_Hetero, self).__init__()
		self.model_list = nn.ModuleList()
		#self.num_path = num_path
		self.num_gnn = num_gnn

		# Initialize GCNs for each metapath
		self.model_list=nn.ModuleList([HeteroBatchv2(hidden, hidden, dropout, layer_s) for _ in range(num_gnn)])
		#self.gcn_list=nn.ModuleList([GraphSAGE(in_dim, hidden, hidden, dropout, layer_s) for _ in range(num_gcn)])
		#self.gcn_list.append(nn.ModuleList([GAT(in_dim, hidden, hidden, dropout,layer_s) for _ in range(num_gcn)]))
		self.attention_list = AttentionH(feat_dim=hidden, atten_dim=attention_dim, num_gcn=self.num_gnn)
		self.dec = nn.Linear(hidden, out_dim)


	def forward(self, graph):
		path_predictions = []
		# Collect predictions from 3 models for the current metapath
		for model in self.model_list:
			model = to_hetero(model, graph.metadata(), aggr='sum')
			path_predictions.append(model(graph.x_dict, graph.edge_index_dict))
		att_layer = self.attention_list
		#print(path_predictions)
		path_predictions = [pred['paper'] for pred in path_predictions]
		attn_output = att_layer(path_predictions)
		final_prediction = self.dec(attn_output)

		return final_prediction

#single
class Single_Hetero(nn.Module):
	def __init__(self, target,metadata, hidden, out_dim, dropout, layer_s):
		super(Single_Hetero, self).__init__()
		self.target = target

		# Initialize GCNs for each metapath
		#self.model_list=nn.ModuleList([HeteroBatch(hidden, hidden, dropout, layer_s) for _ in range(num_gnn)])
		self.model=to_hetero(HeteroBatchv2(hidden, out_dim, dropout, layer_s),metadata, aggr='sum')
		#self.gcn_list=nn.ModuleList([GraphSAGE(in_dim, hidden, hidden, dropout, layer_s) for _ in range(num_gcn)])
		#self.gcn_list.append(nn.ModuleList([GAT(in_dim, hidden, hidden, dropout,layer_s) for _ in range(num_gcn)]))

		#self.dec = nn.Linear(hidden, out_dim)


	def forward(self, graph):
		path_predictions = []
		# Collect predictions from 3 models for the current metapath
		target = self.target

		path_prediction = self.model(graph.x_dict, graph.edge_index_dict)
		

		#return F.log_softmax(path_prediction[target],dim=1)
		return path_prediction
		
class Ogb_batch(nn.Module):
	def __init__(self, mode,target, metadata, num_node, batch_list, num_graph, hidden, out_dim, attention_dim, num_gnn, dropout, layer_s):
		super(Ogb_batch, self).__init__()
		self.target = target
		self.num_metapaths = len(metadata)
		self.num_batch_sizes = len(batch_list)
		self.num_node = num_node
		self.mode = mode
		# Create model for each (metapath, batch size) pair
		self.model_list = nn.ModuleList()
		for i in range(self.num_metapaths):
			for j in range(self.num_batch_sizes):
				# IMPORTANT: use the metadata corresponding exactly to the batch loader i,j
				# e.g., metadata[i][j] or similar
				self.model_list.append(
					Single_Hetero(
						target,
						metadata[i][j].metadata(),  # metadata must match the batch loader i,j
						hidden,
						hidden,
						dropout,
						layer_s
					)
				)
		self.attention_batch = AttentionH(feat_dim=hidden, atten_dim=attention_dim, num_gcn=self.num_batch_sizes)
		self.attention_meta = AttentionH(feat_dim=hidden, atten_dim=attention_dim, num_gcn=self.num_metapaths)
		self.metapath_att = nn.Parameter(torch.randn(self.num_metapaths, out_dim))
		# total number of models: num_metapaths * num_batch_sizes = 4
		self.weights = nn.Parameter(torch.ones(self.num_metapaths))
		self.attention_weights = nn.Parameter(torch.randn(self.num_metapaths * self.num_batch_sizes,num_node, out_dim))
		self.dec = nn.Linear(hidden,out_dim)
	def forward(self, count, batch):
		"""
		Perform forward pass for a single model in model_list on a given batch.
		"""
		#print(batch)
		#print('count', count)

		target = self.target
		pred = self.model_list[count](batch)
		#print('batch_size',batch[target].batch_size)
		#print(batch)
		if self.mode == 'att':
			pred = pred[target]
		else:
			pred = self.dec(pred[target])
			pred = F.log_softmax(pred,dim=1)
		batch = batch[target]
		#node_ids = batch.n_id[:batch.batch_size]
		labels = batch.y[:batch.batch_size]

		return pred[:batch.batch_size], labels


	def attention(self, predict):  # avg_pred: [num_views, D]
		#print('total_pred',len(predict))
		group_sizes = []
		for _ in range(self.num_metapaths):
			group_sizes.append(self.num_batch_sizes)
		#group_sizes = [self.num_batch_sizes, self.num_batch_sizes]  # e.g., first 2 for metapath1, last 2 for metapath2

		start = 0
		all_embeddings = []
		embeddings = []
		if group_sizes[0]>1:
			for i,size in enumerate(group_sizes):
				group = predict[start:start + size]
				min_nodes = min([pred.shape[0] for pred in group])

				# Truncate all predictions to min_nodes
				group = [pred[:min_nodes] for pred in group]
				#print('group metapath',len(group))
				#print(group[0].shape)

				emb= self.attention_batch(group) # get embedding from model
				# weights = torch.softmax(self.weights,dim=0)
				# print('weight',weights)
				# weighted_emb = weights[i]* emb
				embeddings.append(emb)
				
				#embeddings.append(weighted_emb)
				start+=size
			if len(embeddings)>1:
				final_prediction = self.attention_meta(embeddings)
			else:
				final_prediction = embeddings[0]
		else:
			final_prediction = self.attention_meta(predict)
		#final_prediction = torch.mean(torch.stack(embeddings),dim=0)
		fused = self.dec(final_prediction)
		# print(fused.shape)


		# final_pred = self.attention_list[0](predict)
		# fused = self.dec(final_pred)
		return F.log_softmax(fused,dim=1)

class RGCN(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
		super().__init__()
		self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
		self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations)

	def forward(self, x, edge_index, edge_type):
		x = F.relu(self.conv1(x, edge_index, edge_type))
		x = self.conv2(x, edge_index, edge_type)
		return x