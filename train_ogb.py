import numpy as np
import torch
from torch_geometric.loader import NeighborLoader,ClusterData,ClusterLoader
from dataset import load_OgbMag, load_DBLP, load_IMDB, load_ACM, load_Freebase
import argparse
from model_ogb import*
from torch_geometric.utils import to_dense_adj,dense_to_sparse,degree
from torch_geometric.nn import to_hetero
#from torch_geometric.loader import NeighborLoader
from torch_scatter import scatter
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import seaborn as sns
import warnings
import math
from torch_geometric.data import Data, HeteroData
from functools import partial
import torch.nn.functional as F
import os
#import copy
from copy import deepcopy
import time
from collections import defaultdict
import csv

warnings.filterwarnings("ignore")

def delete_file(filepath):

	if os.path.exists(filepath):
		os.remove(filepath)  # Delete the file
		print(f"File '{filepath}' has been deleted.")
		return True
	else:
		print(f"File '{filepath}' does not exist.")
		return False

@torch.no_grad()
def evaluate_batch(model, graphs, train_loader_sets, iter_list, device):
	model.eval()
	acc_list=[]
	auc_list=[]
	loader_iters_sets = [[iter(loader) for loader in loader_set] for loader_set in train_loader_sets]
	total=0
	correct_pred=0
	# Determine how many steps to run
	num_batches = min(min(len(loader) for loader in loader_set)for loader_set in train_loader_sets)
	for batch_idx in range(num_batches):
		small_prediction = []
		large_prediction = []
		small_labels = []
		large_labels = []

		all_preds = []
		all_labels = []
		meta_preds = [[] for imeta in range(len(graphs))]
		meta_labels = [[] for ilabels in range(len(graphs))]
		for metapath_idx, loader_iters in enumerate(loader_iters_sets):
			batch_preds = [[] for imeta in range(len(iter_list))]
			batch_labels = [[] for ilabels in range(len(iter_list))]
			for batch_size_idx, loader_iter in enumerate(loader_iters):
				for _ in range(iter_list[batch_size_idx]):
					try:
						batch = next(loader_iter)
					except StopIteration:
						loader_iters[batch_size_idx] = iter(train_loader_sets[metapath_idx][batch_size_idx])
						batch = next(loader_iters[batch_size_idx])

					model_idx = metapath_idx * len(iter_list) + batch_size_idx
					# print('model: ',model_idx)
					preds, labels = model.forward(model_idx, batch)
					batch_preds[batch_size_idx].append(preds)
					batch_labels[batch_size_idx].append(labels)
			meta_preds[metapath_idx]=(batch_preds)
			meta_labels[metapath_idx]=(batch_labels)
		# print("b1",meta_preds[0][0][0].shape)
		# print("b1",meta_preds[0][0][1].shape)
		# print("b2",meta_preds[0][1][0].shape)					
		final_pred_list = []
		final_label_list = []
		count_final = 0
		for i in range(len(graphs)):
			for j in range(len(iter_list)):
				# print("inside loop start")
				# print(len(meta_preds[i][j]))
				# print(len(meta_preds[i][j][0]))
				# print(meta_preds[i][j][0].shape)
				if len(meta_preds[i][j][0])>1:
					final_pred_list.append(torch.cat(meta_preds[i][j],dim=0))
					final_label_list.append(torch.cat(meta_labels[i][j],dim=0))
				else:
					final_pred_list.append(meta_preds[i][j])
					final_label_list.append(meta_labels[i][j])						
				# print("inside loop",final_pred_list[count_final].shape)
				count_final+=1
		# print("meta1_b1",final_pred_list[0].shape)
		# print("meta1_b2",final_pred_list[1].shape)
		# print("meta2_b1",final_pred_list[2].shape)
		# print("meta2_b2",final_pred_list[3].shape)
		if args.mode == 'att':
			final_preds = model.attention(final_pred_list)
		else:
			final_preds = final_pred_list[0]
			for i in range(1,len(final_pred_list)):
				min_len = min(final_preds.shape[0], final_pred_list[i].shape[0])
				final_preds = final_preds[:min_len] + final_pred_list[i][:min_len]
			final_preds /= len(final_pred_list)
		
		#final_preds = model.attention(torch.stack(final_pred_list, dim=0))
		final_labels = final_label_list[0][:final_preds.shape[0]]

		predicted = final_preds.argmax(dim=1)
		correct = (predicted == final_labels).sum().item()
		#total+=40
		#print(final_labels.size(0))
		#correct_pred+=correct
		acc = correct / final_labels.size(0)
		acc_list.append(acc)
		probs = torch.exp(final_preds).detach().cpu().numpy()
		labels = final_labels[:final_preds.shape[0]].detach().cpu().numpy()
		unique_classes = np.unique(labels)

		if len(unique_classes) < 2:
			auc = float('nan')  # Avoid computing AUC if only one class present
		else:
			labels_bin = label_binarize(labels, classes=unique_classes)
			#auc = roc_auc_score(labels_bin, probs[:, unique_classes], average='macro', multi_class='ovr')

		#auc_list.append(auc)
	#accuracy = correct/total
	accuracy = sum(acc_list)/len(acc_list)
	auc_score = np.nanmean(auc_list)

	return accuracy,auc_score
def run_Ogb(args):
	if args.dataset == 'Ogb':
		_,data, ogb_graph, train_mask, val_mask, test_mask = load_OgbMag(seed=args.seed,feature='uniform')
		graphs = [graph.to(args.device) for graph in ogb_graph]
		data.to(args.device)
		#metadata = [[graphs[0],graphs[0]],[graphs[1],graphs[1]]]
		num_class = ogb_graph[0]['paper'].y.unique().shape[0]
		#print(num_class)
		graphs.append(data)
		#graphs = [graphs[1],graphs[2]]
		#metadata = [[graphs[0],graphs[0],graphs[0]],[graphs[1],graphs[1],graphs[1]],[graphs[2],graphs[2],graphs[2]]]
		#metadata = [[graphs[0],graphs[0],graphs[0]],[graphs[1],graphs[1],graphs[1]]]
		#metadata = [[graphs[1],graphs[1]]]
		#graphs = [ogb_graph[1].to(args.device)]
		#graphs = [graphs[args.choice]]
		metadata = [[graphs[0]],[graphs[1]],[graphs[2]]]
		graphs = [graphs[2]]
		metadata = [[graphs[0]]]

		type = 'paper'
	elif args.dataset == 'DBLP':
		_,data, dblp_graph,train_mask, val_mask, test_mask = load_DBLP(seed=args.seed)
		graphs = [graph.to(args.device) for graph in dblp_graph]
		data.to(args.device)
		graphs.append(data)
		metadata = [[graphs[0],graphs[0],graphs[0]],[graphs[1],graphs[1],graphs[1]],[graphs[2],graphs[2],graphs[2]]]
		#metadata = [[graphs[0],graphs[0]],[graphs[1],graphs[1]]]
		#graphs = [graphs[args.choice]]
		#metadata = [[graphs[0],graphs[0],graphs[0]]]

		#metadata = [[graphs[0],graphs[0]]]
		#graphs = [dblp_graph[0].to(args.device)]
		graphs = [graphs[2]]
		metadata = [[graphs[0]]]
		num_class = dblp_graph[0]['author'].y.unique().shape[0] 
		type = 'author'
	elif args.dataset == 'IMDB':
		_,data, imdb_graph, train_mask, val_mask, test_mask = load_IMDB(seed=args.seed)
		graphs = [graph.to(args.device) for graph in imdb_graph]
		data.to(args.device)
		graphs.append(data)
		metadata = [[graphs[0],graphs[0],graphs[0]],[graphs[1],graphs[1],graphs[1]],[graphs[2],graphs[2],graphs[2]]]
		#metadata = [[graphs[0],graphs[0],graphs[0]],[graphs[1],graphs[1],graphs[1]]]
		#graphs=[graphs[1],graphs[1]]
		#graphs = [graphs[args.choice]]
		#metadata = [[graphs[0],graphs[0],graphs[0]]]
		graphs = [graphs[2]]
		metadata = [[graphs[0]]]
		type = 'movie'
		num_class = imdb_graph[0][type].y.unique().shape[0] 
	elif args.dataset == 'ACM':
		_,data, acm_graph, train_mask, val_mask, test_mask = load_ACM(seed=args.seed)
		graphs = [graph.to(args.device) for graph in acm_graph]
		data.to(args.device)
		graphs.append(data)
		#graphs = [graphs[args.choice]]
		#metadata = [[graphs[0],graphs[0],graphs[0]],[graphs[1],graphs[1],graphs[1]],[graphs[2],graphs[2],graphs[2]]]
		#metadata = [[graphs[0],graphs[0]],[graphs[1],graphs[1]]]
		#graphs=[graphs[1],graphs[1]]
		metadata = [[graphs[0],graphs[0],graphs[0]],[graphs[1],graphs[1],graphs[1]],[graphs[2],graphs[2],graphs[2]]]
		graphs = [graphs[2]]
		metadata = [[graphs[0]]]
		type = 'paper'
		num_class = acm_graph[0][type].y.unique().shape[0]
	elif args.dataset == 'Freebase':
		_,data, free_graph, train_mask, val_mask, test_mask = load_Freebase(seed=args.seed)
		graphs = [graph.to(args.device) for graph in free_graph]
		data.to(args.device)
		#metadata = [[graphs[0],graphs[0]],[graphs[1],graphs[1]]]
		graphs.append(data)
		#graphs = [graphs[0],graphs[2]]
		metadata = [[graphs[0],graphs[0],graphs[0]],[graphs[1],graphs[1],graphs[1]],[graphs[2],graphs[2],graphs[2]]]
		#metadata = [[graphs[0],graphs[0],graphs[0]],[graphs[1],graphs[1],graphs[1]]]
		#graphs=[graphs[1],graphs[1]]
		#graphs = [graphs[2]]
		#metadata = [[graphs[0],graphs[0],graphs[0]]]
		graphs = [graphs[2]]
		metadata = [[graphs[0]]]
		type = 'book'
		num_class = free_graph[0][type].y.unique().shape[0] 
		#print(num_class)

	train_node = train_mask.sum().item()
	val_node = val_mask.sum().item()
	test_node = test_mask.sum().item()

	num_graph = len(graphs)
	neighbor_list = [args.num_neighbor] * args.num_hop
	train_loaders = []
	val_loaders = []
	test_loaders = []
	train_loader_sets = []
	val_loader_sets = []
	test_loader_sets = []
	batch_list = [0.1]
	#batch_list = [batch_list[args.choice]]
	iter_list = [int(max(batch_list)/b) for b in batch_list]

	# for mult in batch_list:
	# 	batch_size_train = math.ceil(mult * train_node)
	# 	batch_size_val = math.ceil(mult * val_node)
	# 	batch_size_test = math.ceil(mult * test_node)

	# 	train_loaders = [NeighborLoader(graph, num_neighbors=neighbor_list, batch_size=batch_size_train, input_nodes=(type,train_mask), shuffle=False,drop_last=False) for graph in graphs]
	# 	val_loaders = [NeighborLoader(graph, num_neighbors=neighbor_list, batch_size=batch_size_val, input_nodes=(type,val_mask), shuffle=False,drop_last=False) for graph in graphs]
	# 	test_loaders = [NeighborLoader(graph, num_neighbors=neighbor_list, batch_size=batch_size_test, input_nodes=(type,test_mask), shuffle=False,drop_last=False) for graph in graphs]

	# 	train_loader_sets.append(train_loaders)
	# 	val_loader_sets.append(val_loaders)
	# 	test_loader_sets.append(test_loaders)
	max_node = int(max(batch_list)*train_node)
	for metapath_graph in graphs:  # one graph per metapath
		train_loaders = []
		val_loaders = []
		test_loaders = []
		
		for mult in batch_list:
			batch_size_train = math.ceil(mult * train_node)
			batch_size_val = math.ceil(mult * val_node)
			batch_size_test = math.ceil(mult * test_node)
			# if batch_size_train%2==1:
			# 	batch_size_train+=1
			# if batch_size_val%2==1:
			# 	batch_size_val+=1
			# if batch_size_test%2==1:
			# 	batch_size_test+=1
			# print(batch_size_train)
			# print(batch_size_val)
			# print(batch_size_test)
			# print(mult)
			# print(train_node)
			# print(val_node)
			# print(test_node)
			train_loader = NeighborLoader(
				metapath_graph, num_neighbors=neighbor_list, batch_size=batch_size_train,
				input_nodes=(type, train_mask), shuffle=False, drop_last=False
			)
			# for i, batch in enumerate(train_loader):
			# 	print(batch[type].batch_size)       # batch size
			# 	print(batch[type].n_id)
			val_loader = NeighborLoader(
				metapath_graph, num_neighbors=neighbor_list, batch_size=batch_size_train,
				input_nodes=(type, val_mask), shuffle=False, drop_last=False
			)
			test_loader = NeighborLoader(
				metapath_graph, num_neighbors=neighbor_list, batch_size=batch_size_train,
				input_nodes=(type, test_mask), shuffle=False, drop_last=False
			)

			train_loaders.append(train_loader)
			val_loaders.append(val_loader)
			test_loaders.append(test_loader)
		
		train_loader_sets.append(train_loaders)
		val_loader_sets.append(val_loaders)
		test_loader_sets.append(test_loaders)
	model = Ogb_batch(mode=args.mode,target=type,metadata=metadata,num_node=max_node,batch_list=batch_list,num_graph=num_graph, hidden=args.hidden, out_dim=num_class, attention_dim=args.attention_dim, 
		num_gnn=args.num_model, dropout=args.dropout, layer_s=args.layer_size).to(args.device)


	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	loss_fcn = torch.nn.NLLLoss()
	best_val = 0
	patience = 0
	num_batch_sizes = len(batch_list)
	num_metapaths = len(graphs)
	# torch.save(model.state_dict(), f'model_state_dict_minmax_{args.dataset}_{args.seed}_{count}.pth')
	model_state = deepcopy(model.state_dict())
	print("training batch...")
	start_time = time.time()
	if args.inference:
		for epoch in range(args.epochs):
			model.train()
			total_loss = 0

			# Create loader iterators
			loader_iters_sets = [[iter(loader) for loader in loader_set] for loader_set in train_loader_sets]

			# Determine how many steps to run
			num_batches = min(min(len(loader) for loader in loader_set)for loader_set in train_loader_sets)

			for batch_idx in range(num_batches):
				optimizer.zero_grad()
				small_prediction = []
				large_prediction = []
				small_labels = []
				large_labels = []

				all_preds = []
				all_labels = []
				meta_preds = [[] for imeta in range(len(graphs))]
				meta_labels = [[] for ilabels in range(len(graphs))]
				for metapath_idx, loader_iters in enumerate(loader_iters_sets):
					batch_preds = [[] for imeta in range(len(batch_list))]
					batch_labels = [[] for ilabels in range(len(batch_list))]
					for batch_size_idx, loader_iter in enumerate(loader_iters):
						for _ in range(iter_list[batch_size_idx]):
							try:
								batch = next(loader_iter)
							except StopIteration:
								#print('stop')
								loader_iters[batch_size_idx] = iter(train_loader_sets[metapath_idx][batch_size_idx])
								batch = next(loader_iters[batch_size_idx])

							model_idx = metapath_idx * len(iter_list) + batch_size_idx
							#print('model: ',model_idx)
							preds, labels = model.forward(model_idx, batch)
							batch_preds[batch_size_idx].append(preds)
							batch_labels[batch_size_idx].append(labels)
					meta_preds[metapath_idx]=(batch_preds)
					meta_labels[metapath_idx]=(batch_labels)
				# print("b1",meta_preds[0][0][0].shape)
				# print("b1",meta_preds[0][0][1].shape)
				# print("b2",meta_preds[0][1][0].shape)					
				final_pred_list = []
				final_label_list = []
				count_final = 0
				for i in range(len(graphs)):
					for j in range(len(batch_list)):
						# print("inside loop start")
						# print(len(meta_preds[i][j]))
						# print(len(meta_preds[i][j][0]))
						# print(meta_preds[i][j][0].shape)
						if len(meta_preds[i][j][0])>1:
							final_pred_list.append(torch.cat(meta_preds[i][j],dim=0))
							final_label_list.append(torch.cat(meta_labels[i][j],dim=0))
						else:
							final_pred_list.append(meta_preds[i][j])
							final_label_list.append(meta_labels[i][j])						
						# print("inside loop",final_pred_list[count_final].shape)
						count_final+=1
				# print("meta1_b1",final_pred_list[0].shape)
				# print("meta1_b2",final_pred_list[1].shape)
				# print("meta2_b1",final_pred_list[2].shape)
				# print("meta2_b2",final_pred_list[3].shape)
				#final_preds = model.attention(final_pred_list)
				stacked_cov = torch.stack([p.mean(dim=0) for p in final_pred_list])
				stacked_cov_mean = torch.matmul(stacked_cov, stacked_cov.T)
				stacked_cov = torch.norm(stacked_cov_mean, p=1) ** 2
				if args.mode == 'att':
					final_preds= model.attention(final_pred_list)
				else:
					final_preds = final_pred_list[0]
					for i in range(1,len(final_pred_list)):
						min_len = min(final_preds.shape[0], final_pred_list[i].shape[0])
						final_preds = final_preds[:min_len] + final_pred_list[i][:min_len]
					final_preds /= len(final_pred_list)

				#final_preds = model.attention(torch.stack(final_pred_list, dim=0))

				final_labels = final_label_list[0][:final_preds.shape[0]]

				# print("final_pred",final_preds.shape)
				# print("final_label",final_labels.shape)
				# Concatenate all predictions and labels
				# final_preds = torch.cat(all_preds, dim=0)   # [total_nodes, num_classes]
				# final_labels = torch.cat(all_labels, dim=0) # [total_nodes]


				loss = loss_fcn(final_preds, final_labels)+ args.lambda_cov * stacked_cov
				loss.backward()
				total_loss += loss.item()

				optimizer.step()


			val_acc,_ = evaluate_batch(model, graphs,val_loader_sets, iter_list, args.device)
			test_acc,_ = evaluate_batch(model,graphs,test_loader_sets, iter_list, args.device)

			#print(f"Epoch {epoch:03d} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | Patience: {patience}")
			if val_acc > best_val:
				best_val = val_acc
				patience = 0
				best_model_state = deepcopy(model.state_dict())
			else:
				patience += 1
				if patience >= args.patience:
					break
	#print('loading...')
	# Load best model
	end_time = time.time()

	# Compute duration
	epoch_time = end_time - start_time
	print(f"Time for one epoch: {epoch_time:.2f} seconds")
	model.load_state_dict(best_model_state)
	#model.load_state_dict(torch.load(f'best_model_state_dict_RGCN_batch_{args.dataset}_{args.seed}.pth'))
	acc,auc = evaluate_batch(model, graphs, test_loader_sets, iter_list, args.device)
	print(f"Test Acc = {acc:.4f}")
	#print(f"Test Auc = {auc:.4f}")
	return (acc,model.state_dict())



def run_RGCN(args):
	if args.dataset == 'Ogb':
		data, ogb_graph, train_mask, val_mask, test_mask = load_OgbMag(seed=0,feature='uniform')
		ogb_graph.append(data)
		selected_graph = ogb_graph[args.choice].to(args.device)
		num_class = selected_graph['paper'].y.unique().shape[0]
		type = 'paper'
	elif args.dataset == 'DBLP':
		_,data, dblp_graph, train_mask, val_mask, test_mask = load_DBLP(seed=0)
		dblp_graph.append(data)
		selected_graph = dblp_graph[args.choice].to(args.device)
		num_class = dblp_graph[0]['author'].y.unique().shape[0] 
		type = 'author'
	elif args.dataset == 'IMDB':
		_,data, dblp_graph, train_mask, val_mask, test_mask = load_IMDB(seed=0)
		dblp_graph.append(data)
		selected_graph = dblp_graph[args.choice].to(args.device)
		num_class = dblp_graph[0]['movie'].y.unique().shape[0] 
		type = 'movie'
	elif args.dataset == 'ACM':
		_,data, dblp_graph, train_mask, val_mask, test_mask = load_ACM(seed=0)
		dblp_graph.append(data)
		selected_graph = dblp_graph[args.choice].to(args.device)
		num_class = dblp_graph[0]['paper'].y.unique().shape[0] 
		type = 'paper'
	elif args.dataset == 'Freebase':
		_,data, dblp_graph, train_mask, val_mask, test_mask = load_Freebase(seed=0)
		dblp_graph.append(data)
		selected_graph = dblp_graph[args.choice].to(args.device)
		num_class = dblp_graph[0]['book'].y.unique().shape[0]
		print(num_class)
		type = 'book'
		
	model = HeteroSAGE(hidden=args.hidden, out_dim=num_class, dropout=args.dropout, layer_s=args.layer_size).to(args.device)
	model = to_hetero(model, selected_graph.metadata(), aggr='sum')

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	#loss_fcn = torch.nn.CrossEntropyLoss()
	loss_fcn = torch.nn.NLLLoss()
	best_val = 0
	patience = 0
	if args.inference:
		for epoch in range(args.epochs):
			model.train()
			optimizer.zero_grad()
			
			
			predict = model(selected_graph.x_dict, selected_graph.edge_index_dict)
			#print(predict)
			paper_pred = predict[type]
			loss = loss_fcn(paper_pred[train_mask], selected_graph[type].y[train_mask])
			loss.backward()
			optimizer.step()
			with torch.no_grad():
				val_output = paper_pred[val_mask]
				val_target = selected_graph[type].y[val_mask]
				val = (val_output.argmax(dim=1) == val_target).float().mean()

			#val = (torch.max(predict[val_mask],dim=1)[1] == selected_graph.y[val_mask]).float().mean()
			if val>best_val:
				torch.save(model.state_dict(), f'RGCN_m{args.choice}_state_dict.pth')
				best_val = val
				patience =0
			else:
				patience+=1
			if patience>=args.patience:
				break
			print("patience:",patience)
			print("best_val:",best_val)
	model.load_state_dict(torch.load(f'RGCN_m{args.choice}_state_dict.pth'))
	@torch.no_grad()
	def test(model):
		model.eval()

		predict = model(selected_graph.x_dict, selected_graph.edge_index_dict)
		paper_pred = predict[type]
		#loss = loss_fcn(predict['paper'][data['paper'].train_mask], data['paper'].y[data['paper'].train_mask])
		test_output = paper_pred[test_mask]
		test_target = selected_graph[type].y[test_mask]
		acc = (test_output.argmax(dim=1) == test_target).float().mean()

		#predict = model(x=selected_graph.x, g=selected_graph.edge_index)[:target_node,:]

		#loss = loss_fcn(predict[test_mask],selected_graph.y[test_mask])
		#print("test_loss:",loss.item())
		#acc = accuracy_score(selected_graph.y[test_mask].cpu().numpy(),torch.max(predict[test_mask],dim=1)[1].cpu().numpy())
		#test_report= classification_report(selected_graph.y[test_mask].cpu().numpy(),torch.max(predict[test_mask],dim=1)[1].cpu().numpy())
		print("RGCN acc:",acc)
		#print("test report",test_report)
		return acc

	acc = test(model)

	return model
def MRGCN(args):
	if args.dataset == 'Ogb':
		data, ogb_graph, train_mask, val_mask, test_mask = load_OgbMag(seed=0,feature='uniform')
		ogb_graph.append(data)
		selected_graph = ogb_graph[args.choice].to(args.device)
		num_class = selected_graph['paper'].y.unique().shape[0]
		graphs = [graph.to(args.device) for graph in ogb_graph]
		type = 'paper'
	elif args.dataset == 'DBLP':
		_,data, dblp_graph,train_mask, val_mask, test_mask = load_DBLP(seed=0)
		dblp_graph.append(data)
		selected_graph = dblp_graph[args.choice].to(args.device)
		num_class = dblp_graph[0]['author'].y.unique().shape[0]
		graphs = [graph.to(args.device) for graph in dblp_graph]
		type = 'author'
	elif args.dataset == 'IMDB':
		data, imdb_graph, train_mask, val_mask, test_mask = load_IMDB(seed=args.seed)
		graphs = [graph.to(args.device) for graph in imdb_graph]
		data.to(args.device)
		graphs.append(data)
		type = 'movie'
		num_class = imdb_graph[0][type].y.unique().shape[0] 
	elif args.dataset == 'ACM':
		data, acm_graph, train_mask, val_mask, test_mask = load_ACM(seed=args.seed)
		graphs = [graph.to(args.device) for graph in acm_graph]
		data.to(args.device)
		graphs.append(data)
		type = 'paper'
		num_class = acm_graph[0][type].y.unique().shape[0]
	elif args.dataset == 'Freebase':
		data, free_graph, train_mask, val_mask, test_mask = load_Freebase(seed=args.seed)
		graphs = [graph.to(args.device) for graph in free_graph]
		data.to(args.device)
		graphs.append(data)
		type = 'book'
		num_class = free_graph[0][type].y.unique().shape[0] 
		#print(num_class)

	model_list = []
	predict_list = []

	for idx, selected_graph in enumerate(graphs):
		model = HeteroSAGE(hidden=args.hidden, out_dim=num_class, dropout=args.dropout, layer_s=args.layer_size).to(args.device)
		model = to_hetero(model, selected_graph.metadata(), aggr='sum')

		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		loss_fcn = torch.nn.NLLLoss()

		best_val = 0
		patience = 0
		best_model_path = f'MRGCN_{args.dataset}_metapath{idx}.pth'

		for epoch in range(args.epochs):
			model.train()
			optimizer.zero_grad()

			predict = model(selected_graph.x_dict, selected_graph.edge_index_dict)
			paper_pred = predict[type]
			loss = loss_fcn(paper_pred[train_mask], selected_graph[type].y[train_mask])
			loss.backward()
			optimizer.step()

			with torch.no_grad():
				val_output = paper_pred[val_mask]
				val_target = selected_graph[type].y[val_mask]
				val = (val_output.argmax(dim=1) == val_target).float().mean()

			if val > best_val:
				best_val = val
				patience = 0
				torch.save(model.state_dict(), best_model_path)
			else:
				patience += 1

			if patience >= args.patience:
				break

			#print(f"[Graph {idx}] Epoch {epoch} | Val Acc: {val.item():.4f} | Best: {best_val:.4f} | Patience: {patience}")

		# Load best model and store predictions
		model.load_state_dict(torch.load(best_model_path))
		model.eval()
		with torch.no_grad():
			logits = model(selected_graph.x_dict, selected_graph.edge_index_dict)[type]
			predict_list.append(logits)
	logits_avg = torch.stack(predict_list).mean(dim=0)
	pred_final = logits_avg[test_mask].argmax(dim=1)
	y_true = graphs[0][type].y[test_mask]
	acc = accuracy_score(y_true.cpu(), pred_final.cpu())
	print("Final Mean Accuracy:", acc)

	return acc

def run_RGCN_batch(args):
	if args.dataset == 'Ogb':
		data, ogb_graph, train_mask, val_mask, test_mask = load_OgbMag(seed=0,feature='')
		ogb_graph.append(data)
		selected_graph = ogb_graph[args.choice].to(args.device)
		num_class = selected_graph['paper'].y.unique().shape[0]
		type = 'paper'
	elif args.dataset == 'DBLP':
		data, dblp_graph,_,_,_, train_mask, val_mask, test_mask = load_DBLP(seed=0)
		dblp_graph.append(data)
		selected_graph = dblp_graph[args.choice].to(args.device)
		num_class = dblp_graph[0]['author'].y.unique().shape[0] 
		type = 'author'
	elif args.dataset == 'IMDB':
		data, ogb_graph, train_mask, val_mask, test_mask = load_IMDB(seed=0)
		ogb_graph.append(data)
		selected_graph = ogb_graph[args.choice].to(args.device)

		type = 'movie'
		num_class = selected_graph[type].y.unique().shape[0]
	elif args.dataset == 'ACM':
		data, ogb_graph, train_mask, val_mask, test_mask = load_ACM(seed=0)
		ogb_graph.append(data)
		selected_graph = ogb_graph[args.choice].to(args.device)
		type = 'paper'
		num_class = selected_graph[type].y.unique().shape[0]
	elif args.dataset == 'Freebase':
		data, ogb_graph, train_mask, val_mask, test_mask = load_Freebase(seed=0)
		ogb_graph.append(data)
		selected_graph = ogb_graph[args.choice].to(args.device)
		type = 'book'
		num_class = selected_graph[type].y.unique().shape[0]
	mult=args.batch_size
	train_node = int(train_mask.sum())
	val_node = int(val_mask.sum())
	test_node = int(test_mask.sum())
	batch_size_train = math.ceil(mult * train_node)
	batch_size_val = math.ceil(mult * val_node)
	batch_size_test = math.ceil(mult * test_node)
	num_neighbors=[args.num_neighbor] * args.num_hop


	train_loader = NeighborLoader(selected_graph,num_neighbors=num_neighbors,batch_size=batch_size_train,
	input_nodes=(type, train_mask),shuffle=False)
	# for i, batch in enumerate(train_loader):
	# 	print(batch[type].batch_size)       # batch size
	# 	print(batch[type].n_id)
	val_loader = NeighborLoader(selected_graph,num_neighbors=num_neighbors,batch_size=batch_size_val,
	input_nodes=(type, val_mask),shuffle=False)
	test_loader = NeighborLoader(selected_graph,num_neighbors=num_neighbors,batch_size=batch_size_test,
	input_nodes=(type, test_mask),shuffle=False)

	
	model = HeteroSAGE(hidden=args.hidden, out_dim=num_class, dropout=args.dropout, layer_s=args.layer_size).to(args.device)
	model = to_hetero(model, selected_graph.metadata(), aggr='sum')

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	#loss_fcn = torch.nn.CrossEntropyLoss()
	loss_fcn = torch.nn.NLLLoss()
	best_val = 0
	patience = 0
	count=0
	if args.inference:
		for epoch in range(args.epochs):
			model.train()
			optimizer.zero_grad()

			for batch in train_loader:
				optimizer.zero_grad()
				#batch = batch.to('cuda:0')
				batch_size = batch[type].batch_size
				out = model(batch.x_dict, batch.edge_index_dict)
				loss = loss_fcn(out[type][:batch_size],batch[type].y[:batch_size])
				loss.backward()
				optimizer.step()
			model.eval()
			correct = 0
			total = 0
			acc_list = []
			with torch.no_grad():
				for batch in val_loader:
					batch_size = batch[type].batch_size
					out = model(batch.x_dict, batch.edge_index_dict)[type]
					pred = out[:batch_size].argmax(dim=1)
					target = batch[type].y[:batch_size]
					correct += (pred == target).sum().item()
					total += batch_size


			val = correct / total
			#val = (torch.max(predict[val_mask],dim=1)[1] == selected_graph.y[val_mask]).float().mean()
			if val>best_val:
				torch.save(model.state_dict(), f'RGCN_batch_m{args.choice}_state_dict.pth')
				best_val = val
				patience =0
			else:
				patience+=1
			if patience>=args.patience:
				break
			print("patience:",patience)
			print("best_val:",best_val)
	model.load_state_dict(torch.load(f'RGCN_batch_m{args.choice}_state_dict.pth'))
	@torch.no_grad()
	def test(model):
		model.eval()

		correct = 0
		total = 0
		for batch in test_loader:
			batch_size = batch[type].batch_size
			out = model(batch.x_dict, batch.edge_index_dict)[type]
			pred = out[:batch_size].argmax(dim=1)
			target = batch[type].y[:batch_size]
			correct += (pred == target).sum().item()
			total += batch_size
		acc = correct / total
		#predict = model(x=selected_graph.x, g=selected_graph.edge_index)[:target_node,:]

		#loss = loss_fcn(predict[test_mask],selected_graph.y[test_mask])
		#print("test_loss:",loss.item())
		#acc = accuracy_score(selected_graph.y[test_mask].cpu().numpy(),torch.max(predict[test_mask],dim=1)[1].cpu().numpy())
		#test_report= classification_report(selected_graph.y[test_mask].cpu().numpy(),torch.max(predict[test_mask],dim=1)[1].cpu().numpy())
		print("acc:",acc)
		#print("test report",test_report)
		return acc

		print('Done')
	acc = test(model)

	return model

def run_cluster(args):
	if args.dataset == 'DBLP':
		homo_data,dataset2, dblp_graph,train_mask, val_mask, test_mask = load_DBLP(seed=args.seed)
		#homo_data=homo_data.to(args.device)
		target_ntype = 'author'
	elif args.dataset == 'Ogb':
		homo_data,dataset2, dblp_graph,train_mask, val_mask, test_mask = load_OgbMag(seed=args.seed)
		#homo_data=homo_data.to(args.device)
		target_ntype = 'paper'
		num_class = dblp_graph[0]['paper'].y.unique().shape[0]
	elif args.dataset == 'IMDB':
		homo_data,dataset2, dblp_graph,train_mask, val_mask, test_mask = load_IMDB(seed=args.seed)
		#homo_data=homo_data.to(args.device)
		target_ntype = 'movie'
		num_class = dblp_graph[0]['movie'].y.unique().shape[0]
	elif args.dataset == 'ACM':
		homo_data,dataset2, dblp_graph,train_mask, val_mask, test_mask = load_ACM(seed=args.seed)
		#homo_data=homo_data.to(args.device)
		target_ntype = 'paper'
		num_class = dblp_graph[0]['paper'].y.unique().shape[0]
	elif args.dataset == 'Freebase':
		homo_data,dataset2, dblp_graph,train_mask, val_mask, test_mask = load_Freebase(seed=args.seed)
		#homo_data=homo_data.to(args.device)
		target_ntype = 'book'
		num_class = dblp_graph[0]['book'].y.unique().shape[0]
	model = RGCN(homo_data.x.size(1), 128, num_class, 
			 int(homo_data.edge_attr.max().item()+1)).to(args.device)
	loss_fcn = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	cluster_data = ClusterData(homo_data, num_parts=200, recursive=False)
	train_loader = ClusterLoader(cluster_data, batch_size=10, shuffle=True)
	best_val = 0
	patience = 0
	for epoch in range(args.epochs):
		model.train()
		total_loss = 0
		for batch in train_loader:
			#print(f"{batch.x.size(0)} nodes, {batch.edge_index.size(1)} edges")
			batch = batch.to(args.device)
			optimizer.zero_grad()
			out = model(batch.x, batch.edge_index, batch.edge_attr)
			loss = loss_fcn(out[batch.train_mask], batch.y[batch.train_mask])
			loss.backward()
			optimizer.step()
			total_loss += loss.item()

		# Evaluation on the full graph
		model.eval()
		with torch.no_grad():
			out = model(homo_data.x.to(args.device), homo_data.edge_index.to(args.device), homo_data.edge_attr.to(args.device))
			prob = F.softmax(out, dim=1)
			pred = prob.argmax(dim=1)

			acc_val = (pred[homo_data.val_mask] == homo_data.y[homo_data.val_mask].to(args.device)).float().mean().item()
			acc_test = (pred[homo_data.test_mask] == homo_data.y[homo_data.test_mask].to(args.device)).float().mean().item()

		#print(f"Epoch {epoch:03d} | Val Acc: {acc_val:.4f} | Test Acc: {acc_test:.4f} | Patience: {patience}")


		
		#val = (torch.max(predict[val_mask],dim=1)[1] == selected_graph.y[val_mask]).float().mean()
		if acc_val>best_val:
			torch.save(model.state_dict(), f'cluster_batch_m{args.choice}_state_dict.pth')
			best_val = acc_val
			patience =0
		else:
			patience+=1
		if patience>=args.patience:
			break

	model.load_state_dict(torch.load(f'cluster_batch_m{args.choice}_state_dict.pth'))
	@torch.no_grad()
	def test(model):
		model.eval()

		out = model(homo_data.x.to(args.device), homo_data.edge_index.to(args.device), homo_data.edge_attr.to(args.device))
		prob = F.softmax(out, dim=1)
		pred = prob.argmax(dim=1)
		test_mask = homo_data.test_mask
		true = homo_data.y[test_mask].to(args.device)
		pred_label = pred[test_mask]

		# Accuracy
		acc = (pred_label == true).float().mean().item()

		# AUC
		
		y_score = prob[test_mask].detach().cpu().numpy()
		y_true = true.detach().cpu().numpy()
		y_true_bin = label_binarize(y_true, classes=list(range(out.shape[1])))


		#auc = roc_auc_score(y_true_bin, y_score, multi_class='ovr')  # multi-class
		print('cluster acc',acc)
		#print('test auc',auc)
		#print("test report",test_report)
		return acc


	acc = test(model)

	return model
def main(args):
	#run_RGCN(args)
	#run_RGCN_batch(args)
	cov=[0.001]
	neighbor = [10,15,20]
	layer_size = [2,3]
	dropout = [0,0.1,0.2,0.5]
	hidden = [64,128,256]
	seed = [0,10,20,30,42]
	choice = [0,1,2,3]
	dataset = ['Freebase']
	# for d in dataset:
	# 	args.dataset=d
	# for c in choice:
	# 	args.choice=c
	# 	run_Ogb(args)
	# for i in seed:
	# 	args.seed=i
	# 	run_Ogb(args)
	run_Ogb(args)
	#run_Ogb_batch(args)
	#run_cluster(args)
	# for c in choice:
	# 	args.choice=c
	# 	for s in seed:
	# 		args.seed = s
	# 		run_Ogb(args)
	# 	run_cluster(args)
	#run_cluster(args)
	result = []
	config = []
	count = 0
	best_acc = 0
	best_result = None
	best_config = None
	best_model = None
	# for i in neighbor:
	# 	args.num_neighbor = i
	# 	for j in layer_size:
	# 		args.layer_size = j
	# 		for k in dropout:
	# 			args.dropout = k
	# 			for z in cov:
	# 				args.lambda_cov = z
	# 				for l in hidden:
	# 					args.hidden = l
	# 					# for s in seed:
	# 					# 	args.seed = s

	# 					(acc,model_state) = run_Ogb(args)
	# 					#torch.save(model_state, f'best_model_state_dict_RGCN_batch_{args.dataset}_{args.seed}.pth')
	# 					#(auc,acc) = run_MGCN(args)
	# 					#(auc,acc,model_state)=run_SeHGNN(args,count)
	# 					if best_acc<acc:
	# 						best_acc = acc
	# 						best_result = acc
	# 						best_config = (i,j,k,z,l)
	# 						best_model = model_state
	# 						torch.save(best_model, f'best_model_state_dict_RGCN_batch2hop_{args.dataset}_{args.seed}.pth')
	# 					result.append(acc)
	# 					config.append((i,j,k,z,l))
	# 					#delete_file('model_state_dict_minmax_'+str(args.dataset)+'_'+str(args.seed)+'_'+str(count)+'.pth')
	# 					#print(f"result{result[count]}")
	# 					print(f"config{config[count]}")
	# 					print(f"best_result:{best_result:.4f}")
	# 					print(f"best_config{best_config}")
	# 					count+=1


if __name__ == "__main__":
	parser = argparse.ArgumentParser("HAN")
	parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='gpu or cpu')
	parser.add_argument('--epochs', type=int, default=3000)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--weight_decay', type=float, default=0.0005)
	parser.add_argument('--dropout', type=float, default=0.1)
	parser.add_argument('--dropout1', type=float, default=0)
	parser.add_argument('--dropout2', type=float, default=0)
	parser.add_argument('--patience',type=int,default=50)
	parser.add_argument('--num_model',type=int,default=3)
	parser.add_argument('--seed',type=int,default=0)
	parser.add_argument('--hidden',type=int,default=64)
	parser.add_argument('--hidden2',type=int,default=32)
	parser.add_argument('--hidden3',type=int,default=64)
	parser.add_argument('--layer_size',type=int,default=2)
	parser.add_argument('--layer_size2',type=int,default=2)
	parser.add_argument('--layer_size3',type=int,default=2)
	parser.add_argument('--choice',type=int,default=0)
	parser.add_argument('--num_hop', type=int, default=2, help="Number of hops")
	parser.add_argument('--num_neighbor', type=int, default=10, help="Number of neighbors per hop")
	parser.add_argument('--batch_size',type=float,default=0.1)
	parser.add_argument('--att',type=int,default=10)
	parser.add_argument('--lambda_cov', type=float, default=0)
	parser.add_argument('--dataset',type=str,default='DBLP')
	parser.add_argument('--mode',type=str,default='att')
	parser.add_argument('--fused',type=bool,default=False)
	parser.add_argument('--attention_dim',type=int,default=8)
	parser.add_argument('--inference', default=True, action='store_false', help='Bool type')

	args = parser.parse_args()
	main(args)