import numpy as np
import torch
from torch_geometric.loader import NeighborLoader
from dataset import load_OgbMag, load_DBLP, load_IMDB, load_ACM, load_Freebase
import argparse
from model_ogb import*
from torch_geometric.utils import to_dense_adj,dense_to_sparse,degree
from torch_geometric.nn import to_hetero
from torch_scatter import scatter
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import random
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
				final_pred_list = []
				final_label_list = []
				count_final = 0
				for i in range(len(graphs)):
					for j in range(len(batch_list)):
						if len(meta_preds[i][j][0])>1:
							final_pred_list.append(torch.cat(meta_preds[i][j],dim=0))
							final_label_list.append(torch.cat(meta_labels[i][j],dim=0))
						else:
							final_pred_list.append(meta_preds[i][j])
							final_label_list.append(meta_labels[i][j])						
						
						count_final+=1

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


def main(args):
	run_Ogb(args)

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
