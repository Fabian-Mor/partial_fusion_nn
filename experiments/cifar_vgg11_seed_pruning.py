import torch
from matplotlib import pyplot as plt
import sys
sys.path.append("../")
from src import data_loader
from src.FusionModel.generalized_pruning.pruning_cnn import *

np.random.seed(0)
torch.random.manual_seed(0)

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
mults = [1, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1]
USE_IMP = False
which_algo = 'stoch_very_high'

features = ''  # if empty then same as 'act', otherwise 'both' or 'weight_based'

save_path = 'saved_compression/cnn'+str(seeds)+'_'+str(mults)+'_'+str(USE_IMP)+'_'+str(which_algo)+features

model_size = [64, 128, 256, 256, 512, 512, 512, 512]

print(save_path)

list_prune = [[] for _ in seeds]
list_fusion_paper = [[] for _ in seeds]
list_post_proc = [[] for _ in seeds]
list_cluster = [[] for _ in seeds]

for i_seed, seed in enumerate(seeds):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    train_loader, test_loader = data_loader.load_cifar10(batch_size=128)
    test_data = []
    for batch in test_loader:
        inputs = batch[0]
        test_data.append(inputs)
    test_data = torch.cat(test_data, dim=0)[:1000]

    model_h = VGG11(manual_chanel_sizes=model_size)
    model_h.load_state_dict(torch.load('saved_compression/' + str(seed) + 'VGG11_' + str(model_size) + '_best.checkpoint'))
    print(model_h)

    model_h.test_model(test_loader)
    for mult in mults:
        small_size = [int(np.round(mult*l)) for l in model_size]
        model_raw_prune = prune_simple(model_h, small_size, importance_type='incoming')
        model_ot_paper = prune_ot_paper(model_h, small_size, test_data)
        model_cluster = prune_clustering(model_h, small_size, test_data, clustering_algo=which_algo)

        val_raw = model_raw_prune.test_model(test_loader, verbose=0)
        val_paper = model_ot_paper.test_model(test_loader, verbose=0)
        val_cluster = model_cluster.test_model(test_loader, verbose=0)

        list_prune[i_seed].append(val_raw)
        list_fusion_paper[i_seed].append(val_paper)
        list_cluster[i_seed].append(val_cluster)


np.save(save_path+'_prune', np.array(list_prune))
np.save(save_path+'_paper', np.array(list_fusion_paper))
np.save(save_path+'_clsuter', np.array(list_cluster))

prunenp = np.mean(np.array(list_prune), axis=0)
papernp = np.mean(np.array(list_fusion_paper), axis=0)
clusternp = np.mean(np.array(list_cluster), axis=0)
print(prunenp)
print(papernp)
print(clusternp)
print(np.std(np.array(list_prune), axis=0))
print(np.std(np.array(list_fusion_paper), axis=0))
print(np.std(np.array(list_cluster), axis=0))