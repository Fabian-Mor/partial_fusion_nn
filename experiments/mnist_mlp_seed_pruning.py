import sys
from matplotlib import pyplot as plt

sys.path.append("../")
from src.FusionModel.generalized_pruning.pruning_mlp import *
from src import data_loader
from src.MLP import Deep_MLP

np.random.seed(0)
torch.random.manual_seed(0)

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
mults = [1, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1]
act = 0
USE_IMP = False
which_algo = 'stoch_very_high'
features = ''  # if empty then same as 'act', otherwise 'both' or 'weight_based'

save_path = 'saved_compression/'+str(seeds)+'_'+str(mults)+'_'+str(act)+'_'+str(USE_IMP)+'_'+str(which_algo)+features

print(save_path)

list_prune = [[] for _ in seeds]
list_post_proc = [[] for _ in seeds]
list_free_act = [[] for _ in seeds]

for i_seed, seed in enumerate(seeds):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    train_loader, test_loader = data_loader.load_mnist()

    test_data = []
    for batch in train_loader:
        inputs = batch[0]
        test_data.append(inputs)
    test_data = torch.cat(test_data, dim=0)[:1000]


    large_size = [100, 100, 100]

    model_a = Deep_MLP(hidden_size_1=large_size[0], hidden_size_2=large_size[1], hidden_size_3=large_size[2], which_act=act)
    model_a.load_model('saved_compression/' + str(seed) + 'deepmlpmnist_' + str(large_size) +'_'+str(act) + '.checkpoint')
    for mult in mults:

        small_size = [int(np.round(mult*l)) for l in large_size]
        model_raw_prune = structured_pruning_ff(model_a, small_size, use_test=True, test_data=test_data, which_act=act)
        model_raw_prune_postproc = structured_pruning_ff(model_a, small_size, post_proc=True, use_test=True, test_data=test_data, which_act=act)
        if features == '' or features == 'act':
            model_act = prune_net_clean(model_a, small_size, test_data, which_act=act, use_reg=False, clustering_algo=which_algo)
        elif features == 'weight_based':
            model_act = prune_net_clean(model_a, small_size, test_data, which_act=act,
                                        use_reg=False, clustering_algo=which_algo)
        elif features == 'both':
            model_act = prune_net_clean(model_a, small_size, test_data, which_act=act, use_reg=False, clustering_algo=which_algo, both=True)

        print('----------- Seed:' + str(seed) + ', Size: ' + str(small_size) + '---------------')
        val_prune = model_raw_prune.test_model(test_loader)
        val_post = model_raw_prune_postproc.test_model(test_loader)
        val_act = model_act.test_model(test_loader)

        list_prune[i_seed].append(val_prune)
        list_post_proc[i_seed].append(val_post)
        list_free_act[i_seed].append(val_act)


np.save(save_path+'_prune', np.array(list_prune))
np.save(save_path+'_post', np.array(list_post_proc))
np.save(save_path+'_free', np.array(list_free_act))

prunenp = np.mean(np.array(list_prune), axis=0)
postnp = np.mean(np.array(list_post_proc), axis=0)
actnp = np.mean(np.array(list_free_act), axis=0)
print(prunenp)
print(postnp)
print(actnp)
print(np.std(np.array(list_prune), axis=0))
print(np.std(np.array(list_post_proc), axis=0))
print(np.std(np.array(list_free_act), axis=0))