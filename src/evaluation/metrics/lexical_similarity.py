import matplotlib.pyplot as plt
from src.evaluation.metrics.js_div import calculate_js_div
from src.evaluation.metrics.dice import calculate_dice_sim
from src.evaluation.metrics.jaccard import calculate_jaccard_index
from src.loaders.load_data import load_data
import numpy as np

def plt_hist(axis, data, hatch, label, bins, col):
    counts, edges = np.histogram(data, bins=bins,range=[0, 1])
    edges = np.repeat(edges, 2)
    hist = np.hstack((0, np.repeat(counts, 2), 0))

    outline, = axis.plot(edges, hist, linewidth=1.3, color=col)
    axis.fill_between(edges, hist, 0,
                      edgecolor=col, hatch=hatch, label=label,
                      facecolor='none')  ## < removes facecolor
    axis.set_ylim(0, None, auto=True)

class LexicalSimilarity:
    def __init__(self):
        self.metrics = ('jaccard', 'dice', 'js-div')
        self.datasets =   ('MSRP', 'Quora','Semeval','STS','PAWS')
        self.jaccard = {}
        self.dice = {}
        self.js_div = {}
        self.labels = {}
        self.pair_ids = {}
        self.sentence1 = {}
        self.sentence2 = {}
        self.difficulty = {}

    def get_accepted_metrics(self):
        return self.metrics
    def get_accepted_datasets(self):
        return self.datasets

    def get_subsets(self, dataset):
        # print(dataset)
        assert dataset in self.get_accepted_datasets()
        if dataset == 'Semeval':
            return  ['train_large', 'test2016', 'test2017']
        elif dataset == 'PAWS':
            return ['train','test']
        else:
            return  ['train', 'dev', 'test']

    def plot_metric_subset_dist(self, metric, dataset,task,plot_folder='',font_size = 12):
        subsets = self.get_subsets(dataset)
        similarities = self.get_metric(metric, dataset, task, subset=None)
        colors = ['g', 'b', 'r']
        if metric=='js-div':
            metric='JSD'
        fig, ax = plt.subplots(1)
        for n, sim_per_pair in enumerate(similarities):
            plt.hist(sim_per_pair, color=colors[n], alpha=0.5, bins=25, label=subsets[n], range=[0, 1])
        if dataset=='Semeval':
            plt.legend(loc='upper left')
        else:
            plt.legend(loc='upper right')
        plt.ylabel('Number of text pairs')
        plt.xlabel(metric)
        if dataset=='Semeval':
            title = dataset + ' ' + task
        else:
            title = dataset
        plt.title('{}'.format(title),fontsize=15)
        for item in ([ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(font_size)
        if plot_folder == '':
            plt.show()
        else:
            plot_path = plot_folder + metric + '_' + dataset + '_' + task + ".pdf"
            plt.tight_layout()
            plt.savefig(plot_path,format='pdf')
            plt.close()

    def plot_metric_label_dist(self, metric, dataset,task,plot_folder='',font_size = 12):
        labels = self.get_labels(dataset,task)
        similarities = self.get_metric(metric, dataset, task, subset=None)
        neg_sim = []
        pos_sim = []
        for lab_per_pair, sim_per_pair in zip(labels,similarities):
            assert len(sim_per_pair)==len(lab_per_pair)
            for i in range(len(sim_per_pair)):
                if lab_per_pair[i]==0:
                    neg_sim.append(sim_per_pair[i])
                elif lab_per_pair[i]==1:
                    pos_sim.append(sim_per_pair[i])
                else:
                    ValueError('{} not allowed as label.'.format(lab_per_pair[i]))
        if metric=='js-div':
            metric='JSD'
        fig, ax = plt.subplots(1)
        plt_hist(ax, neg_sim, '//', 'Negative', 25, 'r')
        plt_hist(ax, pos_sim, '\\\\', 'Positive', 25, 'g')
        ax.legend()
        # plt.hist(neg_sim, ls='solid', color = 'r', lw=3, alpha=0.5,  bins=25, label='Negative', range=[0, 1])
        # plt.hist(pos_sim, ls='dashed',color = 'g', hatch = '|',lw=3, alpha=0.5, bins=25, label='Positive', range=[0, 1])
        if dataset in ['Quora','Semeval']:
            plt.legend(loc='upper left',prop={'size': font_size})
        else:
            plt.legend(loc='upper right',prop={'size': font_size})
        plt.ylabel('Number of text pairs')
        plt.xlabel(metric)
        if dataset=='Semeval':
            title = dataset + ' ' + task
        else:
            title = dataset
        plt.title('{}'.format(title),fontsize=15)
        for item in ([ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(font_size)
        if plot_folder == '':
            plt.show()
        else:
            plot_path = plot_folder + 'col_label_' + metric + '_' + dataset + '_' + task + ".pdf"
            plt.tight_layout()
            plt.savefig(plot_path,format='pdf')
            plt.close()

    def get_labels(self,dataset,task='',subset=None):
        if not dataset in self.labels.keys():
            self.get_metric('jaccard',dataset,task)
        if dataset=='Semeval':
            dataset = '{}_{}'.format(dataset,task)
        if subset is None:
            return self.labels[dataset]
        elif subset in ['train','p_train','train_large']:
            return self.labels[dataset][0]
        elif  subset in ['dev','test2016']:
            return self.labels[dataset][1]
        elif subset in ['test','p_test','test2017']:
            return self.labels[dataset][-1]
        else:
            raise ValueError('{} not accepted value for subset'.format(subset))

    def get_ids(self,dataset,task='',subset=None):
        if dataset=='Semeval':
            dataset = '{}_{}'.format(dataset,task)
        if not dataset in self.labels.keys():
            self.get_metric('jaccard',dataset,task)
        if subset is None:
            return self.pair_ids[dataset]
        elif subset in ['train','p_train','train_large']:
            return self.pair_ids[dataset][0]
        elif  subset in ['dev','test2016']:
            return self.pair_ids[dataset][1]
        elif subset in ['test','p_test','test2017']:
            return self.pair_ids[dataset][-1]
        else:
            raise ValueError('{} not accepted value for subset'.format(subset))

    def get_s1(self,dataset,task='',subset=None):
        if dataset=='Semeval':
            dataset = '{}_{}'.format(dataset,task)
        if not dataset in self.labels.keys():
            self.get_metric('jaccard',dataset,task)
        if subset is None:
            return self.sentence1[dataset]
        elif subset in ['train','p_train','train_large']:
            return self.sentence1[dataset][0]
        elif  subset in ['dev','test2016']:
            return self.sentence1[dataset][1]
        elif subset in ['test','p_test','test2017']:
            return self.sentence1[dataset][-1]
        else:
            raise ValueError('{} not accepted value for subset'.format(subset))

    def get_s2(self,dataset,task='',subset=None):
        if dataset=='Semeval':
            dataset = '{}_{}'.format(dataset,task)
        if not dataset in self.labels.keys():
            self.get_metric('jaccard',dataset,task)
        if subset is None:
            return self.sentence2[dataset]
        elif subset in ['train','p_train','train_large']:
            return self.sentence2[dataset][0]
        elif  subset in ['dev','test2016']:
            return self.sentence2[dataset][1]
        elif subset in ['test','p_test','test2017']:
            return self.sentence2[dataset][-1]
        else:
            raise ValueError('{} not accepted value for subset'.format(subset))


    def get_difficulty(self, dataset, task, metric, split_by='median', subset=None):
        if dataset=='Semeval':
            dataset = '{}_{}'.format(dataset,task)
        if metric == 'jaccard':
            overlapping = self.jaccard[dataset]
        elif metric == 'dice':
            overlapping = self.dice[dataset]
        elif metric == 'js-div':
            overlapping = self.js_div[dataset]
        gold_labels = self.labels[dataset]
        assert metric in ['js-div', 'jaccard', 'dice']
        difficulties = []
        for i,s in enumerate(overlapping):
            overlap = overlapping[i]
            gold_label = gold_labels[i]
            assert len(overlap)==len(gold_label)
            if split_by == 'median':
                criterion = np.median(overlap)
            else:
                criterion = split_by
            difficulty = []
            for o,label in zip(overlap,gold_label):
                if metric == 'js-div':
                    if (o <= criterion) and (label == 0):
                        difficulty.append('Nn')
                    elif (o > criterion) & (label == 0):
                        difficulty.append('No')
                    elif (o > criterion) & (label == 1):
                        difficulty.append('Pn')
                    elif (o <= criterion) & (label == 1):
                        difficulty.append('Po')
                else:
                    if (o <= criterion) & (label == 0):
                        difficulty.append('No')
                    elif (o > criterion) & (label == 0):
                        difficulty.append('Nn')
                    elif (o > criterion) & (label == 1):
                        difficulty.append('Po')
                    elif (o <= criterion) & (label == 1):
                        difficulty.append('Pn')
            difficulties.append(difficulty)
        if subset is None:
            return difficulties
        elif subset in ['train','p_train','train_large']:
            return difficulties[0]
        elif subset in ['dev','test2016']:
            return difficulties[1]
        elif subset in ['test','p_test','test2017']:
            return difficulties[-1]
        else:
            raise ValueError('{} not accepted value for subset'.format(subset))

    def get_metric(self,distance_metric,dataset,task='',subset=None):
        '''
        Load calculated metric scores if existing, otherwise calculate
        :param distance_metric: 
        :param dataset: 
        :param task: 
        :return: nested list with distance/similarity scores depending on metric with outer length of subsets and 
        inner length of example numbers
        '''
        assert distance_metric in self.get_accepted_metrics()
        subsets = self.get_subsets(dataset) # always load all 3 subsets
        opt = {'dataset': dataset, 'datapath': 'data/', 'subsets':subsets,
               'tasks': [task], 'n_gram_embd': False, 'cache': True}
        if dataset=='Semeval':
            dataset = '{}_{}'.format(dataset,task)
        if distance_metric == 'jaccard':
            if dataset not in self.jaccard.keys():
                if dataset not in self.sentence1.keys():
                    data_dict = load_data(opt, numerical=False)
                    R1 = data_dict['R1']
                    R2 = data_dict['R2']
                    pair_ids = []
                    for s,_ in enumerate(subsets):
                        pair_ids.append([i1+'-'+i2 for i1, i2 in zip(data_dict['ID1'][s], data_dict['ID2'][s])])
                    L = data_dict['L']
                    self.sentence1[dataset] = R1
                    self.sentence2[dataset] = R2
                    self.pair_ids[dataset] = pair_ids
                    self.labels[dataset] = L
                else:
                    R1 = self.sentence1[dataset]
                    R2 = self.sentence2[dataset]
                self.jaccard[dataset] = calculate_jaccard_index(R1, R2)
            overlapping = self.jaccard[dataset]
        elif distance_metric == 'dice':
            if dataset not in self.dice.keys():
                if dataset not in self.sentence1.keys():
                    data_dict = load_data(opt, numerical=False)
                    R1 = data_dict['R1']
                    R2 = data_dict['R2']
                    pair_ids = []
                    for s,_ in enumerate(subsets):
                        pair_ids.append([i1+'-'+i2 for i1, i2 in zip(data_dict['ID1'][s], data_dict['ID2'][s])])
                    L = data_dict['L']
                    self.sentence1[dataset] = R1
                    self.sentence2[dataset] = R2
                    self.pair_ids[dataset] = pair_ids
                    self.labels[dataset] = L
                else:
                    R1 = self.sentence1[dataset]
                    R2 = self.sentence2[dataset]
                self.dice[dataset] = calculate_dice_sim(R1,R2)
            overlapping = self.dice[dataset]
        elif distance_metric == 'js-div':
            if dataset not in self.js_div.keys():
                if dataset not in self.sentence1.keys():
                    data_dict = load_data(opt, numerical=False)
                    R1 = data_dict['T1']
                    R2 = data_dict['T2']
                    pair_ids = []
                    for s,_ in enumerate(subsets):
                        pair_ids.append([i1+'-'+i2 for i1, i2 in zip(data_dict['ID1'][s], data_dict['ID2'][s])])
                    L = data_dict['L']
                    self.sentence1[dataset] = R1
                    self.sentence2[dataset] = R2
                    self.pair_ids[dataset] = pair_ids
                    self.labels[dataset] = L
                else:
                    R1 = self.sentence1[dataset]
                    R2 = self.sentence2[dataset]
                self.js_div[dataset] = calculate_js_div(R1,R2)
            overlapping = self.js_div[dataset]
        if subset is None:
            return overlapping
        elif subset in ['train','p_train','train_large']:
            return overlapping[0]
        elif subset in ['dev','test2016']:
            return overlapping[1]
        elif subset in ['test','p_test','test2017']:
            return overlapping[-1]
        else:
            raise ValueError('{} not accepted value for subset'.format(subset))


if __name__=='__main__':

    LexSim = LexicalSimilarity()
    LexSim.get_accepted_metrics()
    LexSim.get_accepted_datasets()

    # LexSim.plot_metric_subset_dist('jaccard','Semeval','A')
    LexSim.plot_metric_subset_dist('js-div',dataset='Semeval',task='B')
    # LexSim.plot_metric_label_dist('js-div',dataset='MSRP',task='B')
