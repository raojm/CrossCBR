import torch
import os
import scipy.sparse as sp 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utility import BundleTrainDataset, BundleTestDataset, print_statistics

# stEmbedding = torch.nn.Embedding(500, 3)
# print(stEmbedding)
# aaaa = torch.LongTensor([[1,2,3,4,5,6,7,8,9], [2,2,3,4,5,6,7,8,9]])
# print(aaaa)                  
# result = stEmbedding(aaaa)
# print(result)
# print(result.shape)


def toNumber(x):
    if x.isdigit():
        return int(x)
    else:
        stSplitResult = x.split('.')
        if len(stSplitResult) == 2 and stSplitResult[0].isdigit() and stSplitResult[1].isdigit():
            return float(x)
        else:
            return 0



class RoDatasets():
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']

        self.bundle_mapping_array = []
        self.bundle_item_orig_data = []
        self.bundle_feature = []
        self.bundle_item = []

        self.item_mapping_array = []

        self.user_mapping_array = []
        self.user_bundle_orig_data = []
        self.user_feature = []
        self.user_bundle = []

        with open(os.path.join("./datasets/RO/orig", 'bundle_item.csv'), 'r') as f:
            for line in f.readlines()[1:]:
                bundle_info_tuple = []
                bundle_mapping_index = -1
                item_mapping_index = -1
                for field_index,szField in enumerate(line[:-1].split(',')):
                    szField = szField.strip('"')
                    field_value = toNumber(szField)
                    if 0==field_index:
                        bundle_mapping_index = len(self.bundle_mapping_array)
                        if field_value in self.bundle_mapping_array:
                            bundle_mapping_index = self.bundle_mapping_array.index(field_value)
                        else:
                            self.bundle_mapping_array.append(field_value)
                        bundle_info_tuple.append(bundle_mapping_index)
                    else:
                        bundle_info_tuple.append(field_value)

                    if field_value > 0 and field_index > 0 and 0==field_index%3:
                        item_mapping_index = len(self.item_mapping_array)
                        if field_value in self.item_mapping_array:
                            item_mapping_index = self.item_mapping_array.index(field_value)
                        else:
                            self.item_mapping_array.append(field_value)
                        # bundle item 列表
                        self.bundle_item.append([bundle_mapping_index, item_mapping_index])
                # bundle feature
                self.bundle_feature.insert(bundle_mapping_index, bundle_info_tuple)
                self.bundle_item_orig_data.append(bundle_info_tuple)

        # print(self.bundle_feature)
        # print(self.bundle_mapping_array)
        # print(self.item_mapping_array)
        # print(self.bundle_item)

        with open(os.path.join("./datasets/RO/orig", 'user_bundle.csv'), 'r') as f:
            for line in f.readlines()[1:]:
                user_info_tuple = []
                user_mapping_index = -1
                for field_index,szField in enumerate(line[:-1].split(',')):
                    szField = szField.strip('"')
                    field_value = toNumber(szField)
                    if 0==field_index:
                        user_mapping_index = len(self.user_mapping_array)
                        if field_value in self.user_mapping_array:
                            user_mapping_index = self.user_mapping_array.index(field_value)
                        else:
                            self.user_mapping_array.append(field_value)
                        user_info_tuple.append(user_mapping_index)
                    else:
                        user_info_tuple.append(field_value)
                # user feature
                self.user_feature.insert(user_mapping_index, user_info_tuple)
                # # user bundle 列表
                bundle_id = user_info_tuple[8]
                self.user_bundle.append([user_mapping_index, self.bundle_mapping_array.index(bundle_id)])
                self.user_bundle_orig_data.append(user_info_tuple)

            # self.user_bundle_orig_data = list(map(lambda s: tuple(aaaa(i.strip('"')) for i in s[:-1].split(',')), f.readlines()[1:]))

        indice = np.array(self.user_bundle_orig_data, dtype=np.int32)
        values = np.ones(len(self.user_bundle_orig_data), dtype=np.float32)

        # 分割数据 成traindata testdata

        self.num_users, self.num_bundles, self.num_items = self.get_data_size()

        b_i_graph = self.get_bi()
        u_i_pairs, u_i_graph = self.get_ui()

        u_b_pairs_train, u_b_graph_train = self.get_ub("train")
        u_b_pairs_val, u_b_graph_val = self.get_ub("tune")
        u_b_pairs_test, u_b_graph_test = self.get_ub("test")

        u_b_for_neg_sample, b_b_for_neg_sample = None, None

        self.bundle_train_data = BundleTrainDataset(conf, u_b_pairs_train, u_b_graph_train, self.num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, conf["neg_num"])
        self.bundle_val_data = BundleTestDataset(u_b_pairs_val, u_b_graph_val, u_b_graph_train, self.num_users, self.num_bundles)
        self.bundle_test_data = BundleTestDataset(u_b_pairs_test, u_b_graph_test, u_b_graph_train, self.num_users, self.num_bundles)

        self.graphs = [u_b_graph_train, u_i_graph, b_i_graph]

        self.train_loader = DataLoader(self.bundle_train_data, batch_size=batch_size_train, shuffle=True, num_workers=10, drop_last=True)
        self.val_loader = DataLoader(self.bundle_val_data, batch_size=batch_size_test, shuffle=False, num_workers=20)
        self.test_loader = DataLoader(self.bundle_test_data, batch_size=batch_size_test, shuffle=False, num_workers=20)


    def get_data_size(self):
        # name = self.name
        # if "_" in name:
        #     name = name.split("_")[0]
        # with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
        #     return [int(s) for s in f.readline().split('\t')][:3]
        return [len(self.user_mapping_array), len(self.bundle_mapping_array), len(self.item_mapping_array)]

    def get_aux_graph(self, u_i_graph, b_i_graph, conf):
        u_b_from_i = u_i_graph @ b_i_graph.T
        u_b_from_i = u_b_from_i.todense()
        bn1_window = [int(i*self.num_bundles) for i in conf['hard_window']]
        u_b_for_neg_sample = np.argsort(u_b_from_i, axis=1)[:, bn1_window[0]:bn1_window[1]]

        b_b_from_i = b_i_graph @ b_i_graph.T
        b_b_from_i = b_b_from_i.todense()
        bn2_window = [int(i*self.num_bundles) for i in conf['hard_window']]
        b_b_for_neg_sample = np.argsort(b_b_from_i, axis=1)[:, bn2_window[0]:bn2_window[1]]

        return u_b_for_neg_sample, b_b_for_neg_sample


    def get_bi(self):
        indice = np.array(self.bundle_item, dtype=np.int32)
        values = np.ones(len(self.bundle_item), dtype=np.float32)
        b_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_bundles, self.num_items)).tocsr()

        print_statistics(b_i_graph, 'B-I statistics')

        return b_i_graph


    def get_ui(self):
        # with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
        #     u_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        #ro里没有u_i 关系 就保持空
        u_i_pairs = [[]]
        indice = np.array(u_i_pairs, dtype=np.int32)
        values = np.ones(len(u_i_pairs), dtype=np.float32)
        u_i_graph = sp.coo_matrix((self.num_users, self.num_items)).tocsr()

        print_statistics(u_i_graph, 'U-I statistics')

        return u_i_pairs, u_i_graph


    def get_ub(self, task):
        # with open(os.path.join(self.path, self.name, 'user_bundle_{}.txt'.format(task)), 'r') as f:
        #     u_b_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        start_index = 0
        end_index = len(self.user_bundle)
        if "train" == task:
            start_index = 0
            end_index = int(len(self.user_bundle)*0.7)
        elif "test" ==  task:
            start_index = int(len(self.user_bundle)*0.7)
            end_index = int(len(self.user_bundle)*0.9)
        else:
            start_index = int(len(self.user_bundle)*0.9)
            end_index = len(self.user_bundle)
        u_b_pairs = self.user_bundle[start_index:end_index]
        indice = np.array(u_b_pairs, dtype=np.int32)
        values = np.ones(len(u_b_pairs), dtype=np.float32)
        u_b_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()

        print_statistics(u_b_graph, "U-B statistics in %s" %(task))

        return u_b_pairs, u_b_graph




