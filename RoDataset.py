import torch
import os
import math
from csv import reader
import scipy.sparse as sp 
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from utility import BundleTrainDataset, BundleTestDataset, print_statistics

# stEmbedding = torch.nn.Parameter(torch.FloatTensor(50, 3))
# stEmbedding = torch.nn.Embedding(50, 3)
# torch.nn.init.xavier_normal_(stEmbedding.weight)
# print(stEmbedding.weight)
# aaaa = torch.LongTensor([[1,2,3,4,5], [3,2,1,4,5], [3,2,1,4,5], [3,2,1,4,5]])
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

        self.bundle_mapping_array = dict()
        self.bundle_item_orig_data = []
        self.bundle_feature = []
        self.bundle_item = []

        self.item_mapping_array = dict()

        self.game_role_dict = dict()
        self.user_mapping_array = dict()
        self.user_bundle_orig_data = []

        self.user_item_ordereddict = OrderedDict()
        
        self.user_feature_orderdict = OrderedDict()
        self.user_bundle_orderdict = OrderedDict()
        
        

        # 处理旧版数据文件
        # self.init_orig_data()

        # 处理新版RO数据 所有数据都在一个csv文件里
        self.init_new_orig_data()

        self.user_feature = list(self.user_feature_orderdict.keys())
        #init_new_orig_data之后通过user_bundle_orderdict转化成list
        self.user_bundle = list(self.user_bundle_orderdict.keys())


        self.num_users, self.num_bundles, self.num_items = self.get_data_size()

        b_i_graph = self.get_bi()
        u_i_pairs, u_i_graph = self.get_ui()

        # 分割数据 成traindata testdata
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

    def init_orig_data(self):
        with open(os.path.join("./datasets/RO/orig", 'bundle_item.csv'), 'r', encoding='UTF-8') as f:
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
                            bundle_mapping_index = self.bundle_mapping_array.get(field_value)
                        else:
                            self.bundle_mapping_array[field_value] = bundle_mapping_index
                        bundle_info_tuple.append(bundle_mapping_index)
                    else:
                        bundle_info_tuple.append(field_value)

                    if field_value > 0 and field_index > 0 and 0==field_index%3:
                        item_mapping_index = len(self.item_mapping_array)
                        if field_value in self.item_mapping_array:
                            item_mapping_index = self.item_mapping_array.get(field_value)
                        else:
                            self.item_mapping_array[field_value] = item_mapping_index
                        # bundle item 列表
                        bundle_item_tuple = [bundle_mapping_index, item_mapping_index]
                        if bundle_item_tuple not in self.bundle_item:
                            self.bundle_item.append(bundle_item_tuple)
                # bundle feature
                self.bundle_feature.insert(bundle_mapping_index, bundle_info_tuple)
                self.bundle_item_orig_data.append(bundle_info_tuple)

        # print(self.bundle_feature)
        # print(self.bundle_mapping_array)
        # print(self.item_mapping_array)
        # print(self.bundle_item)

        with open(os.path.join("./datasets/RO/orig", 'user_bundle.csv'), 'r', encoding='UTF-8') as f:
            for line in f.readlines()[1:]:
                user_info_tuple = []
                user_mapping_index = -1
                for field_index,szField in enumerate(line[:-1].split(',')):
                    szField = szField.strip('"')
                    field_value = toNumber(szField)
                    if 0==field_index:
                        user_mapping_index = len(self.user_mapping_array)
                        if field_value in self.user_mapping_array:
                            user_mapping_index = self.user_mapping_array.get(field_value)
                        else:
                            self.user_mapping_array[field_value] = user_mapping_index
                        user_info_tuple.append(user_mapping_index)
                    else:
                        user_info_tuple.append(field_value)
                # user feature
                self.user_feature.insert(user_mapping_index, user_info_tuple)
                # # user bundle 列表
                bundle_id = user_info_tuple[8]
                user_bundle_item_tuple = (user_mapping_index, self.bundle_mapping_array.get(bundle_id))
                if user_bundle_item_tuple not in self.user_bundle_orderdict:
                    self.user_bundle_orderdict[user_bundle_item_tuple] = len(self.user_bundle_orderdict)
                self.user_bundle_orig_data.append(user_info_tuple)


    def init_new_orig_data(self):
        with open(os.path.join("./datasets/RO/orig", 'bundle_item.csv'), 'r', encoding='UTF-8') as f:
            #跳过第一行 file是可迭代对象
            next(f)
            for line_index, line in enumerate(f):
                if line_index %100 == 0:
                    print("bundle_item line_index:", line_index) 
                bundle_info_tuple = []
                bundle_mapping_index = -1
                item_mapping_index = -1
                for field_index,szField in enumerate(line[:-1].split(',')):
                    szField = szField.strip('"')
                    field_value = toNumber(szField)
                    if 0==field_index:
                        bundle_mapping_index = len(self.bundle_mapping_array)
                        if field_value in self.bundle_mapping_array:
                            bundle_mapping_index = self.bundle_mapping_array.get(field_value)
                        else:
                            self.bundle_mapping_array[field_value] = bundle_mapping_index
                        bundle_info_tuple.append(bundle_mapping_index)
                    else:
                        bundle_info_tuple.append(field_value)

                    if field_value > 0 and field_index > 0 and 0==field_index%3:
                        item_mapping_index = len(self.item_mapping_array)
                        if field_value in self.item_mapping_array:
                            item_mapping_index = self.item_mapping_array.get(field_value)
                        else:
                            self.item_mapping_array[field_value] = item_mapping_index
                        # bundle item 列表
                        bundle_item_tuple = [bundle_mapping_index, item_mapping_index]
                        if bundle_item_tuple not in self.bundle_item:
                            self.bundle_item.append(bundle_item_tuple)
                # bundle feature
                self.bundle_feature.insert(bundle_mapping_index, bundle_info_tuple)
                self.bundle_item_orig_data.append(bundle_info_tuple)

        # print(self.bundle_feature)
        # print(self.bundle_mapping_array)
        # print(self.item_mapping_array)
        # print(self.bundle_item)

        with open(os.path.join("./datasets/RO/orig", 'record_all.csv'), 'r', encoding='UTF-8') as f:
            #跳过第一行 file是可迭代对象
            # next(f)
            file_csv = reader(f)
            # field_index 到item_mapping_index映射
            item_index_dict = OrderedDict()
            for line_index, line in enumerate(file_csv):
                if 0 == line_index:
                    for field_index,szField in enumerate(line):
                        item_id_tmp = 0
                        if szField.startswith("buy"):
                            item_id_tmp = toNumber(szField[3:])
                        elif szField.startswith("cost"):
                            item_id_tmp = toNumber(szField[4:])
                        if item_id_tmp>0 and item_id_tmp in self.item_mapping_array:
                                item_index_dict[field_index] = self.item_mapping_array.get(item_id_tmp)
                    continue
                if line_index > 0 and line_index%10000 == 0:
                    print("record_all line_index:", line_index)
                    # break
                user_info_tuple = []
                user_mapping_index = -1
                is_new_user = False
                for field_index,szField in enumerate(line):
                    # szField = szField.strip("'")
                    field_value = toNumber(szField)
                    user_info_tuple.append(field_value)
                    # if 0==field_index:
                    #     user_mapping_index = len(self.user_mapping_array)
                    #     if field_value in self.user_mapping_array:
                    #         user_mapping_index = self.user_mapping_array.get(field_value)
                    #     else:
                    #         is_new_user = True
                    #         self.user_mapping_array[field_value] = user_mapping_index
                    #     user_info_tuple.append(user_mapping_index)
                    # else:
                    #     user_info_tuple.append(field_value)
                # user feature
                game_role_index = len(self.game_role_dict)
                if user_info_tuple[6] in self.game_role_dict:
                    game_role_index = self.game_role_dict.get(user_info_tuple[6])
                else:
                    self.game_role_dict[user_info_tuple[6]] = game_role_index
                user_feature_value_tuple = (user_info_tuple[4], min(100-1, int(math.pow(user_info_tuple[5], 0.3))), game_role_index,  min(100-1, int(math.pow(user_info_tuple[7], 0.3))), user_info_tuple[11])
                user_feature_value_tuple += tuple(map(int, user_info_tuple[11:12]))
                # user_feature_value_tuple += tuple(map(int, user_info_tuple[22:55]))
                user_feature_value_tuple += tuple(map(int, user_info_tuple[22:24]))
                user_feature_value_tuple += tuple(map(int, user_info_tuple[27:28]))
                # aaaa = list(filter(lambda x: x < 0 or x>=5000, user_feature_value_tuple))
                # if len(aaaa) > 0:
                #     print("aaaa:", aaaa)
                # if is_new_user:
                #     self.user_feature.insert(user_mapping_index, user_feature_value_tuple)
                # # user bundle 列表
                bundle_id = user_info_tuple[9]
                is_bought = user_info_tuple[2]

                user_related_items = []
                for field_index, item_mapping_index in item_index_dict.items():
                    user_item_value = user_info_tuple[field_index]
                    if user_item_value>0:
                        user_related_items.append(item_mapping_index)
                
                if is_bought > 0 or len(user_related_items)>0:
                    #user_mapping_index 换成OrderedDict生成 不同的user_feature_value当不同的user处理
                    user_feature_index = len(self.user_feature_orderdict)
                    if user_feature_value_tuple not in self.user_feature_orderdict:
                        self.user_feature_orderdict[user_feature_value_tuple]= len(self.user_feature_orderdict)
                    else:
                        user_feature_index = self.user_feature_orderdict[user_feature_value_tuple]
                    
                    user_bundle_tuple = (user_feature_index, self.bundle_mapping_array.get(bundle_id))
                    if is_bought > 0 and user_bundle_tuple not in self.user_bundle_orderdict:
                        self.user_bundle_orderdict[user_bundle_tuple]= len(self.user_bundle_orderdict)
                    self.user_bundle_orig_data.append(user_info_tuple)

                    for item_mapping_index_tmp in user_related_items:
                        user_item_tuple = (user_feature_index, item_mapping_index_tmp)
                        if user_item_tuple not in self.user_item_ordereddict:
                            self.user_item_ordereddict[user_item_tuple] = len(self.user_item_ordereddict)


    def get_data_size(self):
        # name = self.name
        # if "_" in name:
        #     name = name.split("_")[0]
        # with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
        #     return [int(s) for s in f.readline().split('\t')][:3]
        return [len(self.user_feature_orderdict), len(self.bundle_mapping_array), len(self.item_mapping_array)]

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




