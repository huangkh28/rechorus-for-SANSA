import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from models.BaseModel import GeneralModel
import sklearn.utils.sparsefuncs as spfuncs
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import inv, spsolve

class SANSA(GeneralModel):
    reader = 'SANSAReader'
    runner = 'BaseRunner'
    
    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.lambda_ = self.lambda_
        self.d = self.d
        self.max_iter = self.max_iter
        self.tol = self.tol
        self.W = None
        self.Z = None

    def _construct_weights(self, X):
        
        U, I = X.shape
        # 计算 A = X.T @ X + lambda * I
        A = X.T @ X + self.lambda_ * np.eye(I)
        # 稀疏化 A
        A = csc_matrix(A)
        # 稀疏 Cholesky 分解 A = L @ D @ L.T
        L = spsolve(diags(np.sqrt(A.diagonal())), A).tocsc()
        L = np.tril(L)
        D = diags(A.diagonal())
        # 初始化 K
        K = 2 * np.eye(I) - L
        for i in range(self.max_iter):
            # 计算残差矩阵 R = I - L @ K
            R = np.eye(I) - L @ K
            # 计算 Frobenius 范数
            res_norm = np.linalg.norm(R)
            if res_norm < self.tol:
                break
            # 这里简化处理，仅为示例，实际可能需要更复杂的更新规则
            K = K + R @ K
        # 计算 W = K @ P，这里假设 P 是随机初始化
        P = np.random.randn(I, self.d)
        self.W = K @ P
        # 计算 Z0 = D^-1 @ W
        D_inv = inv(D)
        Z0 = D_inv @ self.W
        # 计算 r = diag(W.T @ Z0)
        r = np.diag(self.W.T @ Z0)
        # 计算 Z
        self.Z = Z0 / r
        return [self.W.T.tocsr(), self.Z.tocsr()]

    def _build_item_user_matrix(self, corpus):
        if not hasattr(corpus, 'reader') or not hasattr(corpus.reader, 'interaction_matrix'):
            raise ValueError("The dataset object must contain the reader.interaction_matrix attribute.")
        
        # 获取训练集交互矩阵
        X = corpus.reader.interaction_matrix
        
        # 转置为物品-用户矩阵，并转换为CSC格式
        X_T = X.T.tocsc()
        
        return X_T

    def train(self, corpus, flag=True):
        if isinstance(corpus, bool):  # 如果是eval()调用
            return
        
        # 1. 准备物品-用户矩阵
        X_T = self._build_item_user_matrix(corpus)
        
        # 2. 构建权重矩阵
        self.weights = self._construct_weights(X_T)
        del X_T  # 释放内存

    def eval(self):
        """设置为评估模式"""
        pass  # SANSA不需要特殊的评估模式

    def predict(self, feed_dict):
        if self.W is None or self.Z is None:
            raise ValueError("Model has not been fitted yet.")
        return feed_dict @ self.W.T @ self.Z

    def evaluate(self, feed_dict):
        return self.predict(feed_dict)

    def forward(self, feed_dict):
        # 构建稀疏矩阵
        X = self._build_sparse_matrix(feed_dict)
        
        # 预测得分
        scores = self._predict(X)
        return torch.from_numpy(scores.toarray())

    @staticmethod
    def collate_batch(feed_dicts):
        
        batch_dict = {
            'user_id': [],
            'item_id': [],
            'sparse_interaction': []
        }
        
        for feed_dict in feed_dicts:
            if feed_dict is not None:  # 添加空值检查
                batch_dict['user_id'].append(feed_dict['user_id'])
                batch_dict['item_id'].append(feed_dict['item_id'])
                batch_dict['sparse_interaction'].append(feed_dict['sparse_interaction'])
        
        # 转换为numpy数组
        batch_dict['user_id'] = np.array(batch_dict['user_id'])
        batch_dict['item_id'] = np.array(batch_dict['item_id'])
        # sparse_interaction保持为列表形式
        
        return batch_dict
