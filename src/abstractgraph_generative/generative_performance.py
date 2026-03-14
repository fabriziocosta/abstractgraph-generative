from scipy.stats.mstats import gmean
from scipy.stats.mstats import trimmed_mean, trimmed_std
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import train_test_split
from toolz import curry
from toolz import partition_all
import multiprocessing as mp
import numpy as np
from functools import partial
from sklearn.base import clone as sk_clone
import os
import random
import scipy as sp
import matplotlib.pyplot as plt
from joblib import Parallel, delayed




def estimate_mean_and_std_from_quantiles(data):

    """
    Estimate the mean and std of a vector of numbers using quantiles (IQR-based method).
    
    Wan, Xiang, Wenqian Wang, Jiming Liu, and Tiejun Tong. 2014. “Estimating the Sample Mean and Standard Deviation from 
    the Sample Size, Median, Range And/or Interquartile Range.” BMC Medical Research Methodology 14 (135). doi:10.1186/1471-2288-14-135. 

    Parameters:
    data (list or numpy array): A list or array of numerical values.
    
    Returns:
    estimated_mean, estimated_std (float): The estimated mean and std of the data.
    """
    if len(data)==1: 
        return data[0], 0
        
    # Calculate the 25th and 75th percentiles (Q1 and Q3)
    q1 = np.percentile(data, 25)
    m = np.percentile(data, 50)
    q3 = np.percentile(data, 75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Estimate variance using the IQR
    estimated_std = (iqr / 1.34898)  

    estimated_mean = np.mean([q1,m,q3])

    return estimated_mean ,estimated_std

def bootstrap(instances, targets, seed=None):
    if seed is not None: np.random.seed(seed)
    size = len(instances)
    idxs = np.random.choice(size, size=size, replace=True)
    instances_ = instances[idxs]
    targets_ = targets[idxs]
    return instances_, targets_

def resample(instances, targets, size, seed=None):
    if seed is not None: np.random.seed(seed)
    if len(instances)-size < 3:
        return instances, targets
    if size > len(instances): #sample with replacement
        idxs = np.random.choice(len(instances), size=size, replace=True)
    else: #sample without replacement 
        idxs = np.random.choice(len(instances), size=size, replace=False)
    resampled_instances = instances[idxs]
    resampled_targets = np.array(targets)[idxs]
    return resampled_instances, resampled_targets

def class_equalize(instances, targets):
    class_instances = [np.vstack([instance for instance, target in zip(instances, targets) if target == curr_target]) for curr_target in sorted(set(targets))]
    max_size = max(len(class_instances) for class_instances in class_instances)
    resampled_class_instances_list = []
    resampled_targets_list = []
    for target, class_instances in enumerate(class_instances):
        if len(class_instances) < max_size:
            idxs = np.random.choice(len(class_instances), size=max_size, replace=True)
            resampled_class_instances = class_instances[idxs]
        else:
            resampled_class_instances = class_instances
        resampled_class_instances_list.append(resampled_class_instances)
        resampled_targets_list.append([target]*len(resampled_class_instances))
    resampled_instances = np.vstack(resampled_class_instances_list)
    resampled_targets = np.array(sum(resampled_targets_list, []))
    return resampled_instances, resampled_targets

def robust_statistics(vals, n_elements_to_trim=1):
    l = n_elements_to_trim/len(vals)
    mean = trimmed_mean(vals, limits=(l,l))
    std = trimmed_std(vals, limits=(l,l))
    return mean, std

def similarity_entropy(src_instances, dst_instances, metric = 'linear', n_neighbors=10):
    X = np.vstack([src_instances, dst_instances])
    targets = np.asarray([1]*src_instances.shape[0]+[0]*dst_instances.shape[0])
    scale = sp.stats.entropy(np.bincount(targets, minlength=2)/len(targets), base=2)
    K = pairwise_kernels(X, metric=metric)
    knbs = np.argsort(-K, axis=1)[:,:n_neighbors]
    target_knbs = targets[knbs]
    entropies = [sp.stats.entropy(np.bincount(x, minlength=2)/len(x), base=2)/scale for x in target_knbs]
    score = np.mean(entropies)
    return score

def estimate_instance_set_similarity(src_instances, src_targets, dst_instances, dst_targets, metric='cosine', n_neighbors=2):
    similarity_list = []
    target_classes = sorted(set(src_targets))
    for target_class in target_classes:
        loc_src_instances = np.vstack([src_instance for src_instance, src_target in zip(src_instances, src_targets) if src_target == target_class])
        loc_dst_instances = np.vstack([dst_instance for dst_instance, dst_target in zip(dst_instances, dst_targets) if dst_target == target_class])
        similarity = similarity_entropy(loc_src_instances, loc_dst_instances, metric=metric, n_neighbors=n_neighbors)
        similarity_list.append(similarity)
    similarity = gmean(similarity_list)
    return similarity

def adjusted_balanced_accuracy_score(test_targets, predicted_targets):
    return balanced_accuracy_score(test_targets, predicted_targets, adjusted=True)

def make_adjusted_score_func(score):
    def score_func(test_targets, preds):
        random_targets = np.random.choice(test_targets, size=len(test_targets), replace=False)
        random_score = score(test_targets, random_targets)
        perfect_score = score(test_targets, test_targets)
        adjusted_score = (score(test_targets, preds) - random_score)/(perfect_score - random_score)
        adjusted_score = max(0, adjusted_score)
        adjusted_score = min(1, adjusted_score)
        return adjusted_score
    return score_func

@curry
def compute_estimated_predictive_performance_score_func(train_instances, train_targets, test_instances=None, test_targets=None, data_estimator=None, discriminative_performance_func=None, n_rep=10):
    # Prepare a base estimator with n_jobs=1 to avoid nested parallelism warnings
    base_est = data_estimator
    try:
        base_est = sk_clone(base_est)
    except Exception:
        from copy import deepcopy as _deepcopy
        base_est = _deepcopy(base_est)
    try:
        base_est.set_params(n_jobs=1)
    except Exception:
        if hasattr(base_est, "n_jobs"):
            base_est.n_jobs = 1
    scores = []
    for it in range(n_rep):
        X_train, _, y_train, _ = train_test_split(train_instances, train_targets, stratify=train_targets, train_size=.7)
        try:
            est = sk_clone(base_est)
        except Exception:
            from copy import deepcopy as _deepcopy
            est = _deepcopy(base_est)
        predicted_targets = est.fit(X_train, y_train).predict(test_instances)
        score = discriminative_performance_func(test_targets, predicted_targets)
        scores.append(score)
    estimated_predictive_performance_score, estimated_predictive_performance_score_std = robust_statistics(scores, n_elements_to_trim=1)
    return estimated_predictive_performance_score, estimated_predictive_performance_score_std

def discriminative_generative_predictive_performances(data_estimator, generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets, discriminative_performance_func=None, n_rep=10):
    compute_estimated_predictive_performance_score = compute_estimated_predictive_performance_score_func(test_instances=test_instances, test_targets=test_targets, data_estimator=data_estimator, discriminative_performance_func=discriminative_performance_func, n_rep=n_rep)

    predictive_performance_with_real_train, predictive_performance_with_real_train_std = compute_estimated_predictive_performance_score(real_train_instances, real_train_targets)
    #ensure that num of generated_instances is the same as num of real_train_instances
    resampled_generated_instances, resampled_generated_targets = resample(generated_instances, generated_targets, len(real_train_instances))
    predictive_performance_with_generated, predictive_performance_with_generated_std = compute_estimated_predictive_performance_score(resampled_generated_instances, resampled_generated_targets)
    
    predictive_performance_with_real_train_and_reference, predictive_performance_with_real_train_and_reference_std = compute_estimated_predictive_performance_score(np.vstack([real_train_instances,real_reference_instances]), np.hstack([real_train_targets,real_reference_targets]))
    #ensure that num of generated_instances is the same as num of real_reference_instances
    resampled_generated_instances, resampled_generated_targets = resample(generated_instances, generated_targets, len(real_reference_instances))
    predictive_performance_with_real_train_and_generated, predictive_performance_with_real_train_and_generated_std = compute_estimated_predictive_performance_score(np.vstack([real_train_instances,resampled_generated_instances]), np.hstack([real_train_targets,resampled_generated_targets]))
    
    real_gen_instances = np.vstack([real_train_instances,resampled_generated_instances])
    real_gen_target = np.array([1]*len(real_train_instances)+[0]*len(resampled_generated_instances))
    
    predictive_performance_discriminate_real_from_generated_list = []
    for it in range(n_rep):
        train_real_gen_instances, test_real_gen_instances, train_real_gen_targets, test_real_gen_targets = train_test_split(real_gen_instances, real_gen_target, train_size=.7)
        train_real_gen_instances, train_real_gen_targets = class_equalize(train_real_gen_instances, train_real_gen_targets)
        compute_real_gen_estimated_predictive_performance_score = compute_estimated_predictive_performance_score_func(test_instances=test_real_gen_instances, test_targets=test_real_gen_targets, data_estimator=data_estimator, discriminative_performance_func=discriminative_performance_func, n_rep=n_rep)
        predictive_performance_discriminate_real_from_generated, predictive_performance_discriminate_real_from_generated_std = compute_real_gen_estimated_predictive_performance_score(train_real_gen_instances, train_real_gen_targets)
        predictive_performance_discriminate_real_from_generated_list.append(predictive_performance_discriminate_real_from_generated)
    predictive_performance_discriminate_real_from_generated = np.mean(predictive_performance_discriminate_real_from_generated_list)
    predictive_performances = [predictive_performance_with_real_train, predictive_performance_with_generated, predictive_performance_with_real_train_and_reference, predictive_performance_with_real_train_and_generated, predictive_performance_discriminate_real_from_generated]
    return predictive_performances

def std_of_ratio_of_normal_distributions(mu1, std1, mu2, std2):
    return np.sqrt(mu1**2 / mu2**2 * (std1**2 / mu1**2 + std2**2 / mu2**2))

def std_of_product_of_normal_distributions(mu1, std1, mu2, std2):
    return np.sqrt((std1**2 + mu1**2) * (std2**2 + mu2**2) - mu1**2 * mu2**2)

def std_of_sum_of_normal_distributions(mu1, std1, mu2, std2):
    return np.sqrt(std1**2 + std2**2)

def std_of_difference_of_normal_distributions(mu1, std1, mu2, std2):
    return std_of_sum_of_normal_distributions(mu1, std1, mu2, std2)

def std_of_average_of_normal_distributions(mu1, std1, mu2, std2):
    return std_of_sum_of_normal_distributions(mu1, std1, mu2, std2) / 2


class DiscriminativeGenerativeQualityScorer(object):

    def __init__(self, 
        data_estimator=None, 
        discriminative_performance_func=f1_score, 
        n_rep_estimator=3, 
        n_neighbors=3, 
        n_elements_to_trim=1, 
        metric='cosine', 
        verbose=True, 
        parallel=False, 
        n_cpus=None, 
        make_adjusted_score=True, 
        enforce_positive_definite=True,
        enforce_maximum=True):
        self.data_estimator = data_estimator
        if make_adjusted_score: self.effective_discriminative_performance_func = make_adjusted_score_func(discriminative_performance_func)
        else: self.effective_discriminative_performance_func = discriminative_performance_func
        self.n_rep_estimator = n_rep_estimator
        self.n_neighbors = n_neighbors
        self.n_elements_to_trim = n_elements_to_trim
        self.metric = metric
        self.verbose = verbose
        self.parallel = parallel
        self.n_cpus = n_cpus
        self.enforce_positive_definite = enforce_positive_definite
        self.enforce_maximum = enforce_maximum

        self.real_train_instances_list = []
        self.real_train_targets_list = []
        self.real_reference_instances_list = []
        self.real_reference_targets_list = []
        self.test_instances_list = []
        self.test_targets_list = []
        self.generated_instances_list =[]
        self.generated_targets_list = []
        self.similarity_generated_vs_real_train_avg = None
        self.similarity_generated_vs_real_train_std = None
        self.predictive_performance_with_real_train_avg = None
        self.predictive_performance_with_real_train_std = None
        self.predictive_performance_with_generated_avg = None
        self.predictive_performance_with_generated_std = None
        self.predictive_performance_with_real_train_and_reference_avg = None
        self.predictive_performance_with_real_train_and_reference_std = None
        self.predictive_performance_with_real_train_and_generated_avg = None
        self.predictive_performance_with_real_train_and_generated_std = None
        self.predictive_performance_discriminate_real_from_generated_avg = None
        self.predictive_performance_discriminate_real_from_generated_std = None

    def input_train(self, data_mtx, targets):
        self.real_train_instances_list.append(data_mtx)
        self.real_train_targets_list.append(targets)
        return self 
    
    def input_reference(self, data_mtx, targets):
        self.real_reference_instances_list.append(data_mtx)
        self.real_reference_targets_list.append(targets)
        return self 

    def input_test(self, data_mtx, targets):
        self.test_instances_list.append(data_mtx)
        self.test_targets_list.append(targets)
        return self 

    def input_generated(self, data_mtx, targets):
        self.generated_instances_list.append(data_mtx)
        self.generated_targets_list.append(targets)
        return self 

    def input_data(self, generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets):
        self.generated_instances_list.append(generated_instances)
        self.generated_targets_list.append(generated_targets)
        self.real_train_instances_list.append(real_train_instances)
        self.real_train_targets_list.append(real_train_targets)
        self.real_reference_instances_list.append(real_reference_instances)
        self.real_reference_targets_list.append(real_reference_targets)
        self.test_instances_list.append(test_instances)
        self.test_targets_list.append(test_targets)
        return self

    def resample_single(self, instances_list, targets_list, n_iterations=10, use_replacement=False, fraction=0.7):
        working_instances_list = []
        working_targets_list = []
        for instances, targets in zip(instances_list, targets_list):
            for it in range(n_iterations):
                if use_replacement: 
                    working_instances, working_targets = bootstrap(instances, targets, seed=it+1)
                else:
                    size = int(len(targets)*fraction)
                    working_instances, working_targets = resample(instances, targets, size, seed=it+1)
                working_instances = np.array(working_instances)
                working_targets = np.array(working_targets)
                working_instances_list.append(working_instances)
                working_targets_list.append(working_targets)
        return working_instances_list, working_targets_list

    def resample(self, n_iterations=10, use_resampling=False, use_replacement=False, fraction=0.7):
        if use_resampling:
            self.generated_instances_list, self.generated_targets_list = self.resample_single(self.generated_instances_list, self.generated_targets_list, n_iterations=n_iterations, use_replacement=use_replacement, fraction=fraction)
            self.real_train_instances_list, self.real_train_targets_list = self.resample_single(self.real_train_instances_list, self.real_train_targets_list, n_iterations=n_iterations, use_replacement=use_replacement, fraction=fraction)
            self.real_reference_instances_list, self.real_reference_targets_list = self.resample_single(self.real_reference_instances_list, self.real_reference_targets_list, n_iterations=n_iterations, use_replacement=use_replacement, fraction=fraction)
            self.test_instances_list, self.test_targets_list = self.resample_single(self.test_instances_list, self.test_targets_list, n_iterations=n_iterations, use_replacement=use_replacement, fraction=fraction)
        else:
            self.generated_instances_list, self.generated_targets_list = self.resample_single(self.generated_instances_list, self.generated_targets_list, n_iterations=n_iterations, use_replacement=False, fraction=1)
            self.real_train_instances_list, self.real_train_targets_list = self.resample_single(self.real_train_instances_list, self.real_train_targets_list, n_iterations=n_iterations, use_replacement=False, fraction=1)
            self.real_reference_instances_list, self.real_reference_targets_list = self.resample_single(self.real_reference_instances_list, self.real_reference_targets_list, n_iterations=n_iterations, use_replacement=False, fraction=1)
            self.test_instances_list, self.test_targets_list = self.resample_single(self.test_instances_list, self.test_targets_list, n_iterations=n_iterations, use_replacement=False, fraction=1)

    def _compute_performance_indicators_single(self, generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets):
        similarity_generated_vs_real_train = estimate_instance_set_similarity(generated_instances, generated_targets, real_train_instances, real_train_targets, metric=self.metric, n_neighbors=self.n_neighbors)
        predictive_performances = discriminative_generative_predictive_performances(self.data_estimator, generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets, discriminative_performance_func=self.effective_discriminative_performance_func, n_rep=self.n_rep_estimator)
        predictive_performance_with_real_train, predictive_performance_with_generated, predictive_performance_with_real_train_and_reference, predictive_performance_with_real_train_and_generated, predictive_performance_discriminate_real_from_generated = predictive_performances
        if self.verbose: print('\t real:%.2f  gen:%.2f  real_and_ref:%.2f  real_and_gen:%.2f  real_vs_gen:%.2f  similarity:%.2f'%(predictive_performance_with_real_train, predictive_performance_with_generated, predictive_performance_with_real_train_and_reference, predictive_performance_with_real_train_and_generated, predictive_performance_discriminate_real_from_generated, similarity_generated_vs_real_train))
        return similarity_generated_vs_real_train, predictive_performance_with_real_train, predictive_performance_with_generated, predictive_performance_with_real_train_and_reference, predictive_performance_with_real_train_and_generated, predictive_performance_discriminate_real_from_generated
    
    def _compute_performance_indicators(self, data_list):
            predictive_performances_list = [self._compute_performance_indicators_single(*data) for data in data_list]
            return predictive_performances_list

    def _make_data_list(self): 
        data_list = []
        for generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets in zip(self.generated_instances_list, self.generated_targets_list, self.real_train_instances_list, self.real_train_targets_list, self.real_reference_instances_list, self.real_reference_targets_list, self.test_instances_list, self.test_targets_list):
            data = generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets
            data_list.append(data)
        return data_list

    def compute_performance_indicators(self):
        data_list = self._make_data_list()
        if self.parallel is False:
            predictive_performances_list = self._compute_performance_indicators(data_list)
        else:
            # Robust parallel path for notebooks and Py3.12: always use spawn context
            # and send per-item payloads to avoid reliance on inherited globals.
            ctx = mp.get_context("spawn")
            if self.n_cpus is None:
                self.n_cpus = os.cpu_count() or 1
            n_items = len(data_list)
            self.n_cpus = max(1, min(self.n_cpus, n_items))

            pool = ctx.Pool(self.n_cpus)
            try:
                conf = dict(
                    metric=self.metric,
                    n_neighbors=self.n_neighbors,
                    data_estimator=self.data_estimator,
                    discriminative_performance_func=self.effective_discriminative_performance_func,
                    n_rep_estimator=self.n_rep_estimator,
                    verbose=self.verbose,
                )
                payload_iter = (
                    ((gi, gt, rti, rtt, rri, rrt, ti, tt), conf)
                    for (gi, gt, rti, rtt, rri, rrt, ti, tt) in data_list
                )
                results_iter = pool.imap_unordered(
                    _gp_worker_compute_from_data, payload_iter, chunksize=1
                )
                predictive_performances_list = list(results_iter)
            finally:
                pool.close()
                pool.join()
        self._store_predictive_performances_list(predictive_performances_list)
        return self

    def _store_predictive_performances_list(self, predictive_performances_list):
        self.similarity_generated_vs_real_train_list = []
        self.predictive_performance_with_real_train_list = []
        self.predictive_performance_with_generated_list = []
        self.predictive_performance_with_real_train_and_reference_list = []
        self.predictive_performance_with_real_train_and_generated_list = []
        self.predictive_performance_discriminate_real_from_generated_list = []
        for predictive_performances in predictive_performances_list:
            similarity_generated_vs_real_train, predictive_performance_with_real_train, predictive_performance_with_generated, predictive_performance_with_real_train_and_reference, predictive_performance_with_real_train_and_generated, predictive_performance_discriminate_real_from_generated = predictive_performances
            self.similarity_generated_vs_real_train_list.append(similarity_generated_vs_real_train)
            self.predictive_performance_with_real_train_list.append(predictive_performance_with_real_train)
            self.predictive_performance_with_generated_list.append(predictive_performance_with_generated)
            self.predictive_performance_with_real_train_and_reference_list.append(predictive_performance_with_real_train_and_reference)
            self.predictive_performance_with_real_train_and_generated_list.append(predictive_performance_with_real_train_and_generated)
            self.predictive_performance_discriminate_real_from_generated_list.append(predictive_performance_discriminate_real_from_generated)
        self.similarity_generated_vs_real_train_avg, self.similarity_generated_vs_real_train_std = robust_statistics(self.similarity_generated_vs_real_train_list, n_elements_to_trim=self.n_elements_to_trim)
        self.predictive_performance_with_real_train_avg, self.predictive_performance_with_real_train_std = robust_statistics(self.predictive_performance_with_real_train_list, n_elements_to_trim=self.n_elements_to_trim)
        self.predictive_performance_with_generated_avg, self.predictive_performance_with_generated_std = robust_statistics(self.predictive_performance_with_generated_list, n_elements_to_trim=self.n_elements_to_trim)
        self.predictive_performance_with_real_train_and_reference_avg, self.predictive_performance_with_real_train_and_reference_std = robust_statistics(self.predictive_performance_with_real_train_and_reference_list, n_elements_to_trim=self.n_elements_to_trim)
        self.predictive_performance_with_real_train_and_generated_avg, self.predictive_performance_with_real_train_and_generated_std = robust_statistics(self.predictive_performance_with_real_train_and_generated_list, n_elements_to_trim=self.n_elements_to_trim)
        self.predictive_performance_discriminate_real_from_generated_avg, self.predictive_performance_discriminate_real_from_generated_std = robust_statistics(self.predictive_performance_discriminate_real_from_generated_list, n_elements_to_trim=self.n_elements_to_trim)

    def feasibility_condition_enforcement(self, score):
        score = np.nan_to_num(score, posinf=0, neginf=0)
        if self.enforce_positive_definite: score = max(0, score)
        if self.enforce_maximum: score = min(1, score)
        return score

    def post_process_with_feasibility_enforcement(self, score_list):
        score, score_std = estimate_mean_and_std_from_quantiles(score_list)
        score = self.feasibility_condition_enforcement(score)
        score_std = self.feasibility_condition_enforcement(score_std)
        return score, score_std
        
    def get_similarity_list(self):
        return self.similarity_generated_vs_real_train_list

    def similarity(self):
        #similarity: how similar are the distribution of neighbors distances when we consider the neighbors in the real set to a generated instance and when we consider the neighbors in the generated set to a real instance
        #this is a failsafe quality measure that does not depend on the discriminator capacity
        similarity_list = self.get_similarity_list()
        similarity_score, similarity_score_std = self.post_process_with_feasibility_enforcement(similarity_list)
        return similarity_score, similarity_score_std
        
    def get_quality_list(self):
        quality_list = [predictive_performance_with_generated / predictive_performance_with_real_train for predictive_performance_with_generated, predictive_performance_with_real_train in zip(self.predictive_performance_with_generated_list, self.predictive_performance_with_real_train_list)]
        return quality_list

    def quality(self):
        #quality: training on generated data should yield comparable predictive performance on a test set as when training on original data
        quality_list = self.get_quality_list() 
        quality_score, quality_score_std = self.post_process_with_feasibility_enforcement(quality_list)
        return quality_score, quality_score_std
    
    def get_utility_list(self):
        eps = 1e-6
        utility_numerator_list = [max(eps, predictive_performance_with_real_train_and_generated - predictive_performance_with_real_train) for predictive_performance_with_real_train_and_generated, predictive_performance_with_real_train in zip(self.predictive_performance_with_real_train_and_generated_list, self.predictive_performance_with_real_train_list)]
        utility_denominator_list = [max(eps, predictive_performance_with_real_train_and_reference - predictive_performance_with_real_train) for predictive_performance_with_real_train_and_reference, predictive_performance_with_real_train in zip(self.predictive_performance_with_real_train_and_reference_list, self.predictive_performance_with_real_train_list)]
        utility_list = [utility_numerator/utility_denominator for utility_numerator, utility_denominator in zip(utility_numerator_list, utility_denominator_list)]
        return utility_list

    def utility(self):
        #utility: training on original data + generated data should yield comparable increase in predictive performance w.r.t. original data on test as original data + data from same distribution
        utility_list = self.get_utility_list()
        utility_score, utility_score_std = self.post_process_with_feasibility_enforcement(utility_list)
        return utility_score, utility_score_std
        
    def get_indistinguishability_list(self):
        indistinguishability_list = [1 - score for score in self.predictive_performance_discriminate_real_from_generated_list]
        return indistinguishability_list

    def indistinguishability(self):
        #indistinguishability: it should be difficult to accurately discriminate between original data and generated data
        indistinguishability_list = self.get_indistinguishability_list()
        indistinguishability_score, indistinguishability_score_std = self.post_process_with_feasibility_enforcement(indistinguishability_list)
        return indistinguishability_score, indistinguishability_score_std

    def get_exchangeability_list(self):
        quality_list = self.get_quality_list() 
        utility_list = self.get_utility_list()
        indistinguishability_list = self.get_indistinguishability_list()
        similarity_list = self.get_similarity_list()
        exchangeability_list = [np.mean([quality, utility])*np.mean([indistinguishability, similarity]) for quality, utility, indistinguishability, similarity in zip(quality_list, utility_list, indistinguishability_list, similarity_list)]
        return exchangeability_list

    def exchangeability(self):
        exchangeability_list = self.get_exchangeability_list()
        exchangeability_score, exchangeability_score_std = self.post_process_with_feasibility_enforcement(exchangeability_list)
        return exchangeability_score, exchangeability_score_std

    def get_creativity_list(self):
        quality_list = self.get_quality_list() 
        utility_list = self.get_utility_list()
        indistinguishability_list = self.get_indistinguishability_list()
        similarity_list = self.get_similarity_list()
        creativity_list = [np.mean([quality, utility])/(1+np.mean([indistinguishability, similarity])) for quality, utility, indistinguishability, similarity in zip(quality_list, utility_list, indistinguishability_list, similarity_list)]
        return creativity_list

    def creativity(self):
        creativity_list = self.get_creativity_list()
        creativity_score, creativity_score_std = self.post_process_with_feasibility_enforcement(creativity_list)
        return creativity_score, creativity_score_std
        
    def score(self):
        quality, quality_std, utility, utility_std,  indistinguishability, indistinguishability_std, similarity, similarity_std = *self.quality(), *self.utility(), *self.indistinguishability(), *self.similarity()
        exchangeability_score, exchangeability_score_std = self.exchangeability()
        creativity_score, creativity_score_std = self.creativity()
        return exchangeability_score, exchangeability_score_std, creativity_score, creativity_score_std, quality, quality_std, utility, utility_std, indistinguishability, indistinguishability_std, similarity, similarity_std

    def scores(self):
        quality, quality_std, utility, utility_std,  indistinguishability, indistinguishability_std, similarity, similarity_std = *self.quality(), *self.utility(), *self.indistinguishability(), *self.similarity()
        scores = np.array([quality, utility, indistinguishability, similarity]) 
        scores_std = np.array([quality_std, utility_std, indistinguishability_std, similarity_std]) 
        exchangeability_score, exchangeability_score_std = self.exchangeability()
        creativity_score, creativity_score_std = self.creativity()
        predictive_performances = [self.predictive_performance_with_real_train_avg, self.predictive_performance_with_generated_avg, self.predictive_performance_with_real_train_and_reference_avg, self.predictive_performance_with_real_train_and_generated_avg, self.predictive_performance_discriminate_real_from_generated_avg]
        predictive_performances_std = [self.predictive_performance_with_real_train_std, self.predictive_performance_with_generated_std, self.predictive_performance_with_real_train_and_reference_std, self.predictive_performance_with_real_train_and_generated_std, self.predictive_performance_discriminate_real_from_generated_std]
        if self.verbose: print_score(exchangeability_score, creativity_score, scores, predictive_performances, exchangeability_score_std, creativity_score_std, scores_std, predictive_performances_std)
        return exchangeability_score, creativity_score, scores, predictive_performances, exchangeability_score_std, creativity_score_std, scores_std, predictive_performances_std


def print_score(exchangeability_score, creativity_score, scores, predictive_performances, exchangeability_score_std, creativity_score_std, scores_std, predictive_performances_std):
    quality, utility, indistinguishability, similarity = scores
    quality_std, utility_std, indistinguishability_std, similarity_std = scores_std
    predictive_performance_with_real_train_avg, predictive_performance_with_generated_avg, predictive_performance_with_real_train_and_reference_avg, predictive_performance_with_real_train_and_generated_avg, predictive_performance_discriminate_real_from_generated_avg = predictive_performances
    predictive_performance_with_real_train_std, predictive_performance_with_generated_std, predictive_performance_with_real_train_and_reference_std, predictive_performance_with_real_train_and_generated_std, predictive_performance_discriminate_real_from_generated_std = predictive_performances_std
    print('real: %.2f+-%.2f   generated: %.2f+-%.2f   real+reference: %.2f+-%.2f   real+generated: %.2f+-%.2f   real_vs_generated: %.2f+-%.2f'%(predictive_performance_with_real_train_avg, predictive_performance_with_real_train_std, predictive_performance_with_generated_avg, predictive_performance_with_generated_std, predictive_performance_with_real_train_and_reference_avg, predictive_performance_with_real_train_and_reference_std, predictive_performance_with_real_train_and_generated_avg, predictive_performance_with_real_train_and_generated_std, predictive_performance_discriminate_real_from_generated_avg, predictive_performance_discriminate_real_from_generated_std))
    print('quality: %.2f+-%.2f   utility: %.2f+-%.2f   indistinguishability: %.2f+-%.2f   similarity: %.2f+-%.2f'%(quality, quality_std, utility, utility_std, indistinguishability, indistinguishability_std, similarity, similarity_std))
    print('exploitable_exchangeability: %.2f+-%.2f   exploitable_creativity: %.2f+-%.2f'%(exchangeability_score, exchangeability_score_std, creativity_score, creativity_score_std))


def concrete_discriminative_generative_quality_score(generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets, n_iterations=10, use_resampling=False, use_replacement=False, fraction=0.7, data_estimator=ExtraTreesClassifier(n_estimators=100, n_jobs=-1), discriminative_performance_func=adjusted_balanced_accuracy_score, verbose=1, parallel=True):
    #verbose=0 no output; verbose=1 only print_score; verbose=2 print_score and print each iteration
    scorer = DiscriminativeGenerativeQualityScorer(
        data_estimator=data_estimator, 
        discriminative_performance_func=discriminative_performance_func, 
        n_rep_estimator=3, 
        n_neighbors=3, 
        n_elements_to_trim=1, 
        metric='cosine', 
        verbose=verbose>=2, 
        parallel=parallel, 
        n_cpus=None, 
        make_adjusted_score=False, 
        enforce_positive_definite=True)
    scorer.input_data(generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets)
    scorer.resample(n_iterations=n_iterations, use_resampling=use_resampling, use_replacement=use_replacement, fraction=fraction)
    scorer.compute_performance_indicators()
    res = scorer.scores()
    exchangeability_score, creativity_score, scores, predictive_performances, exchangeability_score_std, creativity_score_std, scores_std, predictive_performances_std = res
    return exchangeability_score, creativity_score, scores, predictive_performances, exchangeability_score_std, creativity_score_std, scores_std, predictive_performances_std


def concrete_graph_discriminative_generative_quality_score(vectorizer, generated_graphs, generated_targets, real_train_graphs, real_train_targets, real_reference_graphs, real_reference_targets, test_graphs, test_targets, n_iterations=10, verbose=1, parallel=True, positive_label=1):
    generated_instances = vectorizer.fit_transform(generated_graphs)
    real_train_instances = vectorizer.fit_transform(real_train_graphs)
    real_reference_instances = vectorizer.fit_transform(real_reference_graphs)
    test_instances = vectorizer.fit_transform(test_graphs)
    f1_with_label = partial(f1_score, pos_label=positive_label)
    return concrete_discriminative_generative_quality_score(generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets, n_iterations=n_iterations, use_replacement=False, fraction=0.7, data_estimator=ExtraTreesClassifier(n_estimators=100, n_jobs=-1), discriminative_performance_func=f1_with_label, verbose=verbose, parallel=parallel)


def score_rank(scores_list, n_iter=1000, std_correction_factor=.1):

    def sample_scores(locs, stds): 
        return [np.random.normal(loc=loc, scale=std) for loc, std in zip(locs, stds)]

    def compute_is_dominated_mtx(score_mtx):
        #compute if one model is dominated by another model
        #input: each row is a vector of (sampled) scores for one model
        #output: is_dominated_mtx is nxn matrix with 1 in cell i,j if model i is dominated by model j 
        n_models = score_mtx.shape[0]
        is_dominated_mtx = np.zeros((n_models,n_models))
        for i in range(n_models):
            for j in range(n_models):
                is_dominated_mtx[i,j] = int(np.all(score_mtx[i]<=score_mtx[j])) #compare i with j: mark 1 when i is less on *all* objectives
        return is_dominated_mtx

    n_models = len(scores_list)
    is_dominated_mtx = np.zeros((n_models,n_models))
    for it in range(n_iter):
        score_mtx = np.array([sample_scores(locs, np.array(stds)*std_correction_factor) for locs, stds in scores_list])
        is_dominated_mtx += compute_is_dominated_mtx(score_mtx) #add 1 in cell i,j if model i is dominated by model j 
    is_dominated_counts_mtx = np.sum(is_dominated_mtx, axis=1) #cumulative number of times that model i is dominated by any other model
    ranks = sp.stats.rankdata(is_dominated_counts_mtx)
    return ranks

def discriminative_generative_quality_score_rank(scores_list, n_iter=1000, std_correction_factor=.1):
    #input: scores_list format: each row is a 2-tuple for a single model: (quality, utility, indistinguishability, similarity), (quality_std, utility_std, indistinguishability_std, similarity_std)
    #n_iter is the number of samples sampled from the mean+-std of each score
    #output: array with rank order by quality (1=best quality) of each model in same order as in scores_list
    return score_rank(scores_list, n_iter=n_iter, std_correction_factor=std_correction_factor)

def exploitable_exchangeability_and_exploitable_creativity_rank(scores_list, n_iter=1000, std_correction_factor=.1):
    #input: scores_list format: each row is a 2-tuple for a single model: (exploitable_exchangeability, exploitable_creativity), (exploitable_exchangeability_std, exploitable_creativity_std)
    #n_iter is the number of samples sampled from the mean+-std of each score
    #output: array with rank order by quality (1=best quality) of each model in same order as in scores_list
    return score_rank(scores_list, n_iter=n_iter, std_correction_factor=std_correction_factor)
_GP_DATA = None
_GP_CONF = None

def _gp_worker_compute_by_index(idx: int):
    gi_list, gt_list, rti_list, rtt_list, rri_list, rrt_list, ti_list, tt_list = _GP_DATA
    gi = gi_list[idx]
    gt = gt_list[idx]
    rti = rti_list[idx]
    rtt = rtt_list[idx]
    rri = rri_list[idx]
    rrt = rrt_list[idx]
    ti = ti_list[idx]
    tt = tt_list[idx]
    c = _GP_CONF
    similarity_generated_vs_real_train = estimate_instance_set_similarity(
        gi, gt, rti, rtt, metric=c["metric"], n_neighbors=c["n_neighbors"]
    )
    predictive_performances = discriminative_generative_predictive_performances(
        c["data_estimator"], gi, gt, rti, rtt, rri, rrt, ti, tt,
        discriminative_performance_func=c["discriminative_performance_func"],
        n_rep=c["n_rep_estimator"],
    )
    if c.get("verbose"):
        pr = predictive_performances
        print("\t real:%.2f  gen:%.2f  real_and_ref:%.2f  real_and_gen:%.2f  real_vs_gen:%.2f  similarity:%.2f" %
              (pr[0], pr[1], pr[2], pr[3], pr[4], similarity_generated_vs_real_train))
    return (
        similarity_generated_vs_real_train,
        predictive_performances[0],
        predictive_performances[1],
        predictive_performances[2],
        predictive_performances[3],
        predictive_performances[4],
    )


def _gp_worker_compute_from_data(payload):
    # payload: ((gi, gt, rti, rtt, rri, rrt, ti, tt), conf)
    (gi, gt, rti, rtt, rri, rrt_, ti, tt), c = payload
    similarity_generated_vs_real_train = estimate_instance_set_similarity(
        gi, gt, rti, rtt, metric=c["metric"], n_neighbors=c["n_neighbors"]
    )
    predictive_performances = discriminative_generative_predictive_performances(
        c["data_estimator"], gi, gt, rti, rtt, rri, rrt_, ti, tt,
        discriminative_performance_func=c["discriminative_performance_func"],
        n_rep=c["n_rep_estimator"],
    )
    if c.get("verbose"):
        pr = predictive_performances
        print("\t real:%.2f  gen:%.2f  real_and_ref:%.2f  real_and_gen:%.2f  real_vs_gen:%.2f  similarity:%.2f" %
              (pr[0], pr[1], pr[2], pr[3], pr[4], similarity_generated_vs_real_train))
    return (
        similarity_generated_vs_real_train,
        predictive_performances[0],
        predictive_performances[1],
        predictive_performances[2],
        predictive_performances[3],
        predictive_performances[4],
    )



def plot_expected_gain_weighted_equivalent_data_size(res, title=None, show_weights=False):
    sizes = np.asarray(res.get('sizes', []), float)
    util = np.asarray(res.get('originality', []), float)
    # Optional std arrays for shaded bands (several common key names supported)
    def _get_std(res_dict, keys):
        for k in keys:
            if k in res_dict and res_dict[k] is not None:
                try:
                    return np.asarray(res_dict[k], float)
                except Exception:
                    return None
        return None
    util_std = _get_std(res, (
        'originality_std', 'originality_stds', 'originality_stddev', 'originality_error'
    ))
    eds = np.asarray(res.get('equivalent_data_size_ratios', []), float)
    eds_std = _get_std(res, (
        'eds_std', 'equivalent_data_size_ratios_std', 'equivalent_data_size_std', 'eds_ratios_std'
    ))
    weights = np.asarray(res.get('weights', []), float) if show_weights else None
    hwe = res.get('expected_gain_weighted_equivalent_data_size', None)
    params = res.get('eds_params', None)
    if sizes.size == 0:
        raise ValueError('Result dict is missing sizes/originality/eds.')
    order = np.argsort(sizes)
    sizes = sizes[order]
    util = util[order] if util.size else util
    eds = eds[order] if eds.size else eds
    if util_std is not None and util_std.size == util.size:
        util_std = np.asarray(util_std)[order]
    else:
        util_std = None
    if eds_std is not None and eds_std.size == eds.size:
        eds_std = np.asarray(eds_std)[order]
    else:
        eds_std = None
    if weights is not None and weights.size:
        weights = weights[order]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), constrained_layout=True)
    # originality panel
    ax = axes[0]
    # Light gray band between y=0 and y=1
    ax.axhspan(0.0, 1.0, facecolor='lightgray', alpha=0.2, zorder=0)
    # Std fill (if provided); scale down if too large relative to data range.
    util_std_plot = None
    std_scale = 1.0
    if util_std is not None:
        try:
            data_range = float(np.nanmax(util) - np.nanmin(util))
        except Exception:
            data_range = 0.0
        try:
            smax = float(np.nanmax(util_std))
        except Exception:
            smax = 0.0
        if np.isfinite(data_range) and data_range > 0 and np.isfinite(smax) and smax > 0:
            cap = 0.5 * data_range
            if smax > cap:
                std_scale = cap / smax
        util_std_plot = util_std * std_scale
        lo = util - util_std_plot
        hi = util + util_std_plot
        label = (f'originality std × {std_scale:.2e}' if std_scale < 0.999 else None)
        ax.fill_between(sizes, lo, hi, color='#377eb8', alpha=0.2, linewidth=0, zorder=1, label=label)
    ax.plot(sizes, util, marker='o', color='#377eb8')
    ax.set_title('Originality vs. size')
    ax.set_xlabel('sample size')
    ax.set_ylabel('originality')
    # Include std bands in limits if available
    if util_std_plot is not None:
        ymin = min(0.0, float(np.nanmin(util - util_std_plot)) - 0.05)
        ymax = max(1.05, float(np.nanmax(util + util_std_plot)) + 0.05)
    else:
        ymin = min(0.0, np.nanmin(util) - 0.05)
        ymax = max(1.05, np.nanmax(util) + 0.05)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.grid(True, ls=':')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    if util_std_plot is not None and std_scale < 0.999:
        try:
            ax.legend()
        except Exception:
            pass
    # Annotate originality values
    for xi, yi in zip(sizes, util):
        if np.isfinite(yi):
            ax.annotate(f'{yi:.2f}', (xi, yi), textcoords='offset points', xytext=(6,6), ha='left', va='bottom', fontsize=8, color='#377eb8')
    # EDS panel
    ax = axes[1]
    # Light gray band between y=0 and y=1
    ax.axhspan(0.0, 1.0, facecolor='lightgray', alpha=0.2, zorder=0)
    # Std fill (if provided); scale down if too large relative to data range.
    eds_std_plot = None
    eds_std_scale = 1.0
    if eds_std is not None:
        try:
            eds_range = float(np.nanmax(eds) - np.nanmin(eds))
        except Exception:
            eds_range = 0.0
        try:
            esmax = float(np.nanmax(eds_std))
        except Exception:
            esmax = 0.0
        if np.isfinite(eds_range) and eds_range > 0 and np.isfinite(esmax) and esmax > 0:
            eds_cap = 0.5 * eds_range
            if esmax > eds_cap:
                eds_std_scale = eds_cap / esmax
        eds_std_plot = eds_std * eds_std_scale
        lo = eds - eds_std_plot
        hi = eds + eds_std_plot
        eds_label = (f'EDS std × {eds_std_scale:.2e}' if eds_std_scale < 0.999 else None)
        ax.fill_between(sizes, lo, hi, color='black', alpha=0.15, linewidth=0, zorder=1, label=eds_label)
    if weights is not None and weights.size == eds.size:
        # scale marker size for visibility
        ms = 200 * (weights / (weights.max() if weights.max()>0 else 1.0)) + 20
        ax.scatter(sizes, eds, s=ms, color='black', alpha=0.7, zorder=2)
    else:
        ax.plot(sizes, eds, marker='o', color='black', zorder=2)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    if hwe is not None and np.isfinite(hwe):
        ax.axhline(float(hwe), color="#377eb8", linestyle='--', linewidth=1, label=f'EG-Equivalent Data Size={hwe:.2f}')
    ax.set_title('Equivalent Data Size (EDS) vs. size')
    ax.set_xlabel('sample size')
    ax.set_ylabel('eds ratio')
    ax.grid(True, ls=':')
    # Include std bands in limits if available
    if eds_std_plot is not None:
        ymin = min(0.0, float(np.nanmin(eds - eds_std_plot)) - 0.05)
        ymax = max(1.05, float(np.nanmax(eds + eds_std_plot)) + 0.05)
        ax.set_ylim(bottom=ymin, top=ymax)
    # Annotate EDS values
    for xi, yi in zip(sizes, eds):
        if np.isfinite(yi):
            ax.annotate(f'{yi:.2f}', (xi, yi), textcoords='offset points', xytext=(6,6), ha='left', va='bottom', fontsize=8, color='black')
    try:
        ax.legend()
    except Exception:
        pass
    if title:
        fig.suptitle(title)
    return fig, axes


def _resolve_scorer(scoring, pos_label=1):
    if callable(scoring):
        return scoring
    s = 'f1' if scoring is None else str(scoring).lower()
    if s in ('f1','f1_score'):
        # Default: binary F1 with pos_label (common in the rest of the codebase).
        # Callers who need macro-F1 or another averaging can pass a callable.
        from functools import partial as _partial
        return _partial(f1_score, pos_label=pos_label)
    if s in ('balanced_accuracy','balanced_accuracy_score','balanced'):
        return balanced_accuracy_score
    # Fallback to F1 with the given positive label
    from functools import partial as _partial
    return _partial(f1_score, pos_label=pos_label)

def _clone_estimator(est, *, random_state=None, n_jobs=None):
    try:
        from sklearn.base import clone as sk_clone
        new_est = sk_clone(est)
    except Exception:
        import copy as _copy
        new_est = _copy.deepcopy(est)
    if random_state is not None:
        try:
            new_est.set_params(random_state=int(random_state))
        except Exception:
            if hasattr(new_est, 'random_state'):
                new_est.random_state = int(random_state)
    if n_jobs is not None:
        try:
            new_est.set_params(n_jobs=int(n_jobs))
        except Exception:
            if hasattr(new_est, 'n_jobs'):
                new_est.n_jobs = int(n_jobs)
    return new_est

def _embed_pools(vec, pools_in_order):
    # Fit vec on concatenated graphs to ensure consistent features, then split back per pool.
    try:
        from sklearn.base import clone as sk_clone
        vec_fit = sk_clone(vec)
    except Exception:
        vec_fit = vec
    graphs_concat = []
    lengths = []
    for graphs in pools_in_order:
        graphs_concat.extend(graphs)
        lengths.append(len(graphs))
    X_all = vec_fit.fit_transform(graphs_concat)
    X_all = np.asarray(X_all)
    outs = []
    start = 0
    for L_ in lengths:
        outs.append(X_all[start:start+L_])
        start += L_
    return outs

def _score_from_arrays(X_tr, y_tr, X_te, y_te, *, clf, seed=0, n_jobs=1, scoring=None):
    est = _clone_estimator(clf, random_state=seed, n_jobs=n_jobs) if 'ExtraTreesClassifier' in str(type(clf)) else clf
    y_pred = est.fit(X_tr, y_tr).predict(X_te)
    scorer = _resolve_scorer(scoring)
    return scorer(y_te, y_pred)

def _fit_powerlaw_params(ns, y_real, y_ref, alphas=None):
    # Fit f(n) = a + b*n^{-alpha} by grid search over alpha with least squares for a,b.
    ns = np.asarray(ns, float)
    y1 = np.asarray(y_real, float)
    y2 = np.asarray(y_ref, float)
    n_all = np.concatenate([ns, 2.0*ns])
    y_all = np.concatenate([y1, y2])
    if alphas is None:
        # Avoid extremely small alphas which can amplify inversion noise for EDS
        alphas = np.linspace(0.3, 2.0, 171)
    best = (np.nan, np.nan, 1.0); best_sse = None
    for aexp in alphas:
        x = n_all**(-aexp)
        X = np.stack([np.ones_like(x), x], axis=1)
        try:
            coeff, *_ = np.linalg.lstsq(X, y_all, rcond=None)
        except Exception:
            continue
        a0, b0 = coeff
        resid = y_all - (a0 + b0*x)
        sse = float(np.dot(resid, resid))
        if best_sse is None or sse < best_sse:
            best_sse = sse; best = (float(a0), float(b0), float(aexp))
    return best

def _eds_from_params(ns, y_gen, params, *, delta_rel_floor: float = 1e-3, clip_ratio: float | None = None):
    # Invert f(n) at y_gen to get n_eq, then r = (n_eq - n)/n. Handles sign(b).
    a0, b0, alpha = params
    ns = np.asarray(ns, float)
    y_gen = np.asarray(y_gen, float)
    dalpha = max(float(alpha), 1e-8)
    # Use a relative floor on delta to avoid massive blow-ups when y_gen ~ a0
    if b0 < 0:
        delta = np.maximum(a0 - y_gen, 0.0)
    else:
        delta = np.maximum(y_gen - a0, 0.0)
    scale = max(abs(b0), abs(a0), 1.0)
    delta = np.maximum(delta, float(delta_rel_floor) * scale)
    n_eq = (abs(b0) / delta) ** (1.0 / dalpha)
    r = (n_eq - ns) / np.maximum(ns, 1e-8)
    if clip_ratio is not None:
        r = np.minimum(r, float(clip_ratio))
    return r

def expected_gain_weights(ns, params, beta=1.0, drop_q=None):
    a0, b0, alpha = params
    ns = np.asarray(ns, float)
    f = lambda x: a0 + b0*np.power(x, -alpha)
    delta = np.abs(f(2.0*ns) - f(ns))
    if drop_q is not None and 0 < float(drop_q) < 1:
        thr = np.nanquantile(delta, float(drop_q))
        delta = np.where(delta > thr, delta, 0.0)
    w = np.power(delta, float(beta))
    s = np.sum(w)
    return w/ s if s > 0 else np.zeros_like(w)

def compute_expected_gain_weighted_equivalent_data_size(
    generated_graphs, generated_targets,
    train_graphs, train_targets,
    reference_graphs, reference_targets,
    test_graphs, test_targets,
    *,
    vectorizer=None, classifier=None,
    fracional_size=(1,2,3,4,5,7,10,15),
    n_repeats=3,
    scoring='f1',
    pos_label=1,
    beta=1.0,
    expected_gain_drop_q=0.1,
    backend='loky',
    n_jobs=-1,
    random_state=0,
    eds_delta_rel_floor: float = 1e-3,
    eds_clip_ratio: float | None = None,
):
    """Evaluate per-size originality and overall Expected-Gain-weighted
    Equivalent Data Size (EDS).

    This function is self-contained: it vectorizes graphs once, samples array
    subsets for each (size, repeat), evaluates a classifier on a fixed
    external test set, fits a power-law learning curve on real vs. real+reference,
    computes EDS per size, and aggregates to an Expected-Gain-weighted EDS.

    EDS notion and computation:
    We interpret the performance achieved after adding generated data at size n
    as being equivalent to training with some larger amount of real data. We
    first fit a baseline learning curve on real data and a matched reference
    augmentation, f(n) = a + b · n^(-α). For the observed performance y_gen at
    size n (real + generated), we invert the baseline to find the equivalent
    real-data size n_eq such that f(n_eq) = y_gen. The EDS ratio is then
    r(n) = (n_eq - n) / n, which measures the value of the generated batch in
    “units of an equally sized real batch”: r ≈ 1 means a generated batch is as
    valuable as a same-size real batch; r < 1 means weaker; r > 1 means stronger;
    r < 0 indicates degradation. Inversion handles the sign of b correctly so
    that n_eq is well-defined near the curve's asymptote.

    Args:
        generated_graphs/targets: candidate generated pool and labels.
        train_graphs/targets: real training pool and labels.
        reference_graphs/targets: real reference pool and labels.
        test_graphs/targets: fixed external test set and labels.
        vectorizer: vectorizer object exposing fit_transform/transform. Must be provided.
        classifier: sklearn-compatible classifier exposing fit/predict. Must be provided.
        fracional_size: tuple of divisors to derive sizes = len(train)/div.
        n_repeats: repeats per size.
        scoring: 'f1' | 'balanced_accuracy' or a callable(y_true, y_pred).
        pos_label: positive class for binary F1; macro-F1 is used otherwise.
        beta: Expected-Gain weight exponent.
        expected_gain_drop_q: drop bottom-q Expected-Gain fraction (0 disables).
        backend: joblib backend ('loky' for processes, 'threading' for threads).
        n_jobs: parallel workers for the outer loop (use -1 for all cores).
        random_state: base seed.

    Returns:
        dict with keys:
        - sizes
        - originality
        - originality_std (per-size std across repeats)
        - expected_gain_weighted_equivalent_data_size
        - equivalent_data_size_ratios (EDS per size)
        - eds_std (per-size std across repeats)
        - equivalent_data_size_params
        - weights
    """
    # 1) Defaults and scorer resolution
    if vectorizer is None:
        raise ValueError("vec must be provided (fit_transform/transform methods required)")
    if classifier is None:
        raise ValueError("clf must be provided (fit/predict methods required)")
    scorer = _resolve_scorer(scoring, pos_label=pos_label)

    # 2) Derive evaluation sizes
    n_train = len(train_graphs)
    sizes = [max(1, int(n_train/float(f))) for f in fracional_size][::-1]

    # 3) Embed all pools once using a single vectorizer fit for consistent features
    X_train_pool, X_gen_pool, X_ref_pool, X_test = _embed_pools(vectorizer, [train_graphs, generated_graphs, reference_graphs, test_graphs])
    y_train_pool = np.asarray(train_targets)
    y_gen_pool = np.asarray(generated_targets)
    y_ref_pool = np.asarray(reference_targets)
    y_test = np.asarray(test_targets)

    # 4) Build the full (size, repeat) task list
    base = int(random_state)
    def _sample_idx(n_pool, n, seed):
        r = np.random.default_rng(int(seed))
        if n >= n_pool:
            return r.choice(n_pool, size=n, replace=True)
        return r.choice(n_pool, size=n, replace=False)

    def _run_task(i_sz, n, seed):
        # Sample per-pool arrays
        idx_tr = _sample_idx(len(X_train_pool), n, seed)
        idx_ge = _sample_idx(len(X_gen_pool),   n, seed+1)
        idx_rf = _sample_idx(len(X_ref_pool),   n, seed+2)
        X_tr = X_train_pool[idx_tr]; y_tr = y_train_pool[idx_tr]
        X_trg = np.vstack([X_tr, X_gen_pool[idx_ge]])
        y_trg = np.concatenate([y_tr, y_gen_pool[idx_ge]])
        X_trr = np.vstack([X_tr, X_ref_pool[idx_rf]])
        y_trr = np.concatenate([y_tr, y_ref_pool[idx_rf]])
        # Fit/predict using single-threaded clones to avoid nested oversubscription
        s_r  = _score_from_arrays(X_tr,  y_tr,  X_test, y_test, clf=classifier, seed=seed,    n_jobs=1, scoring=scorer)
        s_rg = _score_from_arrays(X_trg, y_trg, X_test, y_test, clf=classifier, seed=seed+7,  n_jobs=1, scoring=scorer)
        s_rr = _score_from_arrays(X_trr, y_trr, X_test, y_test, clf=classifier, seed=seed+13, n_jobs=1, scoring=scorer)
        return i_sz, s_r, s_rg, s_rr

    tasks = []
    for i_sz, n in enumerate(sizes):
        for rep in range(int(n_repeats)):
            seed = base + 1000*i_sz + rep
            tasks.append((i_sz, n, seed))

    # 5) Execute tasks in parallel and aggregate by size (trimmed mean across repeats)
    parallel_n = int(n_jobs) if int(n_jobs) != 0 else -1
    results = Parallel(n_jobs=parallel_n, backend=str(backend))(delayed(_run_task)(i_sz, n, sd) for (i_sz, n, sd) in tasks)
    by_size_r  = [[] for _ in sizes]
    by_size_rg = [[] for _ in sizes]
    by_size_rr = [[] for _ in sizes]
    for i_sz, s_r, s_rg, s_rr in results:
        by_size_r[i_sz].append(s_r); by_size_rg[i_sz].append(s_rg); by_size_rr[i_sz].append(s_rr)

    y_real, y_gen, y_ref, originalities = [], [], [], []
    originality_stds = []
    eps = 1e-6
    for i_sz in range(len(sizes)):
        # Use scipy trimmed mean/std with 10% symmetric trimming
        m_r  = float(trimmed_mean(np.asarray(by_size_r[i_sz], dtype=float),  limits=(0.1, 0.1)))
        m_rg = float(trimmed_mean(np.asarray(by_size_rg[i_sz], dtype=float), limits=(0.1, 0.1)))
        m_rr = float(trimmed_mean(np.asarray(by_size_rr[i_sz], dtype=float), limits=(0.1, 0.1)))
        y_real.append(m_r); y_gen.append(m_rg); y_ref.append(m_rr)
        originalities.append((m_rg - m_r)/max(eps, m_rr - m_r) if (np.isfinite(m_r) and np.isfinite(m_rr)) else float("nan"))
        # Per-repeat originality std (using per-repeat ratios)
        r_list = []
        for sr, srg, srr in zip(by_size_r[i_sz], by_size_rg[i_sz], by_size_rr[i_sz]):
            denom = max(eps, float(srr) - float(sr))
            r_list.append((float(srg) - float(sr)) / denom)
        if len(r_list) >= 2:
            try:
                r_std = float(trimmed_std(np.asarray(r_list, dtype=float), limits=(0.1, 0.1)))
            except Exception:
                r_std = float(np.nanstd(np.asarray(r_list, dtype=float)))
        else:
            r_std = 0.0
        originality_stds.append(r_std)

    # 6) Fit power law and compute EDS per size
    params = _fit_powerlaw_params(sizes, y_real, y_ref)
    eds = _eds_from_params(sizes, y_gen, params, delta_rel_floor=eds_delta_rel_floor, clip_ratio=eds_clip_ratio)
    # Per-size EDS std via variability of y_gen across repeats (params fixed)
    eds_stds = []
    for i_sz, n in enumerate(sizes):
        eds_rep = []
        for srg in by_size_rg[i_sz]:
            try:
                ed_val = _eds_from_params(
                    np.asarray([n], float),
                    np.asarray([float(srg)], float),
                    params,
                    delta_rel_floor=eds_delta_rel_floor,
                    clip_ratio=eds_clip_ratio,
                )[0]
            except Exception:
                ed_val = float('nan')
            eds_rep.append(float(ed_val))
        if len(eds_rep) >= 2:
            try:
                e_std = float(trimmed_std(np.asarray(eds_rep, dtype=float), limits=(0.1, 0.1)))
            except Exception:
                e_std = float(np.nanstd(np.asarray(eds_rep, dtype=float)))
        else:
            e_std = 0.0
        eds_stds.append(e_std)

    # 7) Expected-Gain-weighted aggregation
    w = expected_gain_weights(sizes, params, beta=beta, drop_q=expected_gain_drop_q)
    hwe = float(np.sum(w*eds)) if np.sum(w) > 0 else float("nan")

    # 8) Return result dictionary (expanded keys included)
    return {
        'sizes': sizes,
        'originality': originalities,
        'originality_std': originality_stds,
        'expected_gain_weighted_equivalent_data_size': hwe,
        'equivalent_data_size_ratios': eds.tolist(),
        'eds_std': eds_stds,
        'equivalent_data_size_params': tuple(params),
        'weights': w.tolist(),
    }
