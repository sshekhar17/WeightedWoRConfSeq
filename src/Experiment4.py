"""
Comparison of propM, propM+CV, and uniform methods on a 'semi-real-world' data
with transactions involving House prices
Data Souce: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
"""
import argparse
from time import time 
from tqdm import tqdm
from math import cos, pi 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tikzplotlib as tpl

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from weightedCSsequential import run_one_expt
from utils import first_threshold_crossing 
from ExperimentBase import *
from constants import ColorsDict, PATH_TO_HOUSING_DATA


class HousingTransactionsAuditor():
    def __init__(self, path_to_data, model=None, train_frac1=0.1, train_frac2=0.1, 
                 prob_error=0.9, f_max=0.7, figname=None, 
                 num_estimators_gt=50):
        """
            Initialize the HousingTransactionsAuditor Class 

            Parameters: 
                path_to_data        :path to the csv file 
                model               :ML model with .fit and .predict methods 
                                        to be used for generating side-info
                train_frac1         :float \in [0,1], fraction of data to be used for generating 
                                        the ground truth using Random Forest 
                train_frac2         :float \in [0,1], fraction of data to be used for training 
                                        model for generating side-info
                prob_error          :float \in [0,1], probability that a transaction has a 
                                        misspecified reported value 
                f_max               :float \in (0,1), an upper bound on the true relative 
                                        misspecified fraction values 
                figname             :string, denoting the name of the figure
                num_estimators_gt     :int, number of trees in the random forest for genrating 
                                        the ground truth 
        """
        self.path_ = path_to_data 
        self.train_frac1=train_frac1
        self.train_frac2=train_frac2
        self.prob_error = prob_error 
        self.f_max = f_max 
        self.num_estimators_gt = num_estimators_gt
        if model is None: 
            self.model = DecisionTreeRegressor()
        else: 
            self.model = model 
        if figname is None:
            self.figname = "../data/Stopping_Times_Housing"
        else:
            self.figname = figname 
        # extract the data from given path 
        self.extract_data()
        # create splits in data for training 
        self.split_data()
        # create the ground truth 
        self.create_ground_truth()
        # create side information 
        self.generate_side_info()
    
    def extract_data(self):
        """
            Extract data from the csv file, and create
            self.X and self.M attributes 
        """
        df = pd.read_csv(self.path_)
        cols = list(df.columns)
        cols.remove('price')
        cols.remove('date')
        X, y = df[cols], df['price']
        # store the extracted information 
        self.X = X 
        self.M = y.to_numpy()

    def split_data(self):
        """
            Split the entire dataset (self.X, self.M) into three parts: 
                * (self.Xtrain1, self.Mtrain1): the first for training a random 
                    forest regressor for generating the ground truth,
                * (self.Xtrain2, self.Mtrain2): the second for training the `model' 
                    for generating the side-information,
                * (self.Xtest, self.Mtest): and the third part for testing our
                    methods for constructing CSs   
        """
        train_frac1, train_frac2=self.train_frac1, self.train_frac2
        # first split for training the `ground truth`
        Xtr1, X2, ytr1, y2 = train_test_split(self.X, self.M, test_size=1-train_frac1)
        self.Xtrain1 = Xtr1
        self.Mtrain1 = ytr1
        # second split for training the AI predictor 
        Xtr2, X3, ytr2, y3 = train_test_split(X2, y2, test_size=1-train_frac2)
        self.Xtrain2 = Xtr2 
        self.Mtr2 = ytr2
        # store the remaining observations for testing  
        self.Xtest = X3 
        self.Mtest = y3 
    
    def create_ground_truth(self):
        """ 
            Generate the ground truth true misspecified fraction 
            values (i.e., f-values) by training a random forest 
            classifier by arbitrarily labelling self.Xtrain1  
        """
        prob_error, f_max = self.prob_error, self.f_max
        # Step1: define a relation between M and (average) true f-values
            # to do this, we first find 20 equally spaced points 
            # withing the range of M values in self.M
        logbins = np.geomspace(self.M.min(), self.M.max(), 20)
        f_intervals = {}
            # to each value in the set 'logbins', we assign an 
            # interval. The interval associated with any value (l)
            # in the set `logbins` represents the range in 
            # which the f-values should lie for all M-values 
            # whose closest element in logbins is 'l'. 
        for i, l in enumerate(logbins):
            a = (i/len(logbins)) 
            # the relation between mean f-values and l  
            mean = 1e-3 + f_max*(cos(4*pi*a)**2)
            f_intervals[l] = [max(0, mean-0.05), mean+0.05]
        # Step 2: generate the f-values for training 
        f_vals = np.zeros(self.Mtrain1.shape)
        for i, m in enumerate(self.Mtrain1): 
            # find the closest element in logbins 
            closest_val = logbins[np.abs(logbins-m).argmin()]
            l_, u_ = f_intervals[closest_val]
            # generate the corresponding element in the interval [l_, u_]
            if np.random.random()<=prob_error:
                f_vals[i] = np.random.random()*(u_-l_) + l_
            else:
                f_vals[i] = np.random.random()*(1e-5)
        # Step3: train a Random forest model on the labelled data 
            # choose a large number of estimators (>=50). Adding more 
            # estimators, or increasing `train_frac1', will lead to a 
            # longer training time. 
        RF = RandomForestRegressor(n_estimators=self.num_estimators_gt,
                                    criterion='absolute_error')
        RF.fit(self.Xtrain1, f_vals)
        # Step4: use the trained model to generate the ground truth 
        self.ftrain2 = RF.predict(self.Xtrain2) 
        self.ftest = RF.predict(self.Xtest)

    def generate_side_info(self):
        """
        Generate the side-information by training the ML model
        (self.model) on the second training data (self.Xtrain2, self.ftrain2)            
        """
        try:
            if not hasattr(self, 'model') or self.model is None:
                print("No model initialized: fall back to DecisionTree")
                self.model = DecisionTreeRegressor()
            # train the model on the second training data 
            self.model.fit(self.Xtrain2, self.ftrain2) 
            # use it to create side information for testing 
            self.Stest = self.model.predict(self.Xtest) 
        except AttributeError:
            raise Exception("First call self.create_ground_truth")

    def one_trial(self, M=None, f=None, S=None, N=300, logical_CS=True, 
                    intersect=True, lambda_max=1.5, beta_max=1.5, alpha=0.05, 
                    epsilon=0.05):
        """
            Run one trial of the experiment comparing the three methods (propM, 
            propM+CV, and uniform) 

            Parameters: 
                M, f, S         :optional, parameters describing the problem instance
                N               :int, number of transactions. 
                                    must be smaller than (1-train_frac1-trainfrac2)*len(self.X)
                                    if M, f, S are None. Otherwise it is set to len(M)
                logical_CS      :bool, if true, use the logical_CS in all three methods
                intersect       :bool, if true, intersect the CSs 
                lambda_max      :float, largest value of bets to be used
                beta_max        :float, largest weight to be given to the control 
                                        variates
                alpha           :float in (0, 1), confidence level of the CS                                        
                epsilon         :float in (0,1), level of accuracy, used for computing 
                                    the stopping times. That is, we stop sampling the 
                                    first time the width of the CS falls below epsilon.
            Returns: 
                L, U, and ST for the three methods (propM, propM+CV, and uniform)
                    L           :numpy array, representing the lower end point of the CS
                    U           :numpy array, representing the upper end point of the CS
                    ST          :int, time for the width of CS (i..e, U-L) to fall below 
                                    epsilon
        """
        if M is None: 
            ratio = min(1, (N+50)/len(self.Mtest)) # slightly larger than needed
            _, M, _, f, _, S = train_test_split(self.Mtest, self.ftest, self.Stest, test_size=ratio) 
            M, f, S = M[:N], f[:N], S[:N]
        else:
            N = len(M) 
        # obtain the results of the three methods (propM, propM+CV, uniform)
        result_propM= run_one_expt(M, f, S, method_name='propM', cs='Bet', lambda_max=lambda_max, beta_max=beta_max, nG=100,
                        use_CV=False, f_over_S_range=None, alpha=alpha, logical_CS=logical_CS, intersect=intersect,
                        return_payoff=False, lambda_strategy=None, cv_max=np.inf, seed=None, 
                        betting_method='kelly')
        result_uniform= run_one_expt(M, f, S, method_name='uniform', cs='Bet', lambda_max=lambda_max, beta_max=beta_max, nG=100,
                        use_CV=False, f_over_S_range=None, alpha=alpha, logical_CS=logical_CS, intersect=intersect,
                        return_payoff=False, lambda_strategy=None, cv_max=np.inf, seed=None, 
                        betting_method='kelly')
        result_CV= run_one_expt(M, f, S, method_name='propM', cs='Bet', lambda_max=lambda_max, beta_max=beta_max, nG=100,
                        use_CV=True, f_over_S_range=None, alpha=alpha, logical_CS=logical_CS, intersect=intersect,
                        return_payoff=False, lambda_strategy=None, cv_max=np.inf, seed=None, 
                        betting_method='kelly')
        # extract the information from the result tuples  
        LpropM, UpropM, STpropM = self.extract_info(result_propM, epsilon)
        Luni, Uuni, STuni = self.extract_info(result_uniform, epsilon)
        Lcv, Ucv, STcv = self.extract_info(result_CV, epsilon)
        return LpropM, UpropM, STpropM, Luni, Uuni, STuni, Lcv, Ucv, STcv

    def extract_info(self, result, epsilon=0.05): 
        """
        Extract the CS and stopping time from the output of 
            `run_one_expt' function 

        Parameters: 
            result:     the output of the funtion `run_one_expt' 
            epsilon:    the threshold for computing the stopping time  
        
        Returns: 
            LowerCS:    numpy array containing the lower end points of 
                            the confidence sequence constructed by 
                            a call to `run_one_expt' function 
            UpperCS:    numpy array containing the upper end points
            st:         int, denoting the stopping time at which the 
                            width of the CS first fell below the 
                            threshold epsilon. 
        """
        grid, Wealth, LowerCS, UpperCS, Transaction_Indices, Error_flag, diagnostics = result
        Width = UpperCS - LowerCS 
        st = first_threshold_crossing(Width,
                                    th=epsilon,
                                    max_time=LowerCS.size,
                                    upward=False)
        return LowerCS, UpperCS, st 

    def get_stopping_times(self, N=300, num_trials=100):
        """
            get the stopping times of the three methods (propM, propM+CV, uniform)
            by running `num_trials` trials of the method, `one_step` 
        """
        # how many trials can be done with one set of data 
        nt1 = len(self.Mtest)//N 
        # initialize 
        StoppingTimes_propM = np.zeros((num_trials,))
        StoppingTimes_uni = np.zeros((num_trials,))
        StoppingTimes_CV = np.zeros((num_trials,))
        # shuffle the data 
        n_ = len(self.Mtest) 
        perm = np.random.permutation(n_) 
        M_, f_, S_ = self.Mtest[perm], self.ftest[perm], self.Stest[perm] 
        for i in tqdm(range(num_trials)):
            if (i+1)%nt1==0:
                #create splits in data for training 
                self.split_data()
                #create the ground truth 
                self.create_ground_truth()
                #create side information 
                self.generate_side_info()
                n_ = len(self.Mtest) 
                perm = np.random.permutation(n_) 
                M_, f_, S_ = self.Mtest[perm], self.ftest[perm], self.Stest[perm] 
            else: 
                i_ = i%nt1 
                M, f, S = M_[i_*N:(i_+1)*N], f_[i_*N:(i_+1)*N], S_[i_*N:(i_+1)*N]
                result = self.one_trial(M=M, f=f, S=S, N=N)
                _, _, st_propM, _, _, st_uni, _, _, st_cv = result 
                StoppingTimes_propM[i] = st_propM
                StoppingTimes_uni[i] = st_uni
                StoppingTimes_CV[i] = st_cv
        # store the results 
        self.StoppingTimes_propM = StoppingTimes_propM
        self.StoppingTimes_uni = StoppingTimes_uni
        self.StoppingTimes_CV = StoppingTimes_CV
    
    def plot_histogram(self, save_fig=False):
        """
            Plot the results obtained by calling the method 
            `self.get_stopping_times'
        """
        assert hasattr(self, 'StoppingTimes_propM')
        assert hasattr(self, 'StoppingTimes_uni')
        assert hasattr(self, 'StoppingTimes_CV')
        color0 = ColorsDict['propM']
        color1 = ColorsDict['propM+CV']
        color2 = ColorsDict['uniform+logical']
        # plot the results 
        plt.figure() 
        plt.hist(self.StoppingTimes_propM, density=True, label='propM', alpha=0.4,
                    color=color0)
        plt.hist(self.StoppingTimes_CV, density=True, label='propM+CV', alpha=0.4, 
                    color=color1)
        plt.hist(self.StoppingTimes_uni, density=True, label='uniform', alpha=0.4, 
                    color=color2)
        rho = self.compute_correlation(return_value=True)
        plt.title(f'Stopping Time Distribution (correlation={rho:.2f})', fontsize=15)
        plt.xlabel('Stopping Times', fontsize=13)
        plt.ylabel('Density', fontsize=13)
        plt.legend()
        # save the figure, if specified
        if save_fig:
            plt.savefig(self.figname+'.png', dpi=450)
            tpl.save(self.figname + '.tex',axis_width=r'\figwidth',
                     axis_height=r'\figheight')
        else:
            plt.show()
 
    def compute_correlation(self, return_value=False):
        """
            Compute the correlation between the side-information (self.Stest) 
            and the ground truth (self.ftest).   
        """
        assert hasattr(self, 'ftest')
        assert hasattr(self, 'Stest')
        f, S = self.ftest, self.Stest
        numerator = np.mean((f-f.mean())*(S-S.mean()))
        denominator = f.std()*S.std()
        if denominator==0:
            print("Undefined")
            correlation = None
        else: 
            correlation = numerator / denominator 
        print(f"\n Correlation between ground truth and side-info: {correlation:.2f}")
        if return_value: 
            return correlation 


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', '-rs', type=int, default=None, help='random seed')
    parser.add_argument('--save_fig', '-s', action='store_true', help='flag to save figures')
    parser.add_argument('-N', type=int, default=250, help='number of transactions')
    parser.add_argument('--num_trials','-nt',  type=int, default=250, help='number of transactions')
    parser.add_argument('--model_type','-mt', choices=('DT', 'RF') , default='DT', 
                        help='choose between decision-tree or random-forest for generating the side information')
    parser.add_argument('--num_estimators', '-ne', type=int, default=10, 
                        help='number of trees, if model_type==RF')
    parser.add_argument('--num_estimators_gt', '-negt', type=int, default=50, 
                        help='number of trees, for the random-forest used for generating the ground truth')
    args = parser.parse_args()

    random_seed = args.random_seed 
    save_fig = args.save_fig 
    N = args.N 
    num_trials = args.num_trials 
    model_type = args.model_type 
    num_estimators = args.num_estimators 
    num_estimators_gt = args.num_estimators_gt
    # set the random seed 
    if random_seed is None:
        random_seed = (int(time())*2953)%10000
    np.random.seed(random_seed)
    print(f"Seed is {random_seed}")
    if save_fig:
        figname = f"../data/Stopping_Times_Housing_seed_{random_seed}"
    else: 
        figname = None 
    if model_type=='DT':
        model = DecisionTreeRegressor()
    elif model_type=='RF':
        model = RandomForestRegressor(n_estimators=num_estimators,
                                       criterion='absolute_error')
    else: 
        print("Only choices for model are DT and RF: falling back to DT!!")
        model = DecisionTreeRegressor()
    # initialize the auditor 
    Auditor = HousingTransactionsAuditor(PATH_TO_HOUSING_DATA,
                                           model=model, 
                                           figname=figname, 
                                           num_estimators_gt=num_estimators_gt)
    Auditor.get_stopping_times(N=N, num_trials=num_trials)
    Auditor.plot_histogram(save_fig=save_fig)

    ST_propM_ = Auditor.StoppingTimes_propM.mean()
    ST_CV_ = Auditor.StoppingTimes_CV.mean()
    print(f"propM: {ST_propM_:.2f}, \t CV: {ST_CV_:.2f}")
    Auditor.compute_correlation()