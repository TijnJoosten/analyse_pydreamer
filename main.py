from cProfile import label
from email.mime import base
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from functools import reduce
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def multi_conv(*arrs, conv = lambda x, y: np.convolve(x, y, mode='full')):
    return reduce(conv, arrs)



def open_file(file_name: str) -> pd.DataFrame:
    data = []
    with open(file_name) as file_in:
        for line in file_in:
            data.append(line.replace('\n', '').split(' '))
    df = pd.DataFrame(data, columns=['time', 'loss', 'step'])

    df = df.astype({'loss': 'float', 'step': 'int32'})
    df['loss'] = df['loss'][df['loss'] < df['loss'].mean() + 10 * df['loss'].std()] 

    return  df

def load_from_csv(file_name: str) ->  pd.DataFrame:
    return pd.read_csv(file_name).iloc[:,1:]


def plot_fitted_line(df: pd.DataFrame):
    plt.plot(df.step, df.iloc[:,1:-1], alpha = 0.35)
    plt.plot(df.step, df['mean'], c='r', label='mean')
    fitted = np.polyfit(df.step, df['mean'], 4)
    linear_model_fn=np.poly1d(fitted)
    # plt.plot(df.step, linear_model_fn(df.step), c='k')
    # plt.fill_between(df.step, df['mean'] - df['stderr'], df['mean'] + df['stderr'], alpha = 0.35)
    plt.legend()
    plt.show()


def plot_all_data(df: pd.DataFrame):
    plt.plot(df.step, df.iloc[:,1:-1], alpha = 0.35)
    plt.plot(df.step, df['mean'], c='r', label='mean')
    plt.fill_between(df.step, df['mean'] - df['stderr'], df['mean'] + df['stderr'], alpha = 0.35)
    plt.legend()
    plt.show()

def compare_two_runs(df1: pd.DataFrame, df2: pd.DataFrame):
    plt.plot(df1.step, df1['mean'], c='b', label='dreamer')
    plt.fill_between(df1.step, df1['mean'] - df1['stderr'], df1['mean'] + df1['stderr'], alpha = 0.35)
    fitted = np.polyfit(df1.step, df1['mean'], 4)
    linear_model_fn=np.poly1d(fitted)
    plt.plot(df1.step, linear_model_fn(df1.step), c='k')

    plt.plot(df2.step, df2['mean'], c='r', label='inst')
    plt.fill_between(df2.step, df2['mean'] - df2['stderr'], df2['mean'] + df2['stderr'], alpha = 0.35)
    fitted = np.polyfit(df2.step, df2['mean'], 4)
    linear_model_fn=np.poly1d(fitted)
    plt.plot(df2.step, linear_model_fn(df2.step), c='k')
    plt.legend()
    plt.show()  

def compare_runs(dfs: List[pd.DataFrame], ylabel: str, title: str):
    shortest_line = dfs[0][0].step.max()
    for df, name in dfs:
        # plt.plot(df.step, df['mean'], label=name, alpha = 0.35)
        
        # plt.fill_between(df.step, df['mean'] - df['stderr'], df['mean'] + df['stderr'], alpha = 0.35)
        # df.step = df.step.apply(lambda x: x + np.random.rand()/10000.0)
        xnew = np.linspace(df.step.min(), df.step.max(), 5000)  
        power_smooth = gaussian_filter1d(df['mean'], sigma=2)
        fitted = np.polyfit(df.step, power_smooth, 10)
        linear_model_fn=np.poly1d(fitted)
        
        y_err = df['mean'] * np.sqrt(1/len(df['mean']) + (df['mean'] - df['mean'].mean())**2 / np.sum((df['mean'] - df['mean'].mean())**2))
        plt.plot(xnew, linear_model_fn(xnew), label=f"{name}")
        plt.fill_between(df.step, linear_model_fn(df.step) - y_err, linear_model_fn(df.step) + y_err, alpha=0.2)

        if df.step.max() < shortest_line:
            shortest_line = df.step.max()

    plt.xlim((0, shortest_line))
    plt.xlabel('Ingame steps')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()  

def get_data(base_path: str, input_files: List[str], metrics: str) -> pd.DataFrame:
    do_stuff = True
    full_df = None
    complete_files = [base_path + name + metrics for name in input_files]
    for file_name in complete_files:
        if do_stuff:
            full_df = open_file(file_name).drop('time', axis=1)
            full_df = full_df[['step', 'loss']]
            do_stuff = False
        else: 
            full_df = pd.merge(full_df, open_file(file_name).drop('time', axis=1), how='outer', on='step')

    full_df = full_df.sort_values(by=['step'])

    return full_df

def calc_metrics(df: pd.DataFrame):
    df['mean'] = df.iloc[:,1:].mean(axis=1)
    df['stderr'] = df.iloc[:,1:-1].std(axis=1)
    # df = df.fillna(df.mean())
    # poly = PolynomialFeatures(degree = 5)
    # X_poly = poly.fit_transform(df.step.to_numpy().reshape((df.step.size, 1)))
    # print(X_poly)
    # print(X_poly.shape)
    # print(df['mean'].shape)
    # # poly.fit(X_poly, df.iloc[:, 1:-2])
    # lin2 = LinearRegression()
    # lin2.fit(X_poly, df['mean'])

    # X_grid = np.arange(min(df.step),max(df.step),0.1)
    # X_grid = X_grid.reshape(len(X_grid),1) 

    # plt.plot(df.step, df['mean'])
    # plt.plot(df.step, lin2.predict(poly.fit_transform(df.step.to_numpy().reshape((df.step.size, 1)))), color = 'red')
    # plt.show()

def save_data(df: pd.DataFrame, name: str):
    df.to_csv(f'data/{name}.csv')


if __name__ == '__main__':
    base_path = 'C:\\Users\\Tijn\\Desktop\\desktopfiles\\programming\\studio projects\\thesis\\pydreamer\mlruns\\0\\'
    
    dreamer_files = ['1ccf707282f34a4a997c2eb54dea3732',
                    '014484d1e6c640e59e80b3920dea6acc',
                    '2e0012ae0f2747c79452dacf6e8399a3',
                    'c865f902a2af42369c359572a505a781',
                    '41b84bc4485d419faf81fce628932954',
                    ]
    
    mean_ccn_instance_files = ['e8ac5bd8238949c9aaf76549d929fe26',
                                '1507955ff2be4bebafd7fc4446cbcac5',
                                '60c5ff2ace7a47f1bb966816fbd7a5ef',
                                'c865f902a2af42369c359572a505a781',
                                '55413d36307943c89e5687bb47d8ed57',
                                ]

    mean_only_files = ['f035065c76a546789df96a6a5fc19f74',
                        '665a4ec1be62444bafe64670f4db81e4',
                        '77eb5cbeb06241af9531f3633a3e1de7',
                        '42efe8fb5caa482698da6de4906fd08d',
                        'ce3c9acb734e4a4498801e6dd0efbdb0',
                        ]

    
    metrics = ['\\metrics\\_loss', '\\metrics\\agent\\policy_value', '\\metrics\\agent\\return']
    metrics2 = ['_loss', '_pv', '_return']

    metric = metrics[1]
    metric2 = metrics2[1]

    df_dreamer = get_data(base_path, dreamer_files, metric) 
    calc_metrics(df_dreamer)
    df_inst = get_data(base_path, mean_ccn_instance_files, metric) 
    calc_metrics(df_inst)
    df_mean = get_data(base_path, mean_only_files, metric) 
    calc_metrics(df_mean)

    # conv2d_noNorm = load_from_csv(f'data/conv2d_noNorm{metric2}.csv')
    # df_dreamer = load_from_csv(f'data/dreamer_normal{metric2}.csv')
    # df_inst = load_from_csv(f'data/cnn_mean_instance{metric2}.csv')
    # df_mean = load_from_csv(f'data/cnn_mean_only{metric2}.csv')

    # df_3d_actnorm = load_from_csv('data/3d_actnorm.csv')
    # df_3d_instancenorm = load_from_csv('data/3d_instancenorm.csv')
    # df_3d_nonorm = load_from_csv('data/3d_nonorm.csv')

    save_data(df_dreamer, 'dreamer_normal_pv')
    save_data(df_inst, 'cnn_mean_instance_pv')
    save_data(df_mean, 'cnn_mean_only_pv')

    plot_fitted_line(df_dreamer)
    plot_fitted_line(df_inst)
    plot_fitted_line(df_mean)
    # plot_fitted_line(conv2d_noNorm)
    # compare_two_runs(df_dreamer, df_inst)

    compare = [
                (df_dreamer, 'DreamerV2'), 
                # (df_inst, 'instance'), 
                (df_mean, 'mean only'),
                # (conv2d_noNorm_return, 'conv2d'),
            ]
    compare_runs(compare, 'reward', 'Reward comparison DreamerV2 and initial conv2d network')
