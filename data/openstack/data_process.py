import sys
sys.path.append('../')

import os
import gc
import pandas as pd
import numpy as np
from logparser import Spell, Drain
import argparse
from tqdm import tqdm
# from logdeep.dataset.session import sliding_window

tqdm.pandas()
pd.options.mode.chained_assignment = None

PAD = 0
UNK = 1
START = 2

data_dir = os.path.expanduser("./")
output_dir = "./"
# log_file = "OpenStack.log"


# In the first column of the log, "-" indicates non-alert messages while others are alert messages.
# def count_anomaly():
#     total_size = 0
#     normal_size = 0
#     with open(data_dir + log_file, encoding="utf8") as f:
#         for line in f:
#             total_size += 1
#             if line.split(' ',1)[0] == '-':
#                 normal_size += 1
#     print("total size {}, abnormal size {}".format(total_size, total_size - normal_size))


# def deeplog_df_transfer(df, features, target, time_index, window_size):
#     """
#     :param window_size: offset datetime https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
#     :return:
#     """
#     agg_dict = {target:'max'}
#     for f in features:
#         agg_dict[f] = _custom_resampler
#
#     features.append(target)
#     features.append(time_index)
#     df = df[features]
#     deeplog_df = df.set_index(time_index).resample(window_size).agg(agg_dict).reset_index()
#     return deeplog_df
#
#
# def _custom_resampler(array_like):
#     return list(array_like)


# def deeplog_file_generator(filename, df, features):
#     with open(filename, 'w') as f:
#         for _, row in df.iterrows():
#             for val in zip(*row[features]):
#                 f.write(','.join([str(v) for v in val]) + ' ')
#             f.write('\n')


def parse_log(input_dir, output_dir, log_file, parser_type):
    log_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'
    regex = [
    ]
    keep_para = False
    if parser_type == "drain":
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.3  # Similarity threshold
        depth = 3  # Depth of all leaf nodes
        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, keep_para=keep_para)
        parser.parse(log_file)
    elif parser_type == "spell":
        tau = 0.55
        parser = Spell.LogParser(indir=data_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex, keep_para=keep_para)
        parser.parse(log_file)

#
# def merge_list(time, activity):
#     time_activity = []
#     for i in range(len(activity)):
#         temp = []
#         assert len(time[i]) == len(activity[i])
#         for j in range(len(activity[i])):
#             temp.append(tuple([time[i][j], activity[i][j]]))
#         time_activity.append(np.array(temp))
#     return time_activity
def sliding_window(raw_data, para):
    """
    split logs into sliding windows/session
    :param raw_data: dataframe columns=[timestamp, label, eventid, time duration]
    :param para:{window_size: seconds, step_size: seconds}
    :return: dataframe columns=[eventids, time durations, label]
    """
    log_size = raw_data.shape[0]
    label_data, time_data = raw_data.iloc[:, 1], raw_data.iloc[:, 0]
    logkey_data, deltaT_data = raw_data.iloc[:, 2], raw_data.iloc[:, 3]
    new_data = []
    start_end_index_pair = set()

    start_time = time_data[0]
    end_time = start_time + para["window_size"]
    start_index = 0
    end_index = 0

    # get the first start, end index, end time
    for cur_time in time_data:
        if cur_time < end_time:
            end_index += 1
        else:
            break

    start_end_index_pair.add(tuple([start_index, end_index]))

    # move the start and end index until next sliding window
    num_session = 1
    while end_index < log_size:
        start_time = start_time + para['step_size']
        end_time = start_time + para["window_size"]
        for i in range(start_index, log_size):
            if time_data[i] < start_time:
                i += 1
            else:
                break
        for j in range(end_index, log_size):
            if time_data[j] < end_time:
                j += 1
            else:
                break
        start_index = i
        end_index = j

        # when start_index == end_index, there is no value in the window
        if start_index != end_index:
            start_end_index_pair.add(tuple([start_index, end_index]))

        num_session += 1
        if num_session % 1000 == 0:
            print("process {} time window".format(num_session), end='\r')

    for (start_index, end_index) in start_end_index_pair:
        dt = deltaT_data[start_index: end_index].values
        dt[0] = 0
        new_data.append([
            time_data[start_index: end_index].values,
            max(label_data[start_index:end_index]),
            logkey_data[start_index: end_index].values,
            dt
        ])

    assert len(start_end_index_pair) == len(new_data)
    print('there are %d instances (sliding windows) in this dataset\n' % len(start_end_index_pair))
    return pd.DataFrame(new_data, columns=raw_data.columns)


if __name__ == "__main__":
    #
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p', default=None, type=str, help="parser type")
    # parser.add_argument('-w', default='T', type=str, help='window size(mins)')
    # parser.add_argument('-s', default='1', type=str, help='step size(mins)')
    # parser.add_argument('-r', default=0.4, type=float, help="train ratio")
    # args = parser.parse_args()
    # print(args)
    #

    ##########
    # Parser #
    #########

    # for log_name in ['openstack_abnormal.log', 'openstack_normal2.log', 'openstack_normal1.log']:
    #     parse_log(data_dir, output_dir, log_name, 'drain')

    #########
    # Count #
    #########
    # count_anomaly()

  ##################
    # Transformation #
    ##################
    # mins
    window_size = 30
    step_size = 6
    # train_ratio = 0.4

    df_normal1 = pd.read_csv(f'{output_dir}openstack_normal1.log_structured.csv')
    df_normal2 = pd.read_csv(f'{output_dir}openstack_normal2.log_structured.csv')
    df = pd.concat([df_normal1, df_normal2], ignore_index=True)


    # # data preprocess
    df['datetime'] = pd.to_datetime(df['Date'] + '-' + df['Time'], format='%Y-%m-%d-%H:%M:%S.%f', errors='coerce')
    print(df['datetime'])
    df = df.dropna(subset=['datetime'])
    df.reset_index(drop=True, inplace=True)
    # print(df['datetime'])

    # df['datetime'] = pd.to_datetime('2023' + '-' + df['Month'] + '-' + df['Day'], format='%Y-%b-%d', errors='coerce') + pd.to_timedelta(df['Time'])
    # df.dropna(subset=['datetime'])
    # print(df['datetime'])

    # df['datetime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S')
    # df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
    df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
    df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
    df['deltaT'].fillna(0)
    # convert time to UTC timestamp
    df['deltaT'] = df['datetime'].apply(lambda t: (t - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))
    df['Label'] = '-'

    # sampling with fixed window
    # features = ["EventId", "deltaT"]
    # target = "Label"
    # deeplog_df = deeplog_df_transfer(df, features, target, "datetime", window_size=args.w)
    # deeplog_df.dropna(subset=[target], inplace=True)

    # sampling with sliding window
    deeplog_df = sliding_window(df[["timestamp", "Label", "EventTemplate", "deltaT"]],
                                para={"window_size": int(window_size), "step_size": int(step_size)}
                                )

    deeplog_df.drop(columns=['timestamp', 'Label', 'deltaT'], axis=1, inplace=True)
    deeplog_df.rename(columns={"EventTemplate": "text"}, inplace=True)
    deeplog_df['text'] = deeplog_df['text'].apply(lambda x: '|'.join(x))
    #########
    # Train #
    #########
    # df_normal =deeplog_df[deeplog_df["Label"] == 0]
    # df_abnormal =deeplog_df[deeplog_df["Label"] == 1]
    # # df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True) #shuffle
    # df_normal.rename(columns={"EventTemplate": "text", "Label": "labels"}, inplace=True)
    # df_abnormal.rename(columns={"EventTemplate": "text", "Label": "labels"}, inplace=True)

    # normal_len = len(deeplog_df)
    # # train_len = int(normal_len * train_ratio)
    # train_len = 10000

    # train = df_normal[:train_len]
    # # train = train.sample(frac=.5, random_state=20) # sample normal data
    # # deeplog_file_generator(os.path.join(output_dir,'train'), train, ["EventId", "deltaT"])
    # train['text'] = train['text'].apply(lambda x: '|'.join(x))
    # train = train.loc[:, ['text', 'labels']]

    # # deeplog_file_generator(os.path.join(output_dir,'train'), train, ["text"])

    print("training size {}".format(len(deeplog_df)))
    deeplog_df.to_csv(output_dir + "train.csv", index=False)