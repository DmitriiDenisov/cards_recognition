import numpy as np
import pandas as pd
from pandas import ExcelWriter
from engine.tools.select_best_threshold import select_best_threshold, count_TP_and_FP_for_df
import os
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATH_PREDICTIONS = os.path.join(PROJECT_PATH, 'resource', 'data', 'predictions_on_test_data.csv')
PATH_LABELS = os.path.join(PROJECT_PATH, 'resource', 'data', 'labels.csv')
PATH_FILES_NAMES = os.path.join(PROJECT_PATH, 'resource', 'data', 'filenames.txt')
PATH_TEST_ANSWERS = os.path.join(PROJECT_PATH, 'resource', 'data', 'true_answers.xlsx')
PATH_CLASSES_IN_SET = os.path.join(PROJECT_PATH, 'resource', 'data', 'classes_in_set.xlsx')

def consolidation_df_for_predictions():
    with open(PATH_FILES_NAMES) as f:
        content = f.readlines()
    filenames = content[0].split(' ')

    best_pred, best_thresh = select_best_threshold([0.4, 0.45, 0.5], filenames, PATH_PREDICTIONS,
                                                   PATH_TEST_ANSWERS, PATH_LABELS, PATH_CLASSES_IN_SET)
    #true_answers = pd.read_excel(PATH_TEST_ANSWERS).set_index('card_id')
    labels = pd.read_csv(PATH_LABELS)
    labels = dict(zip(labels['class_index'], labels['class_name']))
    pred = pd.read_csv(PATH_PREDICTIONS, header=None)
    if isinstance(pred, pd.DataFrame):
        pred = pred.values

    predicted_class_indices = np.argmax(pred, axis=1)
    predictions_without_thresh = [labels[k] for k in predicted_class_indices]
    df = pd.DataFrame({"Filename": filenames,
                            "Predictions": predictions_without_thresh}).set_index('Filename')
    #df = pd.concat([results, true_answers], axis=1, join_axes=[results.index])
    df['Pred_proba_without_thresh'] = np.max(pred, axis=1)
    #df['indicator'] = df['Predictions'] == df['true_type']

    bool_pred = pred > best_thresh
    rows_does_not_contain_more_than_thresh = np.argwhere(bool_pred.any(1) == False)
    predicted_class_indices[rows_does_not_contain_more_than_thresh] = -1
    predictions_with_thresh = [labels[k] for k in predicted_class_indices]
    df['Pred_class_with_thresh_' + str(best_thresh)] = predictions_with_thresh

    writer = ExcelWriter(os.path.join(PROJECT_PATH, 'resource\statistics\consolidation_df.xlsx'))
    df.to_excel(writer, 'Sheet1')
    writer.save()

def consolidation_df_for_predictions_with_all_metrics():
    with open(PATH_FILES_NAMES) as f:
        content = f.readlines()
    filenames = content[0].split(' ')
    true_answers = pd.read_excel(PATH_TEST_ANSWERS).set_index('card_id')
    classes_in_train_set = pd.read_excel(PATH_CLASSES_IN_SET)['class_name_in_training_set'].values
    true_answers.loc[~true_answers['true_type'].isin(classes_in_train_set), 'true_type'] = 'rejected'

    #A = true_answers.groupby(['true_type'])

    labels = pd.read_csv(PATH_LABELS)
    labels = dict(zip(labels['class_index'], labels['class_name']))
    pred = pd.read_csv(PATH_PREDICTIONS, header=None)
    if isinstance(pred, pd.DataFrame):
        pred = pred.values

    #best_pred, best_thresh = select_best_threshold([0.4, 0.45, 0.5], filenames, PATH_PREDICTIONS,
    #                                               PATH_TEST_ANSWERS, PATH_LABELS, PATH_CLASSES_IN_SET)
    best_thresh = 0.45

    predicted_class_indices = np.argmax(pred, axis=1)
    bool_pred = pred > best_thresh
    rows_does_not_contain_more_than_thresh = np.argwhere(bool_pred.any(1) == False)
    predicted_class_indices[rows_does_not_contain_more_than_thresh] = -1
    predictions_with_thresh = [labels[k] for k in predicted_class_indices]
    results = pd.DataFrame({"Filename": filenames,
                            "Predictions": predictions_with_thresh}).set_index('Filename')
    df = pd.concat([results, true_answers], axis=1, join_axes=[results.index])
    df['indicator'] = df['Predictions'] == df['true_type']


    #Creating score df:
    check = true_answers['true_type'].unique()
    df_results = pd.DataFrame({'Unique true_type': df['true_type'].unique()})
    num_els_in_test = true_answers['true_type'].value_counts()
    df_results['num_els_in_test'] = num_els_in_test.loc[df_results['Unique true_type']].values
    TP_list, FP_list, FN_list, TN_list = count_TP_and_FP_for_df(df)
    df_results['TP_list'] = TP_list
    df_results['FP_list'] = FP_list
    df_results['FN_list'] = FN_list
    df_results['TN_list'] = TN_list
    df_results['Precision'] = df_results['TP_list'] / (df_results['TP_list'] + df_results['FP_list'])
    df_results['Precision'].fillna(0, inplace=True)
    df_results['Recall'] = df_results['TP_list'] / (df_results['TP_list'] + df_results['FN_list'])
    df_results['Weighted_Pr'] = df_results['Precision'] * df_results['num_els_in_test'] / df_results['num_els_in_test'].sum()
    df_results['Weighted_Re'] = df_results['Recall'] * df_results['num_els_in_test'] / df_results['num_els_in_test'].sum()

    scores = ['Scores:', '', df_results['TP_list'].sum(), df_results['FP_list'].sum(), df_results['FN_list'].sum(),
              df_results['TN_list'].sum(), df_results['TP_list'].mean() / (df_results['TP_list'].mean() + df_results['FP_list'].mean()),
              df_results['TP_list'].mean() / (df_results['TP_list'].mean() + df_results['FN_list'].mean()),
              df_results['Weighted_Pr'].sum(), df_results['Weighted_Re'].sum()]
    df_results.loc[len(df_results)] = [''] * df_results.shape[1]
    df_results.loc[len(df_results)] = scores


    writer = ExcelWriter(os.path.join(PROJECT_PATH, 'resource', 'statistics', 'consolidation_df_with_metrics.xlsx'))
    df_results.to_excel(writer, 'Sheet1')
    writer.save()

if __name__ == '__main__':
    consolidation_df_for_predictions_with_all_metrics()