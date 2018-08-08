import os
import argparse

from engine.tools.make_consolidation_df import consolidation_df_for_predictions_with_all_metrics

""" Initialize argument parser """
parser = argparse.ArgumentParser(description='Script for calculated metrics for the trained model')
parser.add_argument('-m', '--model', action='store', type=str, default='',
                    help='Name of the model used for obtaining processing results')
parser.add_argument('-b', '--barcode', action='store', type=str, default='',
                    help='Define barcode class which results are analyzed')
args = parser.parse_args()

""" Set paths for project, model to be used, input data, train data and output data"""
barcode = '_' + args.barcode

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'resource', barcode, 'results')
SUPPORT_FILES_PATH = os.path.join(PROJECT_PATH, 'resource', barcode, 'support_files')

""" Calculate metrocs """
consolidation_df_for_predictions_with_all_metrics(args.model, OUTPUT_PATH, SUPPORT_FILES_PATH)
