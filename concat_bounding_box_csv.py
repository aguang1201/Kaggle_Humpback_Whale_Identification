import pandas as pd
import numpy as np

bbox_csv = '/home/room/dataset/Humpback_Whale/bounding_boxes.csv'
bbox_csv_shenxing = '/home/room/dataset/Humpback_Whale/bounding_boxes_shenxing.csv'
bbox_df_new_file = '/home/room/dataset/Humpback_Whale/bounding_boxes_concat.csv'
bbox_df = pd.read_csv(bbox_csv, index_col='Image')
bbox_shenxing_df = pd.read_csv(bbox_csv_shenxing, index_col='Image')
bbox_df_dropped = bbox_df.drop(bbox_shenxing_df.index)
bbox_df_new = pd.concat([bbox_shenxing_df, bbox_df_dropped])
bbox_df_new.to_csv(bbox_df_new_file)
print(f'bbox_df shape is : {bbox_df.shape}')
print(f'bbox_shenxing_df shape is : {bbox_shenxing_df.shape}')
print(f'bbox_df_dropped shape is : {bbox_df_dropped.shape}')
print(f'bbox_df_new shape is : {bbox_df_new.shape}')
