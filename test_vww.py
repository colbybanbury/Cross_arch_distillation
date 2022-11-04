import tensorflow as tf

import tensorflow_datasets as tfds

# # converting to tfds by generating the metadata

# features = tfds.features.FeaturesDict({
#     'image/encoded': tfds.features.Image(shape=(None, None, 3), encoding_format='jpeg'),
#     'image/class/label': tfds.features.ClassLabel(names=['background', 'person']),
#     # 'image/object/bbox': tfds.features.BBox(),
# })

# split_info = tfds.folder_dataset.compute_split_info(
#     out_dir = '/home/colby/Cross_arch_distillation/person_detection_dataset/',
#     filename_template=tfds.core.ShardedFileTemplate('/home/colby/Cross_arch_distillation/person_detection_dataset/',
#                         '{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}', 'vww'),
# )

# tfds.folder_dataset.write_metadata(
#     data_dir='/home/colby/Cross_arch_distillation/person_detection_dataset/',
#     features=features,
#     # Pass the `out_dir` argument of compute_split_info (see section above)
#     # You can also explicitly pass a list of `tfds.core.SplitInfo`.
#     split_infos=split_info,
#     # Pass a custom file name template or use None for the default TFDS
#     # file name template.
#     # filename_template=tfds.core.ShardedFileTemplate('/home/colby/Cross_arch_distillation/person_detection_dataset/',
#     #                     '{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}', 'vww'),

#     # Optionally, additional DatasetInfo metadata can be provided
#     # See:
#     # https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetInfo
#     # description="""Multi-line description."""
#     # homepage='http://my-project.org',
#     supervised_keys=('image', 'label'),
# )

builder = tfds.builder_from_directory('/home/colby/Cross_arch_distillation/person_detection_dataset/')


# Metadata are avalailable as usual
print(builder.info.splits['train'].num_examples)

# Construct the tf.data.Dataset pipeline
ds = builder.as_dataset()['train']
for example in ds.take(1):
    print(example)