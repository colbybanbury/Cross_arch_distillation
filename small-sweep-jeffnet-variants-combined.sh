declare -a variants=("pt_mobilenetv2_100", "pt_mobilenetv2_100_noskip")

proj_name="mbnetv2-residual-test-imagenet"
dataset="imagenet"
seed=1

# for v in "${variants[@]}"
# do

# TFDS_DATA_DIR=gs://imagenet_distill/tensorflow_datasets \
# bash big_vision/run_tpu.sh big_vision.train \
# --config big_vision/configs/jeffnet_flowers_pets.py:proj_name=$proj_name,data=$dataset,variant=$v,seed=$seed, \
# --workdir gs://imagenet_distill/$dataset/arch-feature-sweep/scratch/skip-test/variant$v/seed=$seed/ \
# /
# done

for v in "${variants[@]}"
do
   TFDS_DATA_DIR=gs://imagenet_distill/tensorflow_datasets \
   bash big_vision/run_tpu.sh big_vision.trainers.proj.distill.distill \
   --config big_vision/configs/proj/distill/jeffnet_flowers_pets.py:proj_name=$proj_name,data=$dataset,variant=$v,seed=$seed, \
   --workdir gs://imagenet_distill/$dataset/arch-feature-sweep/distill/skip-test/variant$v/seed$seed/4/ \
   /
done