declare -a scale=("0.25" "0.5" "1.0" "1.5" "2.0" "4.0")
declare -a resolutions=("96", "128", "160", "192", "224")

base_variant="pt_efficientnet_b0"
proj_name="effnet-true-scaling-sweep-pets"
seed=1

for r in "${resolutions[@]}"
do
for i in "${scale[@]}"
do
   w=$i
   d=$i

   TFDS_DATA_DIR=gs://imagenet_distill/tensorflow_datasets \
   bash big_vision/run_tpu.sh big_vision.train \
   --config big_vision/configs/jeffnet_flowers_pets.py:proj_name=$proj_name,data=pet,variant=$base_variant,seed=$seed,width=$w,res=$r \
   --workdir gs://imagenet_distill/pets/arch-feature-sweep/scratch/scaling/variant$v/width$w/res$r/seed$seed \
   /

   TFDS_DATA_DIR=gs://imagenet_distill/tensorflow_datasets \
   bash big_vision/run_tpu.sh big_vision.train \
   --config big_vision/configs/jeffnet_flowers_pets.py:proj_name=$proj_name,data=pet,variant=$base_variant,seed=$seed,depth=$d,res=$r \
   --workdir gs://imagenet_distill/pets/arch-feature-sweep/scratch/scaling/variant$v/depth$d/res$r/seed$seed \
   /

   TFDS_DATA_DIR=gs://imagenet_distill/tensorflow_datasets \
   bash big_vision/run_tpu.sh big_vision.train \
   --config big_vision/configs/jeffnet_flowers_pets.py:proj_name=$proj_name,data=pet,variant=$base_variant,seed=$seed,width=$w,depth=$d,res=$r \
   --workdir gs://imagenet_distill/pets/arch-feature-sweep/scratch/scaling/variant$v/width$w-depth$d/res$r/seed$seed \
   /


   #distill

   TFDS_DATA_DIR=gs://imagenet_distill/tensorflow_datasets \
   bash big_vision/run_tpu.sh big_vision.trainers.proj.distill.distill \
   --config big_vision/configs/proj/distill/jeffnet_flowers_pets.py:proj_name=$proj_name,data=pet,variant=$base_variant,seed=$seed,width=$w,res=$r \
   --workdir gs://imagenet_distill/pets/arch-feature-sweep/distill/scaling/variant$v/width=$w/res$r/seed$seed \
   /

   TFDS_DATA_DIR=gs://imagenet_distill/tensorflow_datasets \
   bash big_vision/run_tpu.sh big_vision.trainers.proj.distill.distill \
   --config big_vision/configs/proj/distill/jeffnet_flowers_pets.py:proj_name=$proj_name,data=pet,variant=$base_variant,seed=$seed,depth=$d,res=$r \
   --workdir gs://imagenet_distill/pets/arch-feature-sweep/distill/scaling/variant$v/depth$d/res$r/seed$seed \
   /

   TFDS_DATA_DIR=gs://imagenet_distill/tensorflow_datasets \
   bash big_vision/run_tpu.sh big_vision.trainers.proj.distill.distill \
   --config big_vision/configs/proj/distill/jeffnet_flowers_pets.py:proj_name=$proj_name,data=pet,variant=$base_variant,seed=$seed,width=$w,depth=$d,res=$r \
   --workdir gs://imagenet_distill/pets/arch-feature-sweep/distill/scaling/variant$v/width=$w-depth$d/res$r/seed$seed \
   /


done
done