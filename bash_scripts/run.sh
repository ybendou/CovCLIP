#########################
# Example usage:  
# inaturalist openset in-domain: ./run.sh -t ncm_std --covariance diagonal  -b clip_rn50 --text-type class_label --rm 0 --runs-open-set 10 --inaturalist --in-domain --seed 1 --epochs 50 --lr -1 --bs 128 --optimizer adamw --scheduler cosine --batch-norm True;
# imagenet in-domain: ./run.sh -t ncm_std --covariance diagonal  -b clip_rn50 --text-type class_label --rm 10 --runs-open-set 10  --in-domain --seed 1 --epochs 50 --lr -1 --bs 128 --optimizer adamw --scheduler cosine --batch-norm True;
#########################

CODE_PATH="" # path to the code

# training parameters
TRAINER=ncm;
COVARIANCE_FORM=diagonal;
WANDB="''"
BACKBONE=clip_rn50;
BATCH_SIZE=256;
TRAINING_BATCH_SIZE=128;
VALIDATION_BATCH_SIZE=1024;
EVALUATION_BATCH_FEW_SHOT_RUNS=100;
SEED=1;
DEVICE='cuda:0';
TRAINING_DEVICE='cuda:0';
OPTIMIZER=adamw;
SCHEDULER=cosine
LR=-1
WD=-1
END_LR_FACTOR=1
TOTAL_EPOCHS=50
BATCH_NORM=True
PCA=-1

# evaluation parameters
EVALUATION_N_WAYS=0;
EVALUATION_MAX_SHOTS_MULTI_CLASS=16; 
EVALUATION_MAX_SHOTS_OPEN_SET=20; 
RUNS_OPENSET=0;
RERUNS_OPENSET=100;
MULTI_CLASS_RUNS=1;
EVALUATION_MULTI_CLASS_VALIDATION_SHOTS=4;
EVALUATION_COOP_SPLIT=0;
INDOMAIN=False
TRAINING_DATASET='semanticfs_imagenet';
VALIDATION_DATASET='semanticfs_imagenet';
TEXTYPE=class_label
INATURALIST=False
# Parse named arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -t|--trainer)
      TRAINER="$2"
      shift # past argument
      shift # past value
      ;;
    -w|--wandb)
      WANDB="$2"
      shift # past argument
      shift # past value
      ;;
    -c|--covariance)
      COVARIANCE_FORM="$2"
      shift # past argument
      shift # past value
      ;;
    -e|--epochs)
      TOTAL_EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    -b|--backbone)
      BACKBONE="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--pca)
      PCA="$2"
      shift # past argument
      shift # past value
      ;;
    -rm|--runs-multi-class)
      MULTI_CLASS_RUNS="$2"
      shift # past argument
      shift # past value
      ;;
    -ro|--runs-open-set)
      RUNS_OPENSET="$2"
      shift # past argument
      shift # past value
      ;;
    -i|--in-domain)
      INDOMAIN=True
      shift # past argument
      ;;
    --text-type)
      TEXTYPE="$2"
      shift # past argument
      shift # past value
      ;;
    --inaturalist)
      INATURALIST=True
      shift # past value
      ;;
    --seed)
      SEED="$2"
      shift # past argument
      shift # past value
      ;; 
    --lr)
      LR="$2"
      shift
      shift
      ;; 
    --wd)
      WD="$2"
      shift
      shift
      ;; 
    --bs)
      BATCH_SIZE="$2"
      shift
      shift
      ;; 
    --coop)
      EVALUATION_COOP_SPLIT="$2"
      shift
      shift
      ;; 
    --optimizer)
      OPTIMIZER="$2"
      shift
      shift
      ;; 
    --scheduler)
      SCHEDULER="$2"
      shift
      shift
      ;; 
    --batch-norm)
      BATCH_NORM="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $key"
      exit 1
      ;;
  esac
done
WORKDIR="" # path to the working directory
FEATURES_TEXT_TRAIN='""'
# dataset parameters
########## Test features based on in-domain and cross-domain ##########
## If in-domain, then use the same dataset for training and testing
if [ $INDOMAIN == "True" ]; then
  echo "--------- IN DOMAIN ----------"; 
  if [ $INATURALIST == "True" ]; then
    echo "--------- iNaturalist ----------"; 
    TEST_DATASET="['inaturalist']";
    TEST_FEATURES="['${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_inaturalist_train_features.pt']";
    SHOTS_TEST_FEATURES='""' 
    QUERIES_TEST_FEATURES='""'
  else
    echo "--------- ImageNet ----------"; 
    TEST_DATASET="['semanticfs_imagenet']";
    TEST_FEATURES="['${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_imagenet_train_features.pt']";
    SHOTS_TEST_FEATURES='""' 
    QUERIES_TEST_FEATURES='""'
  fi
else # cross domain
  echo "--------- CROSS DOMAIN ----------"; 
  TEST_DATASET="['semanticfs_eurosat','semanticfs_sun397','semanticfs_dtd','semanticfs_oxford_flowers','semanticfs_caltech_101','semanticfs_food_101','semanticfs_stanford_cars','semanticfs_ucf101','semanticfs_oxford_pets']" #,'semanticfs_fgvc_aircraft_2013b']";
  TEST_FEATURES="['${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_eurosat_train_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_sun397_train_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_dtd_train_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_oxford_flowers_train_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_caltech_101_train_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_food_101_train_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_stanford_cars_train_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_ucf101_train_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_oxford_pets_train_features.pt']" #,'${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_fgvc_aircraft_2013b_train_features.pt']";
  SHOTS_TEST_FEATURES="['${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_eurosat_train_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_sun397_train_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_dtd_train_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_oxford_flowers_train_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_caltech_101_train_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_food_101_train_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_stanford_cars_train_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_ucf101_train_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_oxford_pets_train_features.pt']" #,'${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_fgvc_aircraft_2013b_train_features.pt']";
  QUERIES_TEST_FEATURES="['${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_eurosat_test_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_sun397_test_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_dtd_test_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_oxford_flowers_test_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_caltech_101_test_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_food_101_test_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_stanford_cars_test_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_ucf101_test_features.pt','${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_oxford_pets_test_features.pt']" #,'${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_fgvc_aircraft_2013b_test_features.pt']";
fi

###### TEXT
if [ $TEXTYPE == "class_label" ]; then
  echo "--------- Text type: Class labels ----------"; 
  if [ $INATURALIST == "True" ]; then
    FEATURES_TEXT=${WORKDIR}/features/text/${BACKBONE}/${BACKBONE}_class_labels_inaturalist.pt 
    FEATURES_TEXT_TRAIN=${WORKDIR}/features/text/${BACKBONE}/${BACKBONE}_class_labels_inaturalist.pt
  else
    FEATURES_TEXT=${WORKDIR}/features/text/${BACKBONE}/${BACKBONE}_class_labels_cross_modal_handcrafted_with_tokensFalse.pt
  fi
elif [ $TEXTYPE == "description" ]; then
    echo "--------- Text type: Class descriptions ----------"; 
  if [ $INATURALIST == "True" ]; then
    FEATURES_TEXT=${WORKDIR}/features/text/${BACKBONE}/${BACKBONE}_gpt3_inaturalist_with_tokensFalse.pt
    FEATURES_TEXT_TRAIN=${WORKDIR}/features/text/${BACKBONE}/${BACKBONE}_gpt3_inaturalist_with_tokensFalse.pt
  else
    FEATURES_TEXT=${WORKDIR}/features/text/${BACKBONE}/${BACKBONE}_gpt3_prompts_cross_modal_description_with_tokensFalse.pt
    FEATURES_TEXT_TRAIN='""'
  fi
fi

if [ $INATURALIST == "True" ]; then
  TRAINING_DATASET=inaturalist
  VALIDATION_DATASET=inaturalist
  FEATURES_TEXT_CLASS_LABELS=${WORKDIR}/features/text/${BACKBONE}/${BACKBONE}_class_labels_inaturalist.pt
  TRAINING_FEATURES=${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_inaturalist_train_features.pt;
else
  TRAINING_DATASET=semanticfs_imagenet
  VALIDATION_DATASET=semanticfs_imagenet
  FEATURES_TEXT_CLASS_LABELS=${WORKDIR}/features/text/${BACKBONE}/${BACKBONE}_class_labels_cross_modal_handcrafted_with_tokensFalse.pt
  TRAINING_FEATURES=${WORKDIR}/features/images/${BACKBONE}/${BACKBONE}_semanticfs_imagenet_train_features.pt;
fi
TAXONOMY_PATH=${WORKDIR}/inaturalist_metadata/taxonomies/taxa.csv
SAVE_FIG_PATH=${WORKDIR}/tmp/

echo "trainer: $TRAINER", "backbone: $BACKBONE", "covariance_form: $COVARIANCE_FORM", "batch-size $BATCH_SIZE", "wandb $WANDB"
cd ${CODE_PATH}fs-text2stats/src/scripts/
export PYTHONPATH=${CODE_PATH}fs-text2stats:$PYTHONPATH # add current directory to path

python main.py --wandb $WANDB --seed $SEED --trainer $TRAINER --evaluation-n-ways $EVALUATION_N_WAYS --evaluation-max-shots-multi-class $EVALUATION_MAX_SHOTS_MULTI_CLASS --evaluation-batch-few-shot-runs $EVALUATION_BATCH_FEW_SHOT_RUNS \
--evaluation-runs-multi-class $MULTI_CLASS_RUNS --evaluation-runs-open-set $RUNS_OPENSET --evaluation-reruns-open-set $RERUNS_OPENSET --batch-size $BATCH_SIZE --evaluation-max-shots-open-set $EVALUATION_MAX_SHOTS_OPEN_SET --evaluation-multi-class-validation-shots $EVALUATION_MULTI_CLASS_VALIDATION_SHOTS \
--training-batch-size $TRAINING_BATCH_SIZE --validation-batch-size $VALIDATION_BATCH_SIZE --lr $LR --end-lr-factor $END_LR_FACTOR --wd $WD --expansion-ratio 1 --scheduler $SCHEDULER --optimizer $OPTIMIZER --epochs $TOTAL_EPOCHS --batch-norm $BATCH_NORM \
--features-path-text $FEATURES_TEXT --features-path-train-text $FEATURES_TEXT_TRAIN  --features-path-train-images $TRAINING_FEATURES --features-text-class-labels $FEATURES_TEXT_CLASS_LABELS \
--features-path-test-images $TEST_FEATURES --features-path-evaluation-shots-images $SHOTS_TEST_FEATURES --features-path-evaluation-queries-images $QUERIES_TEST_FEATURES --taxonomy-path $TAXONOMY_PATH \
--training-dataset $TRAINING_DATASET --validation-dataset $VALIDATION_DATASET --test-dataset $TEST_DATASET \
--device $DEVICE --training-device $TRAINING_DEVICE \
--evaluation-coop-split $EVALUATION_COOP_SPLIT --covariance-form $COVARIANCE_FORM --pca $PCA --save-fig-path $SAVE_FIG_PATH