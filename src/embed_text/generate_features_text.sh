MYSPACE=/hdd/data/
for BACKBONE in clip_rn50 
do 
# gpt3 prompts
python embed_text_class_labels_cross_modal.py --backbone $BACKBONE --text-path ../../data/cross_modal_splits/class_names.json --save-features $MYSPACE/few-shot-inaturalist-hf/features/cross_modal_datasets/text/${BACKBONE}/${BACKBONE}_class_labels_cross_modal_handcrafted
done

