#!/bin/bash
set -e
main_folder=$1
source /media/ebonetto/WindowsData/pose_zebras_sw/vit/bin/activate
# if pre in main_folder then  use res_pre as output folder else use res fi
if [[ $main_folder == *"pre"* ]]; then
    output_folder="res_pre"
elif [[ $main_folder == *"large"* ]]; then
    output_folder="res_large"
else
    output_folder="res"
fi


mkdir -p res
# for folder in main_folder do echo $folder
for folder in $main_folder/*; do
    echo $folder
    dirname=$(basename $folder)
    if [[ $dirname == *"s_"* ]] || [[ $dirname == *"mix"* ]]; then

        if [ ! -f "$output_folder/$dirname""_ap10.txt" ]; then
              bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/ap10_27.py $(find $folder/ -iname best*) > $output_folder/$dirname"_ap10".txt
        fi

        if [ ! -f "$output_folder/$dirname""_apt36k.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/apt36k_27.py $(find $folder/ -iname best*) > $output_folder/$dirname"_apt36k".txt
        fi

        if [ ! -f "$output_folder/$dirname""_ap10_OZ.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/ap10_27_OZ.py $(find $folder/ -iname best*) > $output_folder/$dirname"_ap10_OZ".txt
        fi

        if [ ! -f "$output_folder/$dirname""_apt36k_OZ.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/apt36k_27_OZ.py $(find $folder/ -iname best*) > $output_folder/$dirname"_apt36k_OZ".txt
        fi

        if [ ! -f "$output_folder/$dirname""_ap10_NZ.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/ap10_27_NZ.py $(find $folder/ -iname best*) > $output_folder/$dirname"_ap10_NZ".txt
        fi

        if [ ! -f "$output_folder/$dirname""_apt36k_NZ.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/apt36k_27_NZ.py $(find $folder/ -iname best*) > $output_folder/$dirname"_apt36k_NZ".txt
        fi

        if [ ! -f "$output_folder/$dirname""_zoo.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/zoo_27.py $(find $folder/ -iname best*) > $output_folder/$dirname"_zoo".txt
        fi

        if [ ! -f "$output_folder/$dirname""_300.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/300_27.py $(find $folder/ -iname best*) > $output_folder/$dirname"_300".txt
        fi

        if [ ! -f "$output_folder/$dirname""_grevy_zebras.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/grevy_zebras_27.py $(find $folder/ -iname best*) > $output_folder/$dirname"_grevy_zebras".txt
        fi

        if [ ! -f "$output_folder/$dirname""_unsup_dom_ad_horse.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/unsup_dom_ad_27.py $(find $folder/ -iname best*) > $output_folder/$dirname"_unsup_dom_ad_horse".txt
        fi
    else

        if [ ! -f "$output_folder/$dirname""_ap10.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/ap10.py $(find $folder/ -iname best*) > $output_folder/$dirname"_ap10".txt
	fi

        if [ ! -f "$output_folder/$dirname""_apt36k.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/apt36k.py $(find $folder/ -iname best*) > $output_folder/$dirname"_apt36k".txt
        fi

        if [ ! -f "$output_folder/$dirname""_ap10_OZ.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/ap10_OZ.py $(find $folder/ -iname best*) > $output_folder/$dirname"_ap10_OZ".txt
        fi

        if [ ! -f "$output_folder/$dirname""_apt36k_OZ.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/apt36k_OZ.py $(find $folder/ -iname best*) > $output_folder/$dirname"_apt36k_OZ".txt
        fi

        if [ ! -f "$output_folder/$dirname""_ap10_NZ.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/ap10_N_Z.py $(find $folder/ -iname best*) > $output_folder/$dirname"_ap10_NZ".txt
        fi

        if [ ! -f "$output_folder/$dirname""_apt36k_NZ.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/apt36k_N_Z.py $(find $folder/ -iname best*) > $output_folder/$dirname"_apt36k_NZ".txt
        fi

        if [ ! -f "$output_folder/$dirname""_300.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/300.py $(find $folder/ -iname best*) > $output_folder/$dirname"_300".txt
        fi

        if [ ! -f "$output_folder/$dirname""_grevy_zebras.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/grevy_zebras.py $(find $folder/ -iname best*) > $output_folder/$dirname"_grevy_zebras".txt
        fi

        if [ ! -f "$output_folder/$dirname""_zoo.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/zoo.py $(find $folder/ -iname best*) > $output_folder/$dirname"_zoo".txt
        fi

        if [ ! -f "$output_folder/$dirname""_unsup_dom_ad_horse.txt" ]; then
          bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/unsup_dom_ad.py $(find $folder/ -iname best*) > $output_folder/$dirname"_unsup_dom_ad_horse".txt
        fi
    fi
done
#        bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/learn_syn_anim_27.py $(find $folder/ -iname best*) > $output_folder/$dirname"_learn_syn_an_horse".txt
#        bash ../tools/dist_test.sh ../configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/test/learn_syn_anim.py $(find $folder/ -iname best*) > $output_folder/$dirname"_learn_syn_an_horse".txt
