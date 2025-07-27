#!/usr/bin/env bash

python local/prepare_osa_single_task3.py \
    --csv-file ./data/label_new_data.csv \
    --output-dir ./data3/ \
    --system "你是一个专业的 OSA 诊断医师。" \
    --query-prefixs \
        "请听录音并估算说话人的 AHI: " \
        "请听录音并估算说话人的 Mallampati 分级: " \
        "请听录音并估算说话人的 Friedman 分级: " \
        "请听录音并估算说话人的最大舌宽: " \
        "请听录音并估算说话人的最大舌长: " \
        "请听录音并估算说话人的舌宽长比: " \
        "请听录音并估算说话人的舌面积: " \
        "请听录音并估算说话人的扁桃体分级: " \
        "请听录音并估算说话人的悬雍垂高度: " \
        "请听录音并估算说话人的悬雍垂宽高比: " \
        "请听录音并估算说话人的悬雍垂面积: " \
        "请听录音并估算说话人的咽侧壁最大宽度: " \
    --query-suffixs "" \
    --response-cols \
        AHI Mallampati_level Friedman_level maximum_tongue_width maximum_tongue_length tongue_width_length_ratio tongue_area tonsil_grade uvula_height uvula_width_height_ratio uvula_area maximum_pharyngeal_lateral_wall_width \
    --response-prefixs \
        "说话人的 AHI 是: " \
        "说话人的 Mallampati 分级是: " \
        "说话人的 Friedman 分级是: " \
        "说话人的最大舌宽是: " \
        "说话人的最大舌长是: " \
        "说话人的舌宽长比是: " \
        "说话人的舌面积是: " \
        "说话人的扁桃体分级是: " \
        "说话人的悬雍垂高度是: " \
        "说话人的悬雍垂宽高比是: " \
        "说话人的悬雍垂面积是: " \
        "说话人的咽侧壁最大宽度是: " \
    --audio-dir ./data/osa_nn/new_data/audios/ \
    --audio-indices 1 2 3 4 5 6