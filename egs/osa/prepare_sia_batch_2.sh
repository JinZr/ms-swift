#!/usr/bin/env bash

# "请听录音并估算说话人的 Mallampati 分级: " \
# "请听录音并估算说话人的 Friedman 分级: " \
# "请听录音并估算说话人的最大舌宽: " \
# "请听录音并估算说话人的最大舌长: " \
# "请听录音并估算说话人的舌宽长比: " \
# "请听录音并估算说话人的舌面积: " \
# "请听录音并估算说话人的扁桃体分级: " \
# "请听录音并估算说话人的悬雍垂高度: " \
# "请听录音并估算说话人的悬雍垂宽高比: " \
# "请听录音并估算说话人的悬雍垂面积: " \
# "请听录音并估算说话人的咽侧壁最大宽度: " \

# Mallampati_level Friedman_level maximum_tongue_width maximum_tongue_length tongue_width_length_ratio tongue_area tonsil_grade uvula_height uvula_width_height_ratio uvula_area maximum_pharyngeal_lateral_wall_width

# "说话人的 Mallampati 分级是: " \
# "说话人的 Friedman 分级是: " \
# "说话人的最大舌宽是: " \
# "说话人的最大舌长是: " \
# "说话人的舌宽长比是: " \
# "说话人的舌面积是: " \
# "说话人的扁桃体分级是: " \
# "说话人的悬雍垂高度是: " \
# "说话人的悬雍垂宽高比是: " \
# "说话人的悬雍垂面积是: " \
# "说话人的咽侧壁最大宽度是: " \

python local/prepare_osa_single_task_sia_batch_2.py \
    --csv-file ./data/label_new_data_batch_2.csv \
    --output-dir ./data_sia_batch_2/ \
    --system "你是一名耳鼻喉科主治医师，专攻阻塞性睡眠呼吸暂停（OSA）。请通过对比同一患者坐位与仰卧位朗读同一段文本的语音差异，判断其是否患有重度 OSA。" \
    --query-prefixs \
        "请听录音并估算说话人的 OSA 严重程度: " \
    --query-suffixs "，请重点关注两段语音在高频能量分布、共振峰位置及鼻化程度的差异。" \
    --response-cols \
        AHI \
    --response-prefixs \
        "说话人的 OSA 严重程度是: " \
    --audio-dir ./data/osa_nn/annotation_full/ \
    --audio-indices 1 2 3 4 5 6

python local/prepare_osa_single_task_sia_batch_2_binary_only.py \
    --csv-file ./data/label_new_data_batch_2.csv \
    --output-dir ./data_sia_batch_2_binary_only/ \
    --system "你是一名耳鼻喉科主治医师，专攻阻塞性睡眠呼吸暂停（OSA）。请通过对比同一患者坐位与仰卧位朗读同一段文本的语音差异，判断其是否患有重度 OSA。" \
    --query-prefixs \
        "请听录音并估算说话人的 OSA 严重程度: " \
    --query-suffixs "，请重点关注两段语音在高频能量分布、共振峰位置及鼻化程度的差异。" \
    --response-cols \
        AHI \
    --response-prefixs \
        "说话人的 OSA 严重程度是: " \
    --audio-dir ./data/osa_nn/annotation_full/ \
    --audio-indices 1 2 3 4 5 6