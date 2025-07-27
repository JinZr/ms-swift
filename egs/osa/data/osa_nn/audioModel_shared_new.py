# expect to add more layers after self,projector (deal with (256,74,batch_size) tensor)

from transformers import Wav2Vec2Model, Wav2Vec2ForSequenceClassification
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _HIDDEN_STATES_START_POSITION,
    SequenceClassifierOutput,
    Wav2Vec2PreTrainedModel,
)


class modelForSequenceClassification(Wav2Vec2ForSequenceClassification):
    def __init__(self, config):
        super(modelForSequenceClassification, self).__init__(config)
        self.projector1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.projector2 = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.relu = nn.ReLU()

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
        is_relu=True,
        type="last",  # 'last', 'chosen', 'weighted_sum'
        block_num=-1,
        weight=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_hidden_states = (
            True if self.config.use_weighted_layer_sum else output_hidden_states
        )

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if type == "last":
            if self.config.use_weighted_layer_sum:
                hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
                hidden_states = torch.stack(hidden_states, dim=1)
                norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
                hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            else:
                hidden_states = outputs[0]

            hidden_states = self.projector(hidden_states)

            # if attention_mask is None:
            #     pooled_output = hidden_states.mean(dim=1)
            # else:
            #     padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            #     hidden_states[~padding_mask] = 0.0
            #     pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
            if attention_mask is None:
                pooled_mean = hidden_states.mean(dim=1)
                pooled_std = hidden_states.std(dim=1)
                pooled_output = torch.cat((pooled_mean, pooled_std), dim=1)
            else:
                padding_mask = self._get_feature_vector_attention_mask(
                    hidden_states.shape[1], attention_mask
                )
                hidden_states[~padding_mask] = 0.0
                count_num = padding_mask.sum(dim=1).view(-1, 1)
                pooled_mean = hidden_states.sum(dim=1) / count_num
                pooled_var = (
                    (hidden_states**2).sum(dim=1) - count_num * (pooled_mean**2)
                ) / (count_num - 1)
                pooled_std = pooled_var**0.5
                pooled_output = torch.cat((pooled_mean, pooled_std), dim=1)

        elif type == "chosen":
            hidden_states = outputs.hidden_states[block_num]

            hidden_states = self.projector(hidden_states)

            # if attention_mask is None:
            #     pooled_output = hidden_states.mean(dim=1)
            # else:
            #     padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            #     hidden_states[~padding_mask] = 0.0
            #     pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
            if attention_mask is None:
                pooled_mean = hidden_states.mean(dim=1)
                pooled_std = hidden_states.std(dim=1)
                pooled_output = torch.cat((pooled_mean, pooled_std), dim=1)
            else:
                padding_mask = self._get_feature_vector_attention_mask(
                    hidden_states.shape[1], attention_mask
                )
                hidden_states[~padding_mask] = 0.0
                count_num = padding_mask.sum(dim=1).view(-1, 1)
                pooled_mean = hidden_states.sum(dim=1) / count_num
                pooled_var = (
                    (hidden_states**2).sum(dim=1) - count_num * (pooled_mean**2)
                ) / (count_num - 1)
                pooled_std = pooled_var**0.5
                pooled_output = torch.cat((pooled_mean, pooled_std), dim=1)

        else:
            softmax = nn.Softmax(dim=0)
            weight = softmax(weight)
            hidden_states = 0
            for ct in range(len(outputs.hidden_states)):
                hidden_states = hidden_states + weight[ct] * outputs.hidden_states[ct]

            hidden_states = self.projector1(hidden_states)
            hidden_states = self.relu(hidden_states)
            hidden_states = self.projector2(hidden_states)
            # hidden_states = self.relu(hidden_states)  # this line will make training fail

            if attention_mask is None:
                pooled_mean = hidden_states.mean(dim=1)
                pooled_std = hidden_states.std(dim=1)
                pooled_output = torch.cat((pooled_mean, pooled_std), dim=1)
            else:
                padding_mask = self._get_feature_vector_attention_mask(
                    hidden_states.shape[1], attention_mask
                )
                hidden_states[~padding_mask] = 0.0
                count_num = padding_mask.sum(dim=1).view(-1, 1)
                pooled_mean = hidden_states.sum(dim=1) / count_num
                pooled_var = (
                    (hidden_states**2).sum(dim=1) - count_num * (pooled_mean**2)
                ) / (count_num - 1)
                pooled_std = pooled_var**0.5
                pooled_output = torch.cat((pooled_mean, pooled_std), dim=1)

        if is_relu:
            pooled_output = F.relu(pooled_output)

        return pooled_output
