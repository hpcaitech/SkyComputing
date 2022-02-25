# coding=utf-8

"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import math
import sys

import torch
import torch.nn.functional as F
import torch.nn.init as init
from dllb.registry import LAYER
from torch import nn
from torch.nn import Module
from torch.nn.parameter import Parameter

from .bert import BertConfig


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


# used only for triton inference


def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


# used specifically for training since torch.nn.functional.gelu breaks ONNX export


def bias_gelu_training(bias, y):
    x = bias + y
    return torch.nn.functional.gelu(x)  # Breaks ONNX export


def bias_tanh(bias, y):
    x = bias + y
    return torch.tanh(x)


def swish(x):
    return x * torch.sigmoid(x)


# torch.nn.functional.gelu(x) # Breaks ONNX export
ACT2FN = {
    "gelu": gelu,
    "bias_gelu": bias_gelu,
    "bias_tanh": bias_tanh,
    "relu": torch.nn.functional.relu,
    "swish": swish,
}


class LinearActivation(Module):
    r"""Fused Linear and activation Module."""
    __constants__ = ["bias"]

    def __init__(self, in_features, out_features, act="gelu", bias=True):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_fn = nn.Identity()  #
        self.biased_act_fn = None  #
        #
        self.bias = None
        # For TorchScript
        if isinstance(act, str) or (
            sys.version_info[0] == 2 and isinstance(act, unicode)
        ):
            if bias and not "bias" in act:  # compatibility
                act = "bias_" + act  #
                #
                self.biased_act_fn = ACT2FN[act]

            else:
                self.act_fn = ACT2FN[act]
        else:
            self.act_fn = act
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if not self.bias is None:
            return self.biased_act_fn(self.bias, F.linear(input, self.weight, None))
        else:
            return self.act_fn(F.linear(input, self.weight, self.bias))

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class BertNonFusedLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(BertNonFusedLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = x - u
        s = s * s
        s = s.mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


try:
    import apex

    # apex.amp.register_half_function(apex.normalization.fused_layer_norm, 'FusedLayerNorm')
    import apex.normalization
    from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction

    # apex.amp.register_float_function(apex.normalization.FusedLayerNorm, 'forward')
    # BertLayerNorm = apex.normalization.FusedLayerNorm
    APEX_IS_AVAILABLE = True
except ImportError:
    # BertLayerNorm = BertNonFusedLayerNorm
    APEX_IS_AVAILABLE = False


class BertLayerNorm(Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.shape = torch.Size((hidden_size,))
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.apex_enabled = APEX_IS_AVAILABLE

    @torch.jit.unused
    def fused_layer_norm(self, x):
        return FusedLayerNormAffineFunction.apply(
            x, self.weight, self.bias, self.shape, self.eps
        )

    def forward(self, x):
        if self.apex_enabled and not torch.jit.is_scripting():
            x = self.fused_layer_norm(x)
        else:
            u = x.mean(-1, keepdim=True)
            s = x - u
            s = s * s
            s = s.mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight * x + self.bias
        return x


@LAYER.register_module
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        config = BertConfig.from_dict(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.word_embeddings.weight.dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings, extended_attention_mask


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = torch.reshape(x, new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_key_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = torch.reshape(x, new_x_shape)
        return x.permute(0, 2, 3, 1)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = torch.reshape(context_layer, new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense_act = LinearActivation(
            config.hidden_size, config.intermediate_size, act=config.hidden_act
        )

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


@LAYER.register_module
class BertLayer_Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = BertConfig.from_dict(config)
        self.attention = BertAttention(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        return attention_output, attention_mask


@LAYER.register_module
class BertLayer_Body(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = BertConfig.from_dict(config)
        self.intermediate = BertIntermediate(config)

    def forward(self, attention_output, attention_mask):
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output, attention_output, attention_mask


@LAYER.register_module
class BertLayer_Tail(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = BertConfig.from_dict(config)
        self.output = BertOutput(config)

    def forward(self, intermediate_output, attention_output, attention_mask):
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_mask


@LAYER.register_module
class BertTailForClassification(nn.Module):
    def __init__(self, hidden_dropout_prob, hidden_size, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, logits):
        logits = self.dropout(logits)
        logits = self.classifier(logits)
        logits = logits.view(-1, self.num_classes)
        return logits


@LAYER.register_module
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = BertConfig.from_dict(config)
        self.dense_act = LinearActivation(
            config.hidden_size, config.hidden_size, act="tanh"
        )

    def forward(self, hidden_states, attention_mask):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense_act(first_token_tensor)
        return pooled_output


# class BertPredictionHeadTransform(nn.Module):
#     def __init__(self, config):
#         super(BertPredictionHeadTransform, self).__init__()
#         self.dense_act = LinearActivation(
#             config.hidden_size, config.hidden_size, act=config.hidden_act)
#         self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

#     def forward(self, hidden_states):
#         hidden_states = self.dense_act(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states)
#         return hidden_states


# class BertLMPredictionHead(nn.Module):
#     def __init__(self, config, bert_model_embedding_weights):
#         super(BertLMPredictionHead, self).__init__()
#         self.transform = BertPredictionHeadTransform(config)

#         # The output weights are the same as the input embeddings, but there is
#         # an output-only bias for each token.
#         self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
#                                  bert_model_embedding_weights.size(0),
#                                  bias=False)
#         self.decoder.weight = bert_model_embedding_weights
#         self.bias = nn.Parameter(torch.zeros(
#             bert_model_embedding_weights.size(0)))

#     def forward(self, hidden_states):
#         hidden_states = self.transform(hidden_states)
#         hidden_states = self.decoder(hidden_states) + self.bias
#         return hidden_states


# class BertOnlyMLMHead(nn.Module):
#     def __init__(self, config, bert_model_embedding_weights):
#         super(BertOnlyMLMHead, self).__init__()
#         self.predictions = BertLMPredictionHead(
#             config, bert_model_embedding_weights)

#     def forward(self, sequence_output):
#         prediction_scores = self.predictions(sequence_output)
#         return prediction_scores


# class BertOnlyNSPHead(nn.Module):
#     def __init__(self, config):
#         super(BertOnlyNSPHead, self).__init__()
#         self.seq_relationship = nn.Linear(config.hidden_size, 2)

#     def forward(self, pooled_output):
#         seq_relationship_score = self.seq_relationship(pooled_output)
#         return seq_relationship_score


# class BertPreTrainingHeads(nn.Module):
#     def __init__(self, config, bert_model_embedding_weights):
#         super(BertPreTrainingHeads, self).__init__()
#         self.predictions = BertLMPredictionHead(
#             config, bert_model_embedding_weights)
#         self.seq_relationship = nn.Linear(config.hidden_size, 2)

#     def forward(self, sequence_output, pooled_output):
#         prediction_scores = self.predictions(sequence_output)
#         seq_relationship_score = self.seq_relationship(pooled_output)
#         return prediction_scores, seq_relationship_score


# class BertForPreTraining(BertPreTrainedModel):
#     """BERT model with pre-training heads.
#     This module comprises the BERT model followed by the two pre-training heads:
#         - the masked language modeling head, and
#         - the next sentence classification head.
#     Params:
#         config: a BertConfig class instance with the configuration to build a new model.
#     Inputs:
#         `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
#             with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
#             `extract_features.py`, `run_classifier.py` and `run_squad.py`)
#         `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
#             types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
#             a `sentence B` token (see BERT paper for more details).
#         `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
#             selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#             input sequence length in the current batch. It's the mask that we typically use for attention when
#             a batch has varying length sentences.
#         `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
#             with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
#             is only computed for the labels set in [0, ..., vocab_size]
#         `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
#             with indices selected in [0, 1].
#             0 => next sentence is the continuation, 1 => next sentence is a random sentence.
#     Outputs:
#         if `masked_lm_labels` and `next_sentence_label` are not `None`:
#             Outputs the total_loss which is the sum of the masked language modeling loss and the next
#             sentence classification loss.
#         if `masked_lm_labels` or `next_sentence_label` is `None`:
#             Outputs a tuple comprising
#             - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
#             - the next sentence classification logits of shape [batch_size, 2].
#     Example usage:
#     ```python
#     # Already been converted into WordPiece token ids
#     input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#     input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#     token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
#     config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
#     model = BertForPreTraining(config)
#     masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
#     ```
#     """

#     def __init__(self, config):
#         super(BertForPreTraining, self).__init__(config)
#         self.bert = BertModel(config)
#         self.cls = BertPreTrainingHeads(
#             config, self.bert.embeddings.word_embeddings.weight)
#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids, token_type_ids, attention_mask):
#         encoded_layers, pooled_output = self.bert(
#             input_ids, token_type_ids, attention_mask)
#         sequence_output = encoded_layers[-1]
#         prediction_scores, seq_relationship_score = self.cls(
#             sequence_output, pooled_output)

#         return prediction_scores, seq_relationship_score


# class BertForMaskedLM(BertPreTrainedModel):
#     """BERT model with the masked language modeling head.
#     This module comprises the BERT model followed by the masked language modeling head.
#     Params:
#         config: a BertConfig class instance with the configuration to build a new model.
#     Inputs:
#         `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
#             with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
#             `extract_features.py`, `run_classifier.py` and `run_squad.py`)
#         `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
#             types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
#             a `sentence B` token (see BERT paper for more details).
#         `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
#             selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#             input sequence length in the current batch. It's the mask that we typically use for attention when
#             a batch has varying length sentences.
#         `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
#             with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
#             is only computed for the labels set in [0, ..., vocab_size]
#     Outputs:
#         if `masked_lm_labels` is  not `None`:
#             Outputs the masked language modeling loss.
#         if `masked_lm_labels` is `None`:
#             Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].
#     Example usage:
#     ```python
#     # Already been converted into WordPiece token ids
#     input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#     input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#     token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
#     config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
#     model = BertForMaskedLM(config)
#     masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
#     ```
#     """

#     def __init__(self, config):
#         super(BertForMaskedLM, self).__init__(config)
#         self.bert = BertModel(config)
#         self.cls = BertOnlyMLMHead(
#             config, self.bert.embeddings.word_embeddings.weight)
#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
#         encoded_layers, _ = self.bert(
#             input_ids, token_type_ids, attention_mask)
#         sequence_output = encoded_layers[-1]
#         prediction_scores = self.cls(sequence_output)

#         if masked_lm_labels is not None:
#             loss_fct = CrossEntropyLoss(ignore_index=-1)
#             masked_lm_loss = loss_fct(
#                 prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
#             return masked_lm_loss
#         else:
#             return prediction_scores


# class BertForNextSentencePrediction(BertPreTrainedModel):
#     """BERT model with next sentence prediction head.
#     This module comprises the BERT model followed by the next sentence classification head.
#     Params:
#         config: a BertConfig class instance with the configuration to build a new model.
#     Inputs:
#         `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
#             with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
#             `extract_features.py`, `run_classifier.py` and `run_squad.py`)
#         `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
#             types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
#             a `sentence B` token (see BERT paper for more details).
#         `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
#             selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#             input sequence length in the current batch. It's the mask that we typically use for attention when
#             a batch has varying length sentences.
#         `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
#             with indices selected in [0, 1].
#             0 => next sentence is the continuation, 1 => next sentence is a random sentence.
#     Outputs:
#         if `next_sentence_label` is not `None`:
#             Outputs the total_loss which is the sum of the masked language modeling loss and the next
#             sentence classification loss.
#         if `next_sentence_label` is `None`:
#             Outputs the next sentence classification logits of shape [batch_size, 2].
#     Example usage:
#     ```python
#     # Already been converted into WordPiece token ids
#     input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#     input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#     token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
#     config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
#     model = BertForNextSentencePrediction(config)
#     seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
#     ```
#     """

#     def __init__(self, config):
#         super(BertForNextSentencePrediction, self).__init__(config)
#         self.bert = BertModel(config)
#         self.cls = BertOnlyNSPHead(config)
#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
#         _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
#         seq_relationship_score = self.cls(pooled_output)

#         if next_sentence_label is not None:
#             loss_fct = CrossEntropyLoss(ignore_index=-1)
#             next_sentence_loss = loss_fct(
#                 seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
#             return next_sentence_loss
#         else:
#             return seq_relationship_score


# class BertForSequenceClassification(BertPreTrainedModel):
#     """BERT model for classification.
#     This module is composed of the BERT model with a linear layer on top of
#     the pooled output.
#     Params:
#         `config`: a BertConfig class instance with the configuration to build a new model.
#         `num_labels`: the number of classes for the classifier. Default = 2.
#     Inputs:
#         `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
#             with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
#             `extract_features.py`, `run_classifier.py` and `run_squad.py`)
#         `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
#             types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
#             a `sentence B` token (see BERT paper for more details).
#         `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
#             selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#             input sequence length in the current batch. It's the mask that we typically use for attention when
#             a batch has varying length sentences.
#         `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
#             with indices selected in [0, ..., num_labels].
#     Outputs:
#         if `labels` is not `None`:
#             Outputs the CrossEntropy classification loss of the output with the labels.
#         if `labels` is `None`:
#             Outputs the classification logits of shape [batch_size, num_labels].
#     Example usage:
#     ```python
#     # Already been converted into WordPiece token ids
#     input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#     input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#     token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
#     config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
#     num_labels = 2
#     model = BertForSequenceClassification(config, num_labels)
#     logits = model(input_ids, token_type_ids, input_mask)
#     ```
#     """

#     def __init__(self, config, num_labels):
#         super(BertForSequenceClassification, self).__init__(config)
#         self.num_labels = num_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None):
#         _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
#         pooled_output = self.dropout(pooled_output)
#         return self.classifier(pooled_output)


# class BertForMultipleChoice(BertPreTrainedModel):
#     """BERT model for multiple choice tasks.
#     This module is composed of the BERT model with a linear layer on top of
#     the pooled output.
#     Params:
#         `config`: a BertConfig class instance with the configuration to build a new model.
#         `num_choices`: the number of classes for the classifier. Default = 2.
#     Inputs:
#         `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
#             with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
#             `extract_features.py`, `run_classifier.py` and `run_squad.py`)
#         `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
#             with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
#             and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
#         `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
#             selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#             input sequence length in the current batch. It's the mask that we typically use for attention when
#             a batch has varying length sentences.
#         `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
#             with indices selected in [0, ..., num_choices].
#     Outputs:
#         if `labels` is not `None`:
#             Outputs the CrossEntropy classification loss of the output with the labels.
#         if `labels` is `None`:
#             Outputs the classification logits of shape [batch_size, num_labels].
#     Example usage:
#     ```python
#     # Already been converted into WordPiece token ids
#     input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
#     input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
#     token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
#     config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
#     num_choices = 2
#     model = BertForMultipleChoice(config, num_choices)
#     logits = model(input_ids, token_type_ids, input_mask)
#     ```
#     """

#     def __init__(self, config, num_choices):
#         super(BertForMultipleChoice, self).__init__(config)
#         self.num_choices = num_choices
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         flat_input_ids = input_ids.view(-1, input_ids.size(-1))
#         flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
#         flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
#         _, pooled_output = self.bert(
#             flat_input_ids, flat_token_type_ids, flat_attention_mask)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#         reshaped_logits = logits.view(-1, self.num_choices)

#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(reshaped_logits, labels)
#             return loss
#         else:
#             return reshaped_logits


# class BertForTokenClassification(BertPreTrainedModel):
#     """BERT model for token-level classification.
#     This module is composed of the BERT model with a linear layer on top of
#     the full hidden state of the last layer.
#     Params:
#         `config`: a BertConfig class instance with the configuration to build a new model.
#         `num_labels`: the number of classes for the classifier. Default = 2.
#     Inputs:
#         `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
#             with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
#             `extract_features.py`, `run_classifier.py` and `run_squad.py`)
#         `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
#             types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
#             a `sentence B` token (see BERT paper for more details).
#         `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
#             selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#             input sequence length in the current batch. It's the mask that we typically use for attention when
#             a batch has varying length sentences.
#         `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
#             with indices selected in [0, ..., num_labels].
#     Outputs:
#         if `labels` is not `None`:
#             Outputs the CrossEntropy classification loss of the output with the labels.
#         if `labels` is `None`:
#             Outputs the classification logits of shape [batch_size, sequence_length, num_labels].
#     Example usage:
#     ```python
#     # Already been converted into WordPiece token ids
#     input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#     input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#     token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
#     config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
#     num_labels = 2
#     model = BertForTokenClassification(config, num_labels)
#     logits = model(input_ids, token_type_ids, input_mask)
#     ```
#     """

#     def __init__(self, config, num_labels):
#         super(BertForTokenClassification, self).__init__(config)
#         self.num_labels = num_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         encoded_layers, _ = self.bert(
#             input_ids, token_type_ids, attention_mask)
#         sequence_output = encoded_layers[-1]
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)

#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             # Only keep active parts of the loss
#             if attention_mask is not None:
#                 active_loss = attention_mask.view(-1) == 1
#                 active_logits = logits.view(-1, self.num_labels)[active_loss]
#                 active_labels = labels.view(-1)[active_loss]
#                 loss = loss_fct(active_logits, active_labels)
#             else:
#                 loss = loss_fct(
#                     logits.view(-1, self.num_labels), labels.view(-1))
#             return loss
#         else:
#             return logits


# class BertForQuestionAnswering(BertPreTrainedModel):
#     """BERT model for Question Answering (span extraction).
#     This module is composed of the BERT model with a linear layer on top of
#     the sequence output that computes start_logits and end_logits
#     Params:
#         `config`: a BertConfig class instance with the configuration to build a new model.
#     Inputs:
#         `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
#             with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
#             `extract_features.py`, `run_classifier.py` and `run_squad.py`)
#         `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
#             types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
#             a `sentence B` token (see BERT paper for more details).
#         `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
#             selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#             input sequence length in the current batch. It's the mask that we typically use for attention when
#             a batch has varying length sentences.
#     Outputs:
#          Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
#          position tokens of shape [batch_size, sequence_length].
#     Example usage:
#     ```python
#     # Already been converted into WordPiece token ids
#     input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#     input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#     token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
#     config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
#     model = BertForQuestionAnswering(config)
#     start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
#     ```
#     """

#     def __init__(self, config):
#         super(BertForQuestionAnswering, self).__init__(config)
#         self.bert = BertModel(config)
#         # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
#         # self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.qa_outputs = nn.Linear(config.hidden_size, 2)
#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids, token_type_ids, attention_mask):
#         encoded_layers, _ = self.bert(
#             input_ids, token_type_ids, attention_mask)
#         sequence_output = encoded_layers[-1]
#         logits = self.qa_outputs(sequence_output)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)
#         return start_logits, end_logits
