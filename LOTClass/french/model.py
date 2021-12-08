# =================================================================================
#from transformers import BertModel
#from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers import CamembertModel
from transformers import RobertaPreTrainedModel
from transformers import CamembertTokenizer
from transformers.models.camembert.modeling_camembert import CamembertForMaskedLM
# =================================================================================
from torch import nn, zeros
import sys


# class CamembertPredictionHeadTransform(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         if isinstance(config.hidden_act, str):
#             self.transform_act_fn = ACT2FN[config.hidden_act]
#         else:
#             self.transform_act_fn = config.hidden_act
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

#     def forward(self, hidden_states):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.transform_act_fn(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states)
#         return hidden_states


# class CamembertLMPredictionHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.transform = CamembertPredictionHeadTransform(config)

#         # The output weights are the same as the input embeddings, but there is
#         # an output-only bias for each token.
#         self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         self.bias = nn.Parameter(zeros(config.vocab_size))

#         # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
#         self.decoder.bias = self.bias

#     def forward(self, hidden_states):
#         hidden_states = self.transform(hidden_states)
#         hidden_states = self.decoder(hidden_states)
#         return hidden_states


class CamembertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = CamembertForMaskedLM.from_pretrained("camembert-base")
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.topk = 50 # Maximum number of predictions kept
        self.model.eval()

    def forward(self, input_ids):
        print("MLMHead input shape:", input_ids.size())
        logits = self.model(input_ids)[0]  # The last hidden-state is the first element of the output tuple
        masked_index = (input_ids.squeeze() == self.tokenizer.mask_token_id).nonzero().item()
        logits = logits[0, masked_index, :]
        prob = logits.softmax(dim=0)
        return prob.topk(k=self.topk, dim=0)


class LOTClassModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        #self.bert = BertModel(config, add_pooling_layer=False)
        self.bert = CamembertModel(config, add_pooling_layer=False)
        #self.cls = BertOnlyMLMHead(config)
        self.cls = CamembertOnlyMLMHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        # MLM head is not trained
        for param in self.cls.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, pred_mode, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None):
        bert_outputs = self.bert(input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 head_mask=head_mask,
                                 inputs_embeds=inputs_embeds)
        last_hidden_states = bert_outputs[0]
        if pred_mode == "classification":
            trans_states = self.dense(last_hidden_states)
            trans_states = self.activation(trans_states)
            trans_states = self.dropout(trans_states)
            logits = self.classifier(trans_states)
        elif pred_mode == "mlm":
            logits = self.cls(last_hidden_states)
        else:
            sys.exit("Wrong pred_mode!")
        print(f"Model output size: {logits.size()}")
        print(f"Model output type: {type(logits)}")
        return logits
